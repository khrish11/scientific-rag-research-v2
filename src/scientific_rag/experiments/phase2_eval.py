"""Evidence evaluation + budgeted experiment runner (Phase 2)."""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

from scientific_rag.data.dataset import load_dataset
from scientific_rag.retrieval.baselines import (
    BM25Retriever,
    DenseRetriever,
    build_passages,
    hybrid_search,
    pack_budget,
    rrf_search,
)
from scientific_rag.retrieval.evidence import (
    is_non_textual_evidence,
    normalize_text,
)

logger = logging.getLogger(__name__)
ENC = tiktoken.get_encoding("cl100k_base")

BUDGETS = [256, 512, 1024, 2048, 4096]
TOPK = [1, 3, 5, 10, 20]
ALPHAS = [i / 10 for i in range(11)]


def gold_evidence_sets(paper, passages):
    """Map each annotation's evidence strings to passage IDs (strict mode).

    Returns list of frozensets of passage_id strings; excludes non-textual
    and unmappable evidence (recorded as exclusions)."""
    pid_by_text = {}
    for p in passages:
        pid_by_text.setdefault(normalize_text(p.text), str(p.passage_id))
    sets, excluded = [], 0
    for qa in paper.qas:
        for ann in qa.answers:
            ev = [
                e
                for e in (ann.answer.evidence + ann.answer.highlighted_evidence)
                if not is_non_textual_evidence(e)
            ]
            ids = set()
            ok = True
            for e in ev:
                # strict: exact then normalized-containment match
                found = None
                for p in passages:
                    if p.text == e or normalize_text(p.text) == normalize_text(e):
                        found = str(p.passage_id)
                        break
                if found is None:
                    for p in passages:
                        ne = normalize_text(e)
                        if ne and ne in normalize_text(p.text):
                            found = str(p.passage_id)
                            break
                if found:
                    ids.add(found)
                else:
                    ok = False
            if ids and ok:
                sets.append(frozenset(ids))
            elif ids:
                sets.append(frozenset(ids))  # partial mapping still usable, flagged
                excluded += 1
    return sets


def eval_question(retrieved_ids: set[str], gold_sets: list[frozenset]) -> dict | None:
    """Metrics for one question given retrieved passage-id set."""
    if not gold_sets:
        return None
    rel = set().union(*gold_sets) if gold_sets else set()
    inter = retrieved_ids & rel
    prec = len(inter) / len(retrieved_ids) if retrieved_ids else 0.0
    rec = len(inter) / len(rel) if rel else 0.0
    complete = any(g and g <= retrieved_ids for g in gold_sets)
    partial = any(len(g & retrieved_ids) / len(g) for g in gold_sets if g) if gold_sets else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "complete": complete,
        "partial": max(partial, 0.0),
        "n_rel": len(rel),
    }


def run_subset(
    papers,
    budgets=BUDGETS,
    topk=TOPK,
    alphas=ALPHAS,
    out_path="results/phase2_subset.json",
):
    results = {
        "config": {
            "budgets": budgets,
            "topk": topk,
            "alphas": alphas,
            "tokenizer": "cl100k_base",
            "bm25": {"k1": 1.5, "b": 0.75},
            "dense_model": "sentence-transformers/all-MiniLM-L6-v2",
        },
        "questions": [],
        "aggregate": {},
    }
    # alpha selection on first half of papers (dev), eval on second half
    dev_papers, eval_papers = papers[: len(papers) // 2], papers[len(papers) // 2 :]

    def process(paper_list, alpha=None):
        rows = []
        for paper in paper_list:
            passages = build_passages(paper, ENC)
            if not passages:
                continue
            bm25 = BM25Retriever(passages)
            dense = DenseRetriever(passages)
            gold = gold_evidence_sets(paper, passages)
            for qa in paper.qas:
                q = qa.question_text or qa.question
                t0 = time.perf_counter()
                sp = bm25.search(q, k=20)
                dn = dense.search(q, k=20)
                lat_index = (time.perf_counter() - t0) * 1000
                variants = {
                    "bm25": sp,
                    "dense": dn,
                    "rrf": rrf_search([sp, dn]),
                }
                if alpha is not None:
                    variants["hybrid"] = hybrid_search(sp, dn, alpha)
                for name, ranked in variants.items():
                    for k in topk:
                        ids = {str(p.passage_id) for p, _ in ranked[:k]}
                        m = eval_question(ids, gold)
                        if m:
                            rows.append(
                                {
                                    "paper": paper.id,
                                    "qid": qa.question_id,
                                    "method": name,
                                    "k": k,
                                    "mode": "topk",
                                    "latency_ms": round(lat_index, 2),
                                    **m,
                                }
                            )
                    for b in budgets:
                        sel, used = pack_budget(ranked, b, ENC)
                        ids = {str(p.passage_id) for p in sel}
                        m = eval_question(ids, gold)
                        if m:
                            rows.append(
                                {
                                    "paper": paper.id,
                                    "qid": qa.question_id,
                                    "method": name,
                                    "budget": b,
                                    "mode": "budget",
                                    "tokens_used": used,
                                    "n_sel": len(sel),
                                    "latency_ms": round(lat_index, 2),
                                    **m,
                                }
                            )
        return rows

    # alpha sweep on dev papers
    alpha_scores = {}
    for a in alphas:
        rows = process(dev_papers, alpha=a)
        comp = [r for r in rows if r.get("mode") == "budget" and r.get("budget") == 1024]
        alpha_scores[a] = float(np.mean([r["complete"] for r in comp])) if comp else 0.0
    best_alpha = max(alpha_scores, key=alpha_scores.get)
    results["alpha_sweep_dev"] = {str(k): v for k, v in alpha_scores.items()}
    results["selected_alpha"] = best_alpha

    # final eval on held-out papers with all methods
    final = process(eval_papers, alpha=best_alpha)
    results["questions"] = final

    # aggregate
    agg = defaultdict(lambda: defaultdict(list))
    for r in final:
        key = (r["method"], r.get("k") or r.get("budget"), r["mode"])
        agg[key]["complete"].append(r["complete"])
        agg[key]["recall"].append(r["recall"])
        agg[key]["precision"].append(r["precision"])
        agg[key]["partial"].append(r["partial"])
        agg[key]["latency_ms"].append(r["latency_ms"])
        if r["mode"] == "budget":
            agg[key]["tokens_used"].append(r["tokens_used"])
    for key, d in agg.items():
        results["aggregate"][f"{key[0]}|{key[1]}|{key[2]}"] = {
            kk: {
                "mean": float(np.mean(v)),
                "median": float(np.median(v)) if kk == "tokens_used" else None,
            }
            for kk, v in d.items()
            if v
        }

    Path(out_path).parent.mkdir(exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    ds = load_dataset("validation")
    papers = ds.papers[:n]
    print(f"Running Phase-2 subset experiment on {len(papers)} validation papers")
    res = run_subset(papers, out_path="results/phase2_subset.json")
    print("alpha:", res["selected_alpha"], res["alpha_sweep_dev"])
    for k, v in list(res["aggregate"].items())[:10]:
        print(k, v)
