"""Phase 2.1 full baseline experiment.

Dev tuning (train partition) -> frozen config -> full validation eval.
Metrics: P@k/R@k, MRR, nDCG@k, complete annotated-evidence recovery,
partial coverage, token usage, latency, oracle bounds, bootstrap CIs.
"""

import json
import logging
import math
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
from scientific_rag.retrieval.evidence import is_non_textual_evidence, normalize_text

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
ENC = tiktoken.get_encoding("cl100k_base")
RESULTS = Path("results")
SEED = 42
BUDGETS = [256, 512, 1024, 2048, 4096]
TOPK = [1, 3, 5, 10, 20]


def passage_id_str(p):
    return str(p.passage_id)


def map_evidence_to_ids(paper, passages):
    """Strict evidence->passage-id mapping per annotation.

    Returns (valid_sets, n_excluded_annotations, stats).
    Rule: an annotation's textual evidence set is valid iff every
    non-textual-filtered evidence string maps to >=1 passage (first match
    wins for id assignment; ambiguity counted). Annotations with zero
    textual evidence are excluded. Alternative sets never merged.
    """
    norm_passages = [(normalize_text(p.text), passage_id_str(p)) for p in passages]
    valid_sets = []
    n_annot = n_excluded = 0
    cat = defaultdict(int)
    for qa in paper.qas:
        for ann in qa.answers:
            n_annot += 1
            ev_all = ann.answer.evidence + ann.answer.highlighted_evidence
            textual = [e for e in ev_all if not is_non_textual_evidence(e)]
            n_nontext = len(ev_all) - len(textual)
            cat["nontext_evidence"] += n_nontext
            if not textual:
                n_excluded += 1
                cat["excluded_no_textual"] += 1
                continue
            ids, all_mapped = set(), True
            for e in textual:
                ne = normalize_text(e)
                hit = None
                # exact normalized-equality first, then containment
                for nt, pid in norm_passages:
                    if nt == ne:
                        hit = pid
                        break
                if hit is None:
                    for nt, pid in norm_passages:
                        if ne and ne in nt:
                            hit = pid
                            break
                if hit:
                    ids.add(hit)
                else:
                    all_mapped = False
                    cat["unmapped_evidence"] += 1
            if ids:
                valid_sets.append(frozenset(ids))
                if not all_mapped:
                    cat["partial_mapping_sets"] += 1
            else:
                n_excluded += 1
                cat["excluded_no_mapped"] += 1
    return valid_sets, n_excluded, dict(cat)


def question_eligible(qa_gold_sets):
    return len(qa_gold_sets) > 0


def dcg(rels):
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def metrics_for(ranked_ids, gold_sets, k):
    """Metrics at rank/budget cutoff. gold_sets: list[frozenset[str]]."""
    if not gold_sets:
        return None
    rel = set().union(*gold_sets)
    top = ranked_ids[:k]
    hits = [1 if pid in rel else 0 for pid in top]
    inter = sum(hits)
    prec = inter / k if k else 0.0
    rec = inter / len(rel) if rel else 0.0
    # MRR: first position (1-indexed) containing any gold passage
    mrr = 0.0
    for i, pid in enumerate(top):
        if pid in rel:
            mrr = 1.0 / (i + 1)
            break
    idcg = dcg([1] * min(k, len(rel)))
    ndcg = dcg(hits) / idcg if idcg > 0 else 0.0
    ret_set = set(top)
    complete = any(g <= ret_set for g in gold_sets if g)
    partial = max((len(g & ret_set) / len(g) for g in gold_sets if g), default=0.0)
    return {
        "p": prec,
        "r": rec,
        "mrr": mrr,
        "ndcg": ndcg,
        "complete": float(complete),
        "partial": partial,
        "n_rel": len(rel),
    }


def oracle_fits(gold_sets, budget):
    """True if any complete gold set fits under budget.

    Packing simulation: sort each set's passages by token_count asc, greedily
    add while serialized estimate (sum tok + 2*(n-1)) <= budget. Assumption:
    oracle knows gold passage texts; order optimized greedily by size.
    """
    for g in gold_sets:
        if not g:
            continue
        toks = sorted(gold_tok.get(pid, 10**9) for pid in g)
        if any(t == 10**9 for t in toks):
            continue
        used = 0
        ok = True
        for i, t in enumerate(toks):
            cost = t + (2 if i else 0)
            if used + cost > budget:
                ok = False
                break
            used += cost
        if ok:
            return True
    return False


gold_tok: dict[str, int] = {}  # passage_id_str -> token_count, set per paper in eval loop


def eval_split(papers, methods_cfg, tag, out_rows):
    """Evaluate methods on papers; append per-question rows."""
    stats = {"papers": 0, "questions": 0, "eligible": 0, "excluded": 0}
    t_start = time.time()
    for pi, paper in enumerate(papers):
        passages = build_passages(paper, ENC)
        if not passages:
            continue
        stats["papers"] += 1
        for p in passages:
            gold_tok[passage_id_str(p)] = p.token_count
        # gold sets per question (strict mapping, per-annotation)
        qgold = {}
        for qa in paper.qas:
            sets = []
            for ann in qa.answers:
                ev_all = ann.answer.evidence + ann.answer.highlighted_evidence
                textual = [e for e in ev_all if not is_non_textual_evidence(e)]
                if not textual:
                    continue
                ids = set()
                all_mapped = True
                for e in textual:
                    ne = normalize_text(e)
                    hit = None
                    for p in passages:
                        if normalize_text(p.text) == ne:
                            hit = passage_id_str(p)
                            break
                    if hit is None:
                        for p in passages:
                            if ne and ne in normalize_text(p.text):
                                hit = passage_id_str(p)
                                break
                    if hit:
                        ids.add(hit)
                    else:
                        all_mapped = False
                # an annotation is eligible only if ALL its textual evidence
                # mapped; unmapped components are never silently dropped
                if ids and all_mapped:
                    sets.append(frozenset(ids))
            qgold[qa.question_id] = sets

        bm25 = BM25Retriever(passages)
        dense = DenseRetriever(passages)
        for qa in paper.qas:
            gold_sets = qgold[qa.question_id]
            stats["questions"] += 1
            if not gold_sets:
                stats["excluded"] += 1
                continue
            stats["eligible"] += 1
            q = qa.question_text
            t0 = time.perf_counter()
            sp = bm25.search(q, k=len(passages))
            dn = (
                dense.search_full(q)
                if hasattr(dense, "search_full")
                else dense.search(q, k=len(passages))
            )
            lat_ms = (time.perf_counter() - t0) * 1000
            ranked_variants = {
                "bm25": [p for p, _ in sp],
                "dense": [p for p, _ in dn],
            }
            if "hybrid" in methods_cfg:
                ranked_variants["hybrid"] = [
                    p
                    for p, _ in hybrid_search(
                        sp, dn, methods_cfg["alpha"], k=len(passages), pool_size=len(passages)
                    )
                ]
            if "rrf" in methods_cfg:
                ranked_variants["rrf"] = [
                    p
                    for p, _ in rrf_search(
                        [sp, dn], k=len(passages), rank_constant=methods_cfg.get("rrf_k", 60)
                    )
                ]
            for name, plist in ranked_variants.items():
                # top-k modes
                for k in TOPK:
                    ids = [passage_id_str(p) for p in plist[:k]]
                    m = metrics_for(ids, gold_sets, k)
                    out_rows.append(
                        {
                            "split": tag,
                            "paper": paper.id,
                            "qid": qa.question_id,
                            "method": name,
                            "mode": "topk",
                            "k": k,
                            "latency_ms": lat_ms,
                            **(m or {}),
                        }
                    )
                # budget modes: pack greedily by rank
                for b in BUDGETS:
                    sel, used = pack_budget([(p, 0.0) for p in plist], b, ENC)
                    ids = [passage_id_str(p) for p in sel]
                    # verify serialized token count
                    if sel:
                        ser = "\n\n".join(p.text for p in sel)
                        exact_tok = len(ENC.encode(ser))
                    else:
                        exact_tok = 0
                    m = metrics_for(ids, gold_sets, len(ids))
                    fits = oracle_fits(gold_sets, b)
                    out_rows.append(
                        {
                            "split": tag,
                            "paper": paper.id,
                            "qid": qa.question_id,
                            "method": name,
                            "mode": "budget",
                            "budget": b,
                            "tokens_used": used,
                            "exact_tokens": exact_tok,
                            "n_sel": len(sel),
                            "oracle_possible": fits,
                            "latency_ms": lat_ms,
                            **(m or {}),
                        }
                    )
        if (pi + 1) % 25 == 0:
            logger.info("%s: %d/%d papers", tag, pi + 1, len(papers))
    stats["runtime_s"] = round(time.time() - t_start, 1)
    return stats


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["method"], r.get("k") or r.get("budget"), r["mode"])
        for m in (
            "p",
            "r",
            "mrr",
            "ndcg",
            "complete",
            "partial",
            "latency_ms",
            "tokens_used",
            "exact_tokens",
        ):
            if m in r:
                agg[key][m].append(r[m])
    out = {}
    for key, d in agg.items():
        out[f"{key[0]}|{key[1]}|{key[2]}"] = {
            m: {"mean": float(np.mean(v)), "median": float(np.median(v))} for m, v in d.items() if v
        }
    return out


def bootstrap_ci(rows, metric="complete", budget=1024, n_boot=2000):
    """Paired paper-cluster bootstrap for method comparisons."""
    by_paper = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("mode") == "budget" and r.get("budget") == budget and metric in r:
            by_paper[r["paper"]][r["method"]].append(r[metric])
    papers = sorted(by_paper)
    methods = sorted({m for p in by_paper.values() for m in p})
    paper_means = {}
    for p in papers:
        paper_means[p] = {m: float(np.mean(by_paper[p][m])) for m in methods if m in by_paper[p]}
    rng = np.random.default_rng(SEED)
    cis = {}
    pairs = [("rrf", "dense"), ("hybrid", "dense"), ("rrf", "hybrid"), ("bm25", "dense")]
    for a, b in pairs:
        if a not in methods or b not in methods:
            continue
        diffs = []
        for _ in range(n_boot):
            sample = rng.choice(papers, size=len(papers), replace=True)
            da = [paper_means[p][a] for p in sample if a in paper_means[p]]
            db = [paper_means[p][b] for p in sample if b in paper_means[p]]
            if da and db:
                diffs.append(np.mean(da) - np.mean(db))
        if diffs:
            cis[f"{a}_vs_{b}"] = {
                "diff": float(np.mean(diffs)),
                "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
            }
    return cis


def main():
    RESULTS.mkdir(exist_ok=True)
    # deterministic dev partition: 60 train papers, seed 42
    train = load_dataset("train")
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(train.papers), size=min(60, len(train.papers)), replace=False)
    dev_papers = [train.papers[i] for i in sorted(idx)]
    dev_ids = sorted(p.id for p in dev_papers)
    (RESULTS / "dev_partition.json").write_text(
        json.dumps({"seed": SEED, "n": len(dev_papers), "paper_ids": dev_ids})
    )

    val = load_dataset("validation")

    # ---- dev tuning: alpha sweep + depth + rrf constant at budget 1024 ----
    print("=== DEV TUNING (60 train papers) ===", flush=True)
    # alpha sweep (pool=all, rrf_k=60)
    alpha_scores = {}
    for alpha in [i / 10 for i in range(11)]:
        rows = []
        eval_split(dev_papers, {"hybrid": True, "alpha": alpha}, "dev", rows)
        comp = [
            r["complete"] for r in rows if r.get("mode") == "budget" and r.get("budget") == 1024
        ]
        alpha_scores[alpha] = float(np.mean(comp)) if comp else 0.0
        print(f"alpha={alpha}: complete@1024={alpha_scores[alpha]:.4f}", flush=True)
    # tie-break: max recovery; ties -> prefer larger alpha (dense endpoint)
    best_alpha = max(alpha_scores, key=lambda a: (alpha_scores[a], a))
    print("selected alpha:", best_alpha)

    # RRF constant sensitivity
    rrf_scores = {}
    for rk in [20, 60, 100]:
        rows = []
        eval_split(dev_papers, {"rrf": True, "rrf_k": rk}, "dev", rows)
        comp = [
            r["complete"] for r in rows if r.get("mode") == "budget" and r.get("budget") == 1024
        ]
        rrf_scores[rk] = float(np.mean(comp)) if comp else 0.0
        print(f"rrf_k={rk}: {rrf_scores[rk]:.4f}", flush=True)
    best_rrf_k = max(rrf_scores, key=lambda k: (rrf_scores[k], -abs(k - 60)))

    manifest = {
        "seed": SEED,
        "alpha": best_alpha,
        "rrf_k": best_rrf_k,
        "alpha_scores": {str(k): v for k, v in alpha_scores.items()},
        "rrf_scores": rrf_scores,
        "dev_papers": len(dev_papers),
    }
    (RESULTS / "tuning_manifest.json").write_text(json.dumps(manifest, indent=2))

    # ---- full validation eval with frozen config ----
    print("=== VALIDATION EVAL (281 papers) ===", flush=True)
    val_rows = []
    val_stats = eval_split(
        val.papers,
        {"hybrid": True, "alpha": best_alpha, "rrf": True, "rrf_k": best_rrf_k},
        "validation",
        val_rows,
    )
    print("validation stats:", val_stats)

    agg = aggregate(val_rows)
    cis = bootstrap_ci(val_rows)

    out = {
        "config": manifest,
        "validation_stats": val_stats,
        "aggregate": agg,
        "bootstrap_1024": cis,
    }
    (RESULTS / "phase2_full_results.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "phase2_per_question.json").write_text(json.dumps(val_rows))
    print("DONE. aggregate keys:", len(agg))


if __name__ == "__main__":
    main()
