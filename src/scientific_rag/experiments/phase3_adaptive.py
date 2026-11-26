"""Phase 3: adaptive stopping trajectories + sufficiency diagnostics.

For each eligible validation question and each ranking method, simulate
rank-order acquisition under the shared packing policy and record per-step
features + complete-recovery labels. Offline policies (threshold, margin,
fixed-k, fixed-budget) are evaluated from trajectories — no re-retrieval.

Gold labels are used ONLY for offline supervision/evaluation.
"""

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
    rrf_search,
)
from scientific_rag.retrieval.evidence import is_non_textual_evidence, normalize_text

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
ENC = tiktoken.get_encoding("cl100k_base")
RESULTS = Path("results")
SEED = 42
BUDGETS = [256, 512, 1024, 2048, 4096]
ALPHA = 0.9
RRF_K = 20
DEVICE = "cuda"  # verified RTX 4060


def pid(p):
    return str(p.passage_id)


def gold_sets_for(_paper, passages, qa):
    """Strict per-annotation evidence sets; ALL textual evidence must map."""
    norm = [(normalize_text(p.text), pid(p)) for p in passages]
    sets = []
    for ann in qa.answers:
        textual = [
            e
            for e in (ann.answer.evidence + ann.answer.highlighted_evidence)
            if not is_non_textual_evidence(e)
        ]
        if not textual:
            continue
        ids, ok = set(), True
        for e in textual:
            ne = normalize_text(e)
            hit = next((i for nt, i in norm if nt == ne), None)
            if hit is None:
                hit = next((i for nt, i in norm if ne and ne in nt), None)
            if hit:
                ids.add(hit)
            else:
                ok = False
        if ids and ok:
            sets.append(frozenset(ids))
    return sets


def complete_at(sel_ids, gold_sets):
    s = set(sel_ids)
    return any(g <= s for g in gold_sets if g)


def partial_at(sel_ids, gold_sets):
    s = set(sel_ids)
    return max((len(g & s) / len(g) for g in gold_sets if g), default=0.0)


def simulate(ranked, gold_sets, max_steps=25):
    """Rank-order acquisition; record per-step features + labels.

    ranked: list[(Passage, score)] sorted by rank.
    Returns list of step dicts (1-indexed steps up to max_steps or budget 4096).
    """
    steps = []
    sel, sel_ids = [], []
    used = 0
    scores = [s for _, s in ranked]
    for i, (p, sc) in enumerate(ranked[:max_steps]):
        cost = p.token_count + (2 if sel else 0)
        if used + cost > 4096:
            break
        sel.append(p)
        sel_ids.append(pid(p))
        used += cost
        top = scores[0] if scores else 0.0
        second = scores[1] if len(scores) > 1 else 0.0
        prev = scores[i - 1] if i > 0 else top
        steps.append(
            {
                "step": i + 1,
                "passage_id": pid(p),
                "score": float(sc),
                "top_score": float(top),
                "margin": float(top - second),
                "score_decay": float(prev - sc),
                "rel_drop": float((prev - sc) / abs(prev)) if prev else 0.0,
                "n_selected": len(sel),
                "tokens": used,
                "complete": float(complete_at(sel_ids, gold_sets)),
                "partial": float(partial_at(sel_ids, gold_sets)),
            }
        )
    return steps


def stop_point(steps, policy, param):
    """Return index of last step included under policy, or None."""
    if policy == "fixed_k":
        return min(param, len(steps))
    if policy == "score_threshold":
        for i, st in enumerate(steps):
            if st["score"] < param:
                return i
        return len(steps)
    if policy == "margin":
        for i, st in enumerate(steps):
            if st["margin"] > param:
                return i + 1 if i > 0 else 1
        return len(steps)
    if policy == "rel_drop":
        for i, st in enumerate(steps):
            if i > 0 and st["rel_drop"] > param:
                return i
        return len(steps)
    if policy == "budget":
        n = 0
        for st in steps:
            if st["tokens"] <= param:
                n += 1
        return n
    return len(steps)


def auroc(scores, labels):
    """Rank-based AUROC."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = np.array(labels, dtype=bool)
    npos, nneg = pos.sum(), (~pos).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auprc(scores, labels):
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels, dtype=int)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    rec = tp / max(y.sum(), 1)
    return float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))


def main():
    RESULTS.mkdir(exist_ok=True)
    val = load_dataset("validation")
    policies = {
        "fixed_k": [1, 3, 5, 10, 20],
        "budget": BUDGETS,
        "score_threshold": [0.3, 0.4, 0.5, 0.6, 0.7],
        "margin": [0.05, 0.1, 0.2, 0.3],
        "rel_drop": [0.3, 0.5, 0.7],
    }
    trajectories = []
    eval_rows = []
    t0 = time.time()
    for pi, paper in enumerate(val.papers):
        passages = build_passages(paper, ENC)
        if not passages:
            continue
        bm25 = BM25Retriever(passages)
        dense = DenseRetriever(passages, device=DEVICE, batch_size=32)
        for qa in paper.qas:
            gold = gold_sets_for(paper, passages, qa)
            if not gold:
                continue
            q = qa.question_text
            sp = bm25.search(q, k=len(passages))
            dn = dense.search(q, k=len(passages))
            hyb = hybrid_search(sp, dn, ALPHA, k=len(passages), pool_size=len(passages))
            rrf = rrf_search([sp, dn], k=len(passages), rank_constant=RRF_K)
            for mname, ranked in [("bm25", sp), ("dense", dn), ("hybrid", hyb), ("rrf", rrf)]:
                steps = simulate(ranked, gold)
                for st in steps:
                    st2 = dict(st)
                    st2.update({"paper": paper.id, "qid": qa.question_id, "method": mname})
                    trajectories.append(st2)
                # evaluate policies
                for pol, params in policies.items():
                    for prm in params:
                        n = stop_point(steps, pol, prm)
                        if n == 0:
                            continue
                        last = steps[n - 1]
                        eval_rows.append(
                            {
                                "paper": paper.id,
                                "qid": qa.question_id,
                                "method": mname,
                                "policy": pol,
                                "param": prm,
                                "n_sel": n,
                                "tokens": last["tokens"],
                                "complete": last["complete"],
                                "partial": last["partial"],
                                "top_score": last["top_score"],
                                "margin": last["margin"],
                            }
                        )
        if (pi + 1) % 50 == 0:
            print(f"{pi + 1}/281 papers, {time.time() - t0:.0f}s", flush=True)

    # aggregate: complete recovery + mean tokens per policy/method
    agg = defaultdict(lambda: {"complete": [], "tokens": [], "partial": []})
    for r in eval_rows:
        key = (r["method"], r["policy"], r["param"])
        agg[key]["complete"].append(r["complete"])
        agg[key]["tokens"].append(r["tokens"])
        agg[key]["partial"].append(r["partial"])
    summary = {
        f"{k[0]}|{k[1]}|{k[2]}": {
            "complete": float(np.mean(v["complete"])),
            "mean_tokens": float(np.mean(v["tokens"])),
            "mean_partial": float(np.mean(v["partial"])),
            "n": len(v["complete"]),
        }
        for k, v in agg.items()
    }

    # sufficiency diagnostics: features at each trajectory step -> complete label
    feats = defaultdict(lambda: {"s": [], "y": []})
    for st in trajectories:
        for f in ("top_score", "margin", "score_decay", "rel_drop", "n_selected", "tokens"):
            feats[f]["s"].append(st[f])
            feats[f]["y"].append(st["complete"])
    diag = {}
    for f, d in feats.items():
        diag[f] = {
            "auroc": auroc(d["s"], d["y"]),
            "auprc": auprc(d["s"], d["y"]),
            "prevalence": float(np.mean(d["y"])),
            "n": len(d["y"]),
        }

    out = {
        "config": {
            "alpha": ALPHA,
            "rrf_k": RRF_K,
            "device": DEVICE,
            "budgets": BUDGETS,
            "seed": SEED,
        },
        "runtime_s": round(time.time() - t0, 1),
        "n_questions": len({(r["paper"], r["qid"]) for r in eval_rows}),
        "policy_summary": summary,
        "sufficiency_diagnostics": diag,
    }
    (RESULTS / "phase3_adaptive.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "phase3_trajectories.json").write_text(json.dumps(trajectories))
    (RESULTS / "phase3_eval_rows.json").write_text(json.dumps(eval_rows))
    print("DONE", out["n_questions"], "questions,", out["runtime_s"], "s")
    for k, v in diag.items():
        print(f"{k}: AUROC={v['auroc']:.3f} AUPRC={v['auprc']:.3f}")


if __name__ == "__main__":
    main()
