"""Phase 5.1 canonical evaluation: identical packing/eval for all methods,
plus CPU-vs-GPU dense comparison to resolve the 0.6441-vs-0.631 discrepancy.

Canonical rules:
- eligibility: >=1 annotation with fully-mapped nonempty textual evidence
- packing: greedy rank-order, cost = token_count + 2*(n-1) ["\n\n" sep],
  skip-if-not-fit, never exceed budget, no truncation
- aggregation: question-micro over eligible questions
- budgets: 256/512/1024/2048/4096; methods: bm25, dense_gpu, dense_cpu,
  hybrid(0.9), rrf(k=20)
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import tiktoken

from scientific_rag.data.dataset import load_dataset
from scientific_rag.experiments.phase3_adaptive import gold_sets_for
from scientific_rag.retrieval.baselines import (
    BM25Retriever,
    DenseRetriever,
    build_passages,
    hybrid_search,
    pack_budget,
    rrf_search,
)

logging.basicConfig(level=logging.WARNING)
ENC = tiktoken.get_encoding("cl100k_base")
RESULTS = Path("results")
BUDGETS = [256, 512, 1024, 2048, 4096]
ALPHA, RRF_K = 0.9, 20


def pid(p):
    return str(p.passage_id)


def complete(sel_ids, gold):
    s = set(sel_ids)
    return float(any(g <= s for g in gold if g))


def partial(sel_ids, gold):
    s = set(sel_ids)
    return float(max((len(g & s) / len(g) for g in gold if g), default=0.0))


def main():
    t0 = time.time()
    val = load_dataset("validation")
    rows = []
    n_elig = n_exc = 0
    papers_with_q = set()
    for pi, paper in enumerate(val.papers):
        passages = build_passages(paper, ENC)
        if not passages:
            continue
        bm25 = BM25Retriever(passages)
        dg = DenseRetriever(passages, device="cuda", batch_size=32)
        dc = DenseRetriever(passages, device="cpu", batch_size=64)
        for qa in paper.qas:
            gold = gold_sets_for(paper, passages, qa)
            if not gold:
                n_exc += 1
                continue
            n_elig += 1
            papers_with_q.add(paper.id)
            q = qa.question_text
            sp = bm25.search(q, k=len(passages))
            gn = dg.search(q, k=len(passages))
            cn = dc.search(q, k=len(passages))
            hyb = hybrid_search(sp, gn, ALPHA, k=len(passages),
                                pool_size=len(passages))
            rrf = rrf_search([sp, gn], k=len(passages), rank_constant=RRF_K)
            # CPU-vs-GPU ranking agreement
            g10 = [pid(p) for p, _ in gn[:10]]
            c10 = [pid(p) for p, _ in cn[:10]]
            rank_overlap = len(set(g10) & set(c10)) / max(len(g10), 1)
            for mname, ranked in [("bm25", sp), ("dense_gpu", gn),
                                  ("dense_cpu", cn), ("hybrid", hyb),
                                  ("rrf", rrf)]:
                for b in BUDGETS:
                    sel, used = pack_budget(ranked, b)
                    sids = [pid(p) for p in sel]
                    rows.append({
                        "paper": paper.id, "qid": qa.question_id,
                        "method": mname, "budget": b,
                        "complete": complete(sids, gold),
                        "partial": partial(sids, gold),
                        "tokens": used, "n_sel": len(sel),
                        "n_rel": min(len(g) for g in gold if g),
                        "rank_overlap_gpu_cpu": rank_overlap})
        if (pi + 1) % 50 == 0:
            print(f"{pi+1}/281 papers {time.time()-t0:.0f}s", flush=True)

    # aggregates
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["method"], r["budget"])
        agg[k]["complete"].append(r["complete"])
        agg[k]["partial"].append(r["partial"])
        agg[k]["tokens"].append(r["tokens"])
        agg[k]["n_sel"].append(r["n_sel"])
    summary = {f"{m}|{b}": {
        "complete": float(np.mean(v["complete"])),
        "partial": float(np.mean(v["partial"])),
        "mean_tokens": float(np.mean(v["tokens"])),
        "mean_n_sel": float(np.mean(v["n_sel"])),
        "n_q": len(v["complete"])} for (m, b), v in agg.items()}

    # CPU vs GPU per-question diff @1024
    byq = defaultdict(dict)
    for r in rows:
        if r["budget"] == 1024 and r["method"] in ("dense_gpu", "dense_cpu"):
            byq[(r["paper"], r["qid"])][r["method"]] = r
    flips = sum(1 for v in byq.values()
                if v["dense_gpu"]["complete"] != v["dense_cpu"]["complete"])
    gpu_c = np.mean([v["dense_gpu"]["complete"] for v in byq.values()])
    cpu_c = np.mean([v["dense_cpu"]["complete"] for v in byq.values()])
    overlap = float(np.mean([v["dense_gpu"]["rank_overlap_gpu_cpu"]
                             for v in byq.values()]))
    discrepancy = {"n_questions": len(byq), "flips": flips,
                   "dense_gpu_complete": float(gpu_c),
                   "dense_cpu_complete": float(cpu_c),
                   "mean_top10_overlap": overlap}

    # evidence-set-size breakdown (dense_gpu @1024)
    sz = defaultdict(lambda: {"c": [], "p": []})
    for r in rows:
        if r["method"] == "dense_gpu" and r["budget"] == 1024:
            sz[min(r["n_rel"], 5)]["c"].append(r["complete"])
            sz[min(r["n_rel"], 5)]["p"].append(r["partial"])
    setsize = {str(k): {"complete": float(np.mean(v["c"])),
                        "partial": float(np.mean(v["p"])),
                        "n": len(v["c"])} for k, v in sorted(sz.items())}

    out = {"config": {"alpha": ALPHA, "rrf_k": RRF_K, "budgets": BUDGETS,
                      "packing": "token_count+2*(n-1), skip-if-not-fit",
                      "seed": 42},
           "n_eligible": n_elig, "n_excluded": n_exc,
           "n_papers": len(papers_with_q),
           "runtime_s": round(time.time() - t0, 1),
           "summary": summary, "cpu_gpu_discrepancy": discrepancy,
           "evidence_set_size_dense1024": setsize}
    (RESULTS / "phase5_canonical.json").write_text(json.dumps(out, indent=2))
    (RESULTS / "phase5_canonical_rows.json").write_text(json.dumps(rows))
    print("DONE", n_elig, "eligible,", n_exc, "excluded,",
          len(papers_with_q), "papers,", out["runtime_s"], "s")
    print("cpu_gpu:", discrepancy)
    print("dense@1024:", summary.get("dense_gpu|1024"))
    print("setsize:", setsize)


if __name__ == "__main__":
    main()
