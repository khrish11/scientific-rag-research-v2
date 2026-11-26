# Phase 2 Report — Retrieval Baselines (partial execution)

Branch: `feature/phase-2-retrieval-baselines`. Status: baselines implemented;
subset experiment executed on 12 validation papers (first 6 dev / last 6 eval).
Full benchmark, figures, bootstrap CIs and failure analysis are NOT yet complete.

## Protocol (frozen)
- Task: single-paper QA; paper known in advance. NOT open-domain/multi-paper.
- Representation: original paragraph-level passages (default per Phase 1.1).
- Retriever inputs: question text + paper paragraphs only. No gold answers,
  evidence flags, or sufficiency labels.
- Test split sealed; validation used for baselines (already informed preprocessing).

## Implemented
- `src/scientific_rag/retrieval/baselines.py`: BM25 (Okapi, k1=1.5, b=0.75,
  lowercase \w+ tokens, deterministic tie-break), DenseRetriever
  (sentence-transformers/all-MiniLM-L6-v2, CPU, cosine, normalized embeddings,
  batched, npz cache in data/processed/embeddings/, truncation counted+logged),
  weighted hybrid (min-max normalized union pool top-20, configurable alpha),
  RRF (k=60 configurable), pack_budget (greedy skip-if-not-fit, cl100k_base
  tokens, "\n\n" separator=2 tokens, never exceeds budget).
- `src/scientific_rag/experiments/phase2_eval.py`: gold evidence mapping
  (strict exact/normalized, non-textual excluded), per-question precision/
  recall/complete-recovery/partial, budgets {256..4096}, topk {1,3,5,10,20},
  alpha sweep on dev half, eval on held-out half.

## Subset results (12 papers, 840 question-method rows; NOT full benchmark)
Alpha sweep (dev, budget=1024 complete-recovery): 0.0→0.578, 0.9→0.625,
1.0→0.625; selected alpha=0.9.

At budget=1024 complete annotated-evidence recovery (mean):
bm25 0.429, dense 0.619, rrf 0.762, hybrid(0.9) 0.619.
At topk=10: bm25 0.476, dense 0.667, rrf 0.667, hybrid 0.619.
BM25@4096 budget: complete 0.762.

## Known limitations / not done
- DenseRetriever reloads model per paper (slow); needs singleton caching.
- Gold mapping uses paragraph-level strict match; unmappable evidence excluded.
- No bootstrap CIs, figures, oracle module, or full-split runs yet.
- Latency measured per query incl. index build amortization not separated.
