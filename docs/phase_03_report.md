# Phase 3 Report — Adaptive Baselines, Sufficiency Diagnostics, GPU

Branch: feature/phase-2-retrieval-baselines (continued).

## GPU (verified, local RTX 4060)
- .venv torch 2.10.0+cu126 CUDA OK; sentence-transformers 6.3.0 installed.
- DenseRetriever: device auto/cpu/cuda, batch_size param, device+batch-aware
  cache keys, FP32. BM25/fusion/eval on CPU.
- Benchmark (2000 paragraphs): CPU 41.6 p/s vs CUDA bs32 600.4 p/s (~14.4x);
  peak VRAM 254MB@bs32, 715MB@bs128; consistency max|dsim|=4.6e-7, top-20
  identical. results/gpu_benchmark.json, docs/gpu_acceleration.md.

## Phase 2 reconciliation
Aggregate table = question-micro; bootstrap = paper-macro — different
estimands caused apparent mismatch. Recomputed both (docs/phase_02_reconciliation.md).

## Oracle (verified per-question, all budgets)
256:0.841, 512:0.983, 1024+:1.000. Oracle-possible-but-failed dense@1024:
289/812 (35.6%) — ranking failures, not budget.

## Adaptive experiment (validation, 812 eligible, GPU ~66s)
- fixed_k/budget/score_threshold/margin/rel_drop policies from trajectories.
- rel_drop/margin stopping ~matches full-depth recovery at lower mean tokens;
  cannot exceed continuing retrieval on same ranking (as predicted).
- Faithful Adaptive-k reproduction infeasible -> documented adaptation.

## Sufficiency diagnostics (univariate, trajectories)
Score features AUROC ~0.49-0.53 (chance); n_selected/tokens ~0.75.
Learned estimator NOT yet justified; recommend minimal probe next.
Recommendation: NARROW scope (docs/research_decision.md).
