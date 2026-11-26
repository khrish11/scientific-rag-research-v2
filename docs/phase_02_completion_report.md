# Phase 2.1 Completion Report

Branch: feature/phase-2-retrieval-baselines.

## What was completed
- DenseRetriever refactored to singleton model (loaded once/process), npz
  embedding cache keyed by content hash, truncation logged.
- Embedding caches untracked from git; .gitignore covers data/raw,
  data/processed, *.npz/*.pt/*.bin, caches, venvs. NOTE: previously committed
  .npz files remain in git history objects (not rewritten per policy);
  they contain only numeric vectors — flagged for owner review.
- Evidence eligibility: per-annotation, ALL textual evidence must map;
  non-textual (FLOAT SELECTED / figure-table pointers) excluded from the
  textual requirement; empty-textual annotations ineligible.
- Dev partition: 60 train papers, seed 42 (results/dev_partition.json).
- Tuning frozen before validation: alpha=0.9, rrf_k=20
  (results/tuning_manifest.json). Primary metric: complete recovery @1024.
- Full validation eval: 281 papers, 1005 questions, 812 eligible / 193
  excluded, runtime ~691s. Per-question rows in results/phase2_per_question.json.
- Metrics: P/R@k, MRR, nDCG@k, complete recovery, partial coverage, exact
  serialized token counts, latency, oracle-possible flag.
- Paper-cluster bootstrap (2000 reps, seed 42) @1024 in results.
- Figures in results/figures/ (PNG+PDF).

## Headline results (validation, complete annotated-evidence recovery)
@1024 budget: bm25 0.578, dense 0.644, hybrid(0.9) 0.663, rrf(k=20) 0.661.
@4096 budget: ~0.96 all methods.
Bootstrap @1024: rrf-dense +0.007 [-0.023, 0.037]; hybrid-dense +0.014
[-0.000, 0.029]; rrf-hybrid -0.007 [-0.035, 0.021]; bm25-dense -0.071
[-0.117, -0.023]. Only BM25-vs-dense is clearly separated; fusion methods
are statistically indistinguishable from dense on this split.

## Incomplete / caveats
- Dev tuning re-ran full eval per alpha (14 passes) — correct but slow;
  candidate-pool depth sensitivity evaluated via pool=all (no top-20 cap).
- Latency includes per-question query encode + search; indexing amortized.
- Peak memory not instrumented.
- Eligibility exclusions (193 questions) not yet characterized vs eligible.
- No MRR/nDCG figures; score-feature sufficiency diagnostics not run.
- Test split untouched.
