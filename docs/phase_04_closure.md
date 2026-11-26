# Phase 4 Closure — Research Milestone (Phases 1–4.1)

Repo: github.com/khrish11/scientific-rag-research-v2 (squashed successor of
deleted scientific-rag-research). Branch phase-4-feasibility.
Original per-phase history NOT recoverable from GitHub (repo deleted);
local reflog may retain old objects but was not relied upon.

## Preserved results
- Phase 1/1.1: QASPER ingestion, strict evidence mapping (train 83.5%,
  val 83.0% unique-mapped), paragraph chunking recommendation.
- Phase 2/2.1: BM25/dense/hybrid/RRF baselines; validation 812 eligible /
  193 excluded; @1024 complete recovery bm25 .578 dense .644 hybrid .663
  rrf .661; @4096 ~.96; oracle 1.0@>=1024 (0.841@256, 0.983@512).
- Phase 3: GPU acceleration (14.4x, verified), micro/macro reconciliation,
  adaptive stopping trajectories; score features AUROC~0.5.
- Phase 4/4.1: eligibility audit (exclusions: 60 all-unans, 78 unmapped
  textual, 37 nontextual-only, 18 no-evidence); Model A AUROC .755,
  Model B .723 (worse), no-size .578; Model A stopping fails NI target;
  matched pairs inconclusive (0 natural pairs).
- Untested: Model C semantic probe. Incomplete: faithful Adaptive-k,
  ECE, constructed matched pairs, formal NI test.

## Decision
CLOSE learned sufficiency-stopping investigation. Negative results
preserved in results/phase4_feasibility.json, phase4_1_results.json,
phase3_adaptive.json. Milestone tag: phase-4-closure.
