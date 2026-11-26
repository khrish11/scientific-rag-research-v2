# Phase 5.0 — Minimal Experiment Design (DESIGN ONLY, not implemented)

## Question
Does structure-aware or coverage-aware passage SELECTION beat relevance
ranking for complete annotated-evidence-set recovery under exact token
budgets on QASPER known-paper QA?

## Methods (same candidate pool = all paper paragraphs, same packing)
1. Dense ranking (baseline, GPU)
2. Hybrid 0.9 / RRF k=20 (frozen from Phase 2)
3. MMR (lambda sweep on dev only)
4. Relevance-per-token greedy
5. Submodular coverage (facility-location on MiniLM embeddings,
   GeoRAG-style but query-single, no sub-query generation)
6. Section-aware variant: coverage objective + penalty for same-section
   redundancy (justified by paper structure; the ONE domain modification)

## Protocol
- Eligibility: identical strict rules (812-question population rule
  reapplied; record exclusions).
- Budgets: 256/512/1024/2048/4096; top-k 1/3/5/10/20.
- Dev: 60-paper train partition (seed 42) for lambda/weights only.
- Eval: validation exploratory; test SEALED.
- Metrics: complete recovery (primary), recall, precision, tokens,
  latency; paper-cluster bootstrap 2000 reps.

## Go/no-go (predeclared)
GO only if a selection method beats dense top-k packing by >=3pp complete
recovery at 1024 tokens with bootstrap CI excluding 0, AND the gain is not
fully explained by the fragmentation paper's passage-granularity remedy
(i.e., must add value beyond simply indexing finer).
Otherwise NO-GO -> abandon retrieval-selection direction; the remaining
defensible contribution is the evaluation methodology itself.

## Prior-art risk
HIGH: fragmentation paper + GeoRAG + BEAR already occupy this space.
Experiment is justified only as a replication/extension on QASPER
paragraph-level known-paper setting with complete-set metric.
