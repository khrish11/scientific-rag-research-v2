# Failure Analysis (Phase 2.1, validation @1024-token budget)

Computed from results/phase2_per_question.json (812 eligible questions).

## Oracle gap
- Oracle-possible rate @1024: 1.000 (every eligible question has ≥1 gold
  set fitting the budget — evidence sets are small, ~1-3 paragraphs).
- Oracle-possible-but-failed: 289/812 (35.6%) for dense — retrieval failures,
  not budget impossibility.

## Head-to-head vs dense (per-question, budget mode @1024)
- hybrid > dense: 27 questions; hybrid < dense: 12 questions.
- rrf > dense: 66; rrf < dense: 52.
- Hybrid wins more often than it loses, but margins are small and CIs overlap.

## Evidence-set-size effect (dense, @1024 complete recovery)
- |rel|=1: 0.681, 2: 0.672, 3: 0.485, 4: 0.375, ≥5: 0.077.
- Multi-paragraph evidence is the dominant failure mode: recovery collapses
  as required evidence grows — consistent with independent per-paragraph
  ranking and no cross-paragraph coverage mechanism.

## Token accounting
- Mean exact serialized tokens @1024: bm25 985.2, dense 995.0, hybrid 995.5,
  rrf 998.3; max ≤1018. No budget violations. Dense/hybrid/rrf consume
  slightly more budget than bm25 (more, shorter paragraphs selected).

## Score-feature diagnostic (not a sufficiency estimator)
- High top-score does not guarantee complete recovery; questions with
  oracle-possible-but-failed show ranking failures where gold paragraphs
  score below non-gold ones. Formal feature-level analysis deferred.
