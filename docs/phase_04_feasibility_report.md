# Phase 4 Feasibility Report — Learned Evidence-Sufficiency

Branch: phase-4-feasibility (on v2 repo). Status: PARTIAL — Models A/B done;
Model C, faithful Adaptive-k, matched pairs, ECE, non-inferiority test INCOMPLETE.

## Eligibility audit (validation, 193 excluded of 1005)
reasons: all_unanswerable=60, unmapped_textual=78, nontextual_only=37,
no_evidence=18. Eligible by type: span 483, free-form 217, yes/no 86,
unanswerable 26. Population unchanged (812 eligible).

## Models (train-dev 60 papers, 9399 states; val 76527 states)
- A (tokens+n_sel): AUROC 0.755 [0.745,0.767], AUPRC 0.843, Brier 0.180
- B (all features): AUROC 0.723 [0.713,0.735] — WORSE than A
- B no-size ablation: AUROC 0.578 — retrieval features add ~nothing
- B per-method: dense 0.766, hybrid 0.772, rrf 0.751

## Stopping (Model B, thr=0.764, val)
dense: complete 0.809 @1777 tok; hybrid 0.817 @1816; rrf 0.807 @1905;
bm25 0.560 @1336. Compare fixed-budget dense@1024 complete 0.631 @944 tok
and @2048 0.826 @1939 tok — learned stopping lands between, NOT clearly
better than fixed budget frontier.

## Verdict (preliminary, partial)
NEGATIVE for score-feature learned stopping: adding retrieval features to
context-size baseline HURTS AUROC (0.723 < 0.755). Confirms Phase 3 finding.
Go/no-go leans NO-GO for this formulation; see docs/research_decision.md.
