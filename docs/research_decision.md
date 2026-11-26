# Research Decision (Phase 3, evidence-based)

1. Adaptive retrieval reduces actual token consumption? YES trivially —
   stopping early consumes fewer tokens, but cannot improve recovery on an
   unchanged ranking (confirmed analytically and empirically).
2. Score-based stopping preserves completeness? Partially — rel_drop/margin
   policies matched ~full-depth recovery at lower mean tokens, but gains are
   modest and threshold-sensitive.
3. Do retrieval scores predict completeness? NO — AUROC ~0.49-0.53 (chance).
   Only context length/paragraph count predict (AUROC ~0.75).
4. Multi-paragraph questions fail distinctly? YES — recovery collapses with
   evidence-set size (dense @1024: 1->0.68, >=5->0.08).
5. Is a learned sufficiency estimator justified? WEAK/UNCERTAIN — univariate
   score features carry no signal; a multivariate model MIGHT extract
   interactions, but the burden of proof is now higher. Recommend a small
   feasibility probe before committing to the full method.
6. Go/no-go: NARROW, don't abandon — the interesting problem is
   multi-paragraph evidence recovery under budget; score-only stopping is
   insufficient. Next step should be a minimal learned sufficiency probe
   on dev trajectories, evaluated honestly on validation.
