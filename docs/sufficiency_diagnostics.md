# Sufficiency Diagnostics (Phase 3, validation trajectories)

AUROC/AUPRC of gold-independent features vs complete-recovery label
(n=812 questions x steps; prevalence ~context-dependent):

| feature | AUROC | AUPRC |
|---|---|---|
| top_score | 0.490 | 0.672 |
| margin | 0.488 | 0.670 |
| score_decay | 0.497 | 0.652 |
| rel_drop | 0.534 | 0.694 |
| n_selected | 0.748 | 0.838 |
| tokens | 0.753 | 0.840 |

Finding: retrieval-score features are at CHANCE for predicting complete
evidence recovery; only context size/paragraph count carry signal
(expected: more context -> higher recovery). This WEAKENS the premise
that a lightweight score-based sufficiency estimator will be accurate;
a learned estimator is not yet justified by these univariate signals.
