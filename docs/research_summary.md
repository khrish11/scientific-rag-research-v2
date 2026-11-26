# Research Summary (preliminary, not peer-reviewed)

Title: "An Exploratory Study of Evidence Completeness and Retrieval
Stopping in Scientific Paper Question Answering"

## Question
Can retrieval signals or lightweight learned models predict/stopping
decide when enough evidence is retrieved under token budgets for
scientific paper QA (QASPER, known-paper setting)?

## Answer (preliminary, exploratory validation only)
NO for the learned-stopping formulation tested:
- Retrieval-score features ~ chance (AUROC 0.49-0.58).
- Context size alone AUROC 0.755 beats all feature models (0.723).
- Model A stopping fails the 20%-token-reduction non-inferiority target.
- Complete annotated-evidence recovery is primarily limited by
  multi-paragraph evidence sets (dense @1024: 1 para .68 -> >=5 .08).
- GPU gives ~14x embedding throughput with identical rankings.

## Status
Exploratory. Validation split used for all reported numbers (informed
preprocessing); test split sealed and unused. Not peer-reviewed.
