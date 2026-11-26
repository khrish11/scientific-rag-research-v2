# Prior-Art Comparison vs Our Work

| Work | Dataset | Budgeted complete-evidence metric | Selection method | Overlap |
|---|---|---|---|---|
| Fixed-Budget Evidence Visibility | QASPER+SciFact | YES (complete coverage) | rank top-k | HIGH — same protocol |
| Evidence Fragmentation | QASPER+2 legal | YES (complete-evidence retrieval) | chunking manipulation + 9 remedies | HIGH — same outcome, adds causal analysis |
| Recall Is Not Enough | multi-hop QA | answer-in-context (span-level) | submodular packer | MED — diagnostic + packing |
| GeoRAG | 6 open-domain QA | EM/F1 | submodular coverage | MED — method, not QASPER |
| Decompose Estimate Select | TREC RAG 2024 | alpha-nDCG nuggets | submodular coverage | MED |
| MORSE | RAG QA | subset optimization | multi-objective | MED |
| BEAR | incl. QASPER | coverage-faithfulness | hierarchical allocation | MED-HIGH |
| Ours (Phase 2) | QASPER | complete annotated-evidence-set recovery | BM25/dense/hybrid/RRF + budgets | — |

Unique elements we retain: strict per-annotation alternative evidence-set
evaluation, paper-cluster bootstrap, exact serialized token accounting,
QASPER paragraph provenance. These are evaluation-methodology
contributions, not retrieval-method novelty.
