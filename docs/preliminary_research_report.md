# An Exploratory Study of Evidence Completeness and Retrieval Stopping in Scientific Paper Question Answering

PRELIMINARY TECHNICAL REPORT — not peer-reviewed, not published.
AI-assisted development disclosed (Devin/Cognition agent + human direction).
All numbers exploratory on QASPER validation; test split sealed.

## Abstract
We investigate whether evidence-sufficiency signals can improve complete
annotated-evidence recovery under context-token budgets for single-paper
scientific QA (QASPER). We build strict evidence eligibility, four retrieval
baselines (BM25, dense, weighted hybrid, RRF), exact token-budget packing,
and paper-cluster statistics. Complete-evidence recovery at 1024 tokens:
BM25 .578, dense .644, hybrid .663, RRF .661 (question-micro, 812 eligible).
Recovery collapses for multi-paragraph evidence (dense: 1 para .68,
>=5 paras .08). Learned sufficiency prediction FAILS feasibility: retrieval
features AUROC ~0.5-0.58 vs context-size 0.755; Model B (all features)
underperforms size-only. Learned stopping does not beat the fixed-budget
frontier. We conclude the score-based stopping hypothesis is not supported
and recommend coverage-aware selection as future work (high prior-art risk).

## 1. Motivation
Budgeted RAG for scientific papers: does "when to stop retrieving" help?
Hypothesis unvalidated a priori.

## 2. Related work (verified)
- Fixed-Budget Evidence Visibility (Appl.Sci.2026): complete-evidence
  coverage under budgets on QASPER+SciFact — closest protocol overlap.
- Evidence Fragmentation (Information 2026): causal chunking study on
  QASPER; passage granularity removes dispersion penalty.
- Recall Is Not Enough (arXiv:2607.00725): answer-in-context diagnostic +
  submodular packer for budgeted RAG.
- GeoRAG (2606.29328), Decompose-Estimate-Select (2610.40314), MORSE,
  BEAR (EMNLP'26, incl. QASPER), SEAL-RAG, Adaptive-k (EMNLP'25).
- QASPER (NAACL'21). See docs/prior_art_comparison.md.

## 3. Research questions
RQ1: baseline recovery under budgets? RQ2: do retrieval scores predict
completeness? RQ3: can lightweight learned stopping reduce tokens at equal
recovery? RQ4: where do failures concentrate?

## 4. Dataset & evidence
QASPER v0.3 (CC BY 4.0): 888/281/416 papers, 2593/1005/1451 questions.
Strict mapping: 83% unique-mapped; eligibility requires fully-mapped
textual evidence; 812/1005 eligible (exclusions itemized).

## 5. Methodology
Paragraph-level passages; exact serialized token packing (skip-if-not-fit);
budgets 256-4096; paper-cluster bootstrap 2000; frozen alpha=.9, rrf k=20
tuned on 60-paper train-dev partition (seed 42).

## 6. Static retrieval results (canonical pipeline, n=812)

| method | @256 | @512 | @1024 | @2048 | @4096 |
|---|---|---|---|---|---|
| bm25 | 0.230 | 0.389 | 0.578 | 0.770 | 0.905 |
| dense_gpu | 0.254 | 0.441 | 0.644 | 0.830 | 0.963 |
| hybrid | 0.267 | 0.458 | 0.663 | 0.839 | 0.963 |
| rrf | 0.266 | 0.475 | 0.661 | 0.837 | 0.964 |

dense_cpu identical to dense_gpu (0/812 flips, top-10 overlap 1.000).
CPU-GPU discrepancy RESOLVED: Phase-3 0.631 was a trajectory-
simulation packing artifact; canonical pipeline reproduces 0.6441.

Evidence-set-size (dense@1024, min |gold set|): 1->.695(n=691),
2->.449(n=89), 3->.176(n=17), 4->0.0(n=9), >=5->0.0(n=6).

Runtime: 582s (CPU+GPU dense). Eligible 812/275 papers.

## 7. Completeness diagnostics
Multi-paragraph collapse; oracle-possible 1.0@>=1024 (0.841@256);
score features ~chance; only context size predicts (AUROC .75).

## 8. Learned stopping feasibility
Model A .755 / B .723 / B-nosize .578 AUROC; Model A stopping fails NI.

## 9. Negative findings
Score-based sufficiency prediction and stopping NOT supported.

## 10. Limitations
Validation-informed preprocessing; no Model C; no faithful Adaptive-k;
matched pairs inconclusive; eligibility exclusions not fully
characterized; single dataset/single-paper setting.

## 11. Future questions
Coverage-aware selection (prior-art risk HIGH); section-aware selection;
cross-section multi-paragraph evidence; evaluation-methodology
contributions only.

## 12. References
See docs/phase_05_novelty_audit.md for verified citations/URLs.

## 13. Reproduction
docs/reproducibility.md.
