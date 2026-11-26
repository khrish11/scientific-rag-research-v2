# An Exploratory Study of Evidence Completeness and Retrieval Stopping in Scientific Paper Question Answering

PRELIMINARY TECHNICAL REPORT — not peer-reviewed, not published.
AI-assisted development disclosed (Devin/Cognition agent + human direction).
All numbers exploratory on QASPER validation; test split sealed.

## Abstract
We investigate whether evidence-sufficiency signals can improve complete
annotated-evidence recovery under context-token budgets for single-paper
scientific QA (QASPER). We build strict evidence eligibility, four retrieval
baselines (BM25, dense, weighted hybrid, RRF), estimated token-budget packing
(per-paragraph cl100k counts + separator estimate; empirically verified to
overestimate exact joined serialization by <=32 tokens on 288 contexts),
and paper-cluster statistics. Complete TEXTUAL-evidence recovery at 1024
tokens: BM25 .578, dense .644, hybrid .663, RRF .661 (question-micro, 812
eligible; metric = textual evidence projection — 129/812 eligible questions
also contain non-textual evidence components excluded from the requirement).
Textual-projection recovery collapses for multi-paragraph evidence sets
(dense @1024 by minimum gold-set size: 1 para .695 n=691, 2 .449 n=89,
3 .176 n=17, 4 0.0 n=9, >=5 0.0 n=6). Learned sufficiency prediction FAILS
feasibility: retrieval features AUROC ~0.5-0.58 vs context-size 0.755;
Model B (all features) underperforms size-only. Descriptive feasibility
targets for learned stopping were not met; formal non-inferiority was NOT
evaluated. Model C (semantic probe) remains UNTESTED. We conclude the
score-based stopping hypothesis is not supported and recommend coverage-
aware selection as future work (high prior-art risk).

## 1. Motivation
Budgeted RAG for scientific papers: does "when to stop retrieving" help?
Hypothesis unvalidated a priori.

## 2. Related work (verified; see docs/phase_05_novelty_audit.md)
- Kang et al., "Fixed-Budget Evidence Visibility in Known Documents",
  Applied Sciences 16(22):11285, 2026. doi:10.3390/app162211285 —
  complete-evidence coverage under budgets on QASPER+SciFact.
- "Evidence Fragmentation Is a Distinct Failure Mode in Chunked Retrieval",
  Information 17(11):1129, 2026. doi:10.3390/info17111129 — causal
  chunking study on QASPER; passage granularity removes penalty.
- Bala, "Recall Is Not Enough", arXiv:2607.00725 (under review EACL 2027)
  — answer-in-context diagnostic + submodular packer.
- Zhang et al., GeoRAG, arXiv:2606.29328 — submodular coverage selection.
- Tu et al., "Decompose, Estimate, Select", arXiv:2610.40314 — coverage-
  aware context construction, TREC RAG 2024.
- Ye, Wang & Chen, MORSE, Complex & Intelligent Systems 2026,
  doi:10.1007/s40747-026-02520-z — multi-objective evidence subsets.
- Sun et al., BEAR, EMNLP 2026 main (2026.emnlp-main.1348) — budgeted
  evidence allocation; reports QASPER results.
- Adaptive-k, EMNLP 2025 (aclanthology 2025.emnlp-main.1017).
- Dasigi et al., QASPER, NAACL 2021 (2021.naacl-main.365).

## 3. Research questions
RQ1: baseline recovery under budgets? RQ2: do retrieval scores predict
completeness? RQ3: can lightweight learned stopping reduce tokens at equal
recovery? RQ4: where do failures concentrate?

## 4. Dataset & evidence
QASPER v0.3 (CC BY 4.0): 888/281/416 papers, 2593/1005/1451 questions.
Strict mapping: 83% unique-mapped; eligibility requires >=1 annotation
with fully-mapped nonempty TEXTUAL evidence. 812/1005 eligible; 193
excluded (60 all-unanswerable, 78 unmapped textual, 37 non-textual-only,
18 no-evidence). IMPORTANT: the completeness metric is a TEXTUAL EVIDENCE
PROJECTION — non-textual evidence components (figures/tables/FLOAT
SELECTED) are excluded from the requirement, not counted as failures.
129/812 eligible (15.9%) questions contain such mixed-media annotations.
Alternative evidence sets preserved independently; never merged.

## 5. Methodology
Paragraph-level passages; paper known in advance (single-paper QA).
Packing: greedy rank-order; cost = per-paragraph cl100k token_count +
2*(n-1) separator estimate; skip-if-not-fit; never exceed budget; no
truncation. EMPIRICAL CHECK (288 contexts): estimate mean 748.5 vs exact
joined encode 736.1, max |diff| 32 -> estimate is conservative (over-
estimates), so budgets are never exceeded in expectation but are NOT
exact serialization. Budgets 256-4096; paper-cluster bootstrap 2000 reps;
frozen alpha=.9, rrf k=20 tuned on 60-paper train-dev partition (seed 42).

## 6. Static retrieval results — CANONICAL REPRODUCTION
(results/phase5_canonical.json; dense on CUDA, identical packing/eval)

| method | @256 | @512 | @1024 | @2048 | @4096 |
|---|---|---|---|---|---|
| bm25 | 0.230 | 0.389 | 0.578 | 0.770 | 0.905 |
| dense_gpu | 0.254 | 0.441 | 0.644 | 0.830 | 0.963 |
| hybrid | 0.267 | 0.458 | 0.663 | 0.839 | 0.963 |
| rrf | 0.266 | 0.475 | 0.661 | 0.837 | 0.964 |

dense_cpu identical to dense_gpu (0/812 flips, top-10 overlap 1.000).
CPU-GPU discrepancy RESOLVED: Phase-3 0.631 was a trajectory-
simulation packing artifact; canonical pipeline reproduces Phase-2's
0.6441 exactly. NOTE: Phase-2 numbers were produced by phase2_full.py
(CPU dense, same eligibility/packing policy) — consistent within
rounding; treat phase5_canonical.json as authoritative.

Evidence-set-size (dense@1024, minimum complete gold-set size):
1 para .695 (n=691), 2 .449 (n=89), 3 .176 (n=17), 4 0.0 (n=9),
>=5 0.0 (n=6). NOTE: these buckets use min |gold set|, not the union of
all evidence; denominators differ from earlier reports — do not mix.

Runtime: 582s (CPU+GPU dense). Eligible 812 questions / 275 papers.

## 7. Completeness diagnostics (historical artifacts, Phase 3/4)
Multi-paragraph collapse; oracle-possible 1.0@>=1024 (0.841@256,
0.983@512); score features ~chance (AUROC .49-.53); only context size
predicts (AUROC .75). Sources: results/phase3_adaptive.json,
phase2_per_question.json.

## 8. Learned stopping feasibility (historical artifacts, Phase 4/4.1)
Model A (tokens+n_sel) AUROC .755; Model B (all retrieval features)
.723 — WORSE than size-only; B no-size ablation .578. Per-method B:
dense .766, hybrid .772, rrf .751. Model A stopping (thr .763):
dense .812 complete @1844 mean tokens vs fixed-budget dense@2048
.826 @1939 — descriptive feasibility targets NOT met; formal
non-inferiority NOT evaluated. Model C UNTESTED (never implemented).
Sources: results/phase4_feasibility.json, phase4_1_results.json.
These used trajectory-level (state-level) evaluation — states are NOT
independent questions; interpret AUROCs as state-level, not
question-level.

## 9. Negative findings
Score-based sufficiency prediction and stopping NOT supported by the
tested formulations. No semantic-sufficiency conclusion is drawn —
Model C was never run.

## 10. Limitations
Validation-informed preprocessing (not untouched); no Model C; no
faithful Adaptive-k; matched pairs inconclusive (0 natural pairs);
eligibility exclusions not fully characterized; metric is textual
evidence projection (15.9% of eligible have non-textual components);
packing is an estimate, not exact serialization; trajectory-state AUROCs
are not question-independent; single dataset/single-paper setting.

## 11. Future questions
Coverage-aware selection (prior-art risk HIGH); section-aware selection;
cross-section multi-paragraph evidence; evaluation-methodology
contributions only.

## 12. References
Verified citations inline in Section 2; full list in
docs/phase_05_novelty_audit.md and docs/prior_art_comparison.md.

## 13. Reproduction & provenance
docs/reproducibility.md — every number maps to a results/*.json artifact.
