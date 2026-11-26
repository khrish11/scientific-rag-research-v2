# Phase 5.0 Novelty Audit — Evidence-Set Recovery under Context Budgets

## Verified prior art (abstracts/full text inspected)
1. "Fixed-Budget Evidence Visibility in Known Documents" (Appl. Sci. 2026,
   doi:10.3390/app162211285) — evaluates complete annotated-evidence
   coverage under fixed budgets on QASPER + SciFact with BM25/MiniLM/BGE/
   SPECTER. DIRECT overlap with our Phase 2 protocol.
2. "Evidence Fragmentation Is a Distinct Failure Mode in Chunked Retrieval"
   (Information 2026, doi:10.3390/info17111129) — controlled QASPER study:
   evidence dispersion causally cuts complete-evidence retrieval up to
   .471; passage-granularity indexing removes it at 8-16x index cost;
   9 remedies tested, 5 fail. DIRECT overlap + remedy analysis.
3. arXiv:2607.00725 "Recall Is Not Enough" — answer-in-context diagnostic
   + submodular budgeted packer; under review EACL 2027. Closest conceptual
   match to budgeted evidence-survival evaluation.
4. arXiv:2606.29328 GeoRAG — information-demand coverage optimization,
   submodular facility location, beats MMR/DPP on 6 QA benchmarks.
5. arXiv:2610.40314 "Decompose, Estimate, Select" — coverage-aware
   submodular context construction, TREC RAG 2024 (very recent).
6. MORSE (Complex & Intelligent Systems 2026,
   doi:10.1007/s40747-026-02520-z) — multi-objective evidence subset
   optimization for RAG.
7. BEAR (EMNLP 2026 main, aclanthology 2026.emnlp-main.1348) — budgeted
   evidence allocation; reports results ON QASPER.
8. SEAL-RAG (arXiv:2512.10787) — replace-don't-expand controller;
   references Adaptive-k baseline.
9. QASPER original (NAACL 2021) — dataset + evaluator.

## Assessment
The proposed alternative "Evidence-Set Recovery under Context Budgets in
Scientific Paper QA" is substantially SCOOPED: two 2026 papers already
measure complete-evidence-set coverage on QASPER under budgets, and
coverage/submodular selection is an active 2026 area (GeoRAG, DES, BEAR,
MORSE, Recall-Is-Not-Enough). Remaining narrow gaps: section/structure-
aware evidence selection exploiting scientific-paper hierarchy, and
cross-section multi-paragraph dependencies — but the fragmentation paper
already shows passage granularity largely fixes dispersion, weakening
even that gap.
