# Research Overview

## Research Question

**Evidence-Sufficiency Stopping for Budgeted Hybrid Retrieval in Scientific Paper Question Answering**

### Hypothesis

A lightweight evidence-sufficiency estimator can improve complete supporting-evidence recovery under controlled context-token budgets.

**Status**: Unvalidated. This hypothesis requires experimental validation.

## Research Domain

Scientific literature retrieval-augmented generation (RAG), specifically:
- Question answering over scientific research papers
- Budgeted retrieval with context-token constraints
- Evidence-sufficiency estimation for early stopping
- Hybrid retrieval systems (BM25 + dense retrieval)

## Scope

### In Scope
- NLP papers (QASPER dataset domain)
- Text-only evidence (paragraph-level)
- Evidence-sufficiency estimation
- Budget-constrained retrieval
- Reproducible experimental infrastructure

### Out of Scope
- Medical RAG
- Multi-modal evidence (figures, tables, images)
- Repository-level coding assistants (future work)
- Claims of scientific novelty without experimental validation

## Research Questions

1. Can evidence-sufficiency estimation accurately determine when sufficient evidence has been retrieved?
2. Does early stopping based on sufficiency estimation improve retrieval efficiency without sacrificing evidence coverage?
3. How does the approach compare to fixed-budget retrieval strategies?
4. What are the trade-offs between computational cost and evidence recovery?

## Methodology (Planned)

1. **Baseline**: Fixed-budget retrieval (top-k by score)
2. **Proposed**: Evidence-sufficiency estimator for adaptive stopping
3. **Evaluation**: Evidence coverage, answer quality, computational cost
4. **Controls**: Random seeds, deterministic preprocessing, split isolation

## Current Phase

**Phase 1 Complete**: Dataset preparation and evaluation infrastructure

The repository currently provides:
- QASPER dataset ingestion and validation
- Evidence mapping with robust normalization
- Token-aware chunking strategies
- Research integrity safeguards
- Comprehensive testing and documentation

**Next Phase**: Implementation of retrieval baselines (awaiting research review)

## Ethical Considerations

- Dataset is CC BY 4.0 licensed (redistribution with attribution)
- No sensitive or restricted data
- No human subjects research
- Results will be made reproducible and publicly available

## Limitations

- Domain-specific to NLP papers
- Text-only (no figures/tables)
- Paragraph-level evidence (not sentence-level)
- Unvalidated hypothesis requiring experimental confirmation
