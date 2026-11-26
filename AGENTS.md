# AGENTS.md

This file provides guidance for AI coding assistants working on this repository.

## Project Context

This is an academic research project investigating evidence-sufficiency stopping for budgeted hybrid retrieval in scientific paper question answering. The hypothesis is unvalidated.

## Core Principles

1. **Correctness over features**: Prioritize correctness, reproducibility, and data integrity over adding new features.
2. **No scientific claims**: Do not claim scientific novelty or superior performance.
3. **Evidence-based**: All experimental results must be computed from actual data, not copied from publications.
4. **Research integrity**: Maintain strict separation between train/validation/test splits.

## Code Conventions

### Python Version

- Use Python 3.11+
- Maintain type hints with mypy strict mode
- Follow PEP 8 with Ruff formatting

### Testing

- Write tests for all new functionality
- Use pytest with fixtures for synthetic data
- Do not fabricate test results
- Tests should cover: schema validation, data integrity, split isolation, leakage prevention

### Dependencies

- Keep dependencies minimal
- Prefer established libraries (pydantic, datasets, tiktoken)
- No paid APIs or proprietary services
- No unnecessary GPU requirements

### Data Handling

- Never commit raw dataset files to Git
- Preserve all provenance information (paper IDs, question IDs, source URLs)
- Do not silently discard problematic records
- Handle unanswerable questions explicitly
- Support multiple valid evidence sets

### Evidence Mapping

- Preserve stable paragraph and chunk identifiers
- Document all normalization operations
- Record ambiguous and failed mappings
- Do not silently assign incorrect evidence labels
- Provide inspection utilities for failed mappings

### Chunking

- Use deterministic segmentation
- Record tokenizer version
- Preserve source provenance (section, paragraph)
- Support both paragraph-level and token-aware chunking
- Do not choose final chunking configuration without analyzing evidence-mapping consequences

### Research Integrity

- Ensure no paper overlap across splits
- Never supply gold answers/evidence to retrieval at inference time
- Do not use test data for model selection or threshold tuning
- Separate derived sufficiency labels from retrieval inputs
- Do not expose answer annotations during evaluation

## Repository Structure Decisions

- `configs/`: Configuration files for experiments
- `data/raw/`: Downloaded dataset files (git-ignored)
- `data/processed/`: Processed dataset files (git-ignored)
- `docs/`: Research documentation
- `src/scientific_rag/`: Source code organized by domain
- `tests/`: Comprehensive test suite
- `scripts/`: Utility scripts for common operations
- `experiments/`: Experiment-specific code
- `results/`: Experiment outputs (git-ignored)

## What NOT to Implement Yet

- BM25 retrieval
- Dense retrieval
- Hybrid retrieval
- Evidence-sufficiency controller (the proposed method)
- Web applications or frontends
- Medical RAG (outside current scope)

## Current Status

The repository currently provides:
- Dataset download and ingestion pipeline
- Dataset analysis and statistics generation
- Evidence mapping with normalization
- Token-aware chunking
- Research integrity safeguards
- Comprehensive test suite

Next steps await research review of the dataset preparation foundation.

## When Adding New Features

1. First add tests for the new functionality
2. Implement the feature
3. Run the full test suite
4. Run linting and type checking
5. Update relevant documentation
6. Commit with descriptive message

## Communication

- Be explicit about assumptions
- Document all normalization operations
- Report actual computed statistics, not published numbers
- Clearly distinguish implemented capabilities from proposed future work
