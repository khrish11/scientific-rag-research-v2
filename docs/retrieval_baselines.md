# Retrieval Baselines (Phase 2.1)

Single-paper QA; corpus = original paragraphs. Inputs: question + paper text
only. No gold labels in retrieval path.

- BM25: Okapi, k1=1.5, b=0.75, \w+ lowercase tokens, deterministic tie-break.
- Dense: sentence-transformers/all-MiniLM-L6-v2, CPU, cosine, normalized
  embeddings, batched, npz cache keyed by text hash, truncation logged.
- Hybrid: alpha*norm_dense + (1-alpha)*norm_sparse, min-max normalization,
  union candidate pool (default all paragraphs for known-paper retrieval).
- RRF: score = Σ 1/(k + rank), k=60 default, configurable.
- Packing: greedy rank-order, skip-if-not-fit, serialized cl100k_base tokens,
  "\n\n" separators; budget never exceeded.
