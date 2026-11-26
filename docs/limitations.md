
## Phase 3 additions
- Adaptive-k reproduced only as score/margin/rel-drop adaptations; faithful
  reproduction of the original algorithm not verified.
- Score-feature diagnostics are univariate; no multivariate estimator trained.
- Oracle @256 = 0.841 (not all eligible evidence fits smallest budget).
- Eligibility exclusions (193 questions) still not fully characterized.
- GPU benchmark on 2000 paragraphs; VRAM headroom for larger batches untested.
