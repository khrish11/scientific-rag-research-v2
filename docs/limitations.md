
## Phase 3 additions
- Adaptive-k reproduced only as score/margin/rel-drop adaptations; faithful
  reproduction of the original algorithm not verified.
- Score-feature diagnostics are univariate; no multivariate estimator trained.
- Oracle @256 = 0.841 (not all eligible evidence fits smallest budget).
- Eligibility exclusions (193 questions) still not fully characterized.
- GPU benchmark on 2000 paragraphs; VRAM headroom for larger batches untested.

## Phase 5.1 additions
- Canonical pipeline unifies packing (token_count+2*(n-1), skip-if-not-fit)
  across methods; CPU-vs-GPU dense compared on identical questions.
- Sufficiency predictors (Model A/B) reproduced from Phase 3/4 artifacts;
  Model C remains UNTESTED (never implemented).
- Coverage-aware selection deferred; prior-art risk documented as HIGH.
