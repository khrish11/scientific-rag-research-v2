
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

## Phase 5.1 corrections
- Packing uses ESTIMATED serialized tokens (per-paragraph counts + 2/sep),
  verified <=32-token overestimate vs exact join on 288 contexts; NOT
  exact serialization.
- Primary metric is TEXTUAL EVIDENCE PROJECTION, not full annotated-
  evidence recovery: 129/812 eligible (15.9%) contain non-textual
  evidence components excluded from the requirement.
- Evidence-set-size buckets use MINIMUM gold-set size; denominators
  differ from earlier union-based reports.
- Learned-stopping claim corrected: descriptive feasibility targets not
  met; formal non-inferiority NOT evaluated.
- Model C remains UNTESTED; no semantic sufficiency conclusion.
