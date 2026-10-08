# Add-on Experiment D1 — Official-50 results

## Design and completion

The prespecified readout was run for models 000–049, 12 canonical phases (0–13.75° in 1.25° steps), and Original/Mirror conditions: **1,200/1,200 integrations**. It used FlyVis 1.2.0 at commit `92b3845cc426dd309a1a0e1b3890156c42e14021`, CPU, with 50 frames at dt=0.02 s, 1 s fade-in, and frames 10–49 averaged. ON_A is the fixed Mi1/Tm3→T4a–d source-current contrast; OFF_A is the fixed Tm1/Tm2/Tm4/Tm9→T5a–d contrast; I_ONOFF = ON_A − OFF_A. No fitted source/spatial weights or post-hoc sign flip were used.

All 600 model-phase rows joined exactly to canonical decoded A_flow. All values were finite; 24 prespecified source-target pairs were available per integration and all eight target subtypes were covered. One initial unsaved model-000/phase-0/Original attempt was repeated after a runner QA-count assertion was corrected; the saved dataset contains exactly one successful row for each of the 1,200 unique planned integrations.

## Primary tests

- **P1, mean I_ONOFF versus zero:** mean = 4.524e-08; median = 1.156e-08; between-model SD = 5.995e-07; whole-model bootstrap 95% CI [-1.119e-07, 2.158e-07]; sign-flip permutation p = 0.606, Holm p = 1.000.
- **P2, I_ONOFF versus decoded A_flow:** Spearman ρ = 0.065; whole-model bootstrap 95% CI [-0.216, 0.341]; permutation p = 0.653, Holm p = 1.000. Leave-one-model-out ρ ranged from 0.015 to 0.124, remaining positive in 50/50 omissions. The direction is stable but the association is weak and its interval is broad.
- **P3:** descriptive only, NOT CONFIRMATORY because the sign mapping from the selected signed synaptic currents to decoded-flow sign was not biologically/theoretically prespecified. Concordant signs: 26/50 (52.0%); exact binomial 95% CI [0.374, 0.663]. No p-value or Holm test was used.

## Secondary and exploratory results

The prespecified secondary OFF_A–A_flow association was negative (ρ = -0.357, 95% model-bootstrap CI [-0.606, -0.051], raw p = 0.012, Holm p = 0.048 across four secondary tests). ON_A, absolute imbalance, and phase-variability associations did not survive their family correction. In the eight-stage exploratory family, Tm2 had the largest negative association (ρ = -0.313, raw p = 0.027, Holm p = 0.219); it did not survive correction. These do not replace the primary imbalance test.

## Decision and scope

**D1_INCONCLUSIVE.** The primary P2 estimate is near zero, but its confidence interval is too broad to rule out a meaningful association. P1 also provides no evidence for a systematic ensemble imbalance. The secondary OFF_A association is a candidate for cautious follow-up, but it does not rescue the prespecified ON/OFF imbalance readout.

**D2 recommendation: NO.** The central D1 imbalance readout did not produce a sufficiently clear, robust observational signal to motivate intervention. This does not establish that ON/OFF pathways are irrelevant.

**Graphical interpretation: LEVEL 0** for strengthening the prespecified “ON/OFF asymmetry as candidate contributor” claim; the secondary OFF_A association can be described as an exploratory association only. D1 is observational and cannot support causal wording.

## Stimulus hash note

All 1,200 canonical stimulus image hashes matched the phase manifest exactly. For the BoxEye arrays, this run records the observed byte hashes alongside the manifest values; **0/1,200 observed hashes matched the manifest's `rendered_sha256` values**. The repository does not document the manifest's rendered-array byte serialization. Stimulus pixels, BoxEye parameters (extent 15, kernel 13), input shape, and FlyVis version were fixed and checked; the hash provenance mismatch is retained explicitly as a reproducibility limitation.

## Outputs

Tables are in `data/precomputed/supplementary/biological_constraints/D1/`; figures D1–D5 are in `manuscript/review/archive/addon_figures/D1/` (PNG and PDF). No manuscript, main figure, graphical abstract, training, or D2 intervention was changed or run.
