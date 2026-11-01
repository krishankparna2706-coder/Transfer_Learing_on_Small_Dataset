# P4 MVP decision memo (validation evidence only)

**Date:** 2026-11-01.
**Protocol:** `fewshotlab-v1`.
**Study:** `configs/mvp.yaml`: Flowers102, K ∈ {1, 10}, the four methods, seed 11, full v1 recipe.
**Code:** commit `8f3d541` (clean code tree).
**Device:** Apple M4 Pro GPU (MPS).

**Generated report:** [reports/fewshotlab-v1-mvp/comparison.md](../reports/fewshotlab-v1-mvp/comparison.md).
Every number below regenerates from the stored `predictions_val.csv` files.

> **Validation only, n = 1 run per cell.**
> - Each learned model's checkpoint was selected on these same 1,020 validation images, so the
>   values are optimistic.
> - They are not held-out test results and must not be compared with published Flowers102 test
>   benchmarks.
> - The test split was never accessed.

## Outcomes

| K | Method | Status | Val macro-F1 | Val top-1 | Val top-5 | Selected / run epochs | Wall time |
|---:|---|---|---:|---:|---:|---|---:|
| 1 | scratch | COMPLETED | 0.073 | 0.083 | 0.235 | 34 / 49 | 168 s |
| 1 | frozen | COMPLETED | 0.449 | 0.465 | 0.699 | 33 / 48 | 149 s |
| 1 | partial | COMPLETED | 0.458 | 0.465 | 0.723 | 71 / 86 | 273 s |
| 1 | prototypes | COMPLETED | 0.415 | 0.422 | 0.668 | class means (no epochs) | 6 s |
| 10 | scratch | COMPLETED | 0.477 | 0.485 | 0.757 | 70 / 85 | 723 s |
| 10 | frozen | COMPLETED | 0.853 | 0.853 | 0.964 | 49 / 52 | 290 s |
| 10 | partial | COMPLETED | 0.871 | 0.872 | 0.967 | 22 / 37 | 230 s |
| 10 | prototypes | COMPLETED | 0.780 | 0.781 | 0.931 | class means (no epochs) | 9 s |

**Label budget:** fit labels are 102 at K=1 and 1,020 at K=10. Every run also used the 1,020
validation labels, and the transfer methods add ImageNet pretraining.

**Measured wall time:** about 31 minutes for all 8 jobs (1,854 s). This falls inside the P3
estimate of 0.14–0.82 h.

## Checks performed

- **Evidence:** all 8 runs pass the evidence check: stored metrics regenerate from their
  prediction files, and the hashes match.
- **Sanity checks** (`reports/fewshotlab-v1-mvp/sanity/sanity.json`): all 8 PASS.
  - Labels match the manifest, and every class is scored.
  - Every new head moved.
  - The frozen encoders are byte-identical to IMAGENET1K_V1.
  - Partial changed only layer4 and no BatchNorm running statistic.
  - Scratch updated its BatchNorm statistics.
  - Prototype norms are consistent: 1 at K=1, below 1 at K=10 (not re-normalized).
- **Learning curves:**
  - Every learned run stopped through patience well before the 100-epoch cap.
  - Scratch K=10 shows the expected overfitting pattern: support loss near 0.1, validation loss
    rising after about epoch 30.
  - Frozen K=10 stopped at epoch 52 with its best checkpoint at epoch 49. That improvement was
    under 0.001, so it did not reset patience but was still selected. This is the intended rule:
    min_delta never discards a better checkpoint.
- **Error galleries** (rule fixed in advance: the first 12 misclassified validation images by ID):
  - Errors fall between visually similar species (passion flower ↔ king protea, frangipani ↔ giant
    white arum lily, petunia ↔ silverbush).
  - Some are confidently wrong, which is consistent with uncalibrated model scores.
  - There are no signs of label-mapping mistakes.
- **Augmentation grid:** random crops (scale 0.7–1.0) keep the flower, and flips work. Augmentation
  does not destroy the target.
- **New data observation:** some Flowers images carry visible text. One support image of
  `label_002` (hard-leaved pocket orchid) shows the caption "Paph. micranthum", that species'
  Latin name, which is a potential text shortcut. This has not been audited systematically; it is
  recorded in the data card and the limitations.

## Decision

**Retain the v1 recipe unchanged for P5.**
- No scientific or implementation defect was found.
- The 100-epoch cap was never binding.
- Patience and checkpoint selection behaved as specified.
- Every method obeyed its parameter and BatchNorm policy.
- Scratch's near-chance K=1 result is an observed finding about learning from 102 images without
  pretraining. It is not a bug, and it does not justify changing the recipe.

Per the protocol, no hyperparameter search was run. Tuning on these validation numbers would bias
the later comparison.

## Frozen candidate recipe for P5

- **Recipe:** `configs/protocol-v1.yaml`, sha256 `4657e85fa33b3d3475b58e8cde4a898190d7970cf2da8513486ca5fcaa1c8051`. No change.
- **Matrix:** `configs/full-study.yaml`. Its dry run gives 160 logical slots and 156 unique runs;
  all pre-flight checks PASS (`reports/fewshotlab-v1-full/plan.md`).
- **Fresh runs:** P5 trains the matrix afresh under category `core`. The MVP runs are not
  reused.
- **Before any test access, P5 must still:**
  1. implement and test the corruption generator and scorer;
  2. pick the NEU K=10 seed-11 demo checkpoint and its review threshold on validation;
  3. seal the plan with `lock-test`.

## What remains uncertain

- **Seed variability:** with n = 1 seed, the frozen vs partial gaps (0.009 at K=1, 0.018 at K=10)
  are smaller than any variability we have not yet measured. No ranking among the transfer methods
  is claimed.
- **Held-out performance:** unknown until the sealed P5 test evaluation. The validation values are
  biased upward by checkpoint selection.
- **Repeatability on MPS:** run-to-run repeatability of training on MPS is unmeasured.
- **Shortcuts:** contamination from official-split duplicates (D-016) and possible text shortcuts
  remain disclosed limitations.
