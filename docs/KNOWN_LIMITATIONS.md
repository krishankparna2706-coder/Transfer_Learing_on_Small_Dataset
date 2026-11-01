# Known limitations

This list states what FewShotLab cannot claim. It is updated as phases add evidence.

## Scientific scope

- **Closed-set classification only.** NEU-CLS has no normal or good class, so the project
  classifies known defect types in stored images. It does not detect defective versus normal
  material, localize defects, estimate severity, certify inspection, or show factory performance.
- **K is not the whole label budget.** Checkpoint selection uses the full validation split (1,020
  Flowers or 360 NEU labels), and the transfer methods use ImageNet pretraining.
- **Pretraining overlap cannot be ruled out.** Split audits cannot prove that ImageNet never
  contained related or overlapping images. Target-split leakage and pretraining provenance are
  separate issues.
- **The NEU split is image-level.** Unless acquisition-group metadata turns up (to be checked in
  P1), the split cannot establish generalization to new coils, cameras or production lines.
- **Seeds conflate two kinds of variability.** The five paired seeds combine support choice and
  training randomness (train_seed = support_seed). At Flowers K=10, support variation is zero and
  prototypes produce a single deterministic result.
- **The recipe is a preregistered starting point**, not tuned to optimality for each method.
- **P4 MVP numbers are validation-only, single-seed (n = 1) and optimistic**, because checkpoints
  were selected on the same validation images. They are exploratory, not benchmark results.
- **Some Flowers images contain visible text** (for example a species caption), so a model could in
  principle exploit text. This is not systematically audited.
- **Scores are uncalibrated model scores**, not probabilities that a prediction is correct. A
  low-score review rule is not out-of-distribution detection.

## Reproducibility

- Results are not bitwise reproducible across platforms, devices or library versions. Each claim
  will state its scope: the same device and environment, within a declared tolerance.
- Only an Apple M4 Pro (MPS and CPU) has been used so far. CUDA paths are untested.
- On MPS, deterministic algorithms are requested in warn-only mode (D-029). Inference and checkpoint
  reload reproduced predictions exactly in five diagnostic runs. Whether repeated training runs
  match on MPS has not been measured yet; on CPU they match within the declared tolerance.
- The compute estimate (`reports/compute_estimate.md`) assumes linear scaling with image counts and
  extrapolates NEU scratch and frozen costs from Flowers ratios. It is an estimate, not a
  measurement.
- The test gate is a process control with an audit trail, not cryptographic isolation from the
  repository owner.

## Environment and access (observed 2026-11-01)

- The Python 3.11 lock uses numpy 2.4.6 and scipy 1.17.1, because newer releases require 3.12.
- The official Oxford host was unreachable from this network. The Flowers102 files came from
  Internet Archive captures of the official URLs, accepted only after an exact MD5 match with
  torchvision's record (D-012).
- The Flowers102 usage terms could not be re-checked online. NEU-CLS states no explicit license;
  its authors ask users to cite their work. Neither dataset is redistributed.
- **Audit findings:**
  - The Flowers102 official split contains 2 byte-identical cross-split pairs (one of them
    train↔test), 6 further cross-split near-duplicate candidates, and 1 near-identical pair with
    different labels.
  - NEU-CLS test (patches) contains one pixel-identical pair.
  - Human visual review of the candidates is pending (D-016).
- The project folder sits in iCloud Drive. The virtualenv was moved outside it
  (`~/.local/share/fewshotlab/venv`, linked as `.venv`) to keep Python working (D-014). Raw data
  and caches inside the project still sync, and may be evicted to the cloud by macOS storage
  optimization.
- No license has been chosen for the code yet.
