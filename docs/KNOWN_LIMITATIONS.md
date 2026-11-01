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
- **Scores are uncalibrated model scores**, not probabilities that a prediction is correct. A
  low-score review rule is not out-of-distribution detection.

## Reproducibility

- Results are not bitwise reproducible across platforms, devices or library versions. Each claim
  will state its scope: the same device and environment, within a declared tolerance.
- Only an Apple M4 Pro (MPS and CPU) has been used so far. CUDA paths are untested.
- The test gate is a process control with an audit trail, not cryptographic isolation from the
  repository owner.

## Environment and access (observed 2026-11-01)

- The Python 3.11 lock uses numpy 2.4.6 and scipy 1.17.1, because newer releases require 3.12.
- The Oxford Flowers102 hosts (`www.robots.ox.ac.uk`, `thor.robots.ox.ac.uk`) resolve in DNS, but
  TCP connections from this network time out, while other hosts respond. Flowers acquisition may
  therefore be BLOCKED in P1. No substitute source will be used.
- The GitHub account used here has read-only access to the project repository, so pushes are
  BLOCKED until write access is granted.
- No license has been chosen for the code yet.
