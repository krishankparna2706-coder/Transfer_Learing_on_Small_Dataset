# fewshotlab-v1-full: robustness of the sealed demo checkpoint

> **ROBUSTNESS STUDY on held-out test images under controlled synthetic corruption. Reported apart from the primary benchmark. Each derivative is paired with its clean parent and is not a new test image. The review threshold was fixed on validation before test access; 80 % coverage is a workflow setting, not a safety guarantee.**

Run `fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa` (partial), threshold 0.95, evidence VERIFIED.

| Condition | Macro-F1 | Top-1 | Coverage | Error among accepted | Review rate | Delta macro-F1 vs clean | Changed predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean | 0.975 | 0.975 | 0.944 | 0.009 | 0.056 | (reference) | - |
| brightness-1 | 0.975 | 0.975 | 0.944 | 0.003 | 0.056 | -0.000 | 8 |
| brightness-2 | 0.914 | 0.914 | 0.808 | 0.021 | 0.192 | -0.061 | 24 |
| gaussian_blur-1 | 0.916 | 0.917 | 0.803 | 0.017 | 0.197 | -0.059 | 22 |
| gaussian_blur-2 | 0.731 | 0.761 | 0.656 | 0.089 | 0.344 | -0.244 | 80 |
| gaussian_noise-1 | 0.945 | 0.947 | 0.900 | 0.015 | 0.100 | -0.030 | 15 |
| gaussian_noise-2 | 0.677 | 0.722 | 0.642 | 0.087 | 0.358 | -0.298 | 102 |

Coverage and error among accepted predictions are read together: a low accepted error obtained by reviewing nearly everything comes with low coverage. NEU-CLS has no normal class, so nothing here measures false accepts on defect-free material.
