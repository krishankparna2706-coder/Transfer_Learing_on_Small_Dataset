# Compute estimate

**ESTIMATE (not an observed runtime).** Generated 2026-11-01T14:55:57Z from the measured diagnostic runs listed below. Training durations are not promised.

| Scope | Lower bound (16 epochs) | Upper bound (100 epochs) |
|---|---:|---:|
| MVP, 8 Flowers jobs | 0.14 h | 0.82 h |
| Full matrix, 156 unique runs (training) | 1.56 h | 9.33 h |

Test evaluation of every unique run (P5): about 0.36 h.

## Inputs (measured)

- `flowers102/frozen` from `diagnostic__fewshotlab-v1__flowers102__frozen__k10__s11__cfg09aef7c2/a001` on mps: fit_s_per_image 0.002668, val_s_per_image 0.002626, first_epoch_fit_s 2.866, steady_epoch_fit_s 2.722, steady_epoch_val_s 2.679
- `flowers102/partial` from `diagnostic__fewshotlab-v1__flowers102__partial__k10__s11__cfg506dc445/a001` on mps: fit_s_per_image 0.003231, val_s_per_image 0.002644, first_epoch_fit_s 3.473, steady_epoch_fit_s 3.295, steady_epoch_val_s 2.697
- `flowers102/prototypes` from `diagnostic__fewshotlab-v1__flowers102__prototypes__k10__s11__cfg40f6fea9/a001` on mps: embed_s_per_image 0.002722, val_s_per_image 0.002646
- `flowers102/scratch` from `diagnostic__fewshotlab-v1__flowers102__scratch__k10__s11__cfg06c045da/a001` on mps: fit_s_per_image 0.005687, val_s_per_image 0.002688, first_epoch_fit_s 10.46, steady_epoch_fit_s 5.801, steady_epoch_val_s 2.742
- `neu_cls/partial` from `diagnostic__fewshotlab-v1__neu_cls__partial__k10__s11__cfgbc1dc080/a001` on mps: fit_s_per_image 0.006331, val_s_per_image 0.002107, first_epoch_fit_s 0.4741, steady_epoch_fit_s 0.3799, steady_epoch_val_s 0.7587

## Assumptions

- per-image costs come from the last (steady-state) epoch of each diagnostic run
- learned runs take between 16 (earliest early stop) and 100 epochs
- fit cost scales linearly with support images; validation cost with validation images
- neu_cls scratch fit cost = neu_cls partial cost x Flowers scratch/partial ratio (1.76)
- neu_cls frozen fit cost = neu_cls partial cost x Flowers frozen/partial ratio (0.83)

Not included: data preparation, report generation, failure contingency, corruption study (P5).
