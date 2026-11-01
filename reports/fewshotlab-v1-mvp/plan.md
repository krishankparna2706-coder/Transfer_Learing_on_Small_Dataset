# Study plan: fewshotlab-v1-mvp

8 logical slots, 8 unique runs (0 deterministic duplicates counted once). Category `mvp`. Generated 2026-11-01T15:48:16Z.

## Pre-flight checks

- PASS unique_run_ids: 8 unique of 8
- PASS shared_support_ids: every method at a dataset/K/seed reads the same support manifest
- PASS initialization_policy: scratch starts from random init; frozen/partial/prototypes from IMAGENET1K_V1
- PASS paired_init_seed: init seed depends only on train_seed (= support_seed)
- PASS identical_training_transform_per_dataset: {'flowers102': ['b5bbd7ad654b4be4']}
- PASS single_recipe: 1 distinct recipe(s) across learned slots

## Slots

| dataset | K | method | seed | status | run ID |
|---|---:|---|---:|---|---|
| flowers102 | 1 | scratch | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__scratch__k1__s11__cfg8f124a29` |
| flowers102 | 1 | frozen | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__frozen__k1__s11__cfg5c36cc8c` |
| flowers102 | 1 | partial | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__partial__k1__s11__cfga97c33c5` |
| flowers102 | 1 | prototypes | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__prototypes__k1__s11__cfg2f1baf0f` |
| flowers102 | 10 | scratch | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__scratch__k10__s11__cfg5a3c0b1a` |
| flowers102 | 10 | frozen | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__frozen__k10__s11__cfg9d773405` |
| flowers102 | 10 | partial | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__partial__k10__s11__cfg6c553c7e` |
| flowers102 | 10 | prototypes | 11 | COMPLETED | `mvp__fewshotlab-v1__flowers102__prototypes__k10__s11__cfg07e5ae7e` |
