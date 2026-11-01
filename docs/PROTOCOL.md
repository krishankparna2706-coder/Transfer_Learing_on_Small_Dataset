# FewShotLab protocol v1 (`fewshotlab-v1`)

This is a specification. Every number below is a protocol setting or a dataset fact, **not a
measured result**, and every benchmark metric is NOT_RUN until real runs exist. The
machine-readable constants are in `configs/protocol-v1.yaml`, which the tests check against
`src/fewshotlab/config.py`. Sources are listed in [SPEC_SOURCES.md](SPEC_SOURCES.md).

## 1. Question

How much does an ImageNet-pretrained ResNet18 help when each class has only K ∈ {1, 3, 5, 10}
labeled training images, and how do cost, variability and errors change across four adaptation
methods?

This is supervised limited-label transfer over a fixed class set. It is not episodic novel-class
meta-learning. A negative or inconclusive answer is a valid outcome.

## 2. Datasets and splits

**Flowers102 (main study)** [S1, S2, S11]
- 8,189 images in 102 classes.
- Official split from `setid.mat`:

  | Split | Key | Images |
  |---|---|---:|
  | train | `trnid` | 1,020 |
  | validation | `valid` | 1,020 |
  | test | `tstid` | 6,149 |

- Train and validation hold exactly 10 images per class.
- Labels:
  - Raw MAT labels run 1..102; subtract 1 exactly once.
  - torchvision's `Flowers102` already returns 0..101, so never subtract again.
- The official split is never merged, re-drawn or modified.
- A confirmed cross-split duplicate is documented. Any cleaned variant becomes a separately named
  sensitivity protocol.

**NEU-CLS (secondary, industrial domain)** [S4]
- The original NEU-CLS classification archive from the authors: 1,800 grayscale 200×200 images,
  300 per class.
- Fixed class order:

  | ID | Class | Prefix |
  |---:|---|---|
  | 0 | rolled-in scale | RS |
  | 1 | patches | Pa |
  | 2 | crazing | Cr |
  | 3 | pitted surface | PS |
  | 4 | inclusion | In |
  | 5 | scratches | Sc |

- NEU-DET (detection) and NEU-CLS-64 are rejected. There is no normal or good class.
- The split is project-owned, not official:
  - per class: 180 train-pool, 60 validation, 60 test;
  - overall: 1,080 / 360 / 360;
  - split_seed 2026, created after the duplicate and group audit.
- Exact-duplicate clusters stay in one split. Any count change is published as a revised protocol
  before experiments.

## 3. Label budget (always reported together)

| K | Flowers fit | Flowers validation | NEU fit | NEU validation |
|---|---:|---:|---:|---:|
| 1 | 102 | 1,020 | 6 | 360 |
| 3 | 306 | 1,020 | 18 | 360 |
| 5 | 510 | 1,020 | 30 | 360 |
| 10 | 1,020 | 1,020 | 60 | 360 |

Transfer methods also rely on ImageNet pretraining, which is external data. A run is never
described as learning from only K labels in total.

## 4. Support sets

- **Seeds:** 11, 22, 33, 44, 55.
- **Construction:** for each dataset, seed and class:
  1. sort the class's train IDs;
  2. permute them with a class-local NumPy PCG64 generator seeded from
     SHA-256(dataset, "support", seed, class);
  3. take nested prefixes K = 1 ⊂ 3 ⊂ 5 ⊂ 10.
- **Pairing:** every method receives identical support IDs for a given dataset, K and seed.
- **Unused images:** NEU train-pool images outside the support are never used for fitting,
  prototypes, normalization or thresholds.
- **Flowers K=10:** every seed uses all 10 training images per class. Support variation is
  therefore zero, and the deterministic prototype result counts as one unique run, not five.
- **Authority:** the stored manifests are the record, not a re-run of the RNG. A rerun reads them
  and verifies their hash.

## 5. Preprocessing

Every path:
1. decodes the image safely;
2. applies EXIF orientation consistently;
3. converts to RGB (grayscale is replicated to three channels);
4. scales to [0,1];
5. normalizes with ImageNet mean (0.485, 0.456, 0.406) and std (0.229, 0.224, 0.225).

Interpolation is bilinear with antialiasing.

| Dataset / stage | Transform |
|---|---|
| Flowers train | RandomResizedCrop(224, scale=(0.7, 1.0), ratio=(3/4, 4/3)) + horizontal flip p=0.5 |
| Flowers val / test / prototype support | resize shorter side to 256, center crop 224 |
| NEU train | full-frame resize to 224×224 + horizontal flip p=0.5 + vertical flip p=0.5 |
| NEU val / test / prototype support | full-frame resize to 224×224 |

Additional rules:
- No hue jitter.
- No statistics are fitted on validation or test images.
- No test-time augmentation.
- Scratch, frozen and partial share one training augmentation per dataset.
- Prototypes use the deterministic evaluation transform for both support and query images.
- A cached deterministic-feature variant of `frozen` would be a separately named protocol.

## 6. Methods

| Method | Start | Trainable | BatchNorm |
|---|---|---|---|
| scratch | random init (`weights=None`) | all parameters | normal training statistics |
| frozen | IMAGENET1K_V1 | new `Linear(512, C)` only | encoder in eval mode; running stats frozen |
| partial | IMAGENET1K_V1, fresh each run | layer4 + new head | all running stats frozen; affine params trainable only inside layer4 |
| prototypes | IMAGENET1K_V1, frozen, eval | nothing (no optimizer) | eval mode |

**All methods**
- Module modes are re-applied after every `model.train()` call. Tests check gradients, parameter
  deltas and BatchNorm buffers.
- Every run builds a fresh model from its declared starting weights. There are no warm starts
  across methods, datasets, seeds or K.

**Prototypes**
1. Reject non-finite embeddings, non-finite norms, and norms ≤ 1e-12. Nothing is clamped.
2. L2-normalize each support and query embedding.
3. Each prototype is the arithmetic mean of its class's normalized support embeddings, and is not
   normalized again.
4. Predict the class with the minimum squared Euclidean distance; ties go to the lowest class ID.
5. The display score is softmax(−d²) at scale 1.

This is a nearest-prototype transfer baseline, not an episodically trained Prototypical
Network [S6].

**Optional DINOv2 ViT-S/14** frozen features are a later, separately labeled comparison. Its
different pretraining corpus confounds any architecture-only conclusion.

## 7. Optimization and checkpoint selection

This is the v1 starting recipe, not a claim of optimality.

- **Loss and optimizer:** cross-entropy; AdamW with weight decay 1e-4, betas (0.9, 0.999), eps 1e-8.
  No scheduler. float32.
- **Learning rates:**
  - scratch: 1e-3 for every parameter;
  - frozen: head 1e-3;
  - partial: layer4 1e-4, head 1e-3.
- **Budget and batching:** at most 100 epochs; batch size min(32, number of support images).
  - A final singleton batch is merged into the previous one.
  - No support image is ever dropped (NEU K=1 has six images).
- **Checkpoint:** the highest full-precision validation macro-F1. Exact ties go to lower
  validation cross-entropy, then the earlier epoch.
- **Early stopping:**
  - A separate patience reference resets only when macro-F1 improves on it by ≥ 0.001.
  - Training stops after 15 epochs without such an improvement.
  - min_delta never discards a better checkpoint.
- **Seeds:** train_seed = support_seed. Separate random streams are derived from it for sampling,
  initialization, augmentation and data loading.

## 8. Experiment matrix

- **MVP (P4):** Flowers, K ∈ {1, 10}, all four methods, seed 11, validation only. Eight
  exploratory jobs.
- **Full study (P5):** 2 datasets × 4 K × 4 methods × 5 seeds = 160 logical slots.
  - At most 156 are unique, after deduplicating the identical Flowers K=10 prototype slots and
    any other prototype slots whose support IDs are identical.
- **Bookkeeping:**
  - Debugging and tuning jobs use a separate ledger category.
  - All prespecified completed, failed and invalidated runs are published.
  - No favorable seed is selected.
  - The recipe and selection rules freeze before test access.

## 9. Metrics

These are definitions, not results. They are computed over all C declared classes, with 0
substituted for any undefined class term [S10]:

```
top1 = correct / N
precision_c = TP_c / (TP_c + FP_c)
recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 TP_c / (2 TP_c + FP_c + FN_c)
macro_F1 = mean over classes of F1_c          (primary endpoint)
balanced_accuracy = mean over classes of recall_c
```

Also reported:
- per-class precision, recall, F1 and support;
- raw and row-normalized confusion matrices (true classes as rows);
- Flowers top-5 (secondary).

Metric fractions are stored in [0, 1]. A missing value is null, never 0.

Chance levels are references only, not baselines or thresholds: 1/102 ≈ 0.98 % for Flowers and
1/6 ≈ 16.67 % for NEU.

## 10. Aggregation and uncertainty

For each dataset, K and method, report:
- per-seed values;
- the mean and sample SD;
- the number of distinct runs.

Rules:
- Method differences are paired by dataset, K and seed on identical supports.
- Predictions from repeated runs are never pooled as extra test images.
- Optional intervals must state what variability they cover:
  - a t-interval over seeds;
  - a class-stratified image bootstrap with 2,000 resamples, seed 2026.

## 11. Test gate

The test split is locked. Implemented in P5 (`fewshotlab.engine.seal`, `fewshotlab.engine.test_eval`).

**Before test access**
1. Finish the data audit, the tests and validation-only development.
2. Preview with `lock-test --plan PATH --dry-run`. It writes `reports/<study>/seal_preview.json`,
   seals nothing and lists anything that blocks sealing.
3. Seal with `lock-test --plan PATH`. Sealing refuses unless:
   - every planned slot is COMPLETED with verified evidence (a partial benchmark needs
     `--allow-partial` and is labelled partial);
   - the code tree is clean;
   - no test-split output exists yet;
   - the current code re-scores every sealed run's validation set with exactly the stored
     predictions.
4. The sealed plan (`reports/<study>/sealed_plan.json`) records:
   - the code digest (src/, configs/, lock files), protocol-module hashes, library versions and
     the evaluation device;
   - split-manifest, class-map and support hashes, and the expected test IDs (count and hash);
   - per run: attempt, checkpoint or prototype hash, model-spec, validation-prediction and
     validation-metric hashes, and the training commit;
   - pretrained-weight hashes, metric definitions, the review scorer, calibration (none) and the
     corruption protocol;
   - the demo checkpoint and threshold, the replay manifest hash, the test plan and the report rules.
5. Its SHA-256 and a random audit token go to `reports/<study>/evaluation_ledger.jsonl` and the
   run ledger.

**Test access** (`evaluate --split test --study PATH`)
- Every access first re-derives the plan from the current files. Any difference (a changed
  support, checkpoint, scorer, threshold, corruption protocol, test ID list, code or environment)
  refuses access before any test image is read.
- Each sealed run is evaluated once on its dataset's full test split. `predictions_test.csv` and
  `test_metrics.json` go into the run's attempt folder, and the file hashes go into the evaluation
  ledger. A recorded evaluation is never repeated, and an unrecorded test file blocks evaluation.
- Test aggregates are written only for runs with recorded, hash-matching evaluations.

**After sealing**
- Changing any sealed item invalidates the seal.
- A scientific correction after test access marks the affected results INVALIDATED (history is
  kept) and reruns them under a new version.
- Post-test tuning is a new, exploratory protocol.

The gate is a process control, not cryptographic isolation.

## 12. Industrial replay and corruptions

Implemented in P5 (`fewshotlab.simulation`). Nothing has been run on test yet.

- **Demo checkpoint:** among the NEU K=10, seed 11 runs of the four methods, the one with the
  highest validation macro-F1. Exact ties go to the lexically first method name. All four
  candidates must be COMPLETED and verified. It is chosen and sealed before test access.
- **Review threshold:**
  - Candidate values: {0, 0.05, …, 1}.
  - Score: the maximum class score (softmax of logits; softmax of −d² for prototypes).
    It is uncalibrated.
  - Coverage = the fraction of validation predictions with max score ≥ threshold.
  - Pick the highest threshold that keeps ≥ 80 % coverage on that model's validation predictions.
  - Scores strictly below the threshold are flagged for review. If nothing is accepted, the
    accepted-error rate is null.
  - The 80 % is a workflow setting, not a safety target.
- **Corruptions**, applied to decoded [0,1] pixels at native resolution, before the evaluation
  resize and normalization:

  | Condition | Level 1 | Level 2 | Rule |
  |---|---|---|---|
  | brightness | ×0.6 | ×1.4 | clipped to [0, 1] |
  | Gaussian blur | σ = 1 px | σ = 2 px | separable kernel, 2·ceil(3σ)+1 taps, reflect padding |
  | Gaussian noise | SD 0.03 | SD 0.08 | one draw per decoded pixel, clipped to [0, 1] |

  - Grayscale images (NEU) are corrupted as one channel, then replicated to RGB.
  - Noise seeds are the uint64 from SHA-256 of the JSON list [parent_id, condition, severity, 703].
  - Each derivative records its parameters, seed, implementation and pixel checksum.
  - Clean images and the six conditions are reported separately. Each derivative is paired with
    its clean parent and never counted as a new test image.
  - Derivatives inherit the parent's split and are never used for training.
- **Replay order:** the NEU test IDs are sorted, then permuted once with the generator seeded from
  ["neu_cls", "replay-order", 703]. The clean stream comes first, then each condition over the
  same parents in the same order. The order ignores labels; the true class is copied in for
  demonstration mode. The manifest (`replay_manifest.csv`) is sealed by hash.
- The replay never labels an image good, safe or defect-free. It classifies known defect types in
  stored images.

## 13. Run records and statuses

- **Statuses:** NOT_RUN, RUNNING, COMPLETED, FAILED, INVALIDATED.
- **Each run stores:**
  - `config.resolved.yaml`, `environment.json`, `provenance.json`;
  - `train_history.csv` (learned methods);
  - `best.ckpt` or `prototypes.npz`;
  - `val_metrics.json`, predictions and plots.
- `test_metrics.json` and `predictions_test.csv` exist only after the gate, each recorded in the
  study's evaluation ledger.
- Fixture artifacts carry `is_fixture = true` and are rejected by aggregation.
- The run ID combines protocol, dataset, method, K, seed and a config hash. Nothing is overwritten.

## 14. Change discipline

Any change to data, classes, preprocessing, optimizer, checkpoint rule or metric definition
creates a new protocol or config identity. Old results keep their status, and a stronger model is
never silently substituted into an earlier benchmark.
