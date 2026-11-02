# Seven-minute judge script

An evidence walkthrough, timed to the evaluation protocol (document 02, p.20).

- **Before you start:** open the app with `python -m fewshotlab demo` at least a minute early, so
  the first load (about 17 s) is done.
- **What is live:** training finished before the demonstration. The app only runs inference and
  reads stored, hash-checked evidence.
- **Where the numbers come from:** the sealed reports (`reports/fewshotlab-v1-full/test/`) and
  the rehearsal record (`demo_assets/fewshotlab-v1-full/rehearsal.md`). The app shows the same
  values live from the registry.

| Time | Page and action | Evidence on screen |
|---|---|---|
| 0:00–1:00 | **Overview.** State the problem and the four budgets. | The research question; the path from images to evidence; the four methods |
| 1:00–2:00 | **Data and label budget**, Flowers. Count both label kinds. | Official split: 1,020 / 1,020 / 6,149; manifest and class-map SHA-256; K=1 = 102 support labels + 1,020 validation labels; the real K=1 support images |
| 2:00–3:30 | **Experiment explorer**, Held-out test, both datasets. | Mean ± SD by K, every seed in its own panel, paired differences, run IDs |
| 3:30–5:30 | **Industrial replay**: the clean stream, then Gaussian noise SD 0.08; reveal the truth. | Live predictions with measured latency, corruption parameters, the sealed threshold 0.95, counts with their denominators |
| 5:30–6:30 | **Industrial replay** error gallery, then **Reproducibility**: trace a number. | A real error; macro-F1 recomputed from its prediction file, with the checkpoint hash |
| 6:30–7:00 | **Overview**: "What this app does not do". | No defect-free class; no factory claim; the next experiment |

## 0:00–1:00: the problem

**Say:** "With 1, 3, 5 or 10 labeled images per class, how much does a pretrained representation
help? We adapt one ResNet18 in four ways: scratch, frozen features with a new head, partial
fine-tuning of the last block, and class prototypes. Each class supplies K training images, and a
separate labeled validation set picks each checkpoint. Both budgets are always reported."

## 1:00–2:00: dataset and label budget

**Do:** on the Data page, choose Oxford Flowers102.

**Show:**
- The official split: 1,020 train, 1,020 validation, 6,149 test images, MD5-checked source
  files.
- The manifest and class-map hashes.
- The label budget: one-shot Flowers uses 102 support labels plus 1,020 validation labels.
- The real K=1 support images.

## 2:00–3:30: measured comparisons (held-out test)

**Do:** on the Explorer, choose Split = Held-out test; show Flowers, then NEU.

**Say:** "Every run was sealed before test access and evaluated on test once."

**Read off these values** (macro-F1, mean ± SD over 5 support seeds):

| Dataset, K | scratch | frozen | partial | prototypes |
|---|---:|---:|---:|---:|
| Flowers, K=1 | 0.061 ± 0.008 | 0.406 ± 0.028 | 0.408 ± 0.020 | 0.394 ± 0.020 |
| Flowers, K=10 | 0.411 ± 0.022 | 0.825 ± 0.003 | 0.853 ± 0.017 | 0.751 (1 run) |
| NEU, K=1 | 0.436 ± 0.156 | 0.855 ± 0.055 | 0.852 ± 0.063 | 0.815 ± 0.064 |
| NEU, K=10 | 0.779 ± 0.056 | 0.962 ± 0.009 | 0.970 ± 0.010 | 0.944 ± 0.021 |

**Point to a measured difference with its variation:**
- every transfer method beats scratch in 5 of 5 seeds at every K;
- partial beats frozen in every seed at Flowers K=5 and at NEU K=3 to K=10, and in 4 of 5 seeds
  at Flowers K=10, but not at K=1;
- prototypes trail frozen.

## 3:30–5:30: industrial replay

**Do:**
1. Choose the clean stream in Live inference mode and press Play.
2. After a few images, switch to "stress: Gaussian noise SD 0.08".
3. Turn on Reveal true labels.

**Show:**
- the predicted defect type and model score;
- the review decision against the sealed threshold 0.95;
- the measured latency for each request (about 7 ms end to end on CPU; 1 image per second is
  only the playback pace);
- each noisy image next to its clean parent.

**Say:** "The threshold was chosen on validation before test: the highest value keeping at
least 80 % validation coverage."

**Read coverage together with errors:**

| Stream | Coverage | Error among accepted | Macro-F1 |
|---|---:|---:|---:|
| Clean | 94.4 % | 0.9 % | 0.975 |
| Noise SD 0.08 | 64.2 % | 8.7 % | 0.677 |

## 5:30–6:30: one real error and the trace

**An error:** open the fixed error gallery. Its rule was set in advance: every clean test image
the model gets wrong, sorted by ID.
- `neu_cls/In_228`, an inclusion, is predicted "scratches" at model score 0.989 and accepted. A
  confident error that the review rule does not catch.
- Describe only what is visible in the image. Do not claim to know why the network decided as it
  did.
- **The next experiment:** more inclusion and scratch examples at K=10, then a check on whether
  this confusion shrinks.

**The trace:** on Reproducibility, under "Trace a number", choose NEU-CLS, Held-out test, K=10,
partial, seed 11.
- The app shows macro-F1 0.975 and recomputes it from
  `runs/artifacts/fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa/a001/predictions_test.csv`
  (360 rows): identical.
- It also shows the checkpoint `best.ckpt` (SHA-256 `830dbab5…`) and the seal token.

## 6:30–7:00: limits

**Say:**
- "NEU-CLS has no defect-free class, so this replay cannot show false accepts on normal parts.
  Nothing here is called safe."
- "The corruptions are synthetic; a new camera, real normal products and a factory process need
  their own evaluation."
- "Scores are uncalibrated model scores."

**The next experiment:** a labeled defect-free class and real capture variation.

## If something fails

See [REHEARSAL.md](REHEARSAL.md), "Backup launch steps".
