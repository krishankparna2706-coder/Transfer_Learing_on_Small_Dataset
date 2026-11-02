# Model card: FewShotLab ResNet18 few-shot classifiers (protocol fewshotlab-v1)

## Shared facts

- **Architecture:** torchvision ResNet18 with a replaced classifier, `Linear(512, C)`
  (prototypes have no head).
- **Pretrained weights:** `ResNet18_Weights.IMAGENET1K_V1`, file `resnet18-f37072fd.pth`, SHA-256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. They are fetched only with
  an explicit `weights fetch`.
- **Datasets and class maps:**
  - Flowers102: 102 classes; class-map hash in the sealed plan; display names are a secondary
    source.
  - NEU-CLS: 6 classes.
  - See the [data cards](DATA_CARD_flowers102.md) and [inventory](RELEASE_INVENTORY.md).
- **Preprocessing** (`data/transforms.py`):
  - **Flowers evaluation:** resize to 256, then a 224 center crop. Training adds
    RandomResizedCrop (scale 0.7–1) and a horizontal flip.
  - **NEU:** the full frame resized to 224. Training adds horizontal and vertical flips.
  - **All:** ImageNet mean and std; grayscale replicated to 3 channels.
- **Label budget per run:**
  - support: K images per class (Flowers 102×K, NEU 6×K), drawn with seeds 11/22/33/44/55;
  - validation, used only to choose the checkpoint: 1,020 (Flowers) and 360 (NEU) labels.
- **Training:** AdamW, weight decay 1e-4, at most 100 epochs, patience 15. Selection is the best
  validation macro-F1, with ties going to the lower loss, then the earlier epoch.
- **Evaluation:**
  - every run was sealed (`seal-c6f58862d63b586a0df8dbcf140db0fd`) before test access and
    evaluated on the test split once;
  - macro-F1 is primary; top-1 and balanced accuracy are also reported;
  - values are the mean ± sample SD over distinct runs.
- **Scores:** uncalibrated model scores (softmax of logits, or of negative squared prototype
  distances).

## Methods

Held-out test macro-F1, mean over 5 support seeds. K=3 and K=5, the SDs and the paired
differences are in `reports/fewshotlab-v1-full/test/aggregate_test.md`.

| Method | Starts from | What trains | Flowers K=1 | Flowers K=10 | NEU K=1 | NEU K=10 |
|---|---|---|---:|---:|---:|---:|
| scratch | random initialization | every layer | 0.061 | 0.411 | 0.436 | 0.779 |
| frozen | IMAGENET1K_V1 | the new linear head only (encoder fixed, BatchNorm statistics frozen) | 0.406 | 0.825 | 0.855 | 0.962 |
| partial | IMAGENET1K_V1 | layer4 (learning rate 1e-4) and the head (1e-3); other BatchNorm frozen | 0.408 | 0.853 | 0.852 | 0.970 |
| prototypes | IMAGENET1K_V1 | nothing: each class is the mean of its K normalized embeddings | 0.394 | 0.751 (1 distinct run) | 0.815 | 0.944 |

**Checkpoints:**
- each run keeps `best.ckpt` (learned methods) or `prototypes.npz`;
- all 156 are listed by hash in the sealed plan, and all 156 match on disk;
- the demo checkpoint is partial, NEU, K=10, seed 11 (SHA-256 `830dbab5…`).

## Intended use

- **For:** studying label efficiency with 1 to 10 labeled images per class, on these two
  datasets, under this protocol.
- **Demonstration:** classifying images among six known NEU-CLS defect types, with a human
  review rule.

## Not intended / prohibited interpretations

- **Not a quality-inspection system:**
  - it is not a good-versus-defective decision, a defect locator or a safety claim;
  - NEU-CLS has no defect-free class.
- **Scores are not probabilities.** An image from outside the learned classes still receives a
  score for every class.
- **No transfer of results:** they do not carry over to other cameras, products or datasets
  without new evaluation.
- **Validation numbers are optimistic,** because checkpoints were selected on them.

## Known failure modes

- **Few labels:** scratch is weak at every K; transfer methods lead at every K on both datasets.
- **NEU confusion:** the sealed NEU demo model confuses inclusion with scratches. 3 of its 9
  clean-test errors are accepted with model scores above 0.97 (fixed error gallery,
  [REHEARSAL.md](REHEARSAL.md)).
- **Corruption:** strong blur (σ 2 px) and noise (SD 0.08) cut macro-F1 by 0.24 and 0.30 and send
  about a third of the images to review (`test/robustness.md`).
