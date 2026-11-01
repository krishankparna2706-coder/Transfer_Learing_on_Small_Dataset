# FewShotLab

FewShotLab is a reproducible experiment workbench for one question: **how much does a pretrained
visual representation help when each class has only 1, 3, 5 or 10 labeled training images, and
how do cost, variability and errors change across adaptation methods?**

> **Status:** see [docs/PROGRESS.md](docs/PROGRESS.md) for the current phase and its evidence.
> Until real runs exist, **every benchmark metric is NOT_RUN**. Commands that belong to later
> phases exit with code 3 ("not implemented") and write nothing.

## Scope

- **Main study: Oxford Flowers102**, official split (1,020 train / 1,020 validation / 6,149 test
  images, 102 classes). Every class is present at every shot budget.
- **Secondary study: NEU-CLS** steel-surface defect *type* classification (1,800 grayscale
  200×200 images, 6 defect classes) with a project-owned 180/60/60-per-class split (seed 2026).
  NEU-CLS has no "normal" class, so this is closed-set defect-type classification. It is not
  defect detection, localization or certified inspection.
- **Encoder:** torchvision ResNet18 with `ResNet18_Weights.IMAGENET1K_V1` (512-d features).
- **Four methods:**
  - `scratch`: random init, all layers train.
  - `frozen`: frozen ImageNet encoder plus a new linear head.
  - `partial`: layer4 and the head train; all BatchNorm statistics stay frozen.
  - `prototypes`: nearest class mean of L2-normalized frozen embeddings.
- This is **supervised limited-label transfer** over a fixed class set. It is not an episodic
  novel-class meta-learning benchmark. The prototype method is a nearest-prototype transfer
  baseline, not a trained Prototypical Network.

## The full label budget

"K-shot" counts only the images used for fitting. Checkpoint and recipe selection also use the
whole labeled validation split, and the transfer methods additionally rely on ImageNet
pretraining.

| K | Flowers fit labels | Flowers validation labels | NEU fit labels | NEU validation labels |
|---|---:|---:|---:|---:|
| 1 | 102 | 1,020 | 6 | 360 |
| 3 | 306 | 1,020 | 18 | 360 |
| 5 | 510 | 1,020 | 30 | 360 |
| 10 | 1,020 | 1,020 | 60 | 360 |

Held-out test images (Flowers 6,149; NEU 360) are used once, only after the evaluation plan is
sealed in phase P5.

## Validation versus test

- **Validation** chooses checkpoints and recipe details, so it is part of the label budget.
- **Test** is locked. No config can unlock it. Only the sealed `lock-test` workflow can, after
  every model, threshold and corruption setting has been fixed on validation data.

## Inputs and outputs

- **Input:** one RGB image (grayscale steel images are replicated to three channels).
- **Output:** one of 102 flower classes or 6 defect types, with ranked **model scores** and the
  producing run ID. Scores are not calibrated probabilities.

## Installation

These commands were run on 2026-11-01 (macOS arm64, uv 0.11.15, Python 3.11.15). You need
[uv](https://docs.astral.sh/uv/); it can provide Python 3.11 if it isn't installed.

```bash
uv sync --locked
uv run python -m fewshotlab doctor
uv run pytest -q
```

`uv.lock` is the authoritative lock. `requirements.lock` is exported from it for pip users
(`pip install -r requirements.lock`, then `pip install -e . --no-deps`). The pip route has not
been tested yet.

## Commands

All commands run as `python -m fewshotlab <command>`.

| Command | Phase | Status |
|---|---|---|
| `doctor [--json] [--require-device {cpu,mps,cuda}]` | P0 | implemented |
| `data prepare --dataset {flowers102,neu_cls} [--source PATH] [--allow-download] [--mirror internet-archive] [--cross-check PATH]` | P1 | implemented |
| `data audit --dataset {flowers102,neu_cls} [--near-dup-threshold BITS]` | P1 | implemented |
| `support build --config PATH [--dataset ...]` | P1 | implemented |
| `weights fetch` | P2 | implemented (explicit download of ResNet18 IMAGENET1K_V1) |
| `train --config PATH [--device ...] [--allow-download] [--resume]` | P3 | implemented (support + validation only) |
| `evaluate --run RUN_ID --split {val,test} [--device ...]` | P3 / P5 | `val` re-check implemented; `test` refused until the P5 seal (exit 5) |
| `benchmark --config PATH [--dry-run] [--device ...] [--allow-download]` | P4 | implemented (study plan, pre-flight checks, runs unfinished slots) |
| `lock-test --plan PATH` | P5 | not implemented yet |
| `report [--ledger PATH] [--out DIR] [--study PATH]` | P4 | implemented (validation-only comparison from stored predictions) |
| `demo [--port N]` | P6 | not implemented yet |

Exit codes:
- 0: success
- 1: the command ran and failed
- 2: usage error
- 3: not implemented yet
- 4: blocked by a missing prerequisite (dataset source, weights, device)
- 5: refused by a protocol guard

Nothing is downloaded implicitly. Datasets and pretrained weights are fetched only when a command
is given `--allow-download`.

## Repository layout

| Path | Responsibility |
|---|---|
| `src/fewshotlab/` | the library: CLI, config, data, models, engine, metrics, reporting, simulation |
| `app/` | Streamlit demonstration (P6), calling the same library as the CLI |
| `configs/` | protocol and run/study configs, validated strictly |
| `tests/` | offline CPU tests with generated fixtures |
| `data/manifests/` | immutable dataset and support manifests (no images) |
| `runs/` | append-only run records; only `ledger.jsonl` is tracked |
| `reports/` | analyses generated from stored predictions |
| `demo_assets/` | approved artifacts for the offline judge demo (P7) |
| `docs/` | protocol, decisions, progress, limitations, hardware, spec sources |

## Documentation

- [docs/PROTOCOL.md](docs/PROTOCOL.md): the experiment contract
- [docs/PROGRESS.md](docs/PROGRESS.md): evidence for each phase
- [docs/DECISIONS.md](docs/DECISIONS.md): design decisions
- [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md): what this project cannot claim
- [docs/HARDWARE.md](docs/HARDWARE.md): measured machine facts and the compute-estimate method
- [docs/SPEC_SOURCES.md](docs/SPEC_SOURCES.md): the specification documents and primary references

## Data, weights and licensing

No images, model weights or run outputs are committed. Datasets are acquired from their
original sources under their own terms; see [docs/DATA_CARD_flowers102.md](docs/DATA_CARD_flowers102.md)
and [docs/DATA_CARD_neu_cls.md](docs/DATA_CARD_neu_cls.md), and the generated
[reports/label_budget.md](reports/label_budget.md). No license has been chosen for this code yet.
