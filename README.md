# FewShotLab

FewShotLab is a reproducible experiment workbench for one question: **how much does a pretrained
visual representation help when each class has only 1, 3, 5 or 10 labeled training images, and
how do cost, variability and errors change across adaptation methods?**

> **Status:** see [docs/PROGRESS.md](docs/PROGRESS.md) for the current phase and its evidence.
> Validation numbers are development results. **Held-out test metrics** were measured once, on
> 2026-11-02, under the sealed plan `seal-c6f58862…` (`reports/fewshotlab-v1-full/test/`).
> Commands that belong to later phases exit with code 3 ("not implemented") and write nothing.

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
| `evaluate --split val --run RUN_ID [--device ...]` | P3 | implemented (re-scores a stored run's validation set; a verification, not a new result) |
| `evaluate --split test --study PATH [--run RUN_ID] [--device ...]` | P5 | implemented; refuses (exit 5) unless the study's sealed plan still matches every current file; each sealed run is evaluated once |
| `benchmark --config PATH [--dry-run] [--device ...] [--allow-download]` | P4 | implemented (study plan, pre-flight checks, runs unfinished slots) |
| `lock-test --plan PATH [--dry-run] [--skip-reproduction] [--allow-partial] [--device ...]` | P5 | implemented (`--dry-run` previews the seal and seals nothing) |
| `report [--ledger PATH] [--out DIR] [--study PATH]` | P4 / P5 | implemented (validation comparison and seed aggregates; test tables only after a sealed evaluation) |
| `predict --run RUN_ID --image PATH [--image PATH ...] [--attempt ID] [--top-k N] [--device ...] [--json]` | P6 | implemented (live prediction with one verified stored run: model scores, never a benchmark result; the app uses the same code) |
| `demo [--port N] [--host ADDR] [--study PATH] [--check] [--rehearse] [--device ...]` | P6 / P7 | implemented (offline Streamlit app on 127.0.0.1; `--check` verifies offline that it can start; `--rehearse` runs the whole sealed replay live and offline, times it and records it under `demo_assets/`) |

Exit codes:
- 0: success
- 1: the command ran and failed
- 2: usage error
- 3: not implemented yet
- 4: blocked by a missing prerequisite (dataset source, weights, device)
- 5: refused by a protocol guard

Nothing is downloaded implicitly. Datasets and pretrained weights are fetched only when a command
is given `--allow-download`.

## Reproduce the study

Run every command from the repository root as `uv run python -m fewshotlab ...`. Each step lists
the artifact it produces. The times were measured on an Apple M4 Pro (24 GB).

1. **Install and check:** `uv sync --locked`, then `python -m fewshotlab doctor`.
   - **Produces:** a device report (MPS or CPU).
   - `uv run pytest -q` runs the offline fixture tests in about 4 minutes. Real-data tests skip
     when the data is absent.
2. **Get the data from the original sources:** follow
   [DATA_CARD_flowers102.md](docs/DATA_CARD_flowers102.md) and
   [DATA_CARD_neu_cls.md](docs/DATA_CARD_neu_cls.md). The data is not redistributed here.
3. **Regenerate the manifests:**
   - `data prepare --dataset flowers102 --source DIR_WITH_102flowers.tgz_AND_MATS`
   - `data prepare --dataset neu_cls --source NEU-CLS.rar`
   - **Produces:** `data/manifests/*-v1.csv` and the class maps. The split manifests are
     byte-identical to the committed files, and the class-map identity hashes are identical
     (checked in a clean clone). Flowers display names are added only with `--cross-check PATH`,
     using the Kaggle copy.
   - Then `data audit --dataset neu_cls` (or `flowers102`) → `reports/data_audit/<dataset>/`.
   - Then `support build --config configs/protocol-v1.yaml` → 40 support manifests in
     `data/manifests/supports/`. Their content hashes equal the committed ones.
4. **One run:** `train --config configs/runs/diagnostic/flowers102-scratch-k1-s11.yaml`.
   - **Produces:** `runs/artifacts/<run ID>/a001/` (checkpoint, `predictions_val.csv`,
     `val_metrics.json`, provenance), in about 20 s.
   - Prototype runs also need `weights fetch` once (47 MB, network).
5. **The matrix:** `benchmark --config configs/full-study.yaml`.
   - Runs 156 unique runs (about 5.8 h of compute here). Finished runs are skipped on a rerun.
   - Writes a ledger entry for each run.
6. **Validation report:** `report --study configs/full-study.yaml` →
   `reports/fewshotlab-v1-full/` (comparison, seed aggregates, plots), regenerated from the
   stored predictions.
7. **The test-lock workflow, once:**
   - `lock-test --plan configs/full-study.yaml --dry-run` writes `seal_preview.json` (about 5
     min).
   - `lock-test --plan configs/full-study.yaml` writes `sealed_plan.json`. Commit it before the
     next step.
   - `evaluate --split test --study configs/full-study.yaml` scores every sealed run once and
     runs the robustness study (about 21 min).
   - Run `report --study ...` again → `reports/fewshotlab-v1-full/test/`.
8. **Offline demonstration:** `demo --check`, then `demo`. `demo --rehearse` runs a full timed
   offline rehearsal into `demo_assets/` (about 23 s).

The study in this repository has already completed steps 1 to 8. Its sealed results are in
`reports/fewshotlab-v1-full/`, and [docs/RELEASE.md](docs/RELEASE.md) gives the status, the
release gates and the evidence index.

## Limitations (summary)

- **No normal class:** NEU-CLS has no defect-free class, so false accepts on normal parts are
  not measured. The replay is a simulation on stored images.
- **Scores:** they are uncalibrated model scores. An image outside the learned classes still
  receives a score for every class.
- **Validation numbers:** they are optimistic; only the sealed test numbers are held out.
- **Seeds:** five seeds describe benchmark variability only; no population confidence intervals
  are given.
- **Not in git:** dataset images, checkpoints and weights. Their hashes are in
  [docs/RELEASE_INVENTORY.md](docs/RELEASE_INVENTORY.md).
- **Full list:** [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).

## Demonstration app

```bash
python -m fewshotlab demo --check
```

```bash
python -m fewshotlab demo
```

The first command checks offline, on the CPU, that the app can start. The second serves it on
http://127.0.0.1:8501 for this machine only. It needs no network once the environment is
installed; prototype models also need the cached ImageNet weights (`weights fetch`, once).

**Pages:**
- **Overview:** the question, the path from images to evidence, the evidence status and the
  headline held-out test table.
- **Data and label budget:** split counts and provenance, the label budget for every K, and the
  real support images behind each run.
- **Experiment explorer:** method / K / seed filters, mean ± SD charts, every seed, paired
  differences and every slot's status, for validation or the sealed held-out test.
- **Single image prediction:** a verified stored model scores an uploaded or a validation image.
  The scores are labeled model scores, and for a validation image the page shows its match with
  the stored prediction.
- **Industrial replay:** the sealed NEU stream, clean or stressed, with the sealed threshold's
  review decision. It is verified against the seal before any image is shown.
- **Reproducibility:** the seal, protocol-module hashes, environment, data and weights hashes,
  commands, artifacts, error galleries and limitations.

**Result labels:** every number carries one of VALIDATION, HELD-OUT TEST or LIVE PREDICTION.
Nothing in the app trains a model or changes a dataset. Uploaded images stay in memory for the
session and are never saved or logged.

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
