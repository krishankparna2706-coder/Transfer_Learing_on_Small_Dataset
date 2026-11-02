# Progress record

Each entry lists commands that were actually executed and what was observed.
- **NOT_RUN:** the step has not been executed.
- **BLOCKED:** an external prerequisite is missing.

## Current status

| Area | Status |
|---|---|
| Implementation | P0–P5 tooling complete. Test access runs only through the sealed workflow (`lock-test`, then `evaluate --split test --study`). `demo` and the app exit 3 until P6 |
| Data | Flowers102 PREPARED (official split, MD5-verified) and audited PASS. NEU-CLS PREPARED (project split) and audited PASS. 40 support manifests built |
| Benchmark | Core matrix: **156/156 COMPLETED on validation** (5 seeds; optimistic). Held-out test: **NOT_RUN** (not sealed; awaiting approval). Ledger: 6 diagnostic + 8 MVP + 156 core runs |
| Demo | NOT_RUN (P6). Chosen on validation in the seal preview: partial, NEU K=10 seed 11; review threshold 0.95 |
| Remote push | P0–P5 pushed to `origin/main`; each phase is pushed after its commit |

## P0: foundation (2026-11-01)

### Environment observed
- **Machine:** macOS 26.6.2, Apple M4 Pro (12 CPU cores, 16-core GPU), 24 GiB RAM. No NVIDIA GPU.
  Details are in [HARDWARE.md](HARDWARE.md).
- **Tools:** Python 3.11.15 (uv-managed), uv 0.11.15, git 2.54.0.
- **`uv lock` resolved 80 packages.**
  - Runtime: torch 2.15.0, torchvision 0.29.2, numpy 2.4.6, scipy 1.17.1, pillow 13.0.0,
    pandas 3.1.0, matplotlib 3.11.2, pyyaml 6.0.3, streamlit 1.66.1.
  - Dev: pytest 9.2.0, ruff 0.17.3, scikit-learn 1.9.1.
- **Lock hashes:**
  - `uv.lock`: sha256 `114a0511f2600595b185fd1c5b60f7f626d11dfdc48d66d40f69ae4e8d3f9bd5`
  - `requirements.lock`: sha256 `3b3d00a64dcb29672d5acd23d628437b24bcb701b5e82ba92cf8607f13b9d499`

### Commands executed and observed results

| Command | Observed |
|---|---|
| `uv lock` | "Resolved 80 packages", using CPython 3.11.15 |
| `uv sync` | environment installed into `.venv` |
| `uv export --format requirements-txt --no-emit-project --locked -o requirements.lock` | exit 0; hashed requirements file |
| `uv run python -m fewshotlab --help` | exit 0; lists the 9 top-level commands and the exit codes |
| `uv run python -m fewshotlab doctor` | exit 0. Highlights:<br>• MPS available, probe PASS; CUDA not available; auto → mps<br>• weights not cached; both datasets NOT_PREPARED<br>• benchmark NOT_RUN; no FAIL checks |
| `uv run python -m fewshotlab doctor --require-device cuda` | exit 4: "CUDA was requested, but torch.cuda.is_available() is False … Use --device mps here …" |
| `uv run python -m fewshotlab data prepare --dataset flowers102` | exit 3: "NOT IMPLEMENTED … planned for phase P1. Nothing was run and no files were written." |
| `uv run pytest -q` | 56 passed |
| `uv run ruff check .` | All checks passed (after `ruff check --fix` and `ruff format`) |
| Clean environment: `UV_PROJECT_ENVIRONMENT=<scratchpad>/p0-clean-venv uv sync --locked`, then `uv run --locked pytest -q` and `doctor --json` in that environment | 56 passed; doctor reports no FAIL checks, auto → mps |
| `find data -type f`; search for `*metrics*.json`, `predictions*.csv`, `*.ckpt`, `*.npz` outside `.venv` | only `data/manifests/README.md` (plus a Finder `.DS_Store`); no metric, prediction or checkpoint files |
| `gh api repos/krishankparna2706-coder/Transfer_Learing_on_Small_Dataset --jq .permissions.push` | `false`, so the push is BLOCKED |

### P0 evidence table

| Criterion | Status | Command / check | Observed |
|---|---|---|---|
| Clean environment installs from documented commands | PASS | fresh `uv sync --locked` into an empty venv | imports `fewshotlab 0.1.0`, torch 2.15.0, torchvision 0.29.2 |
| Package imports | PASS | `python -c "import fewshotlab"` (clean env) | version 0.1.0 |
| Doctor reports the actual backend | PASS | `doctor`, `tests/test_p0_doctor.py` | values match `torch.backends`: MPS available, CUDA unavailable |
| Unsupported backend gives an actionable error | PASS | `doctor --require-device cuda` | exit 4; suggests `--device mps` |
| Fixture forward pass returns expected shapes (software fixture) | PASS | `tests/test_p0_fixture_forward.py` | logits (2,102) and (2,6); features (2,512); parameters 11,176,512 / 11,228,838 / 11,179,590 |
| Help for every command; stubs fail clearly | PASS | `tests/test_p0_cli.py` | every `--help` exits 0 and lists its flags; 10 stub invocations exit 3 and write no files |
| Pretrained weights never download implicitly | PASS | `test_pretrained_without_cache_or_permission_fails_without_network` | raises `WeightsUnavailableError` mentioning `--allow-download`; no download call |
| No dataset download or metric report silently created | PASS | `find` checks above; `tests/test_guardrails.py` | none present |
| Strict config schema and protocol file | PASS | `tests/test_p0_config.py` | unknown keys and protocol violations rejected; protocol YAML matches the code |
| pip install route from `requirements.lock` | NOT_RUN | n/a | deferred to the P8 clean-room check |
| Push to origin | BLOCKED | permission check above | the authenticated account lacks write access |
| Push to origin (retried after the user granted write access) | PASS | `git push -u origin main` | `origin/main` = `2c1e9b7` |

**Benchmark metrics: all NOT_RUN. No dataset has been verified.**

### Remaining risks after P0
- Flowers102 hosts were unreachable from this network during planning. P1 retries the official
  source; if that fails, Flowers is marked BLOCKED.
- NEU-CLS must come from the authors' Google Drive link, which may block scripted downloads.
- MPS determinism limits have not been measured yet (P3).

### Next action (P1)
1. Implement `data prepare`, `data audit` and `support build` with fixture tests.
2. Retry the official Flowers102 source.
3. Fetch NEU-CLS from the authors' Google Drive link.

## P1: data acquisition, audit, manifests, supports (2026-11-01)

### What changed
- **Data layer:**
  - manifest and class-map schema (`data/manifest.py`);
  - safe image decoding with perceptual hashes (`data/imageio.py`);
  - safe archive extraction (`data/archive.py`);
  - dataset adapters (`data/flowers.py`, `data/neu.py`);
  - the audit (`data/audit.py`), nested supports (`data/support.py`) and the label budget
    (`data/budget.py`);
  - seeded streams (`seeds.py`) and atomic I/O (`io_utils.py`).
- **CLI:** `data prepare`, `data audit` and `support build` are implemented.
  - `data prepare` gained `--mirror internet-archive` and `--cross-check PATH` (Flowers only).
  - `weights fetch` was added (D-019).
- **Tests:** synthetic-fixture tests (`tests/test_p1_*.py`) and opt-in real-data checks
  (`tests/test_real_data.py`).
- **Docs:** `docs/DATA_CARD_flowers102.md`, `docs/DATA_CARD_neu_cls.md`, decisions D-012 to D-019,
  and `reports/label_budget.md`.

### Commands executed and observed results

| Command | Observed |
|---|---|
| `data prepare --dataset neu_cls --allow-download` (first run) | archive downloaded; validation stopped on `Thumbs.db`. The attempt was moved aside, and the adapter now ignores and lists OS metadata (D-015) |
| `data prepare --dataset neu_cls --allow-download` | PREPARED in 45 s: 1,800 images, 1,080/360/360, manifest sha256 `2f4740b0…`. The archive was byte-identical to the first download (sha256 `a85bcc6b…`) |
| `data prepare --dataset flowers102 --allow-download --mirror internet-archive --cross-check flower_data` | PREPARED in 608 s: 8,189 images, 1,020/1,020/6,149. All three files matched torchvision's MD5s, and labels matched torchvision's loader. Kaggle copy: 8,189 byte-identical, 7,370/7,370 labels agree |
| `data audit --dataset neu_cls` | PASS in 2 s. 1 exact-duplicate group (`Pa_101`=`Pa_105`, both test); 1 near-duplicate pair (the same one); 0 cross-split |
| `data audit --dataset flowers102` | PASS in 11 s:<br>• 4 exact-duplicate groups, 2 cross-split<br>• 25 near-duplicate candidates (8 cross-split, 1 cross-class)<br>• details in the data card |
| `support build --config configs/protocol-v1.yaml` (first run) | FAILED: a validation bug compared class keys in string order. Fixed, with a regression test |
| `support build --config configs/protocol-v1.yaml` | 40 manifests written (2 datasets × 4 K × 5 seeds); label budget generated |
| the same command again | 40 × "verified (unchanged)"; nothing rewritten |
| `uv run pytest -q` | 138 passed, including 6 real-data checks |
| `uv run ruff check .` and `ruff format --check .` | clean |
| search for `test_metrics.json` or `predictions_test.csv` | none |

### P1 evidence table

| Criterion | Status | Command / check | Observed |
|---|---|---|---|
| Tests fail on overlap, corrupt files, unknown labels, duplicate IDs, unsafe paths, missing classes, wrong counts, changed hashes, train/val merging | PASS | `tests/test_p1_manifest.py`, `test_p1_audit.py`, `test_p1_support.py` | each case is rejected with a specific message |
| Exactly K per class; K=1 ⊂ 3 ⊂ 5 ⊂ 10; reproducible across process restarts | PASS | `test_p1_support.py` (including a subprocess rebuild), `test_real_data.py` | identical content hashes; real supports nested and exact |
| Every method gets the same IDs for a dataset/K/seed | PASS | one immutable file per dataset/K/seed; `load_support` verifies its hash | a different membership is refused |
| Flowers official split: 1,020/1,020/6,149, 102 classes, 10/class train and val, label shift once | PASS | `data prepare`, manifest validation, torchvision cross-check, `test_real_data.py` | all hold; MAT labels − 1 matches torchvision for all 8,189 images |
| NEU-CLS: 6 classes × 300, 200×200 grayscale; 180/60/60 with seed 2026 after the duplicate check | PASS | `data prepare`, `test_real_data.py` | holds; the duplicate cluster stays in one split; counts unaffected |
| Corrupt/duplicate audit with method and threshold; near-duplicates reviewed, never auto-deleted | PASS (automated) / NOT_RUN (human review) | `data audit` | reports in `reports/data_audit/`; visual review of candidates is pending |
| Acquisition groups kept together | NOT APPLICABLE | n/a | the sources provide no group metadata; limitation recorded |
| Label-budget table; data cards distinguish official vs project splits | PASS | `reports/label_budget.md`, data cards | table generated from the stored files |
| Fixtures separate from real manifests; real status only after real audits | PASS | fixtures live in temp dirs; real records show PREPARED + audit PASS | yes |
| Original sources only; no unverified substitute | PASS | Flowers via MD5-verified archived official files; NEU from the authors' Drive link | the Kaggle copy is used only as a cross-check |

**Benchmark metrics: all NOT_RUN.**

### Unresolved issues after P1
- Human visual review of the near-duplicate candidates is pending (D-016).
- Flowers usage terms could not be re-checked online. NEU states no license (citation requested).
- iCloud Drive syncs the project folder; the virtualenv was moved out of it (D-014).

### Readiness for P2
Both datasets are prepared, audited and have supports. P2 (transforms and the four methods) needs
the ResNet18 IMAGENET1K_V1 weights; `weights fetch` will download them (46,830,571 bytes).

## P2: comparable model methods and preprocessing (2026-11-01)

### What changed
- **Preprocessing** (`data/transforms.py`, D-020/D-021): the protocol v1 contracts for Flowers
  and NEU (train and eval) and for fixtures, each with a transform ID.
- **Datasets** (`data/datasets.py`): support and validation only; **test records are refused**.
- **Freezing** (`models/policies.py`, D-022): explicit per-method policies, `enter_train_mode`,
  and AdamW parameter groups.
- **Prototypes** (`models/prototypes.py`, D-025): the nearest-prototype rules, with explicit
  rejection of bad embeddings.
- **Persistence** (`models/spec.py`, D-024): the `ModelSpec` saved with every artifact, plus safe
  save/load for checkpoints and prototype files.
- **Registry and cache** (`models/methods.py`, `models/cache.py`): the method registry, shared
  prediction and embedding paths, and content-addressed cache keys.
- **Weights:** `weights fetch` downloaded ResNet18 IMAGENET1K_V1 (46,830,571 bytes; sha256
  `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`, prefix-verified).

### Commands executed and observed results

| Command | Observed |
|---|---|
| `uv run python -m fewshotlab weights fetch` | downloaded in 30 s; SHA-256 prefix matches torchvision's file name; doctor reports `resnet18_weights_cached: PASS` |
| `uv run pytest -q tests/test_p2_*.py` | 56 passed, including 4 checks with the real ImageNet weights |
| `uv run pytest -q` | 194 passed |
| `uv run ruff check .` and `ruff format --check .` | clean |

### P2 evidence table

| Criterion | Status | Test | Observed |
|---|---|---|---|
| Gradient and before/after tests prove only the permitted parameters change; BatchNorm buffers stay identical where frozen | PASS | `test_p2_policies.py`, `test_p2_real_weights.py` | one AdamW step changes exactly: all parameters (scratch); head only (frozen); layer4 + head (partial). Frozen running statistics are byte-identical; the negative control shows `model.train()` alone would change them |
| Hand-computed prototype example; zero, non-finite and norm ≤ 1e-12 embeddings rejected without clamping | PASS | `test_p2_prototypes.py` | prototypes [[0.3, 0.9], [1, 0]]; distances [0.2029437, 0.5857864]; class 0; scores softmax(−d). All 5 rejection cases plus an overflowing norm are refused |
| Save/reload reproduces predictions within a declared tolerance; incompatible class maps rejected | PASS | `test_p2_persistence.py` | CPU tolerance atol 1e-6, observed equal. A NEU model with a Flowers class map, or a different dataset, transform, order or method, is refused. Pickled objects are refused |
| Output dimensions 102 and 6; transforms cannot read test labels | PASS | `test_p2_methods_cache.py`, `test_p2_dataset.py` | shapes (2, 102) and (2, 6); datasets refuse test records (ProtocolRefusal) |
| Fresh initialization; cache keys separate encoder, dataset, class order, transform and support | PASS | `test_p0_fixture_forward.py`, `test_p2_methods_cache.py` | a seeded, isolated init per build; each cache-key component changes the key |
| Same image → identical evaluation tensor; numeric grayscale and normalization check; training transforms keep shape and label | PASS | `test_p2_transforms.py`, `test_p2_dataset.py` | equal tensors; constant gray 128 → (128/255 − mean_c)/std_c in all 3 channels; augmentation reproducible from its labels |

### Remaining numerical caveats
- P2 checks run on CPU. MPS behavior and determinism are measured in P3.
- Tensor resizing differs slightly from PIL resizing (D-020).

**Benchmark metrics: all NOT_RUN.**

## P3: training engine and evidence capture (2026-11-01)

### What changed
- **Training** (`engine/trainer.py`):
  - accepts only `SupportSet`/`ValidationSet`, which refuse any other split;
  - AdamW parameter groups, cross-entropy, float32;
  - selection by float64 macro-F1, with ties to lower validation loss and then the earlier epoch;
  - a separate patience reference (≥ 0.001, patience 15);
  - non-finite loss means FAILED;
  - the reloaded best checkpoint must reproduce its predictions.
- **Batching** (`engine/batching.py`): min(32, N), with the singleton tail merged and nothing
  dropped.
- **Runs** (`engine/runner.py`):
  - verifies the input hashes, uses immutable run IDs with numbered attempts, and writes atomic
    artifacts;
  - computes metrics from the written predictions file;
  - resume continues as a new attempt; finished runs are immutable;
  - `evaluate --split val` writes verification records.
- **Records:** `engine/ledger.py` (append-only), `engine/artifacts.py`, `engine/environment.py`
  (determinism settings) and `engine/provenance.py` (git state, source hash, deterministic source
  snapshot).
- **Test gate** (`engine/gate.py`): test evaluation is refused (exit 5) until P5 seals a plan.
- **Metrics and reporting:**
  - `metrics/core.py`: the protocol definitions;
  - `reporting/aggregate.py`: accepts only COMPLETED, non-fixture runs whose evidence regenerates;
  - `reporting/plots.py` and `reporting/estimate.py`.
- **CLI:** `train [--resume]` and `evaluate` are implemented.
- **Configs:** 5 diagnostic configs, and the 8 MVP configs (**not run**).

### Commands executed and observed results

| Command | Observed |
|---|---|
| `uv run pytest -q` | 230 passed, including P3 fixture training runs on CPU |
| `python -m fewshotlab evaluate --run x --split test` | exit 5: "test evaluation is locked … no file was written" |
| `python -m fewshotlab train --config configs/runs/diagnostic/<5 configs>` on MPS | all 5 COMPLETED. Wall times: 29.6, 18.5, 19.6, 12.4 and 8.1 s. Reload check identical (max score difference 0.0). Details in `docs/HARDWARE.md` |
| `fewshotlab.reporting.estimate.write_estimate` | ESTIMATE written to `reports/compute_estimate.md`:<br>• MVP: 0.14–0.82 h<br>• full matrix: 1.56–9.33 h<br>• test evaluation: about 0.36 h |

The diagnostics ran from the then-uncommitted P3 working tree. Each run saved a
`source_snapshot.tar.gz` with its hash, and its provenance records `dirty: true`. **Their
validation numbers are 2-epoch diagnostics, not results.**

### Artifacts of one real run

Example: `runs/diagnostic__fewshotlab-v1__flowers102__partial__k10__s11__cfg506dc445/a001/`.
- **Inputs and environment:** `config.resolved.yaml`, `environment.json`, `provenance.json`,
  `source_snapshot.tar.gz`.
- **Model:** `model_spec.json` and `best.ckpt` (45 MB).
- **Results:** `train_history.csv`, `predictions_val.csv` (1,020 rows with full score vectors),
  `val_metrics.json`, and `plots/learning_curves.png` and `plots/confusion_val.png`.
- **Status:** `status.json`, with history RUNNING → COMPLETED.

### P3 evidence table

| Criterion | Status | Test / command | Observed |
|---|---|---|---|
| A tiny labeled fixture learns; its metrics carry `is_fixture=true` and aggregation rejects them | PASS | `test_p3_engine.py` (fixture runs, `test_scratch_training_loss_decreases`, `test_fixture_results_are_rejected_by_aggregation`) | all 4 methods COMPLETED above chance on a 3-class fixture; training loss decreases; every fixture record is rejected with reason "software fixture" |
| Synthetic metric example matches hand calculations, including a never-predicted class and zero denominators | PASS | `test_p3_metrics.py` | top-1 0.75, macro-F1 11/15, confusion [[1,1],[0,2]]; zero convention holds; matches scikit-learn |
| Checkpoint reload preserves results; interrupted runs stay RUNNING/FAILED and never enter the completed table | PASS | reload check in every run; `test_interruption_fails_the_attempt_and_resume_continues_exactly` | interrupted attempt FAILED with no metrics file; rejected by aggregation; resume reproduces the uninterrupted run exactly |
| Test evaluation fails before a sealed plan; no test-data dependency in training | PASS | `test_validation_reevaluation_and_test_refusal`, `test_training_never_reads_test_rows`, `test_fit_interfaces_accept_only_support_and_validation`, CLI exit 5 | refused, nothing written; training read only `train`/`val` rows |
| Two same-device deterministic repetitions meet the declared tolerance | PASS (CPU) / NOT_RUN (MPS training) | `test_two_repetitions_on_the_same_device_match` | CPU: identical predictions and metrics; tolerance 1e-6 declared. Repeated MPS training is not yet measured |
| Batches: NEU K=1 (6 images) and a would-be singleton tail | PASS | `test_p3_batching.py` | 6 → one batch; 33 → one batch of 33; nothing dropped for N in {2, 6, 33, 65, 102, 306, 510, 1020} |
| Non-finite loss stops the run and records the cause | PASS | `test_non_finite_loss_marks_the_run_failed` | FAILED with `NonFiniteError`; no metrics written |
| Short real-data diagnostic, separate from planned jobs | PASS | 5 `diagnostic` runs | timing recorded; estimate generated |

**Benchmark metrics: all NOT_RUN** (no MVP or core run exists).

### Remaining risks
- Repeated training on MPS has not been tested for repeatability (warn-only determinism).
- `num_workers` 0 is probably data-bound. The worker count is an execution detail and can be raised
  for P4 if measured to help.
- The near-duplicate visual review (P1) is still pending.

### Next action (P4, awaiting go-ahead)
Run the 8 MVP jobs listed in `configs/mvp/`, one `train` command per config. Validation only;
the test split stays locked.

## P4: validation-only MVP (2026-11-01)

### What changed
- **Study planning** (`engine/study.py`, D-036): study configs expand to run slots, each with the
  same run ID as its standalone config; pre-flight integrity checks run first; deterministic
  prototype duplicates are counted once.
- **`benchmark --config`** (`--dry-run`, `--device`, `--allow-download`): runs unfinished slots in
  order, records failures and continues. Finished slots are never re-run.
- **`report`:**
  - `--study` builds the validation-only comparison from stored records, with sanity checks and
    error galleries;
  - without `--study` it summarizes the ledger.
- **Report modules:** `reporting/study_report.py`, `reporting/sanity.py`, `reporting/errors.py`
  (D-038).
- **Run records:** each run now records trainable and total parameters, total wall time and
  validation-pass time per image.
- **Provenance:** the dirty flag covers code, configs and locks only (D-037).
- **Configs:** `configs/mvp.yaml` (8 jobs), `configs/full-study.yaml` (P5: 160 slots, 156 unique;
  NOT RUN), and the K=1 batch-flow diagnostic.
- **Assisted review of the Flowers near-duplicate candidates** (D-016a; a human still needs to
  confirm).

### Commands executed and observed results
| Command | Observed |
|---|---|
| `benchmark --config configs/mvp.yaml --dry-run` | 8 slots, 8 unique; all 6 pre-flight checks PASS (shared support IDs, unique run IDs, init policy, paired init seed, one training transform, one recipe) |
| `train --config configs/runs/diagnostic/flowers102-scratch-k1-s11.yaml` | diagnostic COMPLETED in 16.6 s; batches 32/32/32/6 (102 images per epoch, 4 steps); losses finite |
| `benchmark --config configs/full-study.yaml --dry-run` | 160 logical slots, 156 unique (4 Flowers K=10 prototype duplicates); all checks PASS; NOT RUN |
| `benchmark --config configs/mvp.yaml` (MPS, commit `8f3d541`, clean code tree) | **8 COMPLETED, 0 failed, 1,854 s** |
| `report --study configs/mvp.yaml` | comparison regenerated from stored predictions:<br>• 8/8 rows pass the evidence check<br>• sanity checks 8/8 PASS; augmentation check PASS<br>• error galleries written locally |
| `report` | ledger summary: diagnostic 6 COMPLETED, mvp 8 COMPLETED; 8 verified MVP runs |
| `uv run pytest -q` | 233 passed. `test_guardrails.py` confirms that no `test_metrics.json` or `predictions_test.csv` exists |

### MVP outcomes

These are **validation only, n = 1, and optimistic because of checkpoint selection**. The full
table is in [MVP_DECISION_MEMO.md](MVP_DECISION_MEMO.md) and
`reports/fewshotlab-v1-mvp/comparison.md`.

| Method | K=1 val macro-F1 | K=10 val macro-F1 |
|---|---:|---:|
| scratch | 0.073 | 0.477 |
| frozen | 0.449 | 0.853 |
| partial | 0.458 | 0.871 |
| prototypes | 0.415 | 0.780 |

**Decision:** the v1 recipe is retained unchanged for P5.

### P4 evidence table

| Criterion | Status | Evidence | Observed |
|---|---|---|---|
| All 8 jobs have explicit statuses; completion requires valid artifact schemas and traceable predictions; failures stay visible | PASS | `reports/fewshotlab-v1-mvp/comparison.md`, `test_p4_report.py` | 8/8 COMPLETED with full artifacts. NOT_RUN and EVIDENCE_FAILED rows stay visible with null metrics (tested) |
| Every metric regenerates from the stored validation predictions | PASS | `verify_evidence` for each row | all 8 regenerate exactly; prediction-file hashes match |
| Comparison via the CLI and a report export; no hard-coded numbers | PASS | `benchmark`, `report --study`; CSV/JSON/MD/plot built from records | the test confirms table values equal the stored `val_metrics.json` |
| Decision memo from validation evidence only | PASS | [MVP_DECISION_MEMO.md](MVP_DECISION_MEMO.md) | recipe retained |
| Test access stays locked; an automated test confirms no test predictions or metrics | PASS | `test_guardrails.py::test_no_test_split_outputs_exist_before_the_gate`; `evaluate --split test` exits 5 | 0 test-split files |
| Shared support hash per dataset/K/seed; unique run IDs; init policy; identical training transforms | PASS | pre-flight checks (`plan.md`) | all PASS |
| Short diagnostic before the full jobs, kept separate | PASS | `diagnostic` ledger category | K=1 batch flow and finite losses verified |
| Runtime, update counts and trainable parameters reported | PASS | comparison table | wall time, optimizer steps and trainable parameters per run |
| Learning curves, confusion matrices and a prespecified error sample inspected; mapping, frozen-head, BatchNorm-drift and augmentation checks | PASS | sanity checks, plots, error galleries | no defects found; visible-text overlay noted |

### What the MVP demonstrates
- The complete path works on real data: data → supports → four methods → validation selection →
  evidence → report.
- At seed 11, the three transfer methods far exceed scratch at both K values.
- More shots (K=10) help every method.

### What remains uncertain
- Variability across seeds (n = 1).
- Held-out test performance.
- Whether the differences among the transfer methods are real.
- Repeatability of training on MPS.

**Benchmark (core) metrics: all NOT_RUN.**

## P5: locked benchmark, simulation scorer and test seal (2026-11-01 to 2026-11-02)

**State at the end of this entry:**
- Implementation is complete.
- The core matrix is complete on **validation**.
- The test seal passed its dry run but is **not sealed**.
- Held-out test metrics are **NOT_RUN**. Sealing and the one-time test evaluation wait for
  explicit approval.

### What changed
- **Corruption generator** (`simulation/corruptions.py`, D-042):
  - brightness ×0.6 / ×1.4, Gaussian blur σ 1 / 2 px, Gaussian noise SD 0.03 / 0.08;
  - applied to decoded [0, 1] pixels at native resolution, before the evaluation resize;
  - grayscale is corrupted as one channel; reflect padding; per-derivative seed and checksum.
  The clean condition reproduces the normal evaluation input exactly.
- **Review scorer** (`simulation/review.py`, D-044):
  - the score is the maximum class score, with threshold grid {0, 0.05, …, 1};
  - it picks the highest threshold that keeps at least 80 % validation coverage;
  - an image goes to review when its score is strictly below the threshold;
  - accepted error is null when nothing is accepted.
- **Demo selection** (`simulation/demo.py`, D-043):
  - the demo cell is NEU K=10, seed 11, and the method with the highest validation macro-F1 wins;
  - ties go to the lexically first method;
  - every candidate must be COMPLETED and verified.
- **Replay manifest** (`simulation/replay.py`, D-045): a label-independent seeded order of the 360
  NEU test parents × 7 streams (2,520 rows), sealed by hash.
- **Test lock** (`engine/seal.py`, D-040/D-041/D-049): `lock-test --plan PATH [--dry-run]`.
  - It seals code, environment, data, supports, every run's checkpoint and validation evidence,
    weights, metrics, scorer, calibration (none), corruptions, demo, threshold, replay, test plan
    and report rules.
  - Before sealing it re-scores every run's validation set with the current code.
  - It writes an audit token to the study's evaluation ledger and the run ledger.
- **Sealed test evaluation** (`engine/test_eval.py`, D-046): `evaluate --split test --study PATH`.
  - It re-derives the plan first and fails closed on any difference.
  - Each sealed run is evaluated once, through `GatedTestDataset` only.
  - The robustness study scores the demo checkpoint on 7 conditions with paired drops versus
    clean.
- **Shared inference loader** (`engine/inference.py`): validation re-evaluation and test use one
  code path.
- **Seed-aggregated reports** (`reporting/benchmark_report.py`, D-048):
  - mean and sample SD over distinct runs, and paired method differences on matched seeds;
  - plots that disclose dataset, K, support and validation budgets, seeds, protocol and n;
  - test tables only after a sealed evaluation (D-047).
- **Guardrails:**
  - every test-split file must be recorded under an intact seal (`audit_test_outputs`);
  - a static check confirms the training path never imports the test gate or the corruption code.

### Commands executed and observed results
| Command | Observed |
|---|---|
| `benchmark --config configs/full-study.yaml` (MPS, commit `64c7d25`, clean code tree) | **156 COMPLETED, 0 failed** over 160 slots (4 deterministic Flowers K=10 prototype duplicates).<br>Ledger span 12.04 h, of which 6.25 h was a system sleep (see below). About 5.8 h of compute |
| Seed-11 repeat check: the 8 MVP runs against the matching core runs | **8/8 identical.** Every predicted class and float32 score in `predictions_val.csv` matches, and the selected epochs and validation macro-F1 are equal. Only `run_id` differs |
| `report --study configs/full-study.yaml` | comparison and seed aggregates regenerated from stored predictions in 89 s.<br>156 runs, 0 sanity failures. Error galleries are written locally and not committed |
| `lock-test --plan configs/full-study.yaml --dry-run` (MPS, merged `main` `51d664d`, clean code tree) | **ready to seal: True**; nothing was sealed. 286 s.<br>156 sealable runs, 4 duplicates, 0 missing, 0 blocking items.<br>Validation re-scored with the merged code: **156/156 identical**, maximum score difference 0.0.<br>Preview content hash `f556c63c…`, written to `seal_preview.json` |
| `ruff check .`, `ruff format --check .`, `pytest -q` (merged `main`, real data and weights present) | all checks pass; **264 passed**, 0 skipped, 181 s |

### Interruption during the matrix
- **What happened:** the Mac was on battery and entered low-power sleep at 1 % charge, at 05:24
  IST on 2026-11-02 (`pmset -g log`). It woke at 11:39 IST on mains power.
- **The process:** the matrix was suspended for 22,486 s, then resumed.
- **The run in progress:** `fewshotlab-v1__neu_cls__frozen__k5__s55__cfg8b5ac2db` completed normally.
  Reloading its selected checkpoint reproduced its validation predictions, and the seal dry run
  re-scored it identically (D-050).
- **Timing:** its ledger wall time (22,593 s) includes the sleep.
  [HARDWARE.md](HARDWARE.md) reports the matrix timing without it.

### Validation results (5 support seeds)

**VALIDATION ONLY:** each learned checkpoint was selected on these validation images, so the
numbers are optimistic. They are not held-out test results.

Macro-F1 is shown as mean ± sample SD over distinct runs. The full tables (top-1, balanced
accuracy, paired differences and the plot) are in `reports/fewshotlab-v1-full/aggregate_val.md`.

| Dataset | K | scratch | frozen | partial | prototypes |
|---|---:|---:|---:|---:|---:|
| Flowers102 | 1 | 0.074 ± 0.009 | 0.439 ± 0.033 | 0.433 ± 0.030 | 0.403 ± 0.023 |
| Flowers102 | 3 | 0.198 ± 0.008 | 0.685 ± 0.011 | 0.695 ± 0.015 | 0.650 ± 0.013 |
| Flowers102 | 5 | 0.292 ± 0.011 | 0.776 ± 0.010 | 0.799 ± 0.009 | 0.724 ± 0.010 |
| Flowers102 | 10 | 0.457 ± 0.028 | 0.850 ± 0.002 | 0.887 ± 0.011 | 0.780 (1 distinct run) |
| NEU-CLS | 1 | 0.451 ± 0.150 | 0.868 ± 0.041 | 0.854 ± 0.057 | 0.816 ± 0.075 |
| NEU-CLS | 3 | 0.621 ± 0.140 | 0.925 ± 0.016 | 0.940 ± 0.020 | 0.889 ± 0.030 |
| NEU-CLS | 5 | 0.803 ± 0.038 | 0.950 ± 0.023 | 0.974 ± 0.018 | 0.927 ± 0.023 |
| NEU-CLS | 10 | 0.776 ± 0.050 | 0.972 ± 0.017 | 0.985 ± 0.007 | 0.949 ± 0.027 |

**Paired differences on matched seeds (validation):**
- **Transfer vs scratch:** every transfer method beats scratch in 5/5 seeds at every K on both
  datasets.
- **Prototypes vs frozen:** prototypes score below frozen in 5/5 seeds at every K on both
  datasets.
- **Partial vs frozen:**
  - partial is higher in 5/5 seeds at Flowers K=5 (+0.022) and K=10 (+0.037), and at NEU K=5
    (+0.024);
  - at K=1 it is not higher (Flowers 3/5 seeds, mean −0.006; NEU 1/5 seeds, mean −0.014).
- **NEU scratch:** it is not higher at K=10 (0.776 ± 0.050) than at K=5 (0.803 ± 0.038). Five
  seeds do not resolve this.

### P5 evidence table

| Criterion | Status | Evidence | Observed |
|---|---|---|---|
| Every planned slot has an explicit status; failures stay visible | PASS | `runs/ledger.jsonl`, `reports/fewshotlab-v1-full/plan.md` | 156 COMPLETED, 4 DEDUPLICATED, 0 FAILED |
| Checkpoints selected on validation only; test locked throughout training | PASS | `test_guardrails.py` (no unsealed test outputs; the training path never imports the gate or the corruption code); `evaluate --split test` exits 5 without a seal | 0 test-split files |
| Seed aggregation over distinct runs: sample SD, paired differences on matched seeds, duplicates counted once, no confidence intervals | PASS | `aggregate_val.*`, `paired_val.csv`, `slots_val.csv`, `test_p5_aggregate.py` | Flowers K=10 prototypes shown as 1 distinct run, SD null |
| Every aggregate regenerates from stored predictions | PASS | `report --study` evidence check | 156/156 |
| Training repeats exactly on MPS | PASS | seed-11 MVP vs core comparison | 8/8 identical predictions and scores |
| Validation re-scored with the current code before sealing | PASS (dry run) | `reports/fewshotlab-v1-full/seal_preview.json` | 156/156 identical, maximum score difference 0.0. All runs trained at `64c7d25`; the 14 sealed source files changed since (P5 tooling) give identical results |
| Complete evaluation plan before any test access: code, environment, data, supports, checkpoints, weights, metrics, scorer, calibration, corruptions, demo, threshold, replay, report rules | READY, not sealed | `seal_preview.json` | ready to seal; 0 blocking items, 0 missing runs; preview content hash `f556c63c…` |
| Any change to a sealed input fails closed | PASS | `test_p5_seal.py` | a changed checkpoint, support, coverage target, corruption parameter, plan file, ledger or replay file is refused |
| Corruptions seeded, checksummed and reproducible; the clean condition equals the normal input | PASS | `test_p5_corruptions.py` | 10 tests pass |
| Review threshold and demo cell chosen on validation only | PASS (dry run) | `seal_preview.json` | demo: partial, NEU K=10 seed 11 (validation macro-F1 0.9917).<br>Threshold 0.95: validation coverage 0.958, 15 of 360 images to review, 1 accepted error (0.29 %) |
| Held-out test metrics (every sealed run, once) | NOT_RUN | locked until the seal is approved | — |
| Robustness study on test (demo run, 7 conditions) | NOT_RUN | locked until the seal is approved | — |

### Known issue (cosmetic)
- `portable_path` resolves the `runs/artifacts` symlink before making a path relative.
- So files inside run folders are recorded as `<outside project>/<name>`. For example, the dry
  run's 156 verification events show `<outside project>/val-001.json` for
  `runs/artifacts/<run>/a001/verifications/val-001.json`.
- The run ID and attempt still identify each file, and no local path leaks.
- **Fix:** after the test evaluation. Sealed code must not change between sealing and evaluation.

### Next action (awaiting approval)
1. `python -m fewshotlab lock-test --plan configs/full-study.yaml` seals the plan. The seal and
   replay files are then committed.
2. `python -m fewshotlab evaluate --split test --study configs/full-study.yaml` evaluates each
   sealed run once on the test split, then runs the robustness study on the demo run.
3. `python -m fewshotlab report --study configs/full-study.yaml` adds the test aggregates and the
   robustness report.

**Held-out test metrics: all NOT_RUN.**
