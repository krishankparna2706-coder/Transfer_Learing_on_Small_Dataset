# Progress record

Each entry lists commands that were actually executed and what was observed.
- **NOT_RUN:** the step has not been executed.
- **BLOCKED:** an external prerequisite is missing.

## Current status

| Area | Status |
|---|---|
| Implementation | P0–P6 complete. The app (`python -m fewshotlab demo`) and `predict` read only verified records. Test evaluation ended with the sealed run (D-051) |
| Data | Flowers102 PREPARED (official split, MD5-verified) and audited PASS. NEU-CLS PREPARED (project split) and audited PASS. 40 support manifests built |
| Benchmark | Core matrix: 156/156 COMPLETED on validation (5 seeds). **Held-out test: 156/156 EVALUATED once** under seal `seal-c6f58862d63b586a0df8dbcf140db0fd` (2026-11-02). Ledger: 6 diagnostic + 8 MVP + 156 core runs |
| Demo | App implemented (P6). P7: the replay scores the sealed NEU stream live with the sealed checkpoint (partial, NEU K=10 seed 11; threshold 0.95), with a labeled stored fallback. Automated offline rehearsal: 2,520 of 2,520 live results equal the sealed ones. Judge script and rehearsal record in [JUDGE_SCRIPT.md](JUDGE_SCRIPT.md) and [REHEARSAL.md](REHEARSAL.md) |
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
- The core matrix is complete on validation.
- After explicit approval, the evaluation plan was sealed (`seal-c6f58862d63b586a0df8dbcf140db0fd`,
  2026-11-02T06:56:38Z). It was committed and pushed (`abe8aa4`) before any test access.
- Every sealed run was then evaluated on the held-out test split **once**, and the robustness
  study ran on the demo checkpoint.

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
| `lock-test --plan configs/full-study.yaml` (after approval; `3025835`, clean code tree) | **SEALED**: 156 runs, audit token `seal-c6f58862d63b586a0df8dbcf140db0fd`, plan SHA-256 `ab6d96cf…`.<br>Validation re-scored first: 156/156 identical. 276 s.<br>The seal was committed and pushed (`abe8aa4`) before any test access |
| `evaluate --split test --study configs/full-study.yaml` (MPS) | **156 runs evaluated now, 0 already recorded**; robustness study on the demo checkpoint evaluated. 1,235 s.<br>156 `predictions_test.csv` and 156 `test_metrics.json` files; every file hash is in `evaluation_ledger.jsonl` |
| `report --study configs/full-study.yaml` (after evaluation) | test aggregates and the robustness report regenerated from the stored, hash-checked test predictions; robustness evidence VERIFIED. Validation reports changed only in their timestamps. 87 s |
| `ruff check .`, `ruff format --check .`, `pytest -q` (after evaluation) | all checks pass; **264 passed**, 0 skipped, 168 s |

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

### Held-out test results (sealed; each run evaluated once)

**HELD-OUT TEST:** every checkpoint, threshold and rule was fixed on validation and sealed before
test access, and each run was evaluated on test once.

Macro-F1 is shown as mean ± sample SD over distinct runs. Full tables:
`reports/fewshotlab-v1-full/test/aggregate_test.md`.

| Dataset | K | scratch | frozen | partial | prototypes |
|---|---:|---:|---:|---:|---:|
| Flowers102 | 1 | 0.061 ± 0.008 | 0.406 ± 0.028 | 0.408 ± 0.020 | 0.394 ± 0.020 |
| Flowers102 | 3 | 0.175 ± 0.011 | 0.657 ± 0.007 | 0.668 ± 0.011 | 0.624 ± 0.003 |
| Flowers102 | 5 | 0.256 ± 0.013 | 0.744 ± 0.002 | 0.766 ± 0.010 | 0.692 ± 0.003 |
| Flowers102 | 10 | 0.411 ± 0.022 | 0.825 ± 0.003 | 0.853 ± 0.017 | 0.751 (1 distinct run) |
| NEU-CLS | 1 | 0.436 ± 0.156 | 0.855 ± 0.055 | 0.852 ± 0.063 | 0.815 ± 0.064 |
| NEU-CLS | 3 | 0.623 ± 0.155 | 0.905 ± 0.019 | 0.937 ± 0.022 | 0.882 ± 0.032 |
| NEU-CLS | 5 | 0.794 ± 0.048 | 0.940 ± 0.021 | 0.960 ± 0.014 | 0.908 ± 0.027 |
| NEU-CLS | 10 | 0.779 ± 0.056 | 0.962 ± 0.009 | 0.970 ± 0.010 | 0.944 ± 0.021 |

**Paired differences on matched seeds (test):**
- **Transfer vs scratch:** every transfer method beats scratch in 5/5 seeds at every K on both
  datasets.
- **Partial vs frozen:**
  - higher in 5/5 seeds at Flowers K=5 (+0.023) and at NEU K=3, 5 and 10 (+0.032, +0.020,
    +0.009); 4/5 at Flowers K=10 (+0.028);
  - Flowers K=3: +0.011 (3/5 seeds);
  - no difference at K=1 (Flowers +0.001, 2/5 seeds; NEU −0.004, 2/5 seeds).
- **Prototypes vs frozen:** prototypes score below frozen in at least 4 of 5 seeds in every cell.
  On Flowers the gap grows with K, from −0.012 at K=1 to −0.073 at K=10.
- **Validation vs test:**
  - Flowers test macro-F1 is below validation in all 16 cells, by 0.008–0.046 (mean 0.028). That
    is expected when checkpoints are selected on validation.
  - On NEU the difference is −0.003 to +0.020 (mean 0.009).
- **Class balance:** the Flowers test split is class-imbalanced, so top-1 and balanced accuracy
  differ (partial K=10: 0.855 vs 0.876). Macro-F1 is the primary metric.

### Robustness study (held-out test, demo checkpoint)

- **What was scored:** synthetic corruptions of the 360 NEU test images, scored once by the sealed
  demo checkpoint (`fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa`) with the sealed
  review threshold 0.95.
- **How it is reported:** apart from the primary benchmark. Each derivative is paired with its
  clean parent and is not a new test image.
- **Source:** `reports/fewshotlab-v1-full/test/robustness.md`.

| Condition | Macro-F1 | Δ vs clean | Coverage | Error among accepted | Changed predictions |
|---|---:|---:|---:|---:|---:|
| clean | 0.975 | — | 0.944 | 0.009 | — |
| brightness ×0.6 | 0.975 | −0.000 | 0.944 | 0.003 | 8 |
| brightness ×1.4 | 0.914 | −0.061 | 0.808 | 0.021 | 24 |
| Gaussian blur σ 1 px | 0.916 | −0.059 | 0.803 | 0.017 | 22 |
| Gaussian blur σ 2 px | 0.731 | −0.244 | 0.656 | 0.089 | 80 |
| Gaussian noise SD 0.03 | 0.945 | −0.030 | 0.900 | 0.015 | 15 |
| Gaussian noise SD 0.08 | 0.677 | −0.298 | 0.642 | 0.087 | 102 |

- **Clean stream:** it reproduces the primary test predictions exactly (`clean_path_consistency`).
- **The threshold on clean test images:** 94.4 % of images stayed automatic, with 0.9 % error
  among accepted. On validation the figures were 95.8 % and 0.3 %.
- **Strong blur and noise:** both cut macro-F1 by 0.24–0.30. They send about a third of the
  images to review and raise the error among accepted to about 9 %.
- **What is not measured:** NEU-CLS has no defect-free class, so nothing here measures false
  accepts on normal material.

### P5 evidence table

| Criterion | Status | Evidence | Observed |
|---|---|---|---|
| Every planned slot has an explicit status; failures stay visible | PASS | `runs/ledger.jsonl`, `reports/fewshotlab-v1-full/plan.md` | 156 COMPLETED, 4 DEDUPLICATED, 0 FAILED |
| Checkpoints selected on validation only; test locked throughout training | PASS | `test_guardrails.py` (no unsealed test outputs; the training path never imports the gate or the corruption code); `evaluate --split test` exits 5 without a seal | 0 test-split files |
| Seed aggregation over distinct runs: sample SD, paired differences on matched seeds, duplicates counted once, no confidence intervals | PASS | `aggregate_val.*`, `paired_val.csv`, `slots_val.csv`, `test_p5_aggregate.py` | Flowers K=10 prototypes shown as 1 distinct run, SD null |
| Every aggregate regenerates from stored predictions | PASS | `report --study` evidence check | 156/156 |
| Training repeats exactly on MPS | PASS | seed-11 MVP vs core comparison | 8/8 identical predictions and scores |
| Validation re-scored with the current code before sealing | PASS | `seal_preview.json`; `sealed_plan.json` attestations | dry run and seal: 156/156 identical, maximum score difference 0.0. All runs trained at `64c7d25`; the 14 sealed source files changed since (P5 tooling) give identical results |
| Complete evaluation plan before any test access: code, environment, data, supports, checkpoints, weights, metrics, scorer, calibration, corruptions, demo, threshold, replay, report rules | PASS | `reports/fewshotlab-v1-full/sealed_plan.json`, `evaluation_ledger.jsonl` | sealed at 06:56:38Z and pushed (`abe8aa4`) before the first test evaluation; 0 missing runs |
| Any change to a sealed input fails closed | PASS | `test_p5_seal.py` | a changed checkpoint, support, coverage target, corruption parameter, plan file, ledger or replay file is refused |
| Corruptions seeded, checksummed and reproducible; the clean condition equals the normal input | PASS | `test_p5_corruptions.py` | 10 tests pass |
| Review threshold and demo cell chosen on validation only | PASS | `sealed_plan.json` | demo: partial, NEU K=10 seed 11 (validation macro-F1 0.9917).<br>Threshold 0.95: validation coverage 0.958, 15 of 360 images to review, 1 accepted error (0.29 %) |
| Held-out test metrics (every sealed run, once) | PASS | `evaluation_ledger.jsonl`, `test/aggregate_test.*` | 156/156 evaluated once; every file hash recorded; tables regenerate from the stored predictions |
| Robustness study on test (demo run, 7 conditions) | PASS | `test/robustness.*` | 7 conditions × 360 images; the clean stream reproduces the primary test predictions exactly; evidence VERIFIED |
| Test-split files exist only under an intact seal | PASS | `test_guardrails.py` (`audit_test_outputs`) | after evaluation: every test-split run file matches its recorded hash, and the test reports sit under the intact seal |

### Known issue (cosmetic)
- `portable_path` resolves the `runs/artifacts` symlink before making a path relative.
- So files inside run folders are recorded as `<outside project>/<name>`. For example, the dry
  run's 156 verification events show `<outside project>/val-001.json` for
  `runs/artifacts/<run>/a001/verifications/val-001.json`.
- The run ID and attempt still identify each file, and no local path leaks.
- **Fix:** possible now that the test evaluation is complete (D-051). It was left alone between
  sealing and evaluation, because sealed code must not change then.

### Next action (P6, awaiting go-ahead)
- Build the Streamlit app on the sealed artifacts: stored predictions, the sealed demo checkpoint,
  the threshold and the replay manifest.
- The app and the replay can be served from the stored test predictions, so no further test
  access is planned.
- Fix the cosmetic `portable_path` label (above). **Fixed in P6 (D-055).**

**Held-out test metrics: EVALUATED once under seal `seal-c6f58862d63b586a0df8dbcf140db0fd`.**

## P6: demonstration app (2026-11-02)

**State at the end of this entry:**
- The app, the `predict` command and the `demo` command are implemented.
- They were checked on the real project in a browser.
- P7 (live replay inference, judge rehearsal) has not started.

### What changed
- **Live prediction** (`engine/predict.py`, D-053):
  - upload validation by content (PNG, JPEG, BMP, TIFF or WebP; one frame; at most 10 MB and
    25 million pixels; nothing written or logged);
  - model identification with status, evidence, schema, checkpoint-hash and class-map checks;
  - prediction through the stored model's own scoring code, with preprocessing and inference
    timed separately;
  - comparison with a run's stored validation predictions.
- **Run registry** (`reporting/run_registry.py`, D-054): the only source of displayed numbers.
  It wraps the report's `slot_values` and adds the seal summary and dataset facts, cached under a
  fingerprint of the ledgers and the sealed plan.
- **Sealed replay view** (`simulation/sealed_replay.py`, D-052): it plays back the stored sealed
  robustness predictions in the sealed order with the sealed threshold. It is verified by
  identity, and every displayed derivative is checksum-verified.
- **Commands:**
  - `predict` (new) scores image files with one verified run;
  - `demo` launches the app on 127.0.0.1 (`app_launch.py`);
  - `demo --check` checks offline, on the CPU, that it can start.
- **App** (`app/`): Overview, Data and label budget, Experiment explorer, Single image prediction,
  Industrial replay, Reproducibility.
  - Every result carries a label: VALIDATION, HELD-OUT TEST, LIVE PREDICTION, NEU HELD-OUT
    REPLAY or SOFTWARE FIXTURE.
  - `.streamlit/config.toml`: local only, no usage statistics, 10 MB uploads, light theme.
- **Fix:** `portable_path` keeps the project spelling through the `runs/artifacts` symlink
  (D-055).
- **Tests:**
  - `tests/test_p6_app.py` (12 tests) and `tests/test_p6_paths.py`;
  - `tests/test_p0_cli.py` now documents the `predict` and `demo` flags and tests the
    unfinished-phase exit code directly, since every command now exists.

### Commands executed and observed results
| Command | Observed |
|---|---|
| `python -m fewshotlab demo --check` (real project, CPU, offline) | **ready, every check PASS:**<br>• streamlit 1.66.1, app file, ImageNet weights cached<br>• validation 160 COMPLETED and test 160 EVALUATED slots<br>• Flowers and NEU default models load on CPU and match their stored validation predictions (max score difference 4.5e-13 and 2.1e-10; tolerance 1e-4)<br>• sealed replay verified: 7 streams × 360 images |
| `python -m fewshotlab predict --run fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa --image .../Cr_1.bmp --top-k 3` | crazing, model score 0.999941 (the stored validation prediction: 0.99994063); 7.0 ms inference on CPU |
| `predict` with a text file named `fake.png` | refused, exit 1: "the file content is not a readable image" |
| `python -m fewshotlab demo`, reviewed in the built-in browser at 1440×900 | all six pages render; three chart defects found and fixed (see below) |
| `ruff check .`, `ruff format --check .`, `pytest -q` | all checks pass; **279 passed**, 0 skipped, 220 s (includes the real-project app smoke test and `demo --check`) |

### Browser review (the app was run)
Checked at 1440×900 in the built-in browser. Screenshots stayed in the session and were not saved
to the repository, since some show dataset images.
- **Defects found and fixed:**
  - **Chart text unreadable.** The Explorer charts drew their text in the dark-theme ink. The
    browser reported a dark theme although the app is configured light; the configured theme now
    wins.
  - **Legend missing.** One chart layer switched the shared legend off.
  - **Last panel clipped.** The per-seed panels overflowed and the last one was cut off; each
    method now gets its own column.
  - **Text cut off.** Six overview steps in one row broke words, and two labels on the replay
    page were truncated.
  - **Flowers names.** Flowers classes showed only `label_NNN`; they now show the display name
    beside the identity.
- **Replay:**
  - playback advanced about one image per second (15 images in about 15 s);
  - each stressed image appears next to its own clean parent;
  - the review decision and every count show their denominator, and true labels stay hidden
    until revealed.
- **An observed failure:** with Gaussian noise SD 0.08, `neu_cls/PS_7` was predicted crazing at
  model score 0.766. It went to human review, and its true class is pitted surface. Its clean
  parent was predicted correctly at 0.999.
- **Prediction:** the validation image `flowers102/image_06738` gave pink primrose (label_001) at
  0.9994. It matched the run's stored validation prediction within 1.7e-09; inference took
  7.9 ms on CPU.
- **Reproducibility:** all 14 protocol modules still match their sealed hashes.

### P6 evidence table

| Criterion | Status | Evidence | Observed |
|---|---|---|---|
| A smoke journey loads the app, selects each core method, renders real or explicitly fixture-labeled outputs and shows provenance | PASS | `test_p6_app.py`: every page on a sealed fixture and on the real project, each method on the prediction page; browser review | fixture pages carry the SOFTWARE FIXTURE label; each method's run ID and its stored-prediction match are shown; the seal token is on Reproducibility |
| Changing dataset or K cannot leave an incompatible cached model active | PASS | models cached by run, attempt, checkpoint hash and device; a prediction is tied to the same plus the image (D-054); class-map check in `model_ref` | a model whose class map differs is refused (test) |
| Invalid uploads, missing checkpoints, empty filters and unavailable hardware do not crash | PASS | upload, identity, empty-filter and empty-project tests; device errors are caught on the page | actionable messages, no stack traces |
| Readable labels, units, legend, statuses, contrast and no clipped charts on the judging display | PASS after fixes | browser review at 1440×900; palette checked with the validator (light and dark adjacent-pair checks PASS; per-seed points use small multiples, since the all-pairs check fails for 4 series) | three chart defects found and fixed |
| App and CLI predictions match for the same input, model and preprocessing within the declared tolerance | PASS | `test_live_predictions_match_stored_validation_and_the_cli`; `demo --check`; the page's own check | fixture: bit-identical for the same batch on CPU; real project: 4.5e-13 and 2.1e-10 against the MPS-made stored predictions (tolerance 1e-4) |
| Offline startup with cached approved assets; prerequisites documented | PASS | `demo --check` (no network); README "Demonstration app" | needs the uv environment, prepared data and run folders; ImageNet weights only for prototype models |
| Numbers come only from the registry; NOT RUN for missing values; no substitutes | PASS | `run_registry`; empty-project test | every page of an empty project shows NOT RUN |
| Uploads: size and pixel limits, content check, no temporary files, no logging, no user model files | PASS | `decode_upload` tests; uploads stay in memory; models load only from the registry | |
| The app never trains, seals or evaluates the test split | PASS | AST guard test; file-hash snapshots around the app tests | no file under runs/, reports/ or data/ changes |

### Next action (P7, awaiting go-ahead)
- Live inference with measured latency on the replay, timed separately from playback.
- The seven-minute judge script and a rehearsal checklist.
- A fixed error gallery, using the observed failure above, and an offline package.

## P7: industrial replay and judge rehearsal (2026-11-02)

**State at the end of this entry:**
- The live replay, the offline rehearsal, the judge script and the rehearsal record are done.
- The automated rehearsal ran on this Mac.
- The timed, spoken seven-minute rehearsal is the presenter's to do.

### What changed
- **Live replay inference** (`simulation/replay_live.py`, D-056):
  - the sealed checkpoint loads only under its sealed hash;
  - each derivative is regenerated and checksum-verified, then scored through the sealed
    evaluation contract;
  - every result is checked against the stored sealed prediction;
  - timing follows document 02, p.16 (below).
- **Offline rehearsal** (`simulation/rehearsal.py`, `demo --rehearse`, D-057):
  - every network connection is refused;
  - all 7 streams are scored live and compared with the stored predictions;
  - the recording goes to `demo_assets/fewshotlab-v1-full/`: `rehearsal.json` and
    `rehearsal.md` are tracked, while `rehearsal_events.csv` (per image) stays local;
  - a `replay_rehearsal` ledger event records the file hashes.
- **Replay page:**
  - live mode by default, with the measured latency of each request and the stored-prediction
    check;
  - a visible fallback to the stored sealed recording, with its timestamp and model ID;
  - a downloadable event log;
  - the fixed error gallery (D-058).
- **Reproducibility page:** "Trace a number" recomputes any plotted macro-F1 from its prediction
  file and shows the checkpoint hash.
- **Docs:** [JUDGE_SCRIPT.md](JUDGE_SCRIPT.md) and [REHEARSAL.md](REHEARSAL.md).
- **Tests:** `tests/test_p7_replay.py` (7 tests).

### Commands executed and observed results
| Command | Observed |
|---|---|
| `python -m fewshotlab demo --rehearse` (CPU, 8 threads, float32, network refused) | **2,520 of 2,520 live predictions equal the stored sealed ones** (360/360 in each stream); stream coverage equals the sealed results.<br>Cold start 120 ms. Forward latency: median 6.12 ms, 95th percentile 6.29 ms (batch 1, 20 warm-up, 200 timed). End to end per image: median 7.02 ms, 95th percentile 11.56 ms (browser excluded). 22.5 s in total |
| App, reviewed in the built-in browser at 1440×900 | live replay: rolled-in scale at 1.0000, this request 15.4 ms end to end, live = stored (difference 0.0); the trace recomputes 1,020 rows to an identical macro-F1 |
| `ruff check .`, `ruff format --check .`, `pytest -q` | all checks pass; **286 passed**, 0 skipped, 242 s |

### P7 evidence table

| Criterion | Status | Evidence | Observed |
|---|---|---|---|
| Replay model, threshold, order and corruption pixels match the sealed P5 artifacts; changed identities fail | PASS | `sealed_replay` checks (P6); `LiveReplay.load` refuses a changed checkpoint hash (test); every derivative is checksum-verified | 2,520 of 2,520 live = stored |
| Corruption metrics stay separate with paired parent IDs; no derivative appears in training | PASS | `_separation_problems` test on the fixture and the real study | benchmark predictions are clean-only; every stream keeps the clean parents' order; supports hold only train IDs |
| Timing separates inference, preprocessing and playback, with the accelerator synchronized | PASS | `replay_live.py`; rehearsal report; replay page | forward and end-to-end figures with samples kept; playback labeled as a presentation pace |
| An offline rehearsal runs start to finish with no hidden downloads; missing artifacts trigger the documented fallback visibly | PASS | `demo --rehearse` under `no_network`; fallback page test | no connection attempted; "FALLBACK: stored sealed predictions" when the model cannot load |
| The fallback uses cached real results with timestamps and never invents performance | PASS | stored sealed predictions with their timestamp and model ID | |
| Checklist, observed timing, failure points, backup launch steps and answers to expected questions saved | PASS | [REHEARSAL.md](REHEARSAL.md) | |
| Judge script with the protocol's timings, a real failure, and training done in advance | PASS | [JUDGE_SCRIPT.md](JUDGE_SCRIPT.md) | `In_228`: inclusion predicted as scratches at 0.989 and accepted |
| Pause, resume and deterministic replay work offline; the output is a defect type or review | PASS | replay page (P6 tests and the browser) | |
| A plotted score reaches its checkpoint and prediction file within a minute | TO VERIFY by the presenter | the trace view on one page | |

### Next action (P8, awaiting go-ahead)
Release: a reproduction guide, the model card, the evidence index and the release checklist
(document 02, p.21).
