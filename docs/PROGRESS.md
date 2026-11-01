# Progress record

Each entry lists commands that were actually executed and what was observed.
- **NOT_RUN:** the step has not been executed.
- **BLOCKED:** an external prerequisite is missing.

## Current status

| Area | Status |
|---|---|
| Implementation | P0, P1 and P2 complete (data tooling, transforms, the four methods, model persistence). P3 (training engine) is next; train, evaluate, benchmark, report and demo exit 3 |
| Data | Flowers102 PREPARED (official split, MD5-verified) and audited PASS. NEU-CLS PREPARED (project split) and audited PASS. 40 support manifests built |
| Benchmark | NOT_RUN (`runs/ledger.jsonl` is empty) |
| Demo | NOT_RUN |
| Remote push | P0 pushed to `origin/main` (`2c1e9b7`) once write access was confirmed |

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
