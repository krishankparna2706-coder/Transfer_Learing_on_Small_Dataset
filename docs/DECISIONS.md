# Decision log

Each entry records what was decided, why, and what it implies. Newest entries go at the bottom.

## D-001 (2026-11-01): Python 3.11 managed by uv
- **Decision:** `requires-python = ">=3.11,<3.12"` with uv project mode (uv 0.11.15). The
  specification proposes 3.11, and a uv-managed CPython 3.11.15 was already installed.
- **Consequence:** on 2026-11-01 the newest numpy (2.5.x) and scipy (1.18.x) require Python ≥ 3.12,
  so the lock resolves numpy 2.4.6 and scipy 1.17.1. torch 2.15.0 and torchvision 0.29.2 publish
  macOS-arm64 wheels compatible with 3.11.

## D-002 (2026-11-01): `uv.lock` is authoritative; `requirements.lock` is exported
- **Decision:** `uv lock` resolves 80 packages into `uv.lock`, a universal lock covering macOS,
  Linux and Windows. `requirements.lock`, the file name used in the blueprint, is generated from it:
  `uv export --format requirements-txt --no-emit-project --locked -o requirements.lock`.
  It includes the dev group and hashes. Never edit either file by hand.

## D-003 (2026-11-01): Repository layout follows blueprint page 12
- **Decision:**
  - `src/fewshotlab/` holds `cli.py` plus the subpackages `data/`, `models/`, `engine/`,
    `metrics/`, `reporting/` and `simulation/`.
  - Other top-level folders: `app/`, `configs/`, `tests/`, `data/manifests/`, `runs/`,
    `reports/`, `demo_assets/` and `docs/`. Each documents its own responsibility.

## D-004 (2026-11-01): Specification PDFs stay local
- **Decision:** the three specification PDFs are gitignored, at the user's request, because the
  GitHub repository is public. `docs/SPEC_SOURCES.md` identifies them by SHA-256.

## D-005 (2026-11-01): Git remote, commits and pushes
- **Decision:**
  - Origin is `krishankparna2706-coder/Transfer_Learing_on_Small_Dataset` (public, created empty
    on 2026-11-01).
  - At the user's request there is one local commit per completed phase.
  - Pushes to `main` happen only when the authenticated GitHub account has write permission
    (`gh api repos/<repo> --jq .permissions.push`); otherwise the push is recorded as BLOCKED.
  - Never force-push.

## D-006 (2026-11-01): No implicit downloads; project-local caches
- **Decision:**
  - Datasets and pretrained weights are fetched only when a command receives `--allow-download`.
  - Weights live in `.cache/torch/checkpoints/` (gitignored) and are verified against
    torchvision's SHA-256 file-name prefix; the full SHA-256 is recorded.
  - Without permission, a missing file raises an actionable error and exits with code 4.

## D-007 (2026-11-01): CLI built on argparse, with explicit exit codes
- **Decision:**
  - The CLI uses stdlib `argparse`; this adds no dependency and gives stable `--help` text.
  - Exit codes:
    - 0: OK
    - 1: failure
    - 2: usage
    - 3: not implemented
    - 4: blocked
    - 5: refused by a protocol guard
  - Commands owned by later phases exit with code 3 and write nothing.

## D-008 (2026-11-01): Strict configuration
- **Decision:**
  - Unknown keys are errors.
  - Protocol constants live once in `fewshotlab.config`, and `configs/protocol-v1.yaml` is
    checked against them.
  - Run categories:
    - `core` and `mvp` must use the v1 recipe exactly;
    - `development` and `diagnostic` may change budgets and are separate ledger categories;
    - `fixture` marks software-test runs (`is_fixture=true`).
  - Configs can never unlock the test split (`test_access` must be `locked`).
  - PyYAML follows YAML 1.1, so exponents need a decimal point (`1.0e-08`). The parser explains
    this when it sees a string.

## D-009 (2026-11-01): Random streams
- **Decision:**
  - Each stream (split, support, init, augmentation, loader order) is seeded from SHA-256 of a
    labeled tuple. Python's `hash()` is never used.
  - Model initialization runs inside `torch.random.fork_rng` seeded with the run's train_seed, so
    it neither consumes nor perturbs any other stream.

## D-010 (2026-11-01): Model structure
- **Decision:** `ResNet18Classifier` has two parts:
  - `encoder`: torchvision `resnet18` with `fc = nn.Identity()`, producing 512-d pooled features;
  - `head`: `Linear(512, C)`.
- **Consequence:**
  - State-dict keys are `encoder.*` and `head.*`.
  - ImageNet weights load into the encoder strictly, after dropping exactly `fc.weight` and
    `fc.bias`.
  - Parameter counts match the blueprint arithmetic: 11,176,512 encoder; 11,228,838 (C=102);
    11,179,590 (C=6).

## D-011 (2026-11-01): Tooling
- **Decision:**
  - Ruff handles both lint and format (line length 100).
  - scikit-learn is a dev-only dependency. It serves as an independent oracle in metric tests;
    project metrics are implemented in `fewshotlab.metrics`.

## D-012 (2026-11-01): Flowers102 official files come from Internet Archive captures, MD5-verified
- **Context:** the official Oxford host did not accept TCP connections from this network.
- **Decision:** with the user's approval, `data prepare --mirror internet-archive` fetches pinned
  Internet Archive captures of the three official URLs. It keeps a file only if its MD5 equals
  torchvision's recorded checksum. All three matched, so the images, labels and split are the
  verified official ones.

## D-013 (2026-11-01): The user's Kaggle copy is used for verification only
- **Context:** `flower_data/` is the Udacity re-packaging (6,552 / 818 / 819, unlabeled test),
  not the official split.
- **Decision:** `--cross-check` compares it with the official files:
  - 8,189 / 8,189 images are byte-identical;
  - 7,370 / 7,370 folder labels agree.
- Because every label agreed, its `cat_to_name.json` supplies display names (display only;
  identity stays `label_001..label_102`).
- The folder is gitignored and never modified.

## D-014 (2026-11-01): The virtualenv lives outside iCloud Drive, with `.venv` a symlink to it
- **Context:** the project folder is inside the iCloud-synced Desktop. iCloud's file provider sets
  the macOS `hidden` flag on items inside dot-folders, and Python 3.11.15 skips hidden `.pth` files.
  The editable install therefore vanished at 19:57.
- **What did not work:** renaming the folder to `.venv.nosync` did not stop iCloud from finishing
  its backlog.
- **Decision:** the environment now lives at `~/.local/share/fewshotlab/venv`, outside the synced
  folder. `.venv` in the project is a symlink to it, so `uv run` works unchanged.
- **Recommended:** move the project out of iCloud Drive (for example to `~/Projects`). That also
  stops roughly 1 GB of data and caches from syncing.

## D-015 (2026-11-01): Operating-system metadata files in archives are ignored and listed
- **Decision:** `Thumbs.db`, `desktop.ini`, `.DS_Store`, `._*` and `__MACOSX/` are not dataset
  content. The NEU adapter ignores them, but lists each one in the prepare record, so nothing is
  dropped silently.
- **Also:** a validation failure now writes a FAILED prepare record. Download provenance is saved to
  `data/raw/<dataset>/acquisition.json` before validation, so a re-validation keeps it.

## D-016 (2026-11-01): Duplicates found by the audits are disclosed, not repaired
- **Flowers102 (official split preserved):**
  - Byte-identical cross-split pairs: `image_07307` train = `image_07309` test, and
    `image_08067` test = `image_08077` val.
  - 6 further cross-split pHash candidates (distance ≤ 4).
  - 1 cross-class near-identical test pair, `image_07303` / `image_07304`.
  - Human visual review is pending.
- **NEU-CLS:** `Pa_101` = `Pa_105` (pixel-identical). The split's cluster rule placed both in test;
  counts unaffected.
- Any deduplicated analysis would be a separately named sensitivity protocol.

## D-017 (2026-11-01): Near-duplicate threshold
- **Decision:** candidates are listed at pHash Hamming distance ≤ 4 of 64 bits for manual review.
  Each audit also records the pair counts at 0, 2, 4, 6, 8, 10 and 12 bits, so the threshold's
  effect is visible. Candidates are never removed automatically.

## D-018 (2026-11-01): Supports require a passing audit and are immutable
- **Decision:** `support build` refuses to run without a PASS audit of the current manifest
  (identified by its hash).
- A stored support file is never overwritten. A rebuild that would change a support set is
  refused; it would need a new protocol version.
- Class keys in support files are strings and are compared as a set. A string sort put "10" before
  "2", which broke validation for the 102 Flowers classes. That bug was fixed, with a regression
  test.

## D-019 (2026-11-01): `weights fetch` command
- **Decision:** `python -m fewshotlab weights fetch` is an explicit, documented way to cache the
  ResNet18 IMAGENET1K_V1 weights for offline use. It is in addition to the spec's command list.
- Every other command still needs `--allow-download` before it may fetch anything.

## D-020 (2026-11-01): One tensor-based preprocessing implementation
- **Decision:** transforms decode to float32 tensors in [0, 1] and apply
  `torchvision.transforms.functional` operations (bilinear, antialiased). The CLI, reports and app
  share this single path.
- **Check:** the Flowers evaluation transform is verified to equal torchvision's own
  `Resize(256) → CenterCrop(224) → Normalize` composition on tensors, bit for bit.
- **Caveat:** results differ slightly from PIL-based resizing; that is why the transform ID pins the
  torchvision and Pillow versions.

## D-021 (2026-11-01): Augmentation randomness is per sample and per epoch
- **Decision:** each training image draws its augmentation from a NumPy generator seeded by
  (augmentation stream, train_seed, epoch, image_id).
- **Draw order is fixed:** RandomResizedCrop parameters (torchvision's algorithm), then the
  horizontal flip, then the vertical flip.
- **Consequence:** results do not depend on DataLoader workers or batch order, and augmentation
  never touches the global torch or NumPy state.

## D-022 (2026-11-01): Freezing policy is enforced through module modes and checked
- **Decision:** `enter_train_mode` replaces `model.train()`:
  - frozen: the encoder goes back to eval mode;
  - partial: every BatchNorm goes back to eval mode.
- **Checks:** `check_train_mode` verifies `requires_grad` flags and modes. Tests prove that one AdamW
  step changes exactly the permitted parameters and no frozen BatchNorm buffer. A negative control
  shows that plain `model.train()` would change them.

## D-023 (2026-11-01): AdamW weight decay applies to every trainable parameter
- **Decision:** the v1 recipe states weight decay 1e-4 without exclusions. It therefore applies to
  BatchNorm affine parameters and biases too (PyTorch's AdamW default behavior).
- Excluding them would be a separately named recipe change.

## D-024 (2026-11-01): Artifact formats are safe to load
- **Learned models:** `best.ckpt` holds the state dict and the spec JSON, loaded with
  `torch.load(weights_only=True)`. A test proves arbitrary pickled objects are refused.
- **Prototypes:** `prototypes.npz` holds the matrix, class names and spec JSON, loaded with
  `allow_pickle=False`.
- **Every model is rebuilt from its spec:** encoder weights identity, class order, transform ID and
  method parameters. A different dataset, class map, class order, transform or method is refused.

## D-025 (2026-11-01): Prototype distances are computed exactly, in chunks
- **Decision:** squared distances use `(q - p)²` summed in query chunks, not the
  `|q|² - 2qp + |p|²` expansion. This avoids cancellation error, and ties go to the lowest class ID.
- **Cache keys:** embedding caches are keyed by encoder SHA-256, transform ID, manifest SHA-256,
  class-map SHA-256 and the ordered image-ID list. Any difference is a cache miss.

## D-026 (2026-11-01): The initialization stream depends only on train_seed
- **Decision:** `init_seed = SHA-256("init", train_seed)`, applied inside an isolated
  `torch.random.fork_rng`.
- **Consequence:** at a given seed every method starts from the same new-head initialization, and
  scratch from the same random encoder. Methods are therefore paired on support IDs, augmentation
  draws, loader order and head initialization.
- **Other streams:** augmentation uses (`augment`, train_seed, epoch, image_id) and loader order
  uses (`loader`, train_seed, epoch).

## D-027 (2026-11-01): How checkpoint selection and patience are implemented
- **Selection:** macro-F1 is compared in float64. A new best requires a higher macro-F1, or an equal
  one with strictly lower validation cross-entropy (sample-weighted, float64). Otherwise the
  earlier epoch is kept.
- **Patience:** the reference starts at −∞, so epoch 1 always sets it. It resets only when
  macro-F1 is at least reference + 0.001. Training stops once 15 consecutive epochs pass without
  such a reset, so the earliest possible stop is after epoch 16.
- **Checkpoints:** `best.ckpt` is written at every new best; the resumable `last.ckpt` after every
  epoch.

## D-028 (2026-11-01): Evidence integrity rules
- **Predictions first:** metrics in `val_metrics.json` are computed from the written
  `predictions_val.csv`, after re-reading it and checking that every validation image appears
  exactly once.
- **Reload check:** the selected `best.ckpt`, or the saved prototypes, is reloaded into a fresh
  model and must reproduce the validation predictions; otherwise the run FAILS.
- **Aggregation:** a run is accepted only if it is COMPLETED, not a fixture, on the right protocol
  and category, and its stored metrics regenerate exactly from its predictions file (with a
  matching hash).

## D-029 (2026-11-01): Determinism settings
- **CPU:** strict `torch.use_deterministic_algorithms(True)`.
- **MPS and CUDA:** the same call with `warn_only=True`, because some kernels have no deterministic
  implementation. This is recorded in `environment.json`.
- **Claim scope:** two CPU repetitions of a fixture run give equal predictions, and scores equal
  within the declared tolerance of 1e-6. No bitwise claim is made across devices or platforms.

## D-030 (2026-11-01): Resume semantics
- **Resume is a new attempt.** `train --resume` continues an interrupted (FAILED or RUNNING) attempt
  as a new attempt that records `resumed_from`. It restores the model, the optimizer state, the
  epoch, the selection and patience state, and the history. It also copies the earlier best
  checkpoint, verified by hash.
- **Exactness:** the random streams are pure functions of (seed, epoch, image), so a resumed CPU
  fixture run reproduces an uninterrupted run exactly (tested).
- **Immutability:** interrupted attempts stay FAILED, and a COMPLETED run can never be re-run
  in place.

## D-031 (2026-11-01): Run identity and the ledger
- **Run ID:** `protocol__dataset__method__kK__sSEED__cfg<config sha256[:8]>`. Non-core categories
  are prefixed, for example `diagnostic__…` or `mvp__…`.
- **Attempts:** each run has numbered attempts `a001`, `a002`, …
- **Ledger:** `runs/ledger.jsonl` is append-only, with `status` events (RUNNING, then COMPLETED or
  FAILED) and `verification` events.

## D-032 (2026-11-01): Execution details outside the recipe
- **Decision:** the evaluation batch size (64) and the DataLoader worker count are execution
  details. Results do not depend on them, by design (per-sample augmentation streams), and they
  are recorded with each run.

## D-033 (2026-11-01): The test gate before P5
- **Decision:** `evaluate --split test` is always refused (exit 5) and computes or writes nothing,
  until P5 implements `lock-test` sealing.
- **Validation re-check:** `evaluate --split val` writes a verification record linked to the
  original attempt. It is not a new result.

## D-034 (2026-11-01): Held-out IDs are read only for disjointness checks
- **Decision:** verifying a support set reads the validation and test **IDs** to prove no
  held-out image is in it. No held-out image, label or pixel enters fitting or selection.
- **Proof:** the training path itself reads only the `train` and `val` rows, which a test proves
  with a spy.

## D-035 (2026-11-01): The resume state is deleted when a run completes
- **Context:** `last.ckpt` (model plus AdamW state, about 113 MB for ResNet18) is only needed to
  resume an unfinished run.
- **Decision:** a COMPLETED run deletes its own `last.ckpt` and records that in provenance. It keeps
  `best.ckpt`, the inference artifact. FAILED attempts keep theirs, so they can still be resumed.
- **Exception:** the five P3 diagnostic runs ran before this rule existed and still hold their
  local `last.ckpt` files. Those are gitignored.
