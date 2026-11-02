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

## D-036 (2026-11-01): Studies expand to run slots; deterministic duplicates are counted once
- **Expansion:** a study config (`configs/mvp.yaml`, `configs/full-study.yaml`) expands
  datasets × K × methods × seeds into run slots. Each slot has the same config hash, and therefore
  the same run ID, as the equivalent standalone config file, so a job cannot run twice under two
  names.
- **Pre-flight checks:**
  - unique run IDs;
  - every method at a dataset/K/seed reads the same support manifest;
  - scratch uses random init, the others IMAGENET1K_V1;
  - the init seed depends only on train_seed;
  - one training transform per dataset;
  - one recipe across the learned slots.
- **Duplicates:** a prototype slot whose support **membership** (IDs per class, ignoring the seed
  label) matches an earlier slot is deterministic, so it is marked DEDUPLICATED and counted once.
  In the full study these are the Flowers K=10 seeds 22–55, giving 160 logical slots and 156
  unique runs.

## D-037 (2026-11-01): The provenance "dirty" flag covers code, configs and locks only
- **Context:** runs append to tracked records (`runs/ledger.jsonl`, reports, manifests). Counting
  those would mark every run after the first as dirty.
- **Decision:** `provenance.code.dirty` reflects changes under `src/`, `configs/`, `tests/`,
  `app/` and the lock and project files. Other changed paths are listed separately. The P4 MVP ran
  from commit `8f3d541` with a clean code tree.

## D-038 (2026-11-01): Validation-report rules
- **Source and banner:** `report --study` rebuilds every table and chart from the ledger and the
  stored prediction files, under a "VALIDATION ONLY" banner. The banner states that checkpoints
  were selected on the same validation images, so the numbers are optimistic and not test results.
- **Status handling:**
  - unfinished or deduplicated slots show their status with `null` metrics;
  - a run whose stored metrics do not regenerate is shown as EVIDENCE_FAILED.
- **What is committed:** per-run learning-curve and confusion plots contain no dataset images and
  are copied next to the report.
- **What stays local:** error galleries (rule fixed in advance: the first 12 misclassified
  validation images in image_id order) and augmentation grids contain dataset photos. They are
  gitignored and never published.

## D-016a (2026-11-01): Assisted review of the Flowers near-duplicate candidates
- **What was done:** while the P4 MVP ran, the 9 cross-split or cross-class pHash candidates were
  rendered side by side and inspected (Claude; a human still needs to confirm).
- **Finding:** all 9 appear to be the same photograph, or the same scene with a minimal change.
  `image_07303` (labeled ball moss) appears to be a foxglove identical to `image_07304`: a probable
  source label error.
- **Decision unchanged:** the official split is preserved, and the findings are disclosed (data
  card).

## D-039 (2026-11-01): Run artifacts live under `runs/artifacts/`, outside iCloud on this machine
- **Context:** the core matrix adds about 5–6 GB of checkpoints, and the project folder syncs to
  iCloud Drive. The user chose to keep run artifacts off iCloud.
- **Decision:**
  - Run folders now resolve to `runs/artifacts/<run_id>/<attempt_id>`.
  - On this Mac, `runs/artifacts` is a symlink to `~/.local/share/fewshotlab/runs` (local disk,
    not synced). Elsewhere it is a normal folder.
  - The ledger and reports stay in the project and in git.
- **Migration:** the 14 earlier runs (6 diagnostic, 8 MVP) were moved there unchanged. The MVP
  report regenerated with identical values afterwards.
- **Note:** ledger events written before this change record `run_dir` as `runs/<run_id>/<attempt>`;
  `run_root()` is authoritative. New events record `runs/artifacts/<run_id>/<attempt>`.

## D-040 (2026-11-01): What the seal compares, and what it only records
- **Content** is re-derived from the current files on every test access and must match exactly:
  - the code digest over `src/`, `configs/` and the lock and project files (`tests/` and `app/`
    are excluded, so the P6 app can be built later), plus per-file hashes of the protocol modules;
  - library versions and the evaluation device;
  - data, support, run, weight, metric, scorer, calibration, corruption, demo, replay, test-plan
    and report-rule entries.
- **Attestations** are recorded once, when sealing, and covered by the plan hash:
  - the git commit and branch;
  - the commits that trained the sealed runs, and the sealed files changed since;
  - the validation reproduction (D-041).
- **Why the commit is not compared:** committing the sealed plan itself moves HEAD. The code digest
  is what identifies the code that evaluates test.

## D-041 (2026-11-01): Validation reproduction before sealing
- **Context:** the core matrix was trained at commit `64c7d25`. The P5 tooling changed sealed code
  afterwards, notably the `ImageTransform.apply_tensor` refactor and a shared inference loader for
  validation and test.
- **Decision:** `lock-test` re-scores every sealed run's validation set with the current code.
  Sealing refuses unless every run's predictions are identical to the stored ones. The results
  (including the largest score difference) are part of the sealed attestations.

## D-042 (2026-11-01): Corruption implementation details
- **Order:** decode, then corrupt at native resolution in [0, 1], then the dataset's evaluation
  resize and normalization. The clean condition goes through the same code and equals the normal
  evaluation input exactly (tested).
- **Grayscale:** images with grayscale content are corrupted as one channel, so the noise stays
  gray, then replicated to three channels.
- **Blur:** separable Gaussian with 2·ceil(3σ)+1 taps and `reflect` padding (the edge pixel is not
  repeated). It is tested against `scipy.ndimage.gaussian_filter(mode="mirror", truncate=3)`.
- **Records:** every derivative records its parameters, seed, implementation ID and a SHA-256 of
  its float32 pixels. The pixel checksums are written during the sealed robustness evaluation,
  because computing them earlier would mean reading test pixels before the seal.

## D-043 (2026-11-01): The demo cell is fixed by the protocol
- **Default:** NEU-CLS, K = 10, support seed 11.
- **Fixtures only:** a study config may name another cell (`demo:`); core and MVP studies that try
  are refused.
- **Selection:** all candidates must be COMPLETED with verified evidence. The highest validation
  macro-F1 wins, and exact ties go to the lexically first method name.

## D-044 (2026-11-01): Review routing
- **Score:** the maximum class score. It is uncalibrated, and no calibration is fitted (sealed as
  `calibration: none`).
- **Threshold:** the highest value in {0, 0.05, …, 1} that keeps at least 80 % coverage on the demo
  run's stored validation predictions. Threshold 0 always qualifies.
- **Rule:** review when the score is strictly below the threshold.
- **Reporting:** coverage and the error among accepted predictions are always reported together.
  The accepted error is null when nothing is accepted.

## D-045 (2026-11-01): Replay manifest
- **Order:** a seeded permutation of the sorted NEU test IDs. It does not depend on labels or
  pixels, so no favourable subset can be picked.
- **Streams:** seven, in a fixed sequence (clean first) over the same parents in the same order.
  The true class is copied from the split manifest for demonstration mode.
- **Sealing:** the CSV is written next to the sealed plan and sealed by its SHA-256. P7 consumes it
  unchanged.

## D-046 (2026-11-01): One result per sealed run
- **Scope:** `evaluate --split test --study PATH` verifies the seal, then evaluates each sealed run
  that has no recorded evaluation, plus the robustness study of the demo checkpoint.
- **Outputs:** they go into the sealed attempt folder, and their hashes into the evaluation ledger.
- **Refusals:**
  - re-running a recorded evaluation is skipped;
  - an unrecorded test file blocks evaluation and sealing;
  - a device other than the sealed one is refused.
- **Isolation:** test images enter a model only through `GatedTestDataset`, which requires a
  verified seal. A static test checks that the trainer, runner, study planner and training
  datasets never import the test gate or the simulation package.

## D-047 (2026-11-01): Test-output audit
- `audit_test_outputs` (run by the repository guardrail test) finds every test-split file under
  `runs/` (symlinks followed), `demo_assets/` and `reports/<study>/test/`.
- **Per-run files** must match a hash recorded by a sealed evaluation event.
- **Report files** need an intact seal for their study.
- `report --study` writes test tables only after a sealed evaluation, so a test folder can never
  appear before the seal.

## D-048 (2026-11-01): Seed aggregation
- **Cells:** per dataset × K × method, the mean and sample SD (ddof = 1) over **distinct** runs. A
  deterministic duplicate (Flowers K=10 prototypes) counts once, so its SD is null.
- **Paired differences:** per dataset × K over matched support seeds (mean, SD, min, max, and the
  number of seeds where A > B).
- **Intervals:** no population confidence interval is computed from five seeds. Validation
  aggregates carry the "VALIDATION ONLY, optimistic" banner. Test aggregates exist only after the
  sealed evaluation.

## D-049 (2026-11-01): Partial seals
- If some planned runs are missing, sealing refuses unless `--allow-partial` is given.
- A partial seal is labelled partial in the plan, the ledger and the CLI output.
- The missing slots and their statuses are part of the sealed content.

## D-050 (2026-11-02): A run interrupted by system sleep is kept
- **Event:**
  - a low-battery system sleep suspended the matrix process for 22,486 s while
    `fewshotlab-v1__neu_cls__frozen__k5__s55__cfg8b5ac2db` was training;
  - the process resumed on wake and the run completed.
- **Decision:** keep the run rather than re-run it.
  - Training is deterministic, and no training decision reads the wall clock: early stopping and
    checkpoint selection use validation metrics only.
  - The run passed the trainer's completion check: reloading its selected checkpoint reproduced
    its validation predictions.
  - The seal dry run re-scored its validation set with the merged code and got identical
    predictions, as it did for all 156 runs.
- **Recorded:** the run's ledger wall time includes the sleep. [HARDWARE.md](HARDWARE.md) reports
  the matrix timing without it.

## D-051 (2026-11-02): Code changes after the sealed evaluation
- **Status:** the sealed evaluation of `fewshotlab-v1-full` is complete. 156 runs and the
  robustness study were each recorded once under `seal-c6f58862d63b586a0df8dbcf140db0fd`.
- **Later code changes:** later phases (the P6 app, the P7 replay) and fixes such as the
  `portable_path` label may change code.
  - The seal then no longer verifies. That blocks any further test access, as intended.
- **The recorded evaluation stays the reference.** The sealed plan, the evaluation ledger and the
  file hashes are kept. Reports regenerate the tables from the stored, hash-checked predictions.
- **Corrections:** a correction that changes any test result follows PROTOCOL §11. The affected
  results are marked INVALIDATED, with history kept, and rerun under a new protocol version.

## D-052 (2026-11-02): The sealed replay view is verified by identity
- **What it is:** after the sealed evaluation, the app plays back the NEU replay
  (`simulation/sealed_replay.py`):
  - the sealed demo checkpoint's stored predictions, from the P5 robustness evaluation;
  - in the sealed order;
  - under the sealed threshold.
  It selects, tunes and reports nothing new.
- **Why identity, not the whole code tree:** the full-tree check (D-051) stops matching as soon
  as later phases change code. Before any test image is read, the view checks instead that:
  - the sealed plan is intact, and the replay-relevant protocol modules (image decoding,
    transforms, corruptions, replay order, review rule) still have their sealed hashes;
  - the corruption protocol, test-ID list, class map, demo checkpoint and replay manifest match
    the plan;
  - the robustness evaluation is recorded once, with every file hash unchanged.
- **Pixels:** each displayed derivative is regenerated from its parent image and must reproduce
  its sealed checksum.
- **Failure:** any difference refuses the view and shows the fallback (the stored aggregates).
- **P7:** P7 adds live inference with measured latency on top of this verified view.

## D-053 (2026-11-02): Live predictions on supplied images
- **One code path:** the app and `python -m fewshotlab predict` share `engine/predict.py`. An
  image is decoded by the datasets' safe decoder, transformed by its model's evaluation
  transform, and scored by the stored model's own `score` function.
- **Accepting a file:** the decoded content decides, not the name or extension.
  - Formats: PNG, JPEG, BMP, TIFF or WebP; one still frame.
  - Limits: at most 10 MB and 25 million pixels, checked before decoding.
  - Error messages never repeat the file name or any pixel content.
  - Uploads stay in memory, are never written, logged or used for training, and no accuracy is
    computed for them.
- **Wording:** scores are labeled "model score" (no calibration protocol exists). Every
  prediction page explains that an image outside the learned classes still receives a score for
  every class.
- **Models:** a model loads only after `model_ref` checks:
  - the run is COMPLETED and passes its evidence check;
  - the run record and model spec are readable, with a supported schema;
  - the checkpoint matches its recorded hash, and the sealed hash in a sealed study;
  - the dataset's class map is the one the model was trained with.
- **Agreement tolerance:** the same image scored on two devices, or in batches of different size,
  may differ by floating-point rounding. The declared tolerance is 1e-4 (absolute, per class
  score). The same batch on the same device is bit-identical.
- **Device:** CPU is the default.

## D-054 (2026-11-02): The app's registry and caches
- **Numbers:** every displayed benchmark number comes from `reporting/run_registry.py`, which
  calls the report's `slot_values` (regenerated from stored prediction files and checked against
  run records or the sealed evaluation ledger). A missing value is shown as NOT RUN, never zero.
- **Registry cache:** the registry is cached under a fingerprint of the run ledger, the
  evaluation ledger and the sealed plan, so a new record refreshes it.
- **Model cache:** loaded models are cached by run, attempt, checkpoint hash and device.
- **Stale predictions:** a live prediction is tied to its run, attempt, checkpoint hash, device and
  image. Changing the dataset, method, K or seed therefore cannot show a stale prediction.
- **Display names:** Flowers display names come from a secondary source and are shown beside the
  class identity (`pink primrose (label_001)`); the identity is unchanged.
- **Colors:** chart colors are categorical slots 1–4 of the validated reference palette, in fixed
  method order, with marker shapes as a second channel. Per-seed points are drawn in one panel
  per method.

## D-055 (2026-11-02): Recorded paths keep their project spelling through symlinks
`portable_path` now tries the project-relative spelling before resolving symlinks. Files under
`runs/artifacts` (a link to local disk, D-039) are recorded as `runs/artifacts/...` instead of
`<outside project>/<name>`. A path that really leaves the project still keeps only its final
component. The change came after the sealed evaluation (D-051); it touches no protocol module.
