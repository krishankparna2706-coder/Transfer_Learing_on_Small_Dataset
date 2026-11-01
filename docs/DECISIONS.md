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
