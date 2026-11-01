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
