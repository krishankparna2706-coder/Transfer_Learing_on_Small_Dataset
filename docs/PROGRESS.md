# Progress record

Each entry lists commands that were actually executed and what was observed.
- **NOT_RUN:** the step has not been executed.
- **BLOCKED:** an external prerequisite is missing.

## Current status

| Area | Status |
|---|---|
| Implementation | P0 complete: package skeleton, locked environment, `doctor`, full CLI interface (later-phase commands exit 3) |
| Data | Flowers102 NOT_PREPARED; NEU-CLS NOT_PREPARED. No dataset verified. |
| Benchmark | NOT_RUN (`runs/ledger.jsonl` is empty) |
| Demo | NOT_RUN |
| Remote push | BLOCKED: the authenticated account has `push: false` on the origin repository |

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
