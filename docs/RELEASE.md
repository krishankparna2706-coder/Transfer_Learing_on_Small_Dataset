# Release: FewShotLab, protocol fewshotlab-v1 (P8)

## Status

| Field | Status |
|---|---|
| **Code readiness** | Complete through P8. All 8 phases are implemented; tests pass (see [PROGRESS.md](PROGRESS.md)). |
| **Dataset availability** | Not redistributed. The data cards give the original sources; manifests are tracked and regenerate identically from the original files (checked in a clean clone, below). |
| **Benchmark execution** | Complete for the planned core matrix: 156 of 156 unique runs COMPLETED on validation, then evaluated once on held-out test under the seal; the robustness study is done. Optional extensions (DINOv2, calibration) are NOT_RUN. |
| **Demo readiness** | The app and the offline rehearsal work: 2,520 of 2,520 live predictions equal the sealed ones. The timed, spoken seven-minute rehearsal is still to be done by the presenter. |

## Release gates (document 02, p.21)

| Gate | Status | Evidence |
|---|---|---|
| **Data:** hashes, class maps, counts, split separation, duplicate and group review recorded | PASS | data cards; `reports/data_audit/`; [RELEASE_INVENTORY.md](RELEASE_INVENTORY.md) (manifests, class maps and 40 supports match the seal) |
| **Software:** sampler, freeze-policy, metric, checkpoint and test-lock tests pass | PASS | 290 tests (`pytest -q`), from P0 to P8 |
| **Science:** sealed protocol, all planned run statuses, complete predictions and computed metrics reconcile | PASS | the sealed plan, the evaluation ledger (156 test evaluations), reports regenerated from stored predictions |
| **Reporting:** no fabricated cells; correct distinct-run counts and uncertainty; limitations next to claims | PASS | registry-only numbers (D-054); duplicates counted once (D-048); [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) |
| **Demo:** cold-start rehearsal, offline fallback, fixed review threshold, clean and corruption separation | PASS (automated) / TO DO (spoken) | [REHEARSAL.md](REHEARSAL.md); `test_p7_replay.py` |
| **Handoff:** README reproduction steps, environment lock, model card, dataset provenance, evidence index | PASS | README "Reproduce the study"; `uv.lock`; [MODEL_CARD.md](MODEL_CARD.md); data cards; this page |

**Open items** (owner decisions, not defects):
- no code license has been chosen;
- the spoken rehearsal is still to do;
- a human review of the Flowers near-duplicate candidates is still to do (D-016).

## Evidence index

| Claim | Evidence |
|---|---|
| Transfer beats scratch at every K on both datasets (held-out test) | `reports/fewshotlab-v1-full/test/aggregate_test.md` (paired differences, 5 of 5 seeds) |
| Test was accessed once, after sealing | `reports/fewshotlab-v1-full/sealed_plan.json`; `evaluation_ledger.jsonl`; commit `abe8aa4`, pushed before the first test event |
| Training repeats exactly on MPS | [PROGRESS.md](PROGRESS.md) P5: 8 of 8 seed-11 runs identical to the MVP runs |
| Robustness and the review rule on corrupted test images | `reports/fewshotlab-v1-full/test/robustness.md` |
| Live replay equals the sealed predictions; timing | `demo_assets/fewshotlab-v1-full/rehearsal.md` |
| App and CLI predictions agree | `tests/test_p6_app.py`; `demo --check` |
| Every artifact hash | [RELEASE_INVENTORY.md](RELEASE_INVENTORY.md) |

## Clean-environment check

A fresh clone of the published repository was synced into a new environment and run there.

This ran on 2026-11-02, on commit `127dc96` cloned from GitHub, with `uv sync --locked` into an
empty environment and the data read from the original files.

| Step | Observed |
|---|---|
| `doctor` | Python 3.11.15, torch 2.15.0; MPS available (probe PASS), CPU fallback |
| Fixture tests | 277 passed, 8 skipped (real-data tests without data), **1 failed**. The P7 real-study test assumed run folders exist wherever the sealed plan does. **Fixed in this release:** it now also requires `runs/artifacts` |
| `data prepare` for both datasets, from the original archives | both split manifests **byte-identical** to the committed ones; both class maps have identical identity hashes (Flowers display names come only from `--cross-check` with the Kaggle copy) |
| `support build --config configs/protocol-v1.yaml` | 40 of 40 support manifests with identical content hashes |
| `data audit --dataset neu_cls` | report written |
| `train --config configs/runs/diagnostic/flowers102-scratch-k1-s11.yaml` | COMPLETED; the run folder holds the full evidence |
| `demo --check` (approved run folders supplied) | ready: models match their stored predictions; sealed replay verified |

- **Two documentation errors found and fixed:**
  - the Flowers `--source` folder must hold `102flowers.tgz` itself;
  - `support build` takes `configs/protocol-v1.yaml`, not the study config.
- **Not repeated in the clone:**
  - the 156-run matrix (about 5.8 h);
  - the sealed test evaluation, which runs once by design.

## Handoff

**Reproduce the main result.**
1. Follow the README section "Reproduce the study".
2. Copy the approved run folders, or retrain with `benchmark --config configs/full-study.yaml`
   (about 5.8 h of compute on an M4 Pro).
3. Run `report --study configs/full-study.yaml`. It regenerates every table from the stored
   predictions.

**Demonstrate.**

```bash
python -m fewshotlab demo --check
```

```bash
python -m fewshotlab demo
```

Then follow [JUDGE_SCRIPT.md](JUDGE_SCRIPT.md).

**Still NOT_RUN or open:**
- the optional extensions (DINOv2, calibration);
- a defect-free class and any factory evaluation;
- the spoken rehearsal;
- the license.

**Before claiming more:** the held-out benchmark is fully executed for this protocol. Any new
model, threshold or dataset needs a new protocol version and a new seal.
