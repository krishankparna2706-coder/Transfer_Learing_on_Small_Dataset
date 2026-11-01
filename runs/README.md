# runs/

Append-only experiment records.

- `ledger.jsonl` (tracked in git): one event per status change or verification for every run
  attempt.
- `artifacts/<run_id>/<attempt_id>/` (never tracked): each attempt's evidence.
  - `config.resolved.yaml`, `environment.json`, `provenance.json`, `model_spec.json`
  - `train_history.csv`, `best.ckpt` or `prototypes.npz`
  - `predictions_val.csv`, `val_metrics.json`, plots
  - `test_metrics.json` and `predictions_test.csv` can only appear after the sealed P5 test gate.

On this machine `artifacts/` is a symlink to `~/.local/share/fewshotlab/runs`, so the gigabytes of
checkpoints stay out of iCloud Drive (decision D-039). On any other machine it is an ordinary
folder, created on first use.
