# runs/

Append-only experiment records (P3 onward). Each run attempt gets its own directory with
`config.resolved.yaml`, `environment.json`, `provenance.json`, `train_history.csv`, `best.ckpt` or
`prototypes.npz`, `predictions_val.csv`, `val_metrics.json` and plots. `test_metrics.json` and
`predictions_test.csv` can only appear after the sealed P5 test gate.

Only this README and `ledger.jsonl` are tracked in git; checkpoints and predictions stay local.
The ledger starts empty: no run has been executed, and the benchmark status is NOT_RUN.
