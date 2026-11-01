# configs/

Versioned configuration. Every file is validated strictly: unknown keys are errors.

- `protocol-v1.yaml`: the study protocol (datasets, expected counts, shots, seeds, methods,
  v1 recipe, metric conventions, test-access rule). The test suite checks it against
  `src/fewshotlab/config.py`.
- `runs/diagnostic/`: short timing diagnostics (ledger category `diagnostic`, 2 epochs). They
  measure cost on the actual machine; their numbers are not results.
- `mvp/`: the eight P4 MVP jobs (Flowers102, K ∈ {1, 10}, four methods, seed 11, full recipe,
  category `mvp`, validation only). Run each one with `python -m fewshotlab train --config <file>`.
- Study configs such as `full-study.yaml` (P5) will live here.
