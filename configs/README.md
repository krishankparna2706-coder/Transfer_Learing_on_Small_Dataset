# configs/

Versioned configuration. Every file is validated strictly: unknown keys are errors.

- `protocol-v1.yaml`: the study protocol (datasets, expected counts, shots, seeds, methods,
  v1 recipe, metric conventions, test-access rule). The test suite checks it against
  `src/fewshotlab/config.py`, so the file and the code cannot drift apart.
- Run configs (P3) and study configs such as `mvp.yaml` and `full-study.yaml` (P4–P5) will live here.
