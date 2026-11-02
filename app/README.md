# app/

The Streamlit demonstration (P6). Start it with `python -m fewshotlab demo`, or check offline that
it can start with `python -m fewshotlab demo --check`.

- `streamlit_app.py`: the entry point; page navigation and the shared sidebar.
- `views/common.py`: cached loaders, the result labels (VALIDATION, HELD-OUT TEST, LIVE
  PREDICTION, NEU HELD-OUT REPLAY, SOFTWARE FIXTURE) and the chart palette.
- `views/overview.py`, `data_budget.py`, `explorer.py`, `predict.py`, `replay.py`,
  `reproducibility.py`: one page each.

The pages only format what the `fewshotlab` library returns:
- **Numbers:** `reporting/run_registry.py` (regenerated from stored, hash-checked predictions).
- **Live predictions:** `engine/predict.py` (the same code as `python -m fewshotlab predict`).
- **The replay:** `simulation/sealed_replay.py` (verified against the sealed plan).

Nothing here trains a model, changes a dataset, seals a plan or evaluates the test split. An
automated check in `tests/test_p6_app.py` enforces this.
