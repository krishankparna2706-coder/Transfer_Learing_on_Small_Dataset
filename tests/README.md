# tests/

Offline CPU tests: `uv run pytest -q`.

- Tests use small generated fixtures in temporary directories, never real datasets, unless marked.
- Markers: `slow`, `network` (opt-in via `FEWSHOTLAB_NETWORK_TESTS=1`), `real_data`, `mps`.
- Fixture artifacts carry `is_fixture=true` and can never enter benchmark tables.
