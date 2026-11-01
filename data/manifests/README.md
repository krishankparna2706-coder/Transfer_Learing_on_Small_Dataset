# data/manifests/

Immutable, versioned metadata. Tracked in git; contains no images.

- `<dataset>-v1.csv`: one row per image with `image_id, relative_path, class_id, class_name,
  split, sha256, group_id` (P1).
- `<dataset>.class_map.json`: integer class IDs and names (P1).
- `<dataset>.prepare.json`: the acquisition record: source, access time, archive hashes, status
  (`PREPARED`, `BLOCKED`, ...) (P1).
- `supports/`: nested K-shot support manifests per dataset, K and seed (P1).

Paths are relative to the data root (`data/raw/` by default, overridable with
`FEWSHOTLAB_DATA_ROOT`). Raw images stay in `data/raw/` and are never committed.
