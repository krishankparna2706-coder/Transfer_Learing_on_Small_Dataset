# Data card: Oxford Flowers102 (main study)

| Field | Value |
|---|---|
| Dataset | Oxford 102 Category Flower Dataset (Nilsback and Zisserman, 2008) [S1] |
| Task | 102-way flower-identity classification (fixed class set) |
| Images | 8,189 RGB JPEGs, 856 distinct sizes (shorter side 500 px in the most common sizes) |
| Official split | `setid.mat`: train (`trnid`) 1,020, validation (`valid`) 1,020, test (`tstid`) 6,149 |
| Per class | train 10, validation 10, test 20 to 238 |
| Manifest | `data/manifests/flowers102-v1.csv`, sha256 `886cf59e92473be726096c16975e82804df0ebe18045274da891ad3747651283` |
| Class map | `data/manifests/flowers102.class_map.json`, sha256 `89aa72740be1596d831d08931cf1d684991eb8fc2f9e90f639042038e56b2cff` |
| Prepared | 2026-11-01T14:28:40Z, record `data/manifests/flowers102.prepare.json` |

## Acquisition (2026-11-01)

The official host (`www.robots.ox.ac.uk`, `thor.robots.ox.ac.uk`) did not accept TCP connections
from this network. The three official files were therefore fetched from **Internet Archive captures
of the official Oxford URLs**. They were kept only because each file's MD5 equals the checksum
recorded in torchvision 0.29.2's `Flowers102` loader:

| File | Bytes | MD5 (= torchvision) | SHA-256 |
|---|---:|---|---|
| `102flowers.tgz` | 344,862,509 | `52808999861908f626f3c1f4e79d11fa` | `2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246` |
| `imagelabels.mat` | 502 | `e0620be6f572b9609742df49c70aed4d` | `4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d` |
| `setid.mat` | 14,989 | `a5357ecc9cb78c4bef273ce3793fc85c` | `46b8678f91fd95d3c8f4feab80d271a6c834a1dd896fe29fd3e6ad9ce5c8dccd` |

The pinned capture URLs are recorded in the prepare record (`acquisition.snapshots`). The tgz was
extracted with Python's `tarfile` `data` filter. Command:

```bash
uv run python -m fewshotlab data prepare --dataset flowers102 --allow-download --mirror internet-archive --cross-check flower_data
```

If the official host is reachable, `--allow-download` alone uses torchvision's own download from
the official URL. Either route must reproduce the MD5s above.

## Labels and identity

- `imagelabels.mat` labels 1..102 are converted to `class_id` 0..101 exactly once.
- The manifest was cross-checked image by image against torchvision's `Flowers102` loader (already
  zero-based) for all three splits.
- Class names `label_001`..`label_102` keep the official 1-based label number and are the identity
  used by every model.
- Human-readable display names (for example `label_001` → "pink primrose") come from `cat_to_name.json`
  in the user-supplied Kaggle copy. They are a secondary source, applied only because every folder
  label in that copy agreed with the official labels. They are display metadata only.

## The user-supplied Kaggle copy (`flower_data/`)

The user downloaded a copy from Kaggle. It is the Udacity re-packaging:
- `train/` holds 6,552 images and `valid/` holds 818, each in class folders 1..102.
- `test/` holds 819 unlabeled images in a flat folder.
- `cat_to_name.json` maps folder labels to flower names.

Its split is **not** the official split, so it was used only for verification
(`--cross-check flower_data`). The copy is gitignored and was not modified.

| Check | Result |
|---|---|
| Images byte-identical to the verified official files | 8,189 / 8,189 |
| Folder labels agreeing with the official labels | 7,370 / 7,370 (819 test images unlabeled) |
| Official images missing from the copy, or foreign files | 0, 0 |

## Audit (2026-11-01T14:30:58Z, `reports/data_audit/flowers102/`)

- **Status:** PASS. No integrity or decode failures; all 8,189 files are JPEG in RGB mode.
- **Exact duplicates:** 4 byte-identical groups. **2 span the official splits:**
  - `image_07307` (train) = `image_07309` (test), both `label_094`;
  - `image_08067` (test) = `image_08077` (val), both `label_031`.
- **Near-duplicate candidates:** 25 pairs, using a 64-bit DCT pHash with Hamming distance ≤ 4.
  - 8 of them cross splits (including the two exact pairs above).
  - Remaining cross-split candidates: `image_01835` (val) ~ `image_01858` (test); `image_01844` (test) ~ `image_01855` (train);
    `image_01929` (val) ~ `image_01959` (test); `image_05746` (train) ~ `image_05768` (val);
    `image_07015` (val) ~ `image_07040` (test); `image_07029` (val) ~ `image_08080` (test).
  - One pair crosses **classes**: `image_07303` (`label_093`) ~ `image_07304` (`label_094`), both in test,
    at distance 0. This is possible label noise or near-identical photos in the source.
  - **A human visual review of these candidates is pending (NOT_RUN).**
- **Group metadata:** none is available.

**Decision (D-016):** the official split is preserved unchanged, as protocol v1 requires. At most a
handful of the 6,149 test images have a twin or near-twin in train or validation, and this is
disclosed with every Flowers result. Any deduplicated analysis would be a separately named
sensitivity protocol.

## Terms and citation

The usage terms could not be re-checked on 2026-11-01 because the official page was unreachable.
FewShotLab never redistributes the images. Cite M.-E. Nilsback and A. Zisserman, "Automated flower
classification over a large number of classes", ICVGIP 2008.

## Limitations

- Flowers102 photos overlap in style with web imagery. Overlap with ImageNet pretraining cannot be
  ruled out.
- The validation set (1,020 labels) is part of the label budget at every K.
- Test classes are imbalanced (20 to 238 images), which is why macro-F1 is the primary endpoint.
