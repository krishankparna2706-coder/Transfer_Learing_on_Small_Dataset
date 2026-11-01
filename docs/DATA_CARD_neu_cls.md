# Data card: NEU-CLS steel-surface defects (secondary study)

| Field | Value |
|---|---|
| Dataset | NEU surface defect database, **NEU-CLS** image-only classification archive (K. Song and Y. Yan) [S4] |
| Task | 6-way **defect-type** classification of stored images (closed set; there is **no normal/good class**) |
| Images | 1,800 grayscale 200×200 BMP files (mode `L`), 300 per class |
| Classes (fixed order) | 0 rolled-in_scale (RS), 1 patches (Pa), 2 crazing (Cr), 3 pitted_surface (PS), 4 inclusion (In), 5 scratches (Sc) |
| Split | **Project-owned, not official**: 180 / 60 / 60 per class = 1,080 / 360 / 360, `split_seed` 2026 |
| Manifest | `data/manifests/neu_cls-v1.csv`, sha256 `2f4740b0b0167ed3ef95421b53da671724755ed918ed09f53912c51b4aa477ca` |
| Class map | `data/manifests/neu_cls.class_map.json`, sha256 `2788b023a33f8359db75c7154dd335c3e21adc9fd2eac9f33a8dac163241fceb` |
| Prepared | 2026-11-01T14:26:38Z, record `data/manifests/neu_cls.prepare.json` |

## Acquisition (2026-11-01)

The file came from the authors' **NEU-CLS** Google Drive link on the source page (file ID
`1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc`). The NEU-DET and NEU-CLS-64 variants are rejected
explicitly by the adapter.

- **Archive:** `NEU-CLS.rar` (RAR v4), 50,733,617 bytes, sha256
  `a85bcc6bfa1a40e76f28079e135694aaf5f6a4ee01203f7eaacff0ebafaa96f8`.
- **Two downloads, identical bytes.** The first request returned Google's virus-scan HTML page; the
  adapter followed only its Drive-hosted confirmation form.
- **Extraction:** bsdtar 3.5.3 (libarchive 3.7.4) into a new, empty folder, after a member-path
  safety check. Result: 1,800 BMP files plus `Thumbs.db`, a Windows thumbnail cache that is
  ignored and listed in the record.
- **Command:**

  ```bash
  uv run python -m fewshotlab data prepare --dataset neu_cls --allow-download
  ```

- **Manual fallback:** download "NEU-CLS" from the source page in a browser, then run
  `data prepare --dataset neu_cls --source <file or folder>`.
- **Structure check:** exactly `RS_1..RS_300`, `Pa_1..Pa_300`, `Cr_*`, `PS_*`, `In_*` and `Sc_*`.
  Every image is 200×200 with grayscale content.

## Split construction

For each class:
1. sort the image IDs (lexicographic);
2. treat each exact-duplicate cluster as one unit;
3. permute the units with NumPy PCG64 seeded from SHA-256(`neu_cls`, `split`, 2026, class_id);
4. assign the cumulative positions 0–179 to train, 180–239 to validation and 240–299 to test.

One exact-duplicate cluster exists: `Pa_101` and `Pa_105` are pixel-identical. The cluster rule
kept them together (both landed in test) and the 180/60/60 counts were unaffected.

## Audit (2026-11-01T14:26:46Z, `reports/data_audit/neu_cls/`)

- **Status:** PASS. No integrity or decode failures; every image is 200×200, mode `L`, BMP.
- **Exact duplicates:** 1 group, `Pa_101` = `Pa_105`, both in test.
- **Near-duplicate candidates:** 1 pair at pHash distance ≤ 4. It is the same pair, within one
  split. No cross-split or cross-class candidates.
- **Group metadata:** none (coil, batch, camera and line are unknown). The split is image-level.

## Terms and citation

No explicit license is stated on the source page (checked 2026-11-01). The authors ask users to
cite their works. FewShotLab never redistributes the images. Cite K. Song and Y. Yan, "A noise
robust method based on completed local binary patterns for hot-rolled steel strip surface
defects", Applied Surface Science 285 (2013) 858–864.

## Limitations

- **Closed-set defect-type classification only.** No normal class exists, so results say nothing
  about detecting defective versus normal material, localization or factory performance.
- **An image-level split cannot establish generalization** to new production lines, coils or
  cameras.
- **The patches test set holds one duplicated image**, counted twice in test metrics. This is
  disclosed with NEU results.
- **The split is a FewShotLab protocol decision**, not an official benchmark split. Comparisons
  with published NEU results are not like-for-like.
