# Change log: fewshotlab 0.1.0, protocol fewshotlab-v1

## Measured findings (from stored predictions; details in PROGRESS.md P4–P7)

- **Transfer vs scratch:** on held-out test, every transfer method beats scratch in 5 of 5 seeds
  at every K, on Flowers102 and on NEU-CLS. At K=10, Flowers macro-F1 is 0.853 for partial
  against 0.411 for scratch.
- **Partial vs frozen:** partial fine-tuning beats frozen features in every seed at Flowers K=5
  and NEU K=3 to K=10, in 4 of 5 at Flowers K=10, in 3 of 5 at Flowers K=3, and not at K=1.
  Prototypes trail frozen in at least 4 of 5 seeds in every cell.
- **Validation optimism:** Flowers test macro-F1 is 0.008–0.046 below validation, the expected
  cost of choosing checkpoints on validation.
- **Robustness:** the sealed NEU demo model loses 0.24 and 0.30 macro-F1 under strong blur and
  strong noise. At threshold 0.95, clean coverage is 94.4 % with 0.9 % error among accepted.
- **Repeatability:** training on MPS repeats exactly (8 of 8 seed-11 runs). The live replay
  equals the sealed predictions (2,520 of 2,520).

## Defects found and corrected

- **Run folders on iCloud:** they moved to local disk (`runs/artifacts`, D-039).
- **Path recording:** `portable_path` lost the project path through that link (D-055).
- **Real-data test:** it failed in a fresh clone without data and now skips with a reason (P5).
- **Aggregate tables:** the test tables were written before any sealed evaluation; they are now
  written only afterwards (P5).
- **App charts:** text was unreadable under the wrong theme, the legend was missing and a panel
  was clipped. The Flowers class identities lacked display names, and labels were truncated (P6).
- **CLI vs app comparison:** the comparison now uses a matching batch size; batch size changes
  CPU rounding by about 1e-10 (P6).
- **Found by the clean-clone check (P8):**
  - a P7 test assumed run folders exist wherever the sealed plan does;
  - the README named the wrong Flowers `--source` folder and the wrong config for
    `support build`.

## Optional enhancements (not required; NOT_RUN)

- The DINOv2 ViT-S/14 comparison and probability calibration.
- A CI workflow on GitHub; the full suite runs locally offline.
- Saved app screenshots: they are not kept, because some show dataset images.

## Open limitations (see KNOWN_LIMITATIONS.md)

- **No normal class:** NEU-CLS has no defect-free class. The corruptions are synthetic. There is
  no factory or camera evaluation.
- **Variability:** five seeds describe benchmark variability only; no population confidence
  intervals are given.
- **Local artifacts:** checkpoints, weights and dataset images are not in git. Their hashes are
  in the release inventory.
- **Owner decisions:** the code license, the spoken seven-minute rehearsal, and a human review of
  the near-duplicate candidates.
