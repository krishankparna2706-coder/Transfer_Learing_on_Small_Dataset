# Data audit: neu_cls

- Generated: 2026-11-01T14:26:46Z (audit v1)
- **Status: PASS**
- Manifest: `neu_cls-v1.csv` sha256 `2f4740b0b0167ed3ef95421b53da671724755ed918ed09f53912c51b4aa477ca` (1800 rows)
- Class map sha256: `2788b023a33f8359db75c7154dd335c3e21adc9fd2eac9f33a8dac163241fceb`
- Source: downloaded from the authors' Google Drive link
- Counts: train 1080, val 360, test 360
- Integrity failures: 0; decode failures: 0
- Dimensions (top): {'200x200': 1800}; modes: {'L': 1800}; formats: {'BMP': 1800}

## Exact duplicates
- Byte-identical groups: 1; pixel-identical groups: 1
- Groups spanning splits: 0; groups spanning classes: 0

## Near-duplicate candidates (for manual review)
- Method: 64-bit DCT perceptual hash (8-bit grayscale, Lanczos resize to 32x32, orthonormal 2-D DCT-II, top-left 8x8 block vs its median); pairs compared by Hamming distance
- Threshold: Hamming distance <= 4 of 64 bits
- Pairs within threshold: 1 (cross-split: 0, cross-class: 0)
- Pairs at or below each distance: {'0': 1, '2': 1, '4': 1, '6': 1, '8': 4, '10': 7, '12': 22}
- Rule: candidates are for manual review only; nothing is removed automatically

## Groups
- none available from the source: image-level splits cannot establish generalization to new acquisition groups (lines, coils, cameras)

## Findings
- no acquisition-group metadata is available; splits are image-level
