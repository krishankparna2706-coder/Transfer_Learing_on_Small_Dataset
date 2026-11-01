# Data audit: flowers102

- Generated: 2026-11-01T14:30:58Z (audit v1)
- **Status: PASS**
- Manifest: `flowers102-v1.csv` sha256 `886cf59e92473be726096c16975e82804df0ebe18045274da891ad3747651283` (8189 rows)
- Class map sha256: `89aa72740be1596d831d08931cf1d684991eb8fc2f9e90f639042038e56b2cff`
- Source: Internet Archive captures of the official Oxford URLs, MD5-verified against torchvision's recorded checksums
- Counts: train 1020, val 1020, test 6149
- Integrity failures: 0; decode failures: 0
- Dimensions (top): {'667x500': 1486, '752x500': 834, '666x500': 350, '750x500': 256, '500x667': 208, '500x752': 128, '667x501': 85, '625x500': 75, '751x500': 71, '500x500': 70}; modes: {'RGB': 8189}; formats: {'JPEG': 8189}

## Exact duplicates
- Byte-identical groups: 4; pixel-identical groups: 4
- Groups spanning splits: 2; groups spanning classes: 0

## Near-duplicate candidates (for manual review)
- Method: 64-bit DCT perceptual hash (8-bit grayscale, Lanczos resize to 32x32, orthonormal 2-D DCT-II, top-left 8x8 block vs its median); pairs compared by Hamming distance
- Threshold: Hamming distance <= 4 of 64 bits
- Pairs within threshold: 25 (cross-split: 8, cross-class: 1)
- Pairs at or below each distance: {'0': 14, '2': 19, '4': 25, '6': 30, '8': 42, '10': 117, '12': 580}
- Rule: candidates are for manual review only; nothing is removed automatically

## Groups
- none available from the source: image-level splits cannot establish generalization to new acquisition groups (lines, coils, cameras)

## Findings
- 2 exact-duplicate group(s) span splits; record a dataset decision (official splits are preserved, never rewritten)
- 8 near-duplicate candidate pair(s) span splits (pHash distance <= 4); manual review required
- no acquisition-group metadata is available; splits are image-level
