# Replay rehearsal: fewshotlab-v1-full

> **Demonstration record, not a benchmark result.** The sealed demo checkpoint scored the sealed NEU replay live, offline, and every result was compared with the stored sealed prediction.

Sequence `rehearsal-20261102T164244Z`, 2026-11-02T16:42:44Z to 2026-11-02T16:43:04Z.

| Item | Observed |
|---|---|
| Seal | `seal-c6f58862d63b586a0df8dbcf140db0fd` (sealed 2026-11-02T06:56:38Z) |
| Model | `fewshotlab-v1__neu_cls__partial__k10__s11__cfg085b60aa` / `a001` (partial, K=10, seed 11); checkpoint SHA-256 `830dbab5018f9bb1...` |
| Review threshold | 0.95 (sealed on validation) |
| Device | cpu; 8 CPU threads; float32; synchronization: none needed (CPU runs synchronously); Darwin arm64 |
| Network | every outgoing connection was refused during the rehearsal; none was attempted |
| Cold start (identify and load the checkpoint) | 120 ms |
| Forward latency (batch 1, 20 warm-up, 200 timed, input [1, 3, 224, 224]) | median 6.12 ms, 95th percentile 6.29 ms |
| Per image, end to end (2,520 images; browser rendering excluded) | median 7.02 ms, 95th percentile 11.56 ms |
| Per image, preprocessing (decode, corrupt, transform) | median 0.93 ms, 95th percentile 5.08 ms |
| Per image, scoring call | median 6.13 ms, 95th percentile 6.52 ms |
| Playback | the app shows one image per second: a presentation pace, not a speed figure |

## Live versus stored sealed predictions

| Stream | Images | Live = stored | Largest score difference | Coverage (live) | Errors among accepted (live) | Coverage (stored) |
|---|---:|---:|---:|---:|---:|---:|
| clean (no corruption) | 360 | 360 | 3.3e-06 | 94.4% | 3 (0.9%) | 94.4% |
| stress: brightness x0.6 | 360 | 360 | 4.5e-06 | 94.4% | 1 (0.3%) | 94.4% |
| stress: brightness x1.4 | 360 | 360 | 4.3e-06 | 80.8% | 6 (2.1%) | 80.8% |
| stress: Gaussian blur sigma 1.0 px | 360 | 360 | 2.8e-06 | 80.3% | 5 (1.7%) | 80.3% |
| stress: Gaussian blur sigma 2.0 px | 360 | 360 | 4.1e-06 | 65.6% | 21 (8.9%) | 65.6% |
| stress: Gaussian noise SD 0.03 | 360 | 360 | 3.0e-06 | 90.0% | 5 (1.5%) | 90.0% |
| stress: Gaussian noise SD 0.08 | 360 | 360 | 3.1e-06 | 64.2% | 20 (8.7%) | 64.2% |

**Result:** all match. Coverage and the error among accepted predictions are read together; NEU-CLS has no defect-free class, so nothing here measures false accepts.

## Fixed error gallery: clean stream (9 errors)

Rule: every clean-stream test image the sealed model gets wrong, sorted by image ID (fixed before looking at the errors).

| Image | True class | Predicted | Model score | Decision |
|---|---|---|---:|---|
| `neu_cls/In_228` | inclusion | scratches | 0.9893 | ACCEPTED (wrong) |
| `neu_cls/In_237` | inclusion | pitted_surface | 0.9249 | human review |
| `neu_cls/In_246` | inclusion | scratches | 0.6609 | human review |
| `neu_cls/In_248` | inclusion | scratches | 0.9202 | human review |
| `neu_cls/In_252` | inclusion | scratches | 0.9727 | ACCEPTED (wrong) |
| `neu_cls/In_8` | inclusion | scratches | 0.9943 | ACCEPTED (wrong) |
| `neu_cls/PS_227` | pitted_surface | inclusion | 0.7334 | human review |
| `neu_cls/PS_4` | pitted_surface | scratches | 0.8850 | human review |
| `neu_cls/PS_71` | pitted_surface | scratches | 0.8820 | human review |
