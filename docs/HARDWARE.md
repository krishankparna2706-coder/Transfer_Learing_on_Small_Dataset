# Hardware and compute planning

## Measured on 2026-11-01 (development machine)

| Item | Observed | Source |
|---|---|---|
| OS | macOS 26.6.2 (Darwin 25.6.0), arm64 | `sw_vers`, `uname -a` |
| CPU | Apple M4 Pro, 12 cores (8 performance + 4 efficiency) | `sysctl` |
| GPU | Apple M4 Pro, 16 cores, Metal 4 | `system_profiler SPDisplaysDataType` |
| Memory | 24 GiB unified (25,769,803,776 bytes) | `sysctl hw.memsize` |
| Free disk | 119 GiB (at planning time) | `df -h` |
| PyTorch devices | MPS built, available, live probe PASS; CUDA not built and not available; torch uses 8 threads by default | `python -m fewshotlab doctor` |

No NVIDIA GPU is present, so the CUDA code paths are untested on this machine.

## Compute estimate method

**Status: NOT_RUN.** No job has been timed yet, and no training duration is promised before one
is measured.

1. **Time a representative job.** Run one timed diagnostic job per method and dataset on the
   actual device. This is the closing step of P3, or a P4 pilot. Record:
   - seconds per training epoch;
   - seconds per validation pass;
   - peak memory, where the backend reports it.
2. **Learned methods.** Estimate

   ```
   hours ≈ Σ_jobs epochs_j × (fit_s_per_epoch + val_s_per_pass) / 3600
   ```

   Early stopping makes epochs_j unknown in advance, so report two bounds:
   - lower: 16 epochs (one epoch plus patience 15);
   - upper: 100 epochs.
3. **Prototypes.** Count embedding extraction for support and validation images only; there are
   no epochs.
4. **Overheads.** Add data preparation, test evaluation (6,149 Flowers images per run),
   reporting, and a stated contingency for failures.
5. **Present it as an estimate.** Give the assumptions, never an observed runtime.
