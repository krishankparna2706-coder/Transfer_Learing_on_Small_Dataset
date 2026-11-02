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

## Measured timing (diagnostic runs, 2026-11-01, device `mps`)

These are five `diagnostic` runs (2 epochs each, K=10, seed 11, `num_workers` 0), recorded in
`runs/ledger.jsonl`. **Their validation numbers are not results.**

| Diagnostic run | Fit, epoch 1 / 2 | Validation pass | Wall time | MPS driver memory |
|---|---:|---:|---:|---:|
| Flowers scratch (1,020 support, 1,020 validation images) | 10.46 s / 5.80 s | 2.74 s | 29.6 s | 2.25 GB |
| Flowers frozen | 2.87 s / 2.72 s | 2.68 s | 18.5 s | 1.17 GB |
| Flowers partial | 3.47 s / 3.30 s | 2.70 s | 19.6 s | 1.17 GB |
| Flowers prototypes (embedding 1,020 support images) | 2.78 s | 2.70 s | 12.4 s | 1.16 GB |
| NEU partial (60 support, 360 validation images) | 0.47 s / 0.38 s | 0.76 s | 8.1 s | 1.34 GB |

- Reloading the selected checkpoint reproduced the validation predictions exactly in all five
  runs (maximum score difference 0.0).
- **Interpretation (unverified):** a frozen-encoder epoch costs about as much as a validation pass.
  This suggests image decoding and resizing on the main process (`num_workers` 0) dominate, not the
  network. Raising the DataLoader worker count (an execution detail, D-032) may shorten wall time.
  That is not yet measured.

## Compute estimate (ESTIMATE, not an observed runtime)

Generated reproducibly by `fewshotlab.reporting.estimate` from the diagnostic timings above. See
`reports/compute_estimate.md`, which includes every input run ID and every assumption.

| Scope | Lower bound (early stop at 16 epochs) | Upper bound (100 epochs) |
|---|---:|---:|
| MVP: 8 Flowers jobs (K ∈ {1, 10}, 4 methods, seed 11) | 0.14 h | 0.82 h |
| Full matrix: 156 unique runs, training | 1.56 h | 9.33 h |
| Full matrix: test evaluation of every unique run (P5) | about 0.36 h | |

**Assumptions:**
- costs scale linearly with image counts;
- NEU scratch and frozen costs are scaled from the Flowers method ratios.

**Not included:** data preparation, reports, failures and the corruption study.

**Caveat:** small NEU epochs carry fixed per-batch overhead (NEU partial fit costs 6.3 ms per image,
against 3.2 ms for Flowers).

**Check against a real run (P4):** the 8 MVP jobs took 1,854 s, about 31 minutes, as measured
by `benchmark`. That is inside the estimated 0.14–0.82 h. The learned runs stopped after 37–86
epochs.

**Check against the full matrix (P5):** `benchmark --config configs/full-study.yaml` ran 156
unique runs on MPS with 0 failures.

| Measure | Observed |
|---|---:|
| Wall clock, first run start to last run end (ledger) | 43,338 s (12.04 h) |
| System sleep inside that span (`pmset -g log`, see below) | 22,486 s (6.25 h) |
| Wall clock excluding the sleep | about 20,850 s (5.8 h) |
| Flowers102: 76 runs | 4.95 h |
| NEU-CLS: 80 runs | 0.84 h |

- **Epochs run:**
  - Flowers: scratch 35–100 (mean 68.2), frozen 38–99 (mean 62.4), partial 29–86 (mean 46.8).
  - NEU: scratch 27–100 (mean 68.3), frozen 28–76 (mean 44.4), partial 19–45 (mean 26.6).
- **Against the estimate:** 5.8 h of compute is inside the estimated 1.56–9.33 h.
- **The sleep:**
  - The Mac was running on battery and entered low-power sleep at 1 % charge. That was 05:24
    IST on 2026-11-02; it woke at 11:39 IST.
  - The process resumed. The run in progress (`neu_cls` frozen K=5 seed 55) completed normally.
  - That run's ledger wall time (22,593 s) includes the sleep. Its epoch timers sum to 113 s, and
    one validation pass took 29.5 s across the wake.
  - Long jobs need mains power. The idle-sleep blocker (`caffeinate -i`) cannot prevent battery or
    lid sleep.

**Test evaluation (P5):**
- **Sealing:** it re-scored validation for 156 runs in 276 s.
- **Evaluation:** `evaluate --split test --study configs/full-study.yaml` then scored the 156
  sealed runs and the 7-condition robustness study in 1,235 s (20.6 min) on MPS.
- **Against the estimate:** about 0.36 h (21.6 min).

## Compute estimate method

1. **Time a representative job.** Run one timed diagnostic job per method and dataset on the
   actual device. Record:
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
