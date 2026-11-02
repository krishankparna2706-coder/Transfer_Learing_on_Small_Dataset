# Rehearsal record and backup plan (P7)

The automated rehearsal is `python -m fewshotlab demo --rehearse`.
- **What it does:** it scores the whole sealed replay live, offline, compares every result with
  the stored sealed prediction, and times it with the protocol of document 02, p.16.
- **Where it writes:**
  - `demo_assets/fewshotlab-v1-full/rehearsal.md` and `rehearsal.json`, both committed;
  - `rehearsal_events.csv`, one row per image, kept local;
  - a `replay_rehearsal` event with the files' hashes in `runs/ledger.jsonl`.
- **Timed rehearsal:** the timed, spoken seven-minute rehearsal ([JUDGE_SCRIPT.md](JUDGE_SCRIPT.md))
  is for the presenter, on the presentation machine.

## Checklist

1. **Mains power** is connected. A battery sleep paused the matrix once (D-050).
2. `python -m fewshotlab demo --check`: every line PASS (offline, CPU).
3. `python -m fewshotlab demo --rehearse`: all 7 streams `live = stored 360/360`.
4. **Start the app** with `python -m fewshotlab demo` at least a minute before presenting. Then
   open each page once, so the 17-second registry load is done.
5. **Walk the script** once, against a clock.
6. **Trace a number** on Reproducibility within one minute. This is a presentation target to
   verify, not a measured claim.
7. **Device:** keep the sidebar device on **cpu**, the rehearsed path. MPS also works, but its
   first request includes warm-up.

## Observed timing (2026-11-02, this Mac, CPU, 8 threads, float32)

From `demo --rehearse`. Network connections were refused throughout, and none was attempted.

| Measure | Observed |
|---|---|
| Cold start: identify and load the sealed checkpoint | 120 ms |
| Forward latency (batch 1, prepared 3×224×224 tensor, 20 warm-up, 200 timed) | median 6.12 ms, 95th percentile 6.29 ms |
| Per image, end to end: decode, corrupt, transform, score, format (2,520 images; no browser) | median 7.02 ms, 95th percentile 11.56 ms |
| Per image, preprocessing / scoring call | median 0.93 ms / 6.13 ms |
| Whole rehearsal: verify, load, time, then 2,520 live images | 22.5 s |
| App first load (registry regenerated from stored predictions) | about 17 s |

- **Playback** at one image per second is a presentation pace, not a speed figure.
- **Latency** on the replay page covers one request and excludes browser rendering.

## Live versus stored (the replay identity check)

All 2,520 live predictions match the stored sealed predictions: 360 of 360 in each of the 7
streams. The coverage of each stream equals the sealed result:

| Stream | Coverage |
|---|---:|
| Clean | 94.4 % |
| Brightness ×0.6 | 94.4 % |
| Brightness ×1.4 | 80.8 % |
| Blur σ 1 px | 80.3 % |
| Blur σ 2 px | 65.6 % |
| Noise SD 0.03 | 90.0 % |
| Noise SD 0.08 | 64.2 % |

## Failure points

**Observed model errors** (the fixed error gallery: every clean test image predicted wrong,
sorted by ID):
- 9 of 360 images.
- 3 were **accepted** with high scores: `In_228` (0.989), `In_252` (0.973) and `In_8` (0.994),
  all inclusion predicted as scratches.
- 6 went to review.
- Under noise SD 0.08, `PS_7` (pitted surface) was predicted crazing at 0.766 and sent to review.

**Operational risks:**
- **Battery or lid sleep** pauses everything. Present on mains power with the lid open.
- **The first app load takes about 17 s.** Pre-open the app.
- **A busy port 8501:** use `--port 8502`.
- **Dark theme:** switching Streamlit to dark at runtime lowers chart contrast. Keep the
  configured light theme.

## Backup launch steps

1. **Live inference fails.** The replay page switches by itself to "FALLBACK: stored sealed
   predictions". These are the predictions recorded by the sealed robustness evaluation, with
   their timestamp (2026-11-02T07:17:45Z) and model ID. No other model is substituted. They can
   also be chosen by hand ("Stored sealed predictions (recording)").
2. **The accelerator fails:** stay on CPU, the rehearsed path.
3. **The app does not start:**
   - run `python -m fewshotlab demo --check` and follow its FAIL lines;
   - meanwhile, present the static reports: `reports/fewshotlab-v1-full/test/aggregate_test.md`,
     `test/robustness.md` and this folder's `rehearsal.md`.
4. **A checkpoint is missing or changed.** The app refuses to load it and shows the measured
   static report; it never swaps models.
5. **No measured run is present** (a fresh clone): the app shows NOT RUN, or the SOFTWARE
   FIXTURE label on fixture data.

## Expected questions

- **Why transfer learning?** A pretrained representation is reused. The paired comparison with
  scratch, at matched seeds, measures whether it helps under the stated label budget.
- **Is one-shot really one label?**
  - It is one training image per class.
  - It also uses the separately reported validation set to pick the checkpoint (1,020 labels for
    Flowers) and external ImageNet pretraining.
- **How were validation labels used?** Only to pick each run's checkpoint and to set the review
  threshold. Both happened before the test seal. Validation numbers are labeled optimistic for
  that reason.
- **Any leakage?**
  - The splits share no image.
  - Support sets come only from the training split, and corrupted derivatives never enter
    training (both are tested).
  - The test split was read once, under a plan sealed and pushed before access (`abe8aa4`).
- **Is the prototype model ProtoNet?** No. It is a frozen pretrained nearest-prototype baseline,
  without episodic metric-learning training.
- **Why not choose the best test model?** Test data measures the locked plan. Choosing by test
  would bias the claim, so selection used validation only.
- **Why not claim 99 %?** Accuracy must come from retained predictions. The success criterion is
  a valid, reproducible experiment, even if the results are modest.
- **Does this work in a factory?** Not shown. We tested a public image set and a synthetic
  replay. New cameras, defect-free products, batch effects and operating requirements need their
  own evaluation. NEU-CLS has no normal class, so false accepts on normal parts are unmeasured.
