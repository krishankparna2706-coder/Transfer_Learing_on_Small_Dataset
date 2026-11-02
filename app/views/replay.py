"""Industrial replay: the sealed NEU stream, scored live by the sealed model, with a recording
of the stored sealed predictions as the visible fallback."""

from __future__ import annotations

import csv
import io
import time

import streamlit as st

from fewshotlab.errors import FewShotLabError
from fewshotlab.io_utils import utc_now
from views import common

PLAY_SECONDS = 1.0
LIVE = "Live inference (sealed model)"
STORED = "Stored sealed predictions (recording)"
EVENT_COLUMNS = (
    "time_utc",
    "source",
    "stream",
    "position",
    "derivative_id",
    "parent_image_id",
    "run_id",
    "predicted_class",
    "model_score",
    "review",
    "true_class",
    "forward_ms",
    "end_to_end_ms",
    "device",
)

reg = common.registry()
device = common.device()
st.title("Industrial replay")
common.badge("replay")
try:
    view = common.replay(reg)
except FewShotLabError as exc:
    common.stop_with(
        f"The sealed replay is not available. {exc}",
        "Fallback: the stored held-out test results stay on the Experiment explorer (Held-out "
        "test). The replay exists only after `lock-test` and `evaluate --split test`, and only "
        "while the sealed files verify.",
    )

identity = view.identity
names = view.class_names
streams = {s.stream_id: s for s in view.streams}
with st.container(border=True):
    cols = st.columns(3)
    cols[0].markdown(
        f"**Data** NEU-CLS held-out test, {len(view.items('clean'))} images per stream  \n"
        f"**Seal** `{identity['audit_token']}`"
    )
    cols[1].markdown(
        f"**Model** `{identity['run_id']}` / `{identity['attempt_id']}`  \n"
        f"{identity['method']}, K={identity['k']}, seed {identity['seed']}; checkpoint SHA-256 "
        f"`{identity['checkpoint_sha256'][:16]}...`"
    )
    cols[2].markdown(
        f"**Review threshold {view.threshold:g}**, sealed on validation  \n"
        "An image goes to human review when its model score is strictly below it."
    )
    st.caption(
        f"Image order: {identity['order_rule']} (replay manifest SHA-256 "
        f"`{identity['replay_manifest_sha256'][:16]}...`). "
        f"Model choice: {identity['selection_rule']}. Training finished before the demonstration; "
        "this page only runs inference and reads stored evidence."
    )

top = st.columns([2, 2, 1])
stream_id = top[0].selectbox(
    "Stream", list(streams), format_func=lambda s: streams[s].label, key="replay_stream"
)
mode = top[1].radio(
    "Predictions",
    [LIVE, STORED],
    horizontal=True,
    key="replay_mode",
    help="Live: the sealed checkpoint scores each shown image now. Recording: the predictions "
    "stored by the sealed P5 robustness evaluation.",
)
reveal = top[2].toggle("Reveal true labels", key="replay_reveal", help="Presenter only.")

recorded = (
    f"recorded {identity['robustness_utc']} by `{identity['run_id']}` / `{identity['attempt_id']}`"
)
live = None
if mode == LIVE:
    try:
        live = common.live_replay(reg, device)
    except (FewShotLabError, RuntimeError, OSError) as exc:
        st.warning(
            f"Live inference is unavailable ({exc}). Showing the stored sealed predictions "
            f"instead, {recorded}. No other model is substituted.",
            title="FALLBACK: stored sealed predictions",
            icon=":material/history:",
        )
else:
    st.info(f"Stored sealed predictions from the P5 robustness evaluation, {recorded}.")

items = view.items(stream_id)
total = len(items)
st.session_state.setdefault("replay_index", 0)
st.session_state.setdefault("replay_playing", False)
st.session_state.replay_index = min(st.session_state.replay_index, total - 1)


def go(index: int) -> None:
    st.session_state.replay_index = max(0, min(total - 1, index))


def toggle_play() -> None:
    st.session_state.replay_playing = not st.session_state.replay_playing
    st.session_state.replay_last = time.monotonic()


buttons = st.columns(5)
buttons[0].button("First", on_click=go, args=(0,), icon=":material/first_page:")
buttons[1].button(
    "Previous",
    on_click=lambda: go(st.session_state.replay_index - 1),
    icon=":material/chevron_left:",
)
buttons[2].button(
    "Pause" if st.session_state.replay_playing else "Play",
    on_click=toggle_play,
    type="primary",
    help="Advances one image per second: a presentation pace, not a throughput figure.",
    icon=":material/pause:" if st.session_state.replay_playing else ":material/play_arrow:",
)
buttons[3].button(
    "Next", on_click=lambda: go(st.session_state.replay_index + 1), icon=":material/chevron_right:"
)
buttons[4].button("Last", on_click=go, args=(total - 1,), icon=":material/last_page:")


@st.fragment(run_every=PLAY_SECONDS if st.session_state.replay_playing else None)
def player() -> None:
    index = st.session_state.replay_index
    if st.session_state.replay_playing:
        now = time.monotonic()
        if now - st.session_state.get("replay_last", 0.0) >= PLAY_SECONDS * 0.9:
            if index < total - 1:
                index += 1
                st.session_state.replay_index = index
                st.session_state.replay_last = now
            else:
                st.session_state.replay_playing = False
                st.rerun()
    item = items[index]
    result = None
    if live is not None:
        cache = st.session_state.setdefault("replay_live", {})
        key = (stream_id, index, device)
        if key not in cache:
            cache[key] = live.infer(item)
        result = cache[key]
    pred = result.pred_class_id if result else item.pred_class_id
    score = result.max_score if result else item.max_score
    flagged = result.review if result else item.review

    study_path, fingerprint = str(common.study_path()), reg["fingerprint"]
    st.progress((index + 1) / total, text=f"Image {index + 1} of {total} in this stream")
    left, middle, right = st.columns([1, 1, 1.2])
    with left:
        st.image(
            common.replay_pixels(study_path, fingerprint, stream_id, index),
            caption=f"{item.derivative_id}: {streams[stream_id].label}",
            width="stretch",
        )
    with middle:
        if stream_id == "clean":
            st.caption(
                "Choose a stress stream to compare an image with its own corrupted version. "
                "Each corrupted image is compared only with its clean parent."
            )
        else:
            parent = view.items("clean")[index]
            st.image(
                common.replay_pixels(study_path, fingerprint, "clean", index),
                caption=(
                    f"its clean parent {parent.parent_image_id}: stored prediction "
                    f"{names[parent.pred_class_id]}, model score {parent.max_score:.3f}"
                ),
                width="stretch",
            )
    with right:
        st.markdown(f"Predicted defect type\n### {names[pred]}")
        st.markdown(f"**Model score** {score:.4f}")
        if flagged:
            st.warning(
                f"Model score below the sealed threshold {view.threshold:g}.",
                title="HUMAN REVIEW",
                icon=":material/person_search:",
            )
        else:
            st.success(
                f"Model score at or above the sealed threshold {view.threshold:g}.",
                title=f"ACCEPTED as {names[pred]}",
                icon=":material/check_circle:",
            )
        if reveal:
            verdict = "correct" if pred == item.true_class_id else "wrong"
            st.markdown(
                f"True label: **{names[item.true_class_id]}**; the prediction is **{verdict}**."
            )
        else:
            st.caption(
                "True label hidden. Revealing it never changes the model, score or threshold."
            )
        if result:
            st.caption(
                f"**Measured on {device}, this request:** preprocessing "
                f"{result.preprocess_ms:.1f} ms, model scoring {result.forward_ms:.1f} ms, end to "
                f"end {result.end_to_end_ms:.1f} ms (browser rendering excluded). Protocol "
                "latency (20 warm-up, 200 timed requests) is in the rehearsal report."
            )
            if result.matches_stored:
                st.caption(
                    "Live result matches the stored sealed prediction (score difference "
                    f"{result.score_difference:.1e})."
                )
            else:
                st.error(
                    "The live result differs from the stored sealed prediction: stored "
                    f"{names[item.pred_class_id]} {item.max_score:.4f}.",
                    icon=":material/error:",
                )
        else:
            st.caption("Source: the stored sealed prediction (P5 robustness evaluation).")

    events = st.session_state.setdefault("replay_events", {})
    source = "live" if result else "stored sealed prediction"
    events.setdefault(
        (source, stream_id, index, device),
        {
            "time_utc": utc_now(),
            "source": source,
            "stream": stream_id,
            "position": item.position,
            "derivative_id": item.derivative_id,
            "parent_image_id": item.parent_image_id,
            "run_id": identity["run_id"],
            "predicted_class": names[pred],
            "model_score": f"{score:.6f}",
            "review": flagged,
            "true_class": names[item.true_class_id],
            "forward_ms": f"{result.forward_ms:.3f}" if result else "",
            "end_to_end_ms": f"{result.end_to_end_ms:.3f}" if result else "",
            "device": device if result else "",
        },
    )

    tally = view.tally(stream_id, index + 1)
    shown = tally["shown"]
    counts = st.columns(4)
    counts[0].metric("Shown", f"{shown} of {tally['of']}", border=True)
    counts[1].metric("Sent to review", f"{tally['reviewed']} of {shown}", border=True)
    counts[2].metric(
        "Accepted (coverage)",
        f"{tally['accepted']} of {shown} ({tally['coverage']:.0%})",
        border=True,
    )
    counts[3].metric(
        "Errors among accepted",
        f"{tally['accepted_errors']} of {tally['accepted']}" if reveal else "hidden",
        help="Accepted images whose known true label differs from the prediction. Shown when "
        "the true labels are revealed.",
        border=True,
    )
    stored = view.summaries[stream_id]
    accepted_error = stored["accepted_error_rate"]
    st.caption(
        "Counts cover the first images of the stream and use the sealed predictions; each live "
        "result above is checked against them. Stored sealed result for the whole stream "
        f"(n = {stored['n_images']}): macro-F1 {stored['macro_f1']:.3f}, coverage "
        f"{stored['coverage']:.1%}, error among accepted "
        + ("n/a" if accepted_error is None else f"{accepted_error:.1%}")
        + ". Read coverage and the error among accepted predictions together: reviewing nearly "
        "everything makes the accepted error small only by accepting little."
    )


player()

events = list(st.session_state.get("replay_events", {}).values())
if events:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=EVENT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(events)
    st.download_button(
        f"Download this session's event log ({len(events)} rows, CSV)",
        data=buffer.getvalue().encode("utf-8"),
        file_name=f"replay_events_{identity['study_id']}.csv",
        mime="text/csv",
        icon=":material/download:",
    )

with st.expander("Fixed error gallery: every clean test image the sealed model gets wrong"):
    clean = view.items("clean")
    where = {item.parent_image_id: n for n, item in enumerate(clean)}
    errors = sorted((i for i in clean if not i.correct), key=lambda i: i.parent_image_id)
    st.caption(
        f"{len(errors)} of {len(clean)} clean test images. The rule was fixed before looking at "
        "the errors: every misclassified clean image, sorted by image ID (stored sealed "
        "predictions). ACCEPTED marks a confident error that the review rule would not catch."
    )
    columns = st.columns(5)
    for n, item in enumerate(errors):
        decision = "human review" if item.review else "ACCEPTED"
        columns[n % 5].image(
            common.replay_pixels(
                str(common.study_path()), reg["fingerprint"], "clean", where[item.parent_image_id]
            ),
            caption=(
                f"{item.parent_image_id}: true {names[item.true_class_id]}, predicted "
                f"{names[item.pred_class_id]} {item.max_score:.3f} ({decision})"
            ),
            width="stretch",
        )

st.caption(
    "Playback runs at one image per second, a presentation pace that says nothing about "
    "throughput; measured latency is shown per image and in the rehearsal report. NEU-CLS has no "
    "defect-free class, so this replay cannot show false accepts on normal material, and no "
    "image is ever called safe."
)
