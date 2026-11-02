"""Industrial replay: the sealed NEU stream played back with the sealed model and threshold."""

from __future__ import annotations

import time

import streamlit as st

from fewshotlab.errors import FewShotLabError
from views import common

PLAY_SECONDS = 1.0

reg = common.registry()
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
        f"Model choice: {identity['selection_rule']}."
    )

top = st.columns([2, 1])
stream_id = top[0].selectbox(
    "Stream", list(streams), format_func=lambda s: streams[s].label, key="replay_stream"
)
reveal = top[1].toggle("Reveal true labels (presenter)", key="replay_reveal")
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
        st.markdown(f"Predicted defect type\n### {names[item.pred_class_id]}")
        st.markdown(f"**Model score** {item.max_score:.4f}")
        if item.review:
            st.warning(
                f"Model score below the sealed threshold {view.threshold:g}.",
                title="HUMAN REVIEW",
                icon=":material/person_search:",
            )
        else:
            st.success(
                f"Model score at or above the sealed threshold {view.threshold:g}.",
                title=f"ACCEPTED as {names[item.pred_class_id]}",
                icon=":material/check_circle:",
            )
        if reveal:
            verdict = "correct" if item.correct else "wrong"
            st.markdown(
                f"True label: **{names[item.true_class_id]}**; the prediction is **{verdict}**."
            )
        else:
            st.caption(
                "True label hidden. Revealing it never changes the model, score or threshold."
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
        f"Stored sealed result for the whole stream (n = {stored['n_images']}): macro-F1 "
        f"{stored['macro_f1']:.3f}, coverage {stored['coverage']:.1%}, error among accepted "
        + ("n/a" if accepted_error is None else f"{accepted_error:.1%}")
        + ". Read coverage and the error among accepted predictions together: reviewing nearly "
        "everything makes the accepted error small only by accepting little."
    )


player()
st.caption(
    "Prediction source: the stored sealed prediction from the P5 robustness evaluation, not live "
    "inference; P7 adds live inference with measured latency. Playback runs at one image per "
    "second, a presentation pace that says nothing about throughput. NEU-CLS has no defect-free "
    "class, so this replay cannot show false accepts on normal material."
)
