"""Single image prediction: one verified stored model scores an uploaded or validation image."""

from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from fewshotlab.data.imageio import open_image
from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.engine import predict
from fewshotlab.errors import FewShotLabError
from fewshotlab.reporting import run_registry
from views import common

reg = common.registry()
study = reg["study"]
paths = common.paths()
device = common.device()

st.title("Single image prediction")
common.badge("live")

controls = st.columns(4)
dataset = controls[0].selectbox(
    "Dataset (domain)", study["datasets"], format_func=common.dataset_name, key="pred_dataset"
)
methods = [m for m in common.METHOD_ORDER if m in study["methods"]]
shots, seeds = sorted(study["shots"]), sorted(study["support_seeds"])
default = run_registry.default_cell(reg, dataset) or (dataset, methods[0], shots[-1], seeds[0])
method = controls[1].selectbox(
    "Method", methods, index=methods.index(default[1]), key=f"pred_method_{dataset}"
)
k = controls[2].selectbox("K", shots, index=shots.index(default[2]), key=f"pred_k_{dataset}")
seed = controls[3].selectbox(
    "Support seed", seeds, index=seeds.index(default[3]), key=f"pred_seed_{dataset}"
)
st.caption(common.budget_caption(reg, dataset))

try:
    ref = common.cell_ref(reg, dataset, method, k, seed)
except FewShotLabError as exc:
    common.stop_with(
        f"No usable model for {dataset} / {method} / K={k} / seed {seed}: {exc}",
        "Choose another cell; benchmark numbers for this cell show NOT RUN on the explorer.",
    )

val_row = run_registry.cell(reg, dataset, method, k, seed)
test_row = next(
    (
        r
        for r in reg["rows"]["test"]
        if (r["dataset"], r["method"], r["k"], r["seed"]) == (dataset, method, k, seed)
    ),
    None,
)
with st.container(border=True):
    fixture = "  **SOFTWARE FIXTURE**" if ref.is_fixture else ""
    st.markdown(f"**Model** `{ref.run_id}` / `{ref.attempt_id}`{fixture}")
    cols = st.columns(4)
    cols[0].markdown(f"{ref.num_classes} classes  \n{ref.method}, K={ref.k}, seed {ref.seed}")
    cols[1].markdown(f"Checkpoint `{ref.artifact_file}`  \nSHA-256 `{ref.artifact_sha256[:16]}...`")
    cols[2].markdown(
        f"Stored validation macro-F1 **{common.fmt(val_row['macro_f1'] if val_row else None)}**  \n"
        "VALIDATION (optimistic)"
    )
    cols[3].markdown(
        f"Stored test macro-F1 **{common.fmt(test_row['macro_f1'] if test_row else None)}**  \n"
        "HELD-OUT TEST (sealed)"
    )


@st.cache_data(show_spinner=False, max_entries=4)
def validation_records(dataset: str) -> list[dict]:
    records = read_manifest(manifest_path(common.paths(), dataset), splits=("val",))
    return [
        {"image_id": r.image_id, "relative_path": r.relative_path, "class_id": r.class_id}
        for r in records
    ]


facts = reg["datasets"][dataset]


def label(class_id: int) -> str:
    return common.class_label(facts, class_id)


source = st.radio(
    "Image",
    ["Upload your own image", "Pick a validation image"],
    horizontal=True,
    key="pred_source",
)
stored, truth = None, None
if source.startswith("Upload"):
    upload = st.file_uploader(
        "PNG, JPEG, BMP, TIFF or WebP; up to 10 MB and 25 million pixels",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"],
        key=f"pred_upload_{dataset}",
    )
    if upload is None:
        st.info(
            "Upload an image to see this model's scores, or pick a validation image.",
            icon=":material/upload:",
        )
        st.stop()
    data = upload.getvalue()
    try:
        image = predict.decode_upload(data)
    except predict.UploadRejected as exc:
        common.stop_with(f"This file was not accepted: {exc}")
    image_key = hashlib.sha256(data).hexdigest()
    caption = "your image (in memory for this session only; not saved)"
else:
    records = validation_records(dataset)
    left, right = st.columns(2)
    class_id = left.selectbox(
        "True class",
        sorted({r["class_id"] for r in records}),
        format_func=lambda c: f"{c}: {label(c)}",
        key=f"pred_vclass_{dataset}",
    )
    choices = [r for r in records if r["class_id"] == class_id]
    number = right.number_input(
        f"Image (1 to {len(choices)})", 1, len(choices), 1, key=f"pred_vimage_{dataset}_{class_id}"
    )
    record = choices[int(number) - 1]
    image = open_image(paths.data_root / record["relative_path"])
    image_key = record["image_id"]
    stored = predict.stored_validation_prediction(paths, ref, record["image_id"])
    truth = class_id
    caption = f"validation image {record['image_id']} (true class {class_id}: {label(class_id)})"

key = (ref.run_id, ref.attempt_id, ref.artifact_sha256, device, image_key)
if st.session_state.get("pred_key") != key:
    try:
        result = predict.predict_images(common.model(ref, device), [image])[0]
    except FewShotLabError as exc:
        common.stop_with(f"The model could not run: {exc}")
    except RuntimeError as exc:
        common.stop_with(
            f"The {device} device failed: {exc}", "Choose cpu in the sidebar and try again."
        )
    st.session_state["pred_key"], st.session_state["pred_result"] = key, result
result = st.session_state["pred_result"]

left, right = st.columns([1, 1.3])
with left:
    st.image(image, caption=caption, width="stretch")
with right:
    st.markdown(f"### {label(result.pred_class_id)}")
    st.markdown(
        f"Highest **{predict.SCORE_LABEL}** {result.max_score:.4f} (class {result.pred_class_id}). "
        f"Scores are uncalibrated: {ref.score_kind}."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"rank": i + 1, "class": label(cid), "class ID": cid, predict.SCORE_LABEL: score}
                for i, (cid, _name, score) in enumerate(result.top)
            ]
        ),
        column_config={
            predict.SCORE_LABEL: st.column_config.ProgressColumn(
                predict.SCORE_LABEL, format="%.4f", min_value=0.0, max_value=1.0
            )
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"Preprocessing {result.preprocess_ms:.1f} ms, inference {result.inference_ms:.1f} ms on "
        f"{result.device}. The first prediction on a device includes one-time warm-up."
    )
    if stored is not None:
        same = predict.compare_scores(result, stored)
        if same["same_class"] and same["within_tolerance"]:
            st.success(
                "Matches the prediction this run stored in its own validation evaluation (the CLI "
                "path): same class, largest score difference "
                f"{same['max_abs_score_difference']:.1e} "
                f"(declared tolerance {same['tolerance']:g}).",
                icon=":material/check_circle:",
            )
        else:
            st.error(
                "Differs from the stored validation prediction: "
                f"same class {same['same_class']}, largest score difference "
                f"{same['max_abs_score_difference']:.1e} (tolerance {same['tolerance']:g}).",
                icon=":material/error:",
            )
        verdict = "correct" if result.pred_class_id == truth else "wrong"
        st.markdown(f"True class: **{label(truth)}**; this prediction is **{verdict}**.")

st.warning(
    f"This model knows only the {ref.num_classes} {common.dataset_name(dataset)} classes. An "
    "image of anything else still receives a score for every one of them, and a high score is "
    "not evidence that the image belongs to a known class. A low-score review rule is not "
    "reliable out-of-distribution detection. No accuracy is computed for an upload, because its "
    "true label is unknown.",
    icon=":material/info:",
)
st.markdown("**The same prediction from the command line**")
st.code(
    f"python -m fewshotlab predict --run {ref.run_id} --attempt {ref.attempt_id} "
    f"--image YOUR_IMAGE --device {device}",
    language="bash",
    wrap_lines=True,
)
