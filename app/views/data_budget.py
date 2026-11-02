"""Data and label budget: split facts and provenance, the label budget, and real support sets."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fewshotlab.data.manifest import read_manifest
from fewshotlab.data.registry import manifest_path
from fewshotlab.data.support import load_support
from views import common

reg = common.registry()
study = reg["study"]
paths = common.paths()

st.title("Data and label budget")
dataset = st.radio(
    "Dataset",
    study["datasets"],
    format_func=common.dataset_name,
    horizontal=True,
    key="data_dataset",
)
facts = reg["datasets"][dataset]
if facts["status"] != "PREPARED":
    common.stop_with(
        f"{dataset} is NOT PREPARED: {facts.get('problem')}",
        f"Run `python -m fewshotlab data prepare --dataset {dataset}` (see the data card).",
    )

counts, ranges = facts["counts"], facts["per_class_range"]
columns = st.columns(4)
columns[0].metric("Classes", facts["classes"], border=True)
columns[1].metric(
    "Training pool",
    f"{counts['train']:,}",
    help=f"{ranges['train'][0]}-{ranges['train'][1]} per class. Support sets are drawn from here.",
    border=True,
)
columns[2].metric(
    "Validation",
    f"{counts['val']:,}",
    help=f"{ranges['val'][0]}-{ranges['val'][1]} per class. Used to choose each run's checkpoint.",
    border=True,
)
columns[3].metric(
    "Held-out test",
    f"{counts['test']:,}",
    help=f"{ranges['test'][0]}-{ranges['test'][1]} per class. Scored once, under the sealed plan.",
    border=True,
)
st.caption(
    f"Images per class: training pool {ranges['train'][0]}-{ranges['train'][1]}, validation "
    f"{ranges['val'][0]}-{ranges['val'][1]}, test {ranges['test'][0]}-{ranges['test'][1]}. "
    "The three splits share no image."
)

st.subheader("Split provenance")
source = facts.get("source") or {}
split = facts.get("split") or {}
if source.get("official_page"):
    files = ", ".join(f["filename"] for f in (source.get("expected_files") or {}).values())
    source_text = (
        f"official files ({files}) from {source['official_page']}, each checked against its "
        "published MD5"
    )
    rule = "the official split shipped with the dataset (setid.mat: train / validation / test)"
else:
    where = source.get("page") or source.get("url")
    source_text = (source.get("mode") or "see the data card") + (f" ({where})" if where else "")
    rule = split.get("procedure") or "see the data card"
st.markdown(
    f"- **Source:** {source_text}\n"
    f"- **Split rule:** {rule}\n"
    f"- **Split manifest:** `{facts['manifest']['file']}`, "
    f"SHA-256 `{facts['manifest']['sha256']}`\n"
    f"- **Class map:** {facts['classes']} classes, SHA-256 `{facts['class_map_sha256']}`\n"
    f"- **Data audit:** {facts['audit']['status']}"
    + (f" ({facts['audit']['generated_utc']})" if facts["audit"].get("generated_utc") else "")
)
with st.expander(f"All {facts['classes']} classes, in class-ID order"):
    st.write(", ".join(f"{i}: {common.class_label(facts, i)}" for i in range(facts["classes"])))
    if facts["display_names"] != facts["class_names"]:
        st.caption(
            "The identity of a class is its ID and its name in parentheses; the readable name "
            "comes from a secondary source and is for display only."
        )

st.subheader("Label budget")
budget = reg["budgets"][dataset]
support = {int(k): n for k, n in budget["support_labels"].items()}
table = [
    {
        "K": k,
        "support labels": support[k],
        "validation labels": budget["validation_labels"],
        "labels used before test": support[k] + budget["validation_labels"],
        "test images (held out)": counts["test"],
    }
    for k in sorted(support)
]
st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
st.caption(
    "K = labeled images per class. Every K-shot result uses two label budgets: the support "
    "labels that train the model and the validation labels that pick its checkpoint. Both are "
    "shown wherever results appear."
)

st.subheader("A real support set")
left, right = st.columns(2)
k = left.radio("K", sorted(support), horizontal=True, key="data_k")
seed = right.selectbox("Support seed", study["support_seeds"], key="data_seed")
try:
    chosen = load_support(paths, dataset, int(k), int(seed))
except (OSError, ValueError) as exc:
    common.stop_with(
        f"No support manifest for {dataset} K={k} seed {seed}: {exc}",
        "Run `python -m fewshotlab support build --config configs/full-study.yaml`.",
    )


@st.cache_data(show_spinner=False, max_entries=4)
def train_paths(dataset: str) -> dict[str, str]:
    records = read_manifest(manifest_path(common.paths(), dataset), splits=("train",))
    return {r.image_id: r.relative_path for r in records}


locations = train_paths(dataset)
class_ids = sorted(int(c) for c in chosen["selected"])
shown = st.multiselect(
    "Classes to show",
    class_ids,
    default=class_ids[: min(6, len(class_ids))],
    format_func=lambda c: f"{c}: {common.class_label(facts, c)}",
    max_selections=12,
    key=f"data_classes_{dataset}",
)
for class_id in shown:
    ids = chosen["selected"][str(class_id)]
    st.markdown(f"**{class_id}: {common.class_label(facts, class_id)}**: {len(ids)} image(s)")
    st.image(
        [common.thumbnail(str(paths.data_root / locations[i])) for i in ids],
        caption=[i.split("/", 1)[-1] for i in ids],
        width=120,
    )
st.caption(
    f"Training images only (train split), {len(class_ids) * int(k):,} in this support set. "
    "Seeds change which images are drawn; for a fixed seed, the K=1 images are also in K=3, 5 "
    "and 10 (nested draws)."
)
