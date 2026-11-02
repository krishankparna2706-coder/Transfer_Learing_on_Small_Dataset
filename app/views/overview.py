"""Overview: the question, the path from images to evidence, and where the evidence stands."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from fewshotlab.reporting.benchmark_report import aggregate
from views import common

reg = common.registry()
study = reg["study"]
counts = common.status_counts(reg)
seal = reg["seal"]

st.title("How much does a pretrained representation help with 1 to 10 labels per class?")
st.markdown(
    "FewShotLab adapts one ResNet18 in four ways when each class has only **K = 1, 3, 5 or 10** "
    "labeled training images, on two separate image domains: flower species (Oxford Flowers102) "
    "and steel-surface defects (NEU-CLS). Every number in this app is regenerated from stored, "
    "hash-checked prediction files."
)

st.subheader("From images to evidence")
steps = [
    ("1. Data", "Fixed, audited splits: a training pool, a validation set, a held-out test set."),
    ("2. Support set", "K labeled images per class from the training pool, drawn with 5 seeds."),
    ("3. Adapt", "Four ways to fit ResNet18 to those images (below)."),
    ("4. Select", "Each run keeps the epoch with the best validation macro-F1."),
    ("5. Seal, then test", "The plan is sealed; every run is scored on the test split once."),
    ("6. Evidence", "Tables, charts and this app read the stored predictions."),
]
for row in (steps[:3], steps[3:]):
    for column, (head, text) in zip(st.columns(3), row, strict=True):
        with column, st.container(border=True, height="stretch"):
            st.markdown(f"**{head}**  \n{text}")

st.subheader("The four methods")
for row in (common.METHOD_ORDER[:2], common.METHOD_ORDER[2:]):
    for column, method in zip(st.columns(2), row, strict=True):
        with column:
            st.markdown(f"**{method}**: {common.METHOD_TEXT[method]}")

st.subheader("Where the evidence stands")
left, middle, right = st.columns(3)
left.metric(
    "Validation runs COMPLETED",
    f"{counts['completed_runs']} of {counts['runs']}",
    help=f"{counts['slots']} planned slots (dataset x K x method x seed); deterministic "
    "duplicate slots point to one run.",
    border=True,
)
middle.metric(
    "Held-out test runs EVALUATED once",
    f"{counts['evaluated_runs']} of {counts['runs']}" if counts["evaluated"] else "NOT RUN",
    help="Each sealed run is evaluated on the test split once.",
    border=True,
)
demo = (seal or {}).get("demo") or {}
right.metric(
    "Sealed demo model (industrial replay)",
    f"{demo['selected']['method']}, NEU K={demo['target']['k']}, seed {demo['target']['seed']}"
    if demo
    else "NOT RUN",
    help="Chosen on validation before the seal: highest validation macro-F1 at NEU K=10, seed 11.",
    border=True,
)

st.subheader("Headline: held-out test macro-F1 at the smallest and largest K")
if not counts["evaluated"]:
    st.info(
        "NOT RUN. Held-out test results appear only after `python -m fewshotlab lock-test --plan "
        "PATH` and `python -m fewshotlab evaluate --split test --study PATH`.",
        icon=":material/lock:",
    )
else:
    common.badge("test")
    shots = sorted(study["shots"])
    for dataset in study["datasets"]:
        rows = [r for r in reg["rows"]["test"] if r["dataset"] == dataset]
        cells = {(a["k"], a["method"]): a for a in aggregate(rows, "macro_f1")}
        table = []
        for k in (shots[0], shots[-1]):
            line = {"K": k}
            for method in common.METHOD_ORDER:
                a = cells.get((k, method))
                if not a or a["mean"] is None:
                    line[method] = "NOT RUN"
                else:
                    sd = "" if a["sd"] is None else f" ± {a['sd']:.3f}"
                    line[method] = f"{a['mean']:.3f}{sd} (n={a['n_distinct']})"
            table.append(line)
        st.markdown(f"**{common.dataset_name(dataset)}**")
        st.caption(common.budget_caption(reg, dataset))
        st.dataframe(pd.DataFrame(table), hide_index=True, width="stretch")
    st.caption(
        "Mean ± sample SD over distinct runs across the support seeds (n = distinct runs). The "
        "Experiment explorer shows every K, every seed, both splits and the paired differences."
    )

st.subheader("What this app does not do")
st.markdown(
    "- **It never trains a model or changes a dataset.** A filter only selects stored results; "
    "an uploaded image is scored and then discarded.\n"
    "- **Scores are model scores, not calibrated probabilities.** An image outside the learned "
    "classes still receives a score for every class, and a low-score review rule is not "
    "reliable out-of-distribution detection.\n"
    "- **The industrial replay is a simulation on stored images.** It does not decide whether a "
    "product is defective (NEU-CLS has no defect-free class), locate a defect or control "
    "machinery."
)
