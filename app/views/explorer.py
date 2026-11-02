"""Experiment explorer: measured comparisons by method, K and seed for one dataset and split."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fewshotlab.reporting.benchmark_report import aggregate, paired
from views import common

reg = common.registry()
study = reg["study"]

st.title("Experiment explorer")
top = st.columns(3)
dataset = top[0].radio(
    "Dataset", study["datasets"], format_func=common.dataset_name, horizontal=True, key="ex_dataset"
)
split = top[1].radio(
    "Split", ["val", "test"], format_func=common.SPLIT_NAMES.get, horizontal=True, key="ex_split"
)
metric = top[2].selectbox(
    "Metric", list(common.METRIC_NAMES), format_func=common.METRIC_NAMES.get, key="ex_metric"
)
filters = st.columns(3)
method_options = [m for m in common.METHOD_ORDER if m in study["methods"]]
methods = filters[0].multiselect(
    "Methods", method_options, default=method_options, key="ex_methods"
)
shots = filters[1].multiselect(
    "K (labeled images per class)", study["shots"], default=study["shots"], key="ex_shots"
)
seeds = filters[2].multiselect(
    "Support seeds", study["support_seeds"], default=study["support_seeds"], key="ex_seeds"
)

common.badge(split)
st.caption(common.budget_caption(reg, dataset))
if not (methods and shots and seeds):
    st.info("Choose at least one method, one K and one seed to compare.", icon=":material/tune:")
    st.stop()

methods = [m for m in common.METHOD_ORDER if m in methods]
shots = sorted(shots)
rows = [
    r
    for r in reg["rows"][split]
    if r["dataset"] == dataset and r["method"] in methods and r["k"] in shots and r["seed"] in seeds
]
measured = [r for r in rows if r[metric] is not None]
metric_name = common.METRIC_NAMES[metric]
if not measured:
    statuses = sorted({r["status"] for r in rows})
    action = (
        "Held-out test values exist only after `lock-test` and `evaluate --split test`."
        if split == "test"
        else "Train the missing runs with `python -m fewshotlab benchmark --config CONFIG`."
    )
    st.info(
        f"NOT RUN: no {metric_name} values for this selection ({', '.join(statuses)}). {action}"
    )
    st.stop()

theme = common.theme()
colors, ink = common.SERIES[theme], common.INK[theme]
position = {k: i for i, k in enumerate(shots)}
k_labels = " : ".join(f"datum.value == {i} ? '{k}'" for k, i in position.items()) + " : ''"
x_axis = alt.Axis(values=list(position.values()), labelExpr=k_labels, grid=False, labelAngle=0)
color_scale = alt.Scale(
    domain=list(common.METHOD_ORDER), range=[colors[m] for m in common.METHOD_ORDER]
)
shape_scale = alt.Scale(
    domain=list(common.METHOD_ORDER), range=[common.SHAPES[m] for m in common.METHOD_ORDER]
)
y_scale = alt.Scale(domain=[0, 1])

cells = [a for a in aggregate(rows, metric) if a["mean"] is not None]
means = pd.DataFrame(
    [
        {
            "method": a["method"],
            "K": a["k"],
            "x": position[a["k"]],
            "mean": a["mean"],
            "sd": a["sd"],
            "low": max(0.0, a["mean"] - (a["sd"] or 0.0)),
            "high": min(1.0, a["mean"] + (a["sd"] or 0.0)),
            "runs": a["n_distinct"],
            "seeds": ", ".join(str(s) for s in a["seeds"]),
        }
        for a in cells
    ]
)

# Direct labels at the last K, nudged apart so they never overlap.
last = means[means["K"] == means["K"].max()].sort_values("mean")
gap, placed = 0.05, []
for value in last["mean"]:
    placed.append(max(value, placed[-1] + gap) if placed else value)
overflow = max(0.0, (placed[-1] if placed else 0) - 0.98)
labels = last.assign(label_y=[p - overflow for p in placed])

axis_style = {
    "gridColor": ink["grid"],
    "domainColor": ink["axis"],
    "tickColor": ink["axis"],
    "labelColor": ink["secondary"],
    "titleColor": ink["secondary"],
    "labelFontSize": 12,
    "titleFontSize": 12,
}
tooltip = [
    alt.Tooltip("method:N", title="method"),
    alt.Tooltip("K:Q", title="K"),
    alt.Tooltip("mean:Q", title=f"mean {metric_name}", format=".3f"),
    alt.Tooltip("sd:Q", title="sample SD", format=".3f"),
    alt.Tooltip("runs:Q", title="distinct runs"),
    alt.Tooltip("seeds:N", title="seeds"),
]
x = alt.X(
    "x:Q",
    title="K: labeled training images per class",
    axis=x_axis,
    scale=alt.Scale(domain=[-0.35, len(shots) - 1 + 1.0]),
)
legend = alt.Legend(
    orient="top", title=None, values=methods, labelColor=ink["primary"], labelFontSize=13
)
base = alt.Chart(means)
error_bars = base.mark_rule(strokeWidth=1.5).encode(
    x=x,
    y=alt.Y("low:Q", scale=y_scale),
    y2="high:Q",
    color=alt.Color("method:N", scale=color_scale, legend=legend),
)
lines = base.mark_line(strokeWidth=2).encode(
    x=x,
    y=alt.Y("mean:Q", scale=y_scale, title=f"{metric_name} (mean over seeds)"),
    color=alt.Color("method:N", scale=color_scale, legend=legend),
    detail="method:N",
)
points = base.mark_point(
    size=110, filled=True, opacity=1, stroke=ink["surface"], strokeWidth=2
).encode(
    x=x,
    y=alt.Y("mean:Q", scale=y_scale),
    color=alt.Color("method:N", scale=color_scale, legend=legend),
    shape=alt.Shape("method:N", scale=shape_scale, legend=legend),
    tooltip=tooltip,
)
direct = (
    alt.Chart(labels)
    .mark_text(align="left", dx=12, fontSize=13, color=ink["primary"])
    .encode(x=x, y=alt.Y("label_y:Q", scale=y_scale), text="method:N")
)
chart = (
    alt.layer(error_bars, lines, points, direct)
    .properties(
        height=400,
        title=alt.TitleParams(
            f"{common.dataset_name(dataset)}: {metric_name} by K ({common.SPLIT_NAMES[split]})",
            subtitle="Lines join the mean over distinct runs; bars show ±1 sample SD.",
            anchor="start",
            color=ink["primary"],
            subtitleColor=ink["secondary"],
        ),
    )
    .configure_axis(**axis_style)
    .configure_view(strokeWidth=0)
)
st.altair_chart(chart, width="stretch")

st.markdown(
    "**Every seed, one panel per method** (same scale). An open marker is a deterministic "
    "duplicate: the same run as another seed, counted once in every mean and SD."
)
seed_rows = pd.DataFrame(
    [
        {
            "method": r["method"],
            "K": r["k"],
            "seed": r["seed"],
            "value": r[metric],
            "x": position[r["k"]] + (sorted(seeds).index(r["seed"]) - (len(seeds) - 1) / 2) * 0.1,
            "run": r["source_run_id"],
            "note": "deterministic duplicate: the same run as another seed"
            if r["duplicate"]
            else "",
        }
        for r in measured
    ]
)
for n, (column, method) in enumerate(zip(st.columns(len(methods)), methods, strict=True)):
    mine = seed_rows[seed_rows["method"] == method]
    own_means = means[means["method"] == method]
    if mine.empty:
        column.markdown(f"**{method}**  \nNOT RUN for this selection")
        continue
    px = alt.X("x:Q", title="K", axis=x_axis, scale=alt.Scale(domain=[-0.5, len(shots) - 0.5]))
    py = alt.Y("value:Q", scale=y_scale, title=metric_name if n == 0 else None)
    dots = (
        alt.Chart(mine)
        .mark_point(
            size=70, stroke=colors[method], strokeWidth=1.5, shape=common.SHAPES[method], opacity=1
        )
        .encode(
            x=px,
            y=py,
            fill=alt.when("datum.note == ''")
            .then(alt.value(colors[method]))
            .otherwise(alt.value(ink["surface"])),
            tooltip=[
                alt.Tooltip("seed:N"),
                alt.Tooltip("K:Q"),
                alt.Tooltip("value:Q", title=metric_name, format=".3f"),
                alt.Tooltip("run:N", title="run ID"),
                alt.Tooltip("note:N", title="note"),
            ],
        )
    )
    trend = (
        alt.Chart(own_means)
        .mark_line(strokeWidth=2, color=colors[method], opacity=0.8)
        .encode(
            x=alt.X("x:Q", axis=x_axis, scale=alt.Scale(domain=[-0.5, len(shots) - 0.5])),
            y=alt.Y("mean:Q", scale=y_scale),
        )
    )
    panel = (
        alt.layer(trend, dots)
        .properties(
            height=240,
            title=alt.TitleParams(method, color=ink["primary"], anchor="start", fontSize=14),
        )
        .configure_axis(**axis_style)
        .configure_view(strokeWidth=0)
    )
    column.altair_chart(panel, width="stretch")

st.subheader("Table view")
table = pd.DataFrame(
    [
        {
            "K": a["k"],
            "method": a["method"],
            f"mean {metric_name}": common.fmt(a["mean"]),
            "sample SD": "n/a (one run)"
            if a["mean"] is not None and a["sd"] is None
            else common.fmt(a["sd"]),
            "distinct runs": a["n_distinct"],
            "min": common.fmt(a["min"]),
            "max": common.fmt(a["max"]),
            "seeds with a value": ", ".join(str(s) for s in a["seeds"]) or "none",
        }
        for a in aggregate(rows, metric)
    ]
)
st.dataframe(table, hide_index=True, width="stretch")

with st.expander("Paired differences on matched seeds (A - B)"):
    pairs = [p for p in paired(rows, metric) if p["n_pairs"]]
    if not pairs:
        st.write("No pair of the selected methods has matched seeds in this selection.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "K": p["k"],
                        "A - B": f"{p['a']} - {p['b']}",
                        "matched seeds": p["n_pairs"],
                        "mean difference": common.fmt(p["mean_diff"]),
                        "SD": common.fmt(p["sd_diff"]),
                        "min": common.fmt(p["min_diff"]),
                        "max": common.fmt(p["max_diff"]),
                        "A better": f"{p['a_better']}/{p['n_pairs']}",
                    }
                    for p in pairs
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Five seeds describe this benchmark's variability across support draws; no population "
            "confidence interval is computed from them."
        )

with st.expander("Every slot, with its status and run ID"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "K": r["k"],
                    "method": r["method"],
                    "seed": r["seed"],
                    "status": r["status"],
                    metric_name: common.fmt(r[metric]),
                    "run ID": r["source_run_id"],
                    "duplicate": "yes (same run as another seed)" if r["duplicate"] else "",
                }
                for r in sorted(
                    rows, key=lambda r: (r["k"], common.METHOD_ORDER.index(r["method"]), r["seed"])
                )
            ]
        ),
        hide_index=True,
        width="stretch",
    )
st.caption(
    f"Every value is regenerated from the stored prediction files (registry generated "
    f"{reg['generated_utc']}, fingerprint {reg['fingerprint'][:12]})."
)
