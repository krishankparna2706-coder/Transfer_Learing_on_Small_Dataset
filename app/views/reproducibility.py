"""Reproducibility: protocol, seal, hashes, environment, commands, artifacts, errors and limits."""

from __future__ import annotations

import platform

import numpy
import pandas as pd
import PIL
import streamlit as st
import torch
import torchvision

from fewshotlab import PROTOCOL_ID, __version__
from fewshotlab.engine import artifacts
from fewshotlab.errors import FewShotLabError
from fewshotlab.hashing import sha256_file, sha256_json
from fewshotlab.metrics.core import classification_metrics
from fewshotlab.paths import portable_path
from views import common

reg = common.registry()
study = reg["study"]
seal = reg["seal"]
paths = common.paths()

st.title("Reproducibility and evidence")
st.markdown("**How to read the labels in this app**")
for kind in ("val", "test", "live", "replay"):
    title, text = common.BADGES[kind]
    st.markdown(f"- **{title}**: {text}")

st.subheader("Study and protocol")
st.markdown(
    f"- Protocol `{PROTOCOL_ID}`, fewshotlab {__version__}\n"
    f"- Study `{study['study_id']}` ({study['category']}) from "
    f"`{portable_path(common.study_path())}`, "
    f"config SHA-256 `{sha256_json(study)}`\n"
    f"- Datasets {', '.join(study['datasets'])}; K {study['shots']}; methods "
    f"{', '.join(study['methods'])}; support seeds {study['support_seeds']}"
)

st.subheader("Sealed evaluation plan")
if seal is None:
    st.info(
        "NOT SEALED. Seal with `python -m fewshotlab lock-test --plan PATH`, then evaluate once "
        "with `python -m fewshotlab evaluate --split test --study PATH`.",
        icon=":material/lock_open:",
    )
else:
    reproduction = seal["validation_reproduction"]
    window = seal["test_window_utc"]
    st.markdown(
        f"- Audit token `{seal['audit_token']}`; plan SHA-256 `{seal['plan_sha256']}`\n"
        f"- Sealed {seal['sealed_utc']} at commit `{seal['git'].get('commit', '?')[:12]}` "
        f"(clean code tree: {not seal['git'].get('dirty', True)}); "
        f"plan file `{seal['plan_file']}`\n"
        f"- {seal['runs']} sealed runs, {seal['duplicates']} deterministic duplicates, "
        f"{seal['missing']} missing; partial seal: {seal['partial']}\n"
        f"- Validation re-scored before sealing: {reproduction.get('runs')} runs, all identical "
        f"{reproduction.get('all_identical')}, largest score difference "
        f"{reproduction.get('max_abs_score_difference')}\n"
        f"- Held-out test: {seal['test_evaluations']} runs evaluated once"
        + (f", {window[0]} to {window[1]}" if window else "")
        + (f"; robustness study {seal['robustness_utc']}" if seal["robustness_utc"] else "")
    )

    st.markdown("**Protocol modules: sealed hash versus the file on disk now**")
    modules = []
    for rel, sealed_hash in seal["protocol_modules"].items():
        path = paths.root / rel
        current = sha256_file(path) if path.is_file() else None
        modules.append(
            {
                "module": rel,
                "sealed SHA-256": (sealed_hash or "missing")[:16],
                "current SHA-256": (current or "missing")[:16],
                "matches": "yes" if current == sealed_hash else "NO",
            }
        )
    st.dataframe(pd.DataFrame(modules), hide_index=True, width="stretch")
    st.caption(
        "Test access ended with the sealed evaluation (D-051). The replay page re-checks the "
        "modules it depends on (image decoding, transforms, corruptions, replay order, review "
        "rule) and refuses to run if any of them changed."
    )

    st.markdown("**Environment: sealed versus this session**")
    sealed_env = seal["environment"]
    now = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
    }
    env_rows = [
        {"item": key, "sealed": str(sealed_env.get(key)), "this session": value}
        for key, value in now.items()
    ]
    env_rows.append(
        {
            "item": "device (sealed evaluation)",
            "sealed": sealed_env.get("device"),
            "this session": common.device(),
        }
    )
    for name, digest in (sealed_env.get("locks") or {}).items():
        path = paths.root / name
        env_rows.append(
            {
                "item": name,
                "sealed": (digest or "missing")[:16],
                "this session": (sha256_file(path) if path.is_file() else "missing")[:16],
            }
        )
    st.dataframe(pd.DataFrame(env_rows), hide_index=True, width="stretch")

    st.markdown("**Data and weights in the sealed plan**")
    data_rows = [
        {
            "dataset": name,
            "split manifest": entry["split_manifest"],
            "manifest SHA-256": entry["split_manifest_sha256"][:16],
            "class map SHA-256": entry["class_map_sha256"][:16],
            "test images (sealed list)": entry["expected_test_ids"]["count"],
        }
        for name, entry in seal["data"].items()
    ]
    st.dataframe(pd.DataFrame(data_rows), hide_index=True, width="stretch")
    for weights_id, entry in seal["weights"].items():
        st.markdown(
            f"- Pretrained weights {weights_id}: `{entry.get('filename')}`, "
            f"SHA-256 `{entry.get('sha256')}`"
        )

st.subheader("Trace a number to its prediction file and checkpoint")
pick = st.columns(5)
t_dataset = pick[0].selectbox(
    "Dataset", study["datasets"], format_func=common.dataset_name, key="trace_dataset"
)
t_split = pick[1].selectbox(
    "Split", ["val", "test"], format_func=common.SPLIT_NAMES.get, key="trace_split"
)
t_k = pick[2].selectbox("K", sorted(study["shots"]), key="trace_k")
t_method = pick[3].selectbox(
    "Method", [m for m in common.METHOD_ORDER if m in study["methods"]], key="trace_method"
)
t_seed = pick[4].selectbox("Seed", sorted(study["support_seeds"]), key="trace_seed")
cell = next(
    r
    for r in reg["rows"][t_split]
    if (r["dataset"], r["method"], r["k"], r["seed"]) == (t_dataset, t_method, t_k, t_seed)
)
if cell["macro_f1"] is None:
    st.info(f"NOT RUN: this cell is {cell['status']}, so there is no number to trace.")
else:
    try:
        ref = common.cell_ref(reg, t_dataset, t_method, t_k, t_seed)
        folder = artifacts.run_root(paths, ref.run_id) / ref.attempt_id
        source = folder / f"predictions_{t_split}.csv"
        rows = artifacts.read_predictions(source)
        facts = reg["datasets"][t_dataset]
        again = classification_metrics(rows["true"], rows["pred"], facts["classes"])["macro_f1"]
        same = "identical" if again == cell["macro_f1"] else "DIFFERENT"
        st.markdown(
            f"- Shown in the app ({common.SPLIT_NAMES[t_split]}): macro-F1 "
            f"**{cell['macro_f1']:.6f}**\n"
            f"- Recomputed now from `{portable_path(source)}` ({len(rows['image_ids']):,} rows, "
            f"SHA-256 `{sha256_file(source)[:16]}...`): **{again:.6f}**, {same}\n"
            f"- Checkpoint `{ref.artifact_file}`, SHA-256 `{ref.artifact_sha256[:16]}...`; run "
            f"`{ref.run_id}` / `{ref.attempt_id}`; protocol `{PROTOCOL_ID}`"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "image": image_id,
                        "true class": common.class_label(facts, int(true)),
                        "predicted": common.class_label(facts, int(pred)),
                        "model score": float(scores.max()),
                    }
                    for image_id, true, pred, scores in list(
                        zip(
                            rows["image_ids"],
                            rows["true"],
                            rows["pred"],
                            rows["scores"],
                            strict=True,
                        )
                    )[:8]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption("The first 8 prediction rows of that file; every row is one image.")
    except (FewShotLabError, OSError) as exc:
        st.error(f"This number cannot be traced: {exc}")

st.subheader("Commands")
st.code(
    "\n".join(
        [
            "uv sync --locked",
            "python -m fewshotlab doctor",
            "python -m fewshotlab data prepare --dataset flowers102   # see docs for sources",
            "python -m fewshotlab data prepare --dataset neu_cls",
            "python -m fewshotlab support build --config configs/full-study.yaml",
            "python -m fewshotlab benchmark --config configs/full-study.yaml",
            "python -m fewshotlab report --study configs/full-study.yaml",
            "python -m fewshotlab lock-test --plan configs/full-study.yaml --dry-run",
            "python -m fewshotlab lock-test --plan configs/full-study.yaml",
            "python -m fewshotlab evaluate --split test --study configs/full-study.yaml",
            "python -m fewshotlab predict --run RUN_ID --image IMAGE",
            "python -m fewshotlab demo --check",
            "python -m fewshotlab demo",
        ]
    ),
    language="bash",
)

st.subheader("Artifacts")
study_dir = paths.reports / study["study_id"]
artifacts = [
    ("run ledger (append-only)", paths.runs / "ledger.jsonl"),
    ("study plan", study_dir / "plan.md"),
    ("validation comparison", study_dir / "comparison.md"),
    ("validation aggregates", study_dir / "aggregate_val.md"),
    ("sealed plan", study_dir / "sealed_plan.json"),
    ("evaluation ledger", study_dir / "evaluation_ledger.jsonl"),
    ("held-out test aggregates", study_dir / "test" / "aggregate_test.md"),
    ("robustness study", study_dir / "test" / "robustness.md"),
]
st.dataframe(
    pd.DataFrame(
        [
            {
                "artifact": label,
                "file": portable_path(path),
                "present": "yes" if path.is_file() else "NOT RUN",
            }
            for label, path in artifacts
        ]
    ),
    hide_index=True,
    width="stretch",
)
st.caption(
    "Each run's files (checkpoint, predictions, metrics, provenance) are in "
    "runs/artifacts/<run ID>/<attempt>/; on this machine that folder links to local disk."
)

st.subheader("Error gallery (validation, local only)")
galleries = sorted((study_dir / "errors").glob("*.png")) if (study_dir / "errors").is_dir() else []
if not galleries:
    st.info("No error gallery on this machine; `report --study PATH` writes them locally.")
else:
    chosen = st.selectbox("Run", galleries, format_func=lambda p: p.stem, key="repro_gallery")
    st.image(str(chosen), caption=f"validation errors of {chosen.stem}", width="stretch")
    st.caption(
        "Galleries are fixed, prespecified samples of validation errors; they are kept out of "
        "the repository because they contain dataset images."
    )

st.subheader("Known limitations")
limits = paths.root / "docs" / "KNOWN_LIMITATIONS.md"
if limits.is_file():
    with st.expander("docs/KNOWN_LIMITATIONS.md", expanded=False):
        st.markdown(limits.read_text(encoding="utf-8"))
st.caption(
    "Stored benchmark numbers never change in this app. Only the Single image prediction page "
    "produces live output, and it is labeled LIVE PREDICTION."
)
