"""FewShotLab demonstration app.

Start it with ``python -m fewshotlab demo`` (or ``streamlit run app/streamlit_app.py`` from the
repository root). Every benchmark number comes from the verified run registry; nothing here
trains a model or changes a dataset.
"""

from __future__ import annotations

import streamlit as st

from views import common

st.set_page_config(page_title="FewShotLab", page_icon=":material/biotech:", layout="wide")

PAGES = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/data_budget.py", title="Data and label budget", icon=":material/dataset:"),
    st.Page("views/explorer.py", title="Experiment explorer", icon=":material/insights:"),
    st.Page("views/predict.py", title="Single image prediction", icon=":material/image_search:"),
    st.Page(
        "views/replay.py", title="Industrial replay", icon=":material/precision_manufacturing:"
    ),
    st.Page("views/reproducibility.py", title="Reproducibility", icon=":material/fact_check:"),
]

page = st.navigation(PAGES)
registry = common.registry()
common.sidebar(registry)
if registry["study"]["category"] == "fixture":
    common.badge("fixture")
page.run()
