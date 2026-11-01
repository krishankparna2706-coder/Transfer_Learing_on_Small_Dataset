"""FewShotLab: reproducible transfer-learning experiments with few labeled images.

The package is the single source of truth for the CLI (``python -m fewshotlab``) and the
Streamlit app. Nothing in this package claims or hard-codes benchmark results.
"""

__all__ = ["PROTOCOL_ID", "__version__"]

__version__ = "0.1.0"
PROTOCOL_ID = "fewshotlab-v1"
