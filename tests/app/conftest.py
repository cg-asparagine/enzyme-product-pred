"""Put the repo root on sys.path so the `app` package imports in tests.

`app/` is a script directory (the Streamlit GUI), not an installed package, so —
like the model dirs in tests/models/conftest.py — its root must be added to the
path for `from app import ...` to resolve.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
