"""Put each ``models/<Name>/`` script dir on ``sys.path`` so its uniquely-named
inner package is importable in tests (mirrors the pyright ``extraPaths`` list)."""

import sys
from pathlib import Path

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

for _name in ("ESM2-650M-frozen-ReactionT5",):
    _path = _MODELS_DIR / _name
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
