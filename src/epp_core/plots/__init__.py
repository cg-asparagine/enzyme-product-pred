"""Plot rendering for evaluation reports.

Uses matplotlib's object-oriented ``Figure`` API (no global ``pyplot`` state /
backend), so figures render headlessly and save straight to PNG.
"""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure


def save_figure(fig: Figure, path: str | Path) -> str:
    """Save a figure to ``path`` (creating parents) and return the path string."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return str(path)
