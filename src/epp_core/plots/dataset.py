"""Plots for the dataset / split EDA report.

Pure rendering: every function takes already-aggregated values and returns a
matplotlib ``Figure`` (object-oriented API, no global pyplot state), to be saved
with :func:`epp_core.plots.save_figure`. Aggregation lives in
:mod:`epp_core.report.dataset`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure

_BLUE = "#3b7dd8"
_GREEN = "#2ca02c"
_PURPLE = "#9467bd"
_ORANGE = "#ff7f0e"
_RED = "#d62728"
# Stable colours for the three splits wherever they are drawn together.
_SPLIT_COLORS = {"train": _BLUE, "valid": _ORANGE, "test": _GREEN}


def _hist(
    values: Sequence[float],
    *,
    bins: int,
    title: str,
    xlabel: str,
    color: str = _BLUE,
    logy: bool = False,
    clip: float | None = None,
) -> Figure:
    fig = Figure(figsize=(5.4, 3.6))
    ax = fig.subplots()
    data = [min(v, clip) for v in values] if clip is not None else list(values)
    ax.hist(data, bins=bins, color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(xlabel + (f" (clipped at {clip:g})" if clip is not None else ""))
    ax.set_ylabel("Count" + (" (log)" if logy else ""))
    if logy:
        ax.set_yscale("log")
    return fig


def plot_sequence_length_hist(seq_lens: Sequence[int], clip: float | None = 1500) -> Figure:
    """Distribution of enzyme sequence lengths (one count per unique enzyme)."""
    return _hist(
        seq_lens,
        bins=50,
        title="Enzyme sequence length",
        xlabel="residues",
        color=_PURPLE,
        clip=clip,
    )


def plot_cluster_size_hist(sizes: Sequence[int], clip: float | None = 50) -> Figure:
    """Distribution of cluster sizes (enzymes per cluster), log-scaled counts."""
    return _hist(
        sizes,
        bins=min(50, max(int(max(sizes, default=1)), 1)),
        title="Enzyme cluster sizes",
        xlabel="enzymes per cluster",
        color=_GREEN,
        logy=True,
        clip=clip,
    )


def plot_reaction_token_lengths(
    reactant_tokens: Sequence[int], product_tokens: Sequence[int]
) -> Figure:
    """Overlaid histograms of reactant- vs product-side SMILES token counts."""
    fig = Figure(figsize=(5.4, 3.6))
    ax = fig.subplots()
    ax.hist(
        list(reactant_tokens), bins=40, color=_BLUE, alpha=0.6, label="reactants", edgecolor="white"
    )
    ax.hist(
        list(product_tokens), bins=40, color=_ORANGE, alpha=0.6, label="products", edgecolor="white"
    )
    ax.set_title("Reaction size (SMILES tokens)")
    ax.set_xlabel("tokens")
    ax.set_ylabel("Count")
    ax.legend()
    return fig


def plot_molecule_counts(counts: Mapping[str, Mapping[int, int]]) -> Figure:
    """Grouped bars: how many reactions have N reactant / N product molecules."""
    fig = Figure(figsize=(5.4, 3.6))
    ax = fig.subplots()
    keys = sorted({n for series in counts.values() for n in series})
    width = 0.8 / max(len(counts), 1)
    for i, (label, series) in enumerate(counts.items()):
        color = _BLUE if "react" in label else _ORANGE
        ax.bar(
            [k + i * width for k in keys],
            [series.get(k, 0) for k in keys],
            width=width,
            label=label,
            color=color,
        )
    ax.set_xticks([k + width * (len(counts) - 1) / 2 for k in keys])
    ax.set_xticklabels([str(k) for k in keys])
    ax.set_title("Molecules per reaction side")
    ax.set_xlabel("number of molecules")
    ax.set_ylabel("Count")
    ax.legend()
    return fig


def plot_ec_families(counts: Mapping[str, int]) -> Figure:
    """Bar chart of EC top-level class counts (examples per EC class)."""
    fig = Figure(figsize=(6.0, 3.6))
    ax = fig.subplots()
    labels = list(counts.keys())
    ax.bar(labels, [counts[k] for k in labels], color=_BLUE)
    ax.set_title("Enzyme families (EC top-level class)")
    ax.set_ylabel("Examples")
    ax.tick_params(axis="x", rotation=30)
    for x, k in enumerate(labels):
        ax.text(x, counts[k], f"{counts[k]:,}", ha="center", va="bottom", fontsize=7)
    return fig


def plot_split_composition(counts: Mapping[str, Mapping[str, int]]) -> Figure:
    """Grouped bars of train/valid/test row counts for each split strategy."""
    fig = Figure(figsize=(5.4, 3.6))
    ax = fig.subplots()
    splits = ["train", "valid", "test"]
    width = 0.8 / max(len(counts), 1)
    for i, (strategy, by_split) in enumerate(counts.items()):
        ax.bar(
            [j + i * width for j in range(len(splits))],
            [by_split.get(s, 0) for s in splits],
            width=width,
            label=strategy,
        )
    ax.set_xticks([j + width * (len(counts) - 1) / 2 for j in range(len(splits))])
    ax.set_xticklabels(splits)
    ax.set_title("Split composition (examples)")
    ax.set_ylabel("Examples")
    ax.legend()
    return fig


def plot_enzyme_leakage(leakage: Mapping[str, float]) -> Figure:
    """Bar chart: fraction of test examples whose enzyme also appears in train,
    per split strategy. The motivating contrast for the enzyme split."""
    fig = Figure(figsize=(5.0, 3.6))
    ax = fig.subplots()
    labels = list(leakage.keys())
    vals = [leakage[k] for k in labels]
    ax.bar(labels, vals, color=[_RED if v > 0.01 else _GREEN for v in vals])
    ax.set_ylim(0, 1)
    ax.set_title("Enzyme leakage: test enzymes also seen in train")
    ax.set_ylabel("fraction of test examples")
    ax.tick_params(axis="x", rotation=15)
    for x, v in enumerate(vals):
        ax.text(x, v + 0.02, f"{v:.1%}", ha="center", va="bottom", fontsize=9)
    return fig
