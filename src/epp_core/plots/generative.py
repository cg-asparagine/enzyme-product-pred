"""Plots for generative (reaction-product) evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from matplotlib.figure import Figure


def plot_top_k_accuracy(ks: Sequence[int], accuracies: Sequence[float]) -> Figure:
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.bar([str(k) for k in ks], accuracies, color="#3b7dd8")
    ax.set_xlabel("k")
    ax.set_ylabel("Top-k accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Top-k accuracy")
    for x, y in zip(range(len(ks)), accuracies, strict=False):
        ax.text(x, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    return fig


def plot_coverage_curve(ks: Sequence[int], coverages: Sequence[float]) -> Figure:
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.plot(list(ks), list(coverages), marker="o", color="#2ca02c")
    ax.set_xlabel("k")
    ax.set_ylabel("Per-product recall")
    ax.set_ylim(0, 1)
    ax.set_title("Coverage (recall) vs k")
    return fig


def plot_tanimoto_hist(similarities: Sequence[float]) -> Figure:
    fig = Figure(figsize=(5, 4))
    ax = fig.subplots()
    ax.hist(list(similarities), bins=20, range=(0, 1), color="#9467bd", edgecolor="white")
    ax.set_xlabel("Tanimoto similarity (top-1 prediction vs reference)")
    ax.set_ylabel("Count")
    ax.set_title("Top-1 prediction similarity distribution")
    return fig
