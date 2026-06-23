"""Unit tests for enzyme sequence clustering (pure logic, synthetic sequences)."""

from __future__ import annotations

import pytest

from epp_core.data.cluster import cluster_sequences, jaccard, kmer_set

# A base sequence, a near-duplicate (two point substitutions) and an unrelated one.
_BASE = "ACDEFGHIKLMNPQRSTVWY" * 4
_NEAR = ("ACDEFGHIKLMNPQRSTVWY" * 4).replace("FGH", "FYH").replace("RST", "RWT")
_FAR = "WVTSRQPNMLKIHGFEDCAY" * 4


def _partition(assignment: dict[str, int]) -> set[frozenset[str]]:
    """Cluster ids -> the set of member groups, ignoring the (arbitrary) id values."""
    groups: dict[int, set[str]] = {}
    for key, cid in assignment.items():
        groups.setdefault(cid, set()).add(key)
    return {frozenset(g) for g in groups.values()}


def test_kmer_set_basic() -> None:
    assert kmer_set("ABCDE", 3) == {"ABC", "BCD", "CDE"}
    assert kmer_set("AB", 3) == {"AB"}  # shorter than k -> whole-sequence token
    assert kmer_set("", 3) == frozenset()
    with pytest.raises(ValueError):
        kmer_set("ABC", 0)


def test_jaccard() -> None:
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset("ab"), frozenset("cd")) == 0.0
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
    assert jaccard(frozenset("abc"), frozenset("abd")) == pytest.approx(2 / 4)


def test_uniprot_method_is_one_cluster_per_id() -> None:
    seqs = {"a": _BASE, "b": _BASE, "c": _FAR}
    assert len(set(cluster_sequences(seqs, method="uniprot").values())) == 3


def test_exact_method_groups_identical_sequences() -> None:
    seqs = {"a": _BASE, "b": _BASE, "c": _FAR}
    assert _partition(cluster_sequences(seqs, method="exact")) == {
        frozenset({"a", "b"}),
        frozenset({"c"}),
    }


def test_kmer_clusters_similar_and_separates_dissimilar() -> None:
    seqs = {"base": _BASE, "near": _NEAR, "far": _FAR}
    parts = _partition(cluster_sequences(seqs, method="kmer", k=3, threshold=0.3))
    assert parts == {frozenset({"base", "near"}), frozenset({"far"})}


def test_kmer_high_threshold_only_groups_identical() -> None:
    seqs = {"a": _BASE, "b": _BASE, "near": _NEAR, "far": _FAR}
    parts = _partition(cluster_sequences(seqs, method="kmer", k=3, threshold=0.99))
    assert frozenset({"a", "b"}) in parts  # identical always merge
    assert frozenset({"near"}) in parts  # near-dup split off at high threshold
    assert frozenset({"far"}) in parts


def test_kmer_is_deterministic_and_order_independent() -> None:
    a = {"x": _BASE, "y": _NEAR, "z": _FAR}
    b = {"z": _FAR, "y": _NEAR, "x": _BASE}  # same content, different insertion order
    assert _partition(cluster_sequences(a, method="kmer", k=3, threshold=0.3)) == _partition(
        cluster_sequences(b, method="kmer", k=3, threshold=0.3)
    )


@pytest.mark.parametrize("kwargs", [{"method": "nope"}, {"threshold": 0.0}, {"threshold": 1.5}])
def test_invalid_args_raise(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        cluster_sequences({"a": _BASE}, **kwargs)
