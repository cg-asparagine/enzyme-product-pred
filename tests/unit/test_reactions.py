"""Tests for epp_core.chem.reactions: splitting, canonicalization, and keys."""

from epp_core.chem.reactions import (
    canonical_molecules,
    join_molecules,
    reaction_key,
    split_reaction,
    undirected_reaction_key,
)


def test_split_reaction_basic():
    assert split_reaction("CC=O.[H+]>>CCO") == (["CC=O", "[H+]"], ["CCO"])


def test_split_reaction_single_molecule_each_side():
    assert split_reaction("CCO>>CC=O") == (["CCO"], ["CC=O"])


def test_split_reaction_rejects_malformed():
    assert split_reaction("CCO") is None  # no '>>'
    assert split_reaction("CCO>>CC=O>>X") is None  # double reaction
    assert split_reaction(">>CCO") is None  # empty reactant side
    assert split_reaction("CCO>>") is None  # empty product side
    assert split_reaction("CCO>CC=O") is None  # single '>' (agent form)


def test_canonical_molecules_canonicalizes_each():
    # "OCC" and "CCO" are the same molecule; both canonicalize identically.
    assert canonical_molecules(["OCC"]) == canonical_molecules(["CCO"])


def test_canonical_molecules_returns_none_if_any_invalid():
    assert canonical_molecules(["CCO", "((("]) is None


def test_join_molecules_is_sorted_and_order_independent():
    assert join_molecules(["CCO", "CC=O"]) == join_molecules(["CC=O", "CCO"])
    assert "." in join_molecules(["CCO", "CC=O"])


def test_reaction_key_order_independent_within_side():
    assert reaction_key(["CC=O", "[H+]"], ["CCO"]) == reaction_key(["[H+]", "CC=O"], ["CCO"])


def test_reaction_key_is_direction_aware():
    assert reaction_key(["CCO"], ["CC=O"]) != reaction_key(["CC=O"], ["CCO"])


def test_undirected_key_collapses_forward_and_reverse():
    assert undirected_reaction_key(["CCO"], ["CC=O"]) == undirected_reaction_key(["CC=O"], ["CCO"])


def test_undirected_key_distinguishes_different_reactions():
    assert undirected_reaction_key(["CCO"], ["CC=O"]) != undirected_reaction_key(
        ["CCCO"], ["CCC=O"]
    )
