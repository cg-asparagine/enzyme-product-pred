import pytest

from epp_core.eval.metrics.generative import (
    coverage_at_k,
    exact_set_match,
    novelty,
    per_molecule_validity,
    top_k_accuracy,
    uniqueness,
    validity,
)

# reaction 0: true product CCO, predicted at rank 1
# reaction 1: true product benzene, predicted at rank 2
REFS = [["CCO"], ["c1ccccc1"]]
PREDS = [["CCO", "CCC"], ["CCC", "c1ccccc1"]]


def test_top_k_accuracy_known_values():
    assert top_k_accuracy(REFS, PREDS, 1) == pytest.approx(0.5)
    assert top_k_accuracy(REFS, PREDS, 2) == pytest.approx(1.0)


def test_coverage_at_k_known_values():
    assert coverage_at_k(REFS, PREDS, 1) == pytest.approx(0.5)
    assert coverage_at_k(REFS, PREDS, 2) == pytest.approx(1.0)


def test_exact_set_match_requires_full_product_set():
    refs = [["CCO"], ["CCO", "CC=O"]]
    preds = [["CCO"], ["CCO", "CCO.CC=O"]]
    # reaction 0: top-1 "CCO" matches the singleton set.
    # reaction 1: top-1 "CCO" is only half the set; top-2 "CCO.CC=O" matches.
    assert exact_set_match(refs, preds, 1) == pytest.approx(0.5)
    assert exact_set_match(refs, preds, 2) == pytest.approx(1.0)


def test_exact_set_match_is_order_independent():
    refs = [["CC=O", "CCO"]]
    preds = [["CCO.CC=O"]]  # same set, different textual order
    assert exact_set_match(refs, preds, 1) == pytest.approx(1.0)


def test_validity_counts_invalid():
    assert validity([["CCO", "((("], ["c1ccccc1"]]) == pytest.approx(2 / 3)


def test_per_molecule_validity_scores_each_molecule():
    # "CCO.(((": one valid molecule (CCO), one invalid ((( -> 1/2.
    assert per_molecule_validity([["CCO.((("]]) == pytest.approx(0.5)


def test_uniqueness_dedupes_canonically():
    # "OCC" canonicalizes to the same molecule as "CCO".
    assert uniqueness([["CCO", "CCO", "OCC"]]) == pytest.approx(1 / 3)


def test_novelty_excludes_training_molecules():
    assert novelty([["CCO", "CCC"]], {"CCO"}) == pytest.approx(0.5)


def test_top_k_accuracy_empty_is_zero():
    assert top_k_accuracy([], [], 5) == 0.0
    assert exact_set_match([], [], 5) == 0.0
