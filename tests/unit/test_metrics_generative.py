import pytest

from epp_core.eval.metrics.generative import (
    coverage_at_k,
    exact_set_match,
    f1_at_k,
    novelty,
    per_molecule_validity,
    precision_at_k,
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


def test_precision_at_k_known_values():
    # k=1: reaction 0 predicts CCO (1/1 correct), reaction 1 predicts CCC (0/1) -> 0.5.
    # k=2: each reaction has 1 correct of 2 distinct predictions -> 0.5.
    assert precision_at_k(REFS, PREDS, 1) == pytest.approx(0.5)
    assert precision_at_k(REFS, PREDS, 2) == pytest.approx(0.5)


def test_f1_at_k_known_values():
    # k=1: reaction 0 F1=1.0, reaction 1 F1=0.0 -> mean 0.5.
    # k=2: each reaction precision=0.5, recall=1.0 -> F1=2/3 -> mean 2/3.
    assert f1_at_k(REFS, PREDS, 1) == pytest.approx(0.5)
    assert f1_at_k(REFS, PREDS, 2) == pytest.approx(2 / 3)


def test_per_k_metrics_split_multi_molecule_product_sides():
    # The true product side is two molecules; the model predicts them dot-joined.
    # The per-k metrics must compare molecules, not the joined side string (else a
    # correct multi-molecule prediction scores zero, as it did before the fix).
    refs = [["CCO", "CC=O"]]
    preds = [["CCO.CC=O", "CCO.CCC"]]
    assert top_k_accuracy(refs, preds, 1) == pytest.approx(1.0)  # both true molecules present
    assert coverage_at_k(refs, preds, 1) == pytest.approx(1.0)  # 2/2 recovered
    assert precision_at_k(refs, preds, 1) == pytest.approx(1.0)  # 2 predicted, both correct
    assert f1_at_k(refs, preds, 1) == pytest.approx(1.0)
    # k=2 pools in CCC (wrong), so precision drops to 2 correct of 3 predicted molecules.
    assert precision_at_k(refs, preds, 2) == pytest.approx(2 / 3)
    assert coverage_at_k(refs, preds, 2) == pytest.approx(1.0)


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
