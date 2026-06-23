"""Generative metrics for enzymatic reaction-product prediction.

Correctness is defined by *canonical-SMILES* equality: predictions and
references are canonicalized before comparison so equivalent structures match.
A reaction's reference is the set of product-side molecules; a prediction is a
ranked candidate product side (each candidate may itself be a dot-joined
multi-molecule set). Invalid predictions are excluded from correctness/novelty
but still counted in the validity denominator.
"""

from __future__ import annotations

from collections.abc import Iterable

from epp_core.chem.smiles import canonicalize, is_valid, tanimoto


def _canon_set(smiles_list: Iterable[str]) -> set[str]:
    return {c for c in (canonicalize(s) for s in smiles_list) if c is not None}


def _canon_frozenset(smiles_list: Iterable[str]) -> frozenset[str]:
    return frozenset(_canon_set(smiles_list))


def _ranked_canon(predictions: Iterable[str]) -> list[str]:
    """Canonicalize predictions, drop invalids, dedupe — preserving rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for s in predictions:
        c = canonicalize(s)
        if c is not None and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _top_k_molecules(predictions: Iterable[str], k: int) -> set[str]:
    """Canonical individual product molecules drawn from the top-k predicted sides.

    A prediction is a product *side* — possibly several molecules joined by ``.``
    (e.g. a transferase's product plus a released cofactor). The per-k recovery
    metrics below compare *molecules* (the references are individual product
    molecules too), so each top-k side is split into its molecules before the set
    operations; comparing a dot-joined side string against single-molecule
    references would never match a multi-molecule product.
    """
    molecules: set[str] = set()
    for side in list(predictions)[:k]:
        for mol in side.split("."):
            c = canonicalize(mol)
            if c is not None:
                molecules.add(c)
    return molecules


def top_k_accuracy(references: list[list[str]], predictions: list[list[str]], k: int) -> float:
    """Fraction of reactions with at least one true product molecule in the top-k sides."""
    if not references:
        return 0.0
    hits = 0
    for refs, preds in zip(references, predictions, strict=False):
        if _canon_set(refs) & _top_k_molecules(preds, k):
            hits += 1
    return hits / len(references)


def coverage_at_k(references: list[list[str]], predictions: list[list[str]], k: int) -> float:
    """Mean per-reaction recall (sensitivity) of true product molecules in the top-k sides.

    Per reaction: ``|true ∩ top-k molecules| / |true|`` — the fraction of a
    reaction's true product molecules recovered. Averaged over reactions that have
    at least one reference.
    """
    recalls: list[float] = []
    for refs, preds in zip(references, predictions, strict=False):
        ref_set = _canon_set(refs)
        if not ref_set:
            continue
        topk = _top_k_molecules(preds, k)
        recalls.append(len(ref_set & topk) / len(ref_set))
    return sum(recalls) / len(recalls) if recalls else 0.0


def precision_at_k(references: list[list[str]], predictions: list[list[str]], k: int) -> float:
    """Mean per-reaction precision of the molecules predicted across the top-k sides.

    Per reaction: ``|true ∩ top-k molecules| / |top-k molecules|`` — of the
    distinct valid product molecules predicted across the top-k sides, the fraction
    that are correct. Averaged over reactions that produced at least one valid
    molecule.
    """
    precisions: list[float] = []
    for refs, preds in zip(references, predictions, strict=False):
        ref_set = _canon_set(refs)
        topk = _top_k_molecules(preds, k)
        if not topk:
            continue
        precisions.append(len(ref_set & topk) / len(topk))
    return sum(precisions) / len(precisions) if precisions else 0.0


def f1_at_k(references: list[list[str]], predictions: list[list[str]], k: int) -> float:
    """Mean per-reaction F1 — the harmonic mean of precision@k and sensitivity@k.

    Computed per reaction then averaged (macro), over reactions that have at least
    one reference and one valid predicted molecule.
    """
    f1s: list[float] = []
    for refs, preds in zip(references, predictions, strict=False):
        ref_set = _canon_set(refs)
        topk = _top_k_molecules(preds, k)
        if not ref_set or not topk:
            continue
        tp = len(ref_set & topk)
        precision = tp / len(topk)
        recall = tp / len(ref_set)
        denom = precision + recall
        f1s.append(2 * precision * recall / denom if denom else 0.0)
    return sum(f1s) / len(f1s) if f1s else 0.0


def exact_set_match(references: list[list[str]], predictions: list[list[str]], k: int) -> float:
    """Fraction of reactions whose top-k holds a prediction matching the full product set.

    Stricter than :func:`top_k_accuracy` (which counts any single correct
    product): here a predicted product side must equal the reference side
    exactly, order-independently. Each prediction string is split on ``.`` into
    its molecules before comparison.
    """
    if not references:
        return 0.0
    hits = 0
    for refs, preds in zip(references, predictions, strict=False):
        ref_set = _canon_frozenset(refs)
        for pred in preds[:k]:
            pred_set = _canon_frozenset(pred.split("."))
            if pred_set and pred_set == ref_set:
                hits += 1
                break
    return hits / len(references)


def validity(prediction_lists: list[list[str]]) -> float:
    """Fraction of all predicted strings that are valid SMILES."""
    total = valid = 0
    for preds in prediction_lists:
        for s in preds:
            total += 1
            if is_valid(s):
                valid += 1
    return valid / total if total else 0.0


def per_molecule_validity(prediction_lists: list[list[str]]) -> float:
    """Fraction of individual predicted molecules that are valid SMILES.

    Each prediction may be a dot-joined product set; this splits on ``.`` and
    scores each molecule, so a candidate with 3/4 parseable molecules counts as
    partially valid rather than wholly invalid.
    """
    total = valid = 0
    for preds in prediction_lists:
        for pred in preds:
            for mol in pred.split("."):
                if not mol:
                    continue
                total += 1
                if is_valid(mol):
                    valid += 1
    return valid / total if total else 0.0


def uniqueness(prediction_lists: list[list[str]]) -> float:
    """Mean per-reaction ratio of unique valid predictions to valid predictions."""
    ratios: list[float] = []
    for preds in prediction_lists:
        canon = [c for c in (canonicalize(s) for s in preds) if c is not None]
        if not canon:
            continue
        ratios.append(len(set(canon)) / len(canon))
    return sum(ratios) / len(ratios) if ratios else 0.0


def novelty(prediction_lists: list[list[str]], train_smiles: Iterable[str]) -> float:
    """Fraction of valid predictions not present in the training set."""
    train = _canon_set(train_smiles)
    total = novel = 0
    for preds in prediction_lists:
        for s in preds:
            c = canonicalize(s)
            if c is None:
                continue
            total += 1
            if c not in train:
                novel += 1
    return novel / total if total else 0.0


def tanimoto_to_reference_distribution(
    references: list[list[str]], predictions: list[list[str]]
) -> list[float]:
    """Per-reaction Tanimoto of the top-1 prediction to its closest reference."""
    sims: list[float] = []
    for refs, preds in zip(references, predictions, strict=False):
        top = _ranked_canon(preds)[:1]
        if not top or not refs:
            continue
        best = max((tanimoto(top[0], r) or 0.0) for r in refs)
        sims.append(best)
    return sims
