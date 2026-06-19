"""SMILES utilities (RDKit-backed): the single source of truth for validity,
canonicalization, and similarity used everywhere in the repo.

Every SMILES comparison in the metrics engine routes through :func:`canonicalize`
so that equivalent structures compare equal.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

# RDKit logs a warning to stderr on every unparseable SMILES; we handle parse
# failures explicitly via ``None`` returns, so silence the noise.
RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]  # rdkit ships no stubs

_DEFAULT_RADIUS = 2
_DEFAULT_NBITS = 2048


def is_valid(smiles: str) -> bool:
    """True if RDKit can parse ``smiles`` into a molecule."""
    return bool(smiles) and Chem.MolFromSmiles(smiles) is not None


def canonicalize(smiles: str, isomeric: bool = True) -> str | None:
    """Return RDKit canonical SMILES, or ``None`` if it cannot be parsed.

    Canonical isomeric form: ``Chem.MolToSmiles(mol, isomericSmiles=True)``.
    """
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=isomeric)


def num_atoms(smiles: str) -> int | None:
    """Heavy-atom count, or ``None`` if ``smiles`` is invalid."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return mol.GetNumAtoms() if mol is not None else None


@lru_cache(maxsize=16)
def _morgan_generator(radius: int, n_bits: int) -> Any:
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)


@lru_cache(maxsize=8192)
def _morgan_fp(smiles: str, radius: int, n_bits: int) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _morgan_generator(radius, n_bits).GetFingerprint(mol)


def morgan_fp(smiles: str, radius: int = _DEFAULT_RADIUS, n_bits: int = _DEFAULT_NBITS) -> Any:
    """Morgan (ECFP-like) fingerprint bit vector, or ``None`` if invalid."""
    return _morgan_fp(smiles, radius, n_bits)


def tanimoto(
    smiles_a: str,
    smiles_b: str,
    radius: int = _DEFAULT_RADIUS,
    n_bits: int = _DEFAULT_NBITS,
) -> float | None:
    """Morgan-fingerprint Tanimoto similarity in ``[0, 1]``.

    Returns ``None`` if either SMILES is invalid.
    """
    fp_a = _morgan_fp(smiles_a, radius, n_bits)
    fp_b = _morgan_fp(smiles_b, radius, n_bits)
    if fp_a is None or fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))
