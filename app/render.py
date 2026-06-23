"""RDKit SMILES → PNG rendering for the GUI (pure, cached, no Streamlit).

Kept free of Streamlit so it can be unit-tested without a UI. Uses RDKit's Cairo
backend, which returns PNG bytes directly — exactly what ``st.image`` accepts.
A SMILES may be dot-joined (a multi-molecule reactant or product side); RDKit
parses it into one molecule with disconnected fragments and draws them together.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from rdkit import Chem
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D


def _draw(
    mol: Any,
    width: int,
    height: int,
    highlight_atoms: list[int] | None = None,
    highlight_bonds: list[int] | None = None,
) -> bytes:
    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms or [],
        highlightBonds=highlight_bonds or [],
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


@lru_cache(maxsize=2048)
def mol_to_png(smiles: str, width: int = 320, height: int = 240) -> bytes | None:
    """PNG bytes for ``smiles``; ``None`` if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return _draw(mol, width, height)


@lru_cache(maxsize=2048)
def diff_to_png(
    reference_smiles: str, target_smiles: str, width: int = 320, height: int = 240
) -> bytes | None:
    """Render ``target_smiles`` with the parts that differ from ``reference_smiles``
    highlighted — i.e. the atoms/bonds outside their maximum common substructure,
    which is where the reaction changes the molecule. Falls back to a plain render
    of the target if either SMILES is unparseable or the MCS cannot be computed.
    """
    target = Chem.MolFromSmiles(target_smiles) if target_smiles else None
    if target is None:
        return None
    reference = Chem.MolFromSmiles(reference_smiles) if reference_smiles else None
    if reference is None:
        return _draw(target, width, height)
    try:
        result = rdFMCS.FindMCS([reference, target], timeout=5)
        pattern = Chem.MolFromSmarts(result.smartsString) if result.smartsString else None
        common = set(target.GetSubstructMatch(pattern)) if pattern is not None else set()
        atoms = [a.GetIdx() for a in target.GetAtoms() if a.GetIdx() not in common]
        bonds = [
            b.GetIdx()
            for b in target.GetBonds()
            if b.GetBeginAtomIdx() not in common or b.GetEndAtomIdx() not in common
        ]
        return _draw(target, width, height, atoms, bonds)
    except Exception:
        # MCS is best-effort decoration; never let it break the (plain) render.
        return _draw(target, width, height)
