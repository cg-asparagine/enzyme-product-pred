"""Reaction-SMILES helpers built on the single SMILES source of truth.

EnzymeMap stores reactions as ``reactants>>products`` SMILES, dot-separated on
each side. These utilities split a reaction into its molecule lists,
canonicalize each side, and derive deterministic keys for deduplication and for
leakage checks (EnzymeMap includes the reverse of every reaction, so a reaction
and its reverse must hash to the same *undirected* key).
"""

from __future__ import annotations

from epp_core.chem.smiles import canonicalize

#: Separator between the two sides of an undirected reaction key (never appears
#: in SMILES, so it can't collide with molecule content).
_UNDIRECTED_SEP = "|"


def split_reaction(reaction: str) -> tuple[list[str], list[str]] | None:
    """Split ``"r1.r2>>p1.p2"`` into ``(["r1", "r2"], ["p1", "p2"])``.

    Returns ``None`` if the string is not a single ``>>`` reaction with at least
    one molecule on each side. Agent/reagent blocks (``r>a>p``) are absent from
    EnzymeMap and are treated as malformed.
    """
    parts = reaction.split(">>")
    if len(parts) != 2:
        return None
    reactants = [m for m in parts[0].split(".") if m]
    products = [m for m in parts[1].split(".") if m]
    if not reactants or not products:
        return None
    return reactants, products


def canonical_molecules(molecules: list[str]) -> list[str] | None:
    """Canonicalize each molecule via :func:`epp_core.chem.smiles.canonicalize`.

    Returns ``None`` if *any* molecule fails to parse, so a reaction with one bad
    component is dropped wholesale rather than silently truncated.
    """
    out: list[str] = []
    for mol in molecules:
        canon = canonicalize(mol)
        if canon is None:
            return None
        out.append(canon)
    return out


def join_molecules(molecules: list[str]) -> str:
    """Dot-join a molecule list, sorted, so the result is order-independent."""
    return ".".join(sorted(molecules))


def reaction_key(reactants: list[str], products: list[str]) -> str:
    """Direction-aware dedup key ``"<sorted reactants>>><sorted products>"``.

    Rows with the same reactant set and product set collapse; the reverse
    reaction gets a *different* key (use :func:`undirected_reaction_key` to
    collapse direction).
    """
    return f"{join_molecules(reactants)}>>{join_molecules(products)}"


def undirected_reaction_key(reactants: list[str], products: list[str]) -> str:
    """Direction-collapsed key: a reaction and its reverse map to the same value.

    Used to keep forward/reverse twins together in one train/valid/test split.
    """
    a = join_molecules(reactants)
    b = join_molecules(products)
    lo, hi = sorted((a, b))
    return f"{lo}{_UNDIRECTED_SEP}{hi}"
