"""Train/valid/test split assignment.

For v1 we use a simple **grouped-random** split: whole groups are assigned to
train/valid/test, so all rows sharing a group key land in the same split. Keying
the group on the direction-collapsed reaction identity
(:func:`epp_core.chem.reactions.undirected_reaction_key`) means a reaction's
forward/reverse twins — and the same reaction reported for different organisms —
never straddle the split, which is the dominant leakage mode in EnzymeMap.

Roadmap: scaffold- and similarity-based splits (compound similarity via
Morgan/Tanimoto and, once enzyme sequences are available, sequence similarity)
for honest generalization tests. They will live here alongside
``grouped_random_split`` behind a strategy argument.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence

import pandas as pd


def grouped_random_split(
    groups: Sequence[Hashable],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[Hashable, str]:
    """Map each unique group key to ``"train"`` / ``"valid"`` / ``"test"``.

    Whole groups are assigned together (never split across sets) and the
    assignment is deterministic given ``seed``. ``fractions`` apply to the count
    of *unique groups*, not rows.
    """
    if abs(sum(fractions) - 1.0) >= 1e-9:
        raise ValueError(f"fractions must sum to 1.0, got {fractions}")
    unique = sorted(set(groups), key=str)
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_train = int(n * fractions[0])
    n_valid = int(n * fractions[1])
    assignment: dict[Hashable, str] = {}
    for i, group in enumerate(unique):
        if i < n_train:
            assignment[group] = "train"
        elif i < n_train + n_valid:
            assignment[group] = "valid"
        else:
            assignment[group] = "test"
    return assignment


def assign_splits(
    df: pd.DataFrame,
    group_col: str,
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    split_col: str = "split",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split_col`` column (default ``"split"``)
    assigned by :func:`grouped_random_split` over the values in ``df[group_col]``.

    Whole groups are kept together, so ``group_col`` never straddles the partition.
    Passing a distinct ``split_col`` lets several independent splits — e.g. a
    reaction-grouped ``split`` and an enzyme-cluster ``enzyme_split`` — coexist as
    separate columns of the same dataframe.
    """
    mapping = grouped_random_split(df[group_col].tolist(), fractions, seed)
    out = df.copy()
    out[split_col] = out[group_col].map(lambda group: mapping[group])
    return out
