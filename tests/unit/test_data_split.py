import pandas as pd
import pytest

from epp_core.data.split import assign_splits, grouped_random_split


def test_grouped_random_split_assigns_whole_groups():
    groups = [f"g{i}" for i in range(100)]
    mapping = grouped_random_split(groups, fractions=(0.8, 0.1, 0.1), seed=0)
    assert set(mapping.values()) <= {"train", "valid", "test"}
    counts = {s: sum(1 for v in mapping.values() if v == s) for s in ("train", "valid", "test")}
    assert counts == {"train": 80, "valid": 10, "test": 10}


def test_grouped_random_split_is_deterministic():
    groups = [f"g{i}" for i in range(50)]
    assert grouped_random_split(groups, seed=7) == grouped_random_split(groups, seed=7)


def test_grouped_random_split_seed_changes_assignment():
    groups = [f"g{i}" for i in range(50)]
    assert grouped_random_split(groups, seed=1) != grouped_random_split(groups, seed=2)


def test_assign_splits_keeps_group_together():
    # Three rows share group "A" (a forward/reverse/organism cluster); they must
    # all land in the same split.
    df = pd.DataFrame({"key": ["A", "A", "A", "B", "C", "D", "E", "F", "G", "H"], "x": range(10)})
    out = assign_splits(df, group_col="key", fractions=(0.6, 0.2, 0.2), seed=3)
    assert len(set(out.loc[out["key"] == "A", "split"])) == 1
    assert set(out["split"]) <= {"train", "valid", "test"}


def test_grouped_random_split_rejects_bad_fractions():
    with pytest.raises(ValueError, match="sum to 1.0"):
        grouped_random_split(["a", "b"], fractions=(0.5, 0.3, 0.1))
