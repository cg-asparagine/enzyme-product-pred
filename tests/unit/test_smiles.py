from epp_core.chem.smiles import canonicalize, is_valid, num_atoms, tanimoto


def test_is_valid_true_for_valid(valid_smiles):
    assert all(is_valid(s) for s in valid_smiles)


def test_is_valid_false_for_invalid(invalid_smiles):
    assert not any(is_valid(s) for s in invalid_smiles)


def test_canonicalize_idempotent(valid_smiles):
    for s in valid_smiles:
        c = canonicalize(s)
        assert c is not None
        assert canonicalize(c) == c


def test_canonicalize_order_invariant():
    assert canonicalize("OCC") == canonicalize("CCO")
    assert canonicalize("C1=CC=CC=C1") == canonicalize("c1ccccc1")


def test_canonicalize_invalid_returns_none():
    assert canonicalize("(((") is None
    assert canonicalize("") is None


def test_tanimoto_self_is_one(valid_smiles):
    for s in valid_smiles:
        assert tanimoto(s, s) == 1.0


def test_tanimoto_bounds():
    sim = tanimoto("CCO", "CCCO")
    assert sim is not None
    assert 0.0 <= sim <= 1.0


def test_tanimoto_invalid_returns_none():
    assert tanimoto("CCO", "(((") is None


def test_num_atoms_counts_heavy_atoms():
    assert num_atoms("CCO") == 3
    assert num_atoms("(((") is None
