from epp_core.chem.tokenize import atom_tokenize, detokenize, reconstructs, tokenize_to_str


def test_roundtrip_reconstructs_input(valid_smiles):
    for s in valid_smiles:
        assert "".join(atom_tokenize(s)) == s, s
        assert reconstructs(s)
        assert detokenize(tokenize_to_str(s)) == s


def test_multichar_atoms_are_single_tokens():
    # The whole point of the atom-level tokenizer: Br/Cl stay intact.
    assert atom_tokenize("CCBr") == ["C", "C", "Br"]
    assert atom_tokenize("ClCCl") == ["Cl", "C", "Cl"]


def test_bracket_atoms_kept_whole():
    assert atom_tokenize("[O-]C(=O)C")[0] == "[O-]"
    assert "[C@H]" in atom_tokenize("C[C@H](N)C(=O)O")
    assert "[nH]" in atom_tokenize("c1cc[nH]c1")


def test_detokenize_removes_spaces():
    assert detokenize("C C ( = O ) O") == "CC(=O)O"
