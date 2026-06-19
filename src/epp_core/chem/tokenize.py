"""Atom-level SMILES tokenization (Schwaller et al. molecular-transformer scheme).

The regex keeps multi-character atoms (``Br``, ``Cl``) and bracket atoms
(``[nH]``, ``[O-]``) as single tokens, which is exactly the property a stock
word-piece tokenizer lacks. We use it both to measure atom-token length (for
filtering over-long reactions) and to derive the atom tokens added to a base
model's vocabulary.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

#: Atom/symbol-level SMILES tokenizer pattern (single capturing group).
SMILES_TOKEN_PATTERN = (
    r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|"
    r"\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|%[0-9]{2}|[0-9])"
)
_SMILES_REGEX = re.compile(SMILES_TOKEN_PATTERN)


def atom_tokenize(smiles: str) -> list[str]:
    """Split a SMILES string into atom/symbol tokens."""
    return _SMILES_REGEX.findall(smiles)


def tokenize_to_str(smiles: str) -> str:
    """Space-separated tokenization."""
    return " ".join(atom_tokenize(smiles))


def detokenize(tokenized: str) -> str:
    """Inverse of :func:`tokenize_to_str` — simply remove the spaces."""
    return "".join(tokenized.split())


def reconstructs(smiles: str) -> bool:
    """True if the tokenizer round-trips ``smiles`` without losing characters."""
    return "".join(atom_tokenize(smiles)) == smiles


def derive_atom_tokens(corpus_smiles: Iterable[str]) -> list[str]:
    """Multi-character atom/bracket tokens occurring in the corpus (e.g. Br, Cl,
    [nH], [O-]). Single-character symbols are left to the base tokenizer."""
    tokens: set[str] = set()
    for smiles in corpus_smiles:
        tokens.update(tok for tok in atom_tokenize(smiles) if len(tok) > 1)
    return sorted(tokens)


def _roundtrips(tokenizer: Any, token: str) -> bool:
    ids = tokenizer(token, add_special_tokens=False).input_ids
    return detokenize(tokenizer.decode(ids)) == token


def augment_tokenizer(tokenizer: Any, corpus_smiles: Iterable[str]) -> list[str]:
    """Add only the atom tokens a base tokenizer fails to round-trip; return them.

    For SMILES-native tokenizers (MolT5, ReactionT5) this adds nothing, so the
    model embeddings don't need resizing. The conditional augmentation remains
    for a future base model whose tokenizer splits multi-character atoms.
    """
    existing = set(tokenizer.get_vocab())
    candidates = [tok for tok in derive_atom_tokens(corpus_smiles) if tok not in existing]
    to_add = [tok for tok in candidates if not _roundtrips(tokenizer, tok)]
    if to_add:
        tokenizer.add_tokens(to_add)
    return to_add
