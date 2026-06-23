"""Model + dataset registry backing the Streamlit GUI (``app/streamlit_app.py``).

The trained reaction model is wrapped behind a small :class:`Adapter` exposing a
uniform ``predict`` method, so the UI stays model-agnostic. Heavy ML imports
(torch / transformers / the model package) happen lazily inside
:func:`load_adapter` — importing this module is cheap, so it is safe to import
from tests.

Today there is one model shape — ``generative`` (reactant SMILES + enzyme →
ranked product SMILES) — but the registry is structured so more can drop in.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from epp_core.chem.smiles import canonicalize, tanimoto
from epp_core.chem.tokenize import detokenize
from epp_core.data import load_reactions

REPO_ROOT = Path(__file__).resolve().parents[1]

GENERATIVE = "generative"

# Put each model's script dir on sys.path so its uniquely-named inner package
# imports (mirrors tests/models/conftest.py). Cheap — triggers no heavy imports.
_MODEL_DIRS = ("ESM2-650M-frozen-ReactionT5",)
for _name in _MODEL_DIRS:
    _path = REPO_ROOT / "models" / _name
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _abs(path: str) -> str:
    """Resolve a repo-relative path against the repo root (absolute paths pass through)."""
    p = Path(path)
    return str(p if p.is_absolute() else REPO_ROOT / p)


# --------------------------------------------------------------------------- #
# Static specs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DatasetSpec:
    key: str
    processed_dir: str
    label: str


DATASETS: dict[str, DatasetSpec] = {
    "EnzymeMap_with_seq": DatasetSpec(
        "EnzymeMap_with_seq",
        "data/EnzymeMap_with_seq/processed",
        "EnzymeMap v2 / BRENDA 2023 — reactions with enzyme sequences",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    package: str  # uniquely-named inner package, e.g. "esm2_reactiont5"
    config_class: str  # config dataclass name in <package>.config
    checkpoint: str  # default model dir (a from_pretrained_dir-style layout)
    dataset: str  # default DatasetSpec key (its test split is the "corresponding test set")
    blurb: str


MODELS: dict[str, ModelSpec] = {
    "ESM2-650M-frozen-ReactionT5": ModelSpec(
        "ESM2-650M-frozen-ReactionT5",
        GENERATIVE,
        "esm2_reactiont5",
        "TrainConfig",
        "models/ESM2-650M-frozen-ReactionT5/checkpoints",
        "EnzymeMap_with_seq",
        "ReactionT5 fine-tuned with a frozen ESM-2 650M enzyme-sequence embedding "
        "prepended to the encoder as a soft enzyme token.",
    ),
}

#: Split columns a processed reactions table may carry, with a human label. The
#: reaction-grouped ``split`` is the v1 split; ``enzyme_split`` is the honest
#: new-enzyme (sequence-cluster) split.
SPLIT_COLUMNS: dict[str, str] = {
    "split": "Reaction split (random, grouped on reaction)",
    "enzyme_split": "Enzyme split (new-enzyme generalization)",
}


# --------------------------------------------------------------------------- #
# Dataset → selectable entries
# --------------------------------------------------------------------------- #
@dataclass
class ReactionEntry:
    """One selectable reaction: its substrate/products plus the enzyme + conditions."""

    index: int
    reactant_smiles: str  # may be dot-joined (multiple substrate/cofactor molecules)
    product_smiles: str  # ground-truth product side (dot-joined); "" if unknown
    ec_num: str = ""
    organism: str = ""
    uniprot_id: str = ""
    sequence: str = ""
    seq_len: int = 0
    direction: str = ""
    reaction_id: str = ""

    @property
    def reactants(self) -> list[str]:
        return [m for m in self.reactant_smiles.split(".") if m]

    @property
    def references(self) -> list[str]:
        return [m for m in self.product_smiles.split(".") if m]

    @property
    def label(self) -> str:
        smi = self.reactant_smiles
        if len(smi) > 42:
            smi = smi[:39] + "…"
        ec = f"EC {self.ec_num}" if self.ec_num else "EC —"
        org = self.organism[:24] if self.organism else "—"
        uid = self.uniprot_id or "—"
        return f"#{self.index} · {ec} · {org} · {uid} — {smi}"


@dataclass(frozen=True)
class EnzymeOption:
    """A dataset enzyme selectable for a custom reaction (reuses its cached embedding)."""

    uniprot_id: str
    ec_num: str
    organism: str
    sequence: str
    seq_len: int

    @property
    def label(self) -> str:
        ec = f"EC {self.ec_num}" if self.ec_num else "EC —"
        return f"{self.uniprot_id} · {ec} · {self.organism[:30]} ({self.seq_len} aa)"


def load_reactions_df(
    processed_dir: str, split: str | None, split_col: str = "split"
) -> pd.DataFrame:
    """Load a processed reactions table, optionally filtered to one split."""
    df = load_reactions(_abs(processed_dir), split, split_col)
    return cast(pd.DataFrame, df)


def build_entries(df: pd.DataFrame) -> list[ReactionEntry]:
    """Turn a reactions dataframe into selectable entries (one per reaction row).

    Requires a ``reactant_smiles`` column; ``product_smiles`` and the enzyme
    columns (``ec_num``, ``organism``, ``uniprot_id``, ``sequence``, ``seq_len``,
    ``direction``, ``reaction_id``) are used when present.
    """
    if "reactant_smiles" not in df.columns:
        raise ValueError("Expected a 'reactant_smiles' column.")
    entries: list[ReactionEntry] = []
    for i, row in enumerate(df.to_dict("records")):
        entries.append(
            ReactionEntry(
                index=i,
                reactant_smiles=str(row.get("reactant_smiles", "")),
                product_smiles=str(row.get("product_smiles", "")),
                ec_num=str(row.get("ec_num", "")),
                organism=str(row.get("organism", "")),
                uniprot_id=str(row.get("uniprot_id", "")),
                sequence=str(row.get("sequence", "")),
                seq_len=int(row.get("seq_len", 0) or 0),
                direction=str(row.get("direction", "")),
                reaction_id=str(row.get("reaction_id", "")),
            )
        )
    return entries


def enzyme_options(df: pd.DataFrame, limit: int = 400) -> list[EnzymeOption]:
    """Distinct enzymes (by UniProt id) from a reactions table, for the custom picker."""
    cols = [
        c for c in ("uniprot_id", "ec_num", "organism", "sequence", "seq_len") if c in df.columns
    ]
    if "uniprot_id" not in cols:
        return []
    seen: dict[str, EnzymeOption] = {}
    for row in cast(pd.DataFrame, df[cols]).to_dict("records"):
        uid = str(row.get("uniprot_id", ""))
        if not uid or uid in seen:
            continue
        seq = str(row.get("sequence", ""))
        seen[uid] = EnzymeOption(
            uniprot_id=uid,
            ec_num=str(row.get("ec_num", "")),
            organism=str(row.get("organism", "")),
            sequence=seq,
            seq_len=int(row.get("seq_len", 0) or 0) or len(seq),
        )
        if len(seen) >= limit:
            break
    return list(seen.values())


# --------------------------------------------------------------------------- #
# Prediction comparison helpers (pure; used by the results view)
# --------------------------------------------------------------------------- #
def _canon_set(smiles: list[str]) -> set[str]:
    return {c for c in (canonicalize(s) for s in smiles) if c is not None}


@dataclass
class Match:
    is_exact: bool  # predicted product SET equals the true product SET (order-independent)
    best_tanimoto: float | None  # similarity of the predicted side to its nearest true product


def match_products(prediction: str, references: list[str]) -> Match:
    """How a predicted product side relates to the true products: exact (canonical)
    set match, and the closest true product by Morgan-Tanimoto similarity."""
    pred_set = _canon_set(prediction.split("."))
    if not references:
        return Match(False, None)
    ref_set = _canon_set(references)
    is_exact = bool(pred_set) and pred_set == ref_set
    best = -1.0
    for ref in references:
        sim = tanimoto(prediction, ref)
        if sim is not None and sim > best:
            best = sim
    return Match(is_exact, best if best >= 0 else None)


def recovered_products(predictions: list[str], references: list[str]) -> tuple[set[str], set[str]]:
    """``(recovered, all_true)`` canonical product molecules — recovered are the true
    products appearing anywhere in the predicted candidate set."""
    ref_set = _canon_set(references)
    pred_set = _canon_set([m for p in predictions for m in p.split(".")])
    return ref_set & pred_set, ref_set


# --------------------------------------------------------------------------- #
# Loaded-model adapter
# --------------------------------------------------------------------------- #
@dataclass
class Adapter:
    spec: ModelSpec
    checkpoint_dir: str
    model: Any
    tokenizer: Any
    config: Any
    device: str
    embeddings: dict[str, np.ndarray]
    data_mod: Any  # esm2_reactiont5.data (format_input)
    emb_mod: Any  # esm2_reactiont5.embeddings (embed_sequences, select_device)

    @property
    def kind(self) -> str:
        return self.spec.kind

    def has_embedding(self, uniprot_id: str) -> bool:
        return bool(uniprot_id) and uniprot_id in self.embeddings

    def embedding_for(self, uniprot_id: str | None, sequence: str | None) -> np.ndarray:
        """Embedding for an enzyme: the cached vector for a known UniProt id, else
        embed ``sequence`` live with ESM-2 (a heavy, one-off model load)."""
        if uniprot_id and uniprot_id in self.embeddings:
            return self.embeddings[uniprot_id]
        if sequence:
            vectors = self.emb_mod.embed_sequences([sequence], device=self.device)
            return cast(np.ndarray, vectors[0])
        raise KeyError("No cached embedding for this enzyme and no sequence to embed.")

    def predict(
        self,
        reactant_smiles: str,
        embedding: np.ndarray,
        *,
        k: int | None = None,
        num_beams: int | None = None,
    ) -> list[str]:
        """Ranked product-side SMILES for one reaction, conditioned on the enzyme embedding."""
        import torch

        cfg = self.config
        k = k or cfg.num_return_sequences
        num_beams = max(num_beams or cfg.num_beams, k)
        enc = self.tokenizer(
            [self.data_mod.format_input(reactant_smiles)],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_input_length,
        ).to(self.device)
        emb = torch.as_tensor(
            np.asarray(embedding, dtype=np.float32)[None, :],
            dtype=torch.float32,
            device=self.device,
        )
        generated = self.model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            protein_embedding=emb,
            num_beams=num_beams,
            num_return_sequences=k,
            max_length=cfg.max_target_length,
        )
        # ReactionT5's tokenizer can emit internal spaces when decoding; a space
        # terminates a SMILES for RDKit, so strip whitespace from each candidate.
        decoded = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return [detokenize(s) for s in decoded]


def load_adapter(spec: ModelSpec, model_dir: str | None = None) -> Adapter:
    """Lazily import the model's package, load the trained checkpoint + embedding
    cache, and wrap it. This is the heavy call (imports torch/transformers and
    reads weights); the GUI caches it with ``st.cache_resource``.
    """
    from transformers import AutoTokenizer

    config_mod = cast(Any, importlib.import_module(f"{spec.package}.config"))
    model_mod = cast(Any, importlib.import_module(f"{spec.package}.model"))
    data_mod = cast(Any, importlib.import_module(f"{spec.package}.data"))
    emb_mod = cast(Any, importlib.import_module(f"{spec.package}.embeddings"))

    config = getattr(config_mod, spec.config_class)()
    resolved = _abs(model_dir or spec.checkpoint)
    device = emb_mod.select_device() if config.device == "auto" else config.device

    tokenizer = AutoTokenizer.from_pretrained(resolved)
    model = model_mod.ReactionT5WithProtein.from_pretrained_dir(resolved, config.esm_dim)
    model.to(device)
    model.eval()
    embeddings = emb_mod.load_embeddings(_abs(config.embedding_cache))
    return Adapter(spec, resolved, model, tokenizer, config, device, embeddings, data_mod, emb_mod)
