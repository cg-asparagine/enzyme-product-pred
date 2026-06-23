"""Streamlit model-explorer GUI. Launch with ``just gui`` (or
``uv run streamlit run app/streamlit_app.py``).

Flow: pick a model → pick a reaction from a dataset split (or build a custom one) →
see the enzyme, the substrate, and the true products → hit Predict to run the
trained model and compare its predicted product structures side by side.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit runs this file by path, putting only app/ on sys.path; add the repo
# root so the `app` package (and its sibling modules) import cleanly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from app import registry, render  # noqa: E402
from app.registry import (  # noqa: E402
    DATASETS,
    MODELS,
    SPLIT_COLUMNS,
    ReactionEntry,
)

st.set_page_config(page_title="Enzyme reaction-product explorer", page_icon="🧬", layout="wide")

_PLACEHOLDER_SMILES = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin, as a generic substrate
_MAX_LISTED = 500  # cap reactions shown in the picker (filter to narrow)


# --------------------------------------------------------------------------- #
# Cached loaders / inference (heavy work runs once per unique input)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model…")
def get_adapter(model_name: str, model_dir: str) -> registry.Adapter:
    return registry.load_adapter(MODELS[model_name], model_dir or None)


@st.cache_data(show_spinner=False)
def dataset_entries(dataset_key: str, split: str, split_col: str) -> list[ReactionEntry]:
    ds = DATASETS[dataset_key]
    return registry.build_entries(registry.load_reactions_df(ds.processed_dir, split, split_col))


@st.cache_data(show_spinner="Indexing enzymes…")
def dataset_enzymes(dataset_key: str) -> list[registry.EnzymeOption]:
    ds = DATASETS[dataset_key]
    return registry.enzyme_options(registry.load_reactions_df(ds.processed_dir, None))


@st.cache_data(show_spinner="Predicting products…")
def predict(
    model_name: str, model_dir: str, reactant: str, uniprot_id: str, sequence: str, k: int
) -> list[str]:
    adapter = get_adapter(model_name, model_dir)
    embedding = adapter.embedding_for(uniprot_id or None, sequence or None)
    return adapter.predict(reactant, embedding, k=k)


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def show_structure(smiles: str, caption: str | None = None, reference: str | None = None) -> None:
    """Render a SMILES (optionally diff-highlighted vs a reference) or warn if invalid."""
    png = render.diff_to_png(reference, smiles) if reference else render.mol_to_png(smiles)
    if png is None:
        st.warning(f"⚠️ Could not parse SMILES:\n\n`{smiles}`")
    else:
        st.image(png, caption=caption, width="stretch")


def show_molecule_set(
    smiles_list: list[str], reference: str | None = None, caption_prefix: str = ""
) -> None:
    """Render each molecule of a (reactant or product) side in a small grid."""
    if not smiles_list:
        st.info("—")
        return
    cols = st.columns(min(3, len(smiles_list)))
    for i, smi in enumerate(smiles_list):
        with cols[i % len(cols)]:
            caption = f"{caption_prefix}{i + 1}" if caption_prefix else None
            show_structure(smi, caption=caption, reference=reference)
            st.code(smi, language=None)


def show_enzyme(entry: ReactionEntry) -> None:
    st.markdown("#### Enzyme & conditions")
    cols = st.columns(4)
    cols[0].markdown(f"**EC number**\n\n`{entry.ec_num or '—'}`")
    cols[1].markdown(f"**Organism**\n\n{entry.organism or '—'}")
    cols[2].markdown(f"**UniProt**\n\n`{entry.uniprot_id or '—'}`")
    cols[3].markdown(f"**Direction**\n\n{entry.direction or '—'}")
    if entry.sequence:
        with st.expander(f"Protein sequence — {entry.seq_len or len(entry.sequence)} aa"):
            st.code(entry.sequence, language=None)


# --------------------------------------------------------------------------- #
# Input-source selection
# --------------------------------------------------------------------------- #
def choose_test_entry(dataset_key: str) -> ReactionEntry | None:
    c1, c2 = st.columns(2)
    split_col = c1.selectbox(
        "Split column", list(SPLIT_COLUMNS), format_func=lambda c: SPLIT_COLUMNS[c]
    )
    split = c2.selectbox("Split", ["test", "valid", "train"], index=0)
    try:
        entries = dataset_entries(dataset_key, split, split_col)
    except Exception as exc:  # missing/unbuilt dataset, schema mismatch, …
        st.error(f"Could not load the dataset split: {exc}")
        return None
    if not entries:
        st.warning("No reactions in this split.")
        return None

    flt = st.text_input("Filter — EC number / organism / UniProt id substring").strip().lower()
    if flt:
        entries = [e for e in entries if flt in f"{e.ec_num} {e.organism} {e.uniprot_id}".lower()]
    if not entries:
        st.info("No reactions match the filter.")
        return None

    shown = entries[:_MAX_LISTED]
    extra = f"; showing first {_MAX_LISTED}" if len(entries) > _MAX_LISTED else ""
    idx = st.selectbox(
        f"Reaction ({len(entries)} match{'es' if len(entries) != 1 else ''}{extra})",
        range(len(shown)),
        format_func=lambda i: shown[i].label,
    )
    return shown[int(idx)]


def custom_entry(dataset_key: str) -> ReactionEntry | None:
    reactant = st.text_input(
        "Reactant SMILES (substrate + any cofactors, dot-joined)",
        placeholder=_PLACEHOLDER_SMILES,
    ).strip()

    mode = st.radio(
        "Enzyme", ["Pick a dataset enzyme", "Paste a protein sequence"], horizontal=True
    )
    uniprot_id = sequence = ec_num = organism = ""
    seq_len = 0
    if mode == "Pick a dataset enzyme":
        try:
            enzymes = dataset_enzymes(dataset_key)
        except Exception as exc:
            st.error(f"Could not load enzymes: {exc}")
            return None
        if not enzymes:
            st.info("No dataset enzymes available.")
            return None
        eidx = st.selectbox(
            f"Enzyme ({len(enzymes)} available)",
            range(len(enzymes)),
            format_func=lambda i: enzymes[i].label,
        )
        enz = enzymes[int(eidx)]
        uniprot_id, sequence = enz.uniprot_id, enz.sequence
        ec_num, organism, seq_len = enz.ec_num, enz.organism, enz.seq_len
    else:
        st.caption(
            "Pasting a raw sequence embeds it with frozen ESM-2 650M on Predict — "
            "a heavy one-off model load/download."
        )
        sequence = st.text_area("Protein sequence (amino acids)", height=100).strip().upper()
        ec_num = st.text_input("EC number (optional, for display)").strip()
        organism = st.text_input("Organism (optional, for display)").strip()
        seq_len = len(sequence)

    if not reactant:
        st.info("Enter a reactant SMILES to begin.")
        return None
    if mode == "Paste a protein sequence" and not sequence:
        st.info("Paste a protein sequence (or switch to a dataset enzyme).")
        return None
    return ReactionEntry(
        index=0,
        reactant_smiles=reactant,
        product_smiles="",
        ec_num=ec_num,
        organism=organism,
        uniprot_id=uniprot_id,
        sequence=sequence,
        seq_len=seq_len,
    )


def choose_entry(dataset_key: str) -> ReactionEntry | None:
    source = st.radio("Input source", ["Dataset split", "Custom reaction"], horizontal=True)
    if source == "Dataset split":
        return choose_test_entry(dataset_key)
    return custom_entry(dataset_key)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
def render_predictions(entry: ReactionEntry, preds: list[str]) -> None:
    if not preds:
        st.warning("Model produced no candidates for this reaction.")
        return
    refs = entry.references
    if refs:
        recovered, all_true = registry.recovered_products(preds, refs)
        st.success(f"Recovered **{len(recovered)} / {len(all_true)}** true product molecule(s).")
    st.caption(
        f"{len(preds)} candidate product side(s), in rank order. "
        "Highlight marks the change vs. the substrate."
    )
    cols = st.columns(min(3, len(preds)))
    for i, pred in enumerate(preds):
        with cols[i % len(cols)]:
            show_structure(pred, reference=entry.reactant_smiles)
            badge = ""
            if refs:
                m = registry.match_products(pred, refs)
                if m.is_exact:
                    badge = "✅ **exact set match**"
                elif m.best_tanimoto is not None:
                    badge = f"Tanimoto to nearest true: **{m.best_tanimoto:.2f}**"
            st.markdown(f"**#{i + 1}** · {badge}" if badge else f"**#{i + 1}**")
            st.code(pred, language=None)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🧬 Enzyme reaction-product explorer")

    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox("Model", list(MODELS))
        spec = MODELS[model_name]
        st.caption("generative · reactants + enzyme → product SMILES")
        st.write(spec.blurb)
        with st.expander("Advanced"):
            model_dir = st.text_input(
                "Checkpoint dir override", value="", placeholder=spec.checkpoint
            ).strip()
            k = st.slider("Candidate products (beam outputs to return)", 1, 10, 5)
        st.divider()
        st.caption(
            "The model loads on first Predict and is cached for the session. "
            "Dataset enzymes reuse precomputed ESM-2 embeddings (fast)."
        )

    dataset_key = spec.dataset
    st.caption(f"Dataset: **{DATASETS[dataset_key].label}**")

    entry = choose_entry(dataset_key)
    if entry is None:
        return

    st.divider()
    show_enzyme(entry)

    st.divider()
    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Substrate (reactants)")
        show_molecule_set(entry.reactants)
    with right:
        st.markdown("#### Ground-truth products")
        if entry.references:
            st.caption(
                f"{len(entry.references)} product(s) — changed region highlighted vs. substrate"
            )
            show_molecule_set(
                entry.references, reference=entry.reactant_smiles, caption_prefix="product "
            )
        else:
            st.info("No ground truth for this input (custom reaction).")

    st.divider()
    sig = (model_name, model_dir, entry.reactant_smiles, entry.uniprot_id, entry.sequence)
    clicked = st.button("🔮 Predict products", type="primary")
    # Keep results visible across reruns until the reaction changes; otherwise wait
    # for an explicit Predict click.
    if not clicked and st.session_state.get("predicted_sig") != sig:
        return
    st.session_state["predicted_sig"] = sig

    st.markdown("### Predicted products")
    try:
        preds = predict(
            model_name, model_dir, entry.reactant_smiles, entry.uniprot_id, entry.sequence, k
        )
    except FileNotFoundError as exc:
        st.error(f"No trained checkpoint found — train the model first.\n\n{exc}")
        return
    except KeyError as exc:
        st.warning(str(exc).strip('"'))
        return
    except Exception as exc:
        st.exception(exc)
        return
    render_predictions(entry, preds)


main()
