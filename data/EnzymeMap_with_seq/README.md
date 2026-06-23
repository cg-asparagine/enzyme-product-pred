# EnzymeMap_with_seq

EnzymeMap reactions restricted to those with a **UniProt enzyme**, with the
protein **amino-acid sequence** attached. An example is a unique
**(canonical reaction, UniProt ID)** pair — the substrate/cofactor → product
transformation together with the sequence of the enzyme that catalyzes it.

This is the sequence-conditioned cut of [`EnzymeMap`](../EnzymeMap/README.md);
it reuses the same raw CSV and the same reaction-building / split logic.

## How it's built (`prepare.py`)

```
just data-prep EnzymeMap_with_seq
```

1. Read EnzymeMap's raw CSV (`../EnzymeMap/raw/enzymemap_v2_brenda2023.csv`).
2. Keep rows with `steps == "single"`, `quality >= 0.3`, and
   `protein_db ∈ {uniprot, swissprot}` (both are UniProtKB accessions; `genbank`
   is excluded). Rows listing several accessions are exploded to one row each.
3. Build canonical reactions and **dedupe on (reaction, UniProt ID)** so each
   reaction–enzyme pair appears once.
4. **Fetch the sequence** for every unique accession from the UniProt REST API
   (`epp_core.data.fetch_sequences`), cached to `raw/uniprot_sequences.json`
   (git-ignored; written per batch, so re-runs resume and don't re-hit the API).
5. **Recover** accessions the live endpoint drops (obsolete / secondary) from the
   **UniParc archive** (`epp_core.data.uniparc_sequences`, one lookup per missing
   accession, cached to `raw/uniparc_sequences.json`). UniParc keeps every
   sequence ever seen, so almost all are recovered; the few that remain are
   dropped. Counts are in the manifest (`n_resolved_direct`,
   `n_recovered_uniparc`, `n_dropped_no_seq`).
6. Assign the v1 grouped-random 80/10/10 split (seed 42) on the direction-collapsed
   reaction identity → **`split`** (forward/reverse twins stay together).
7. Cluster enzymes by sequence similarity (`epp_core.data.cluster_sequences`,
   k-mer Jaccard) and assign a second grouped 80/10/10 split over whole clusters →
   **`enzyme_split`** (homologous enzymes never straddle splits).
8. Write `processed/reactions.parquet` + `build_manifest.json` (both git-ignored).

## Processed schema (`reactions.parquet`)

Everything in [`EnzymeMap`](../EnzymeMap/README.md), plus the enzyme columns:

`reaction_id`, `reactant_smiles`, `product_smiles`, `source`, `n_reactants`,
`n_products`, `src_n_tokens`, `tgt_n_tokens`, `split`, `rxn_idx`, `ec_num`,
`organism`, `direction`, `quality`, `natural`, `steps`, **`uniprot_id`**,
**`sequence`** (amino-acid string), **`seq_len`**, **`enzyme_cluster`** (int
sequence-similarity cluster id), **`enzyme_split`** (train/valid/test over enzyme
clusters).

## Two splits — which to use

| column | held-out unit | leakage controlled | use it to measure |
|--------|---------------|--------------------|-------------------|
| `split` | undirected reaction | reaction identity | generalization to **new reactions** |
| `enzyme_split` | enzyme cluster (k-mer Jaccard) | enzyme + close homologs | generalization to **new enzymes** |

For a sequence-conditioned model, `enzyme_split` is the honest test: its train and
test enzymes are non-homologous, so the model can't score by recognizing an enzyme
it trained on. The trade-off is that some reactions recur across `enzyme_split`
sets (only enzymes are held out). The clustering method/threshold and per-split
counts are recorded in `build_manifest.json → enzyme_split`. Render a full
breakdown with `just data-report EnzymeMap_with_seq`.

## Notes / limitations

- `sequence` comes from UniProtKB (live `accessions` endpoint), with obsolete /
  secondary accessions recovered from the UniParc archive. After both, only a
  couple of accessions remain unresolvable; those pairs are dropped. See
  `build_manifest.json` → `n_resolved_direct` / `n_recovered_uniparc` /
  `n_dropped_no_seq`.
- No sequence-length filter is applied yet; use `seq_len` to filter if needed.
- Conditioning a model on the sequence (e.g. an ESM/ProtT5 encoder alongside the
  reaction) is the motivating use case — see the roadmap in
  [CLAUDE.md](../../CLAUDE.md).

## Load

```python
from epp_core.data import load_reactions

# reaction split (default)
train = load_reactions("data/EnzymeMap_with_seq/processed", split="train")
# new-enzyme generalization test
train = load_reactions("data/EnzymeMap_with_seq/processed", split="train", split_col="enzyme_split")
train[["reactant_smiles", "product_smiles", "uniprot_id", "sequence"]].head()
```
