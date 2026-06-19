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
5. Drop pairs whose accession didn't resolve (obsolete / demerged / secondary
   accessions — the count is recorded in the manifest).
6. Assign a grouped-random 80/10/10 split (seed 42) on the direction-collapsed
   reaction identity (forward/reverse twins stay together).
7. Write `processed/reactions.parquet` + `build_manifest.json` (both git-ignored).

## Processed schema (`reactions.parquet`)

Everything in [`EnzymeMap`](../EnzymeMap/README.md), plus the enzyme columns:

`reaction_id`, `reactant_smiles`, `product_smiles`, `source`, `n_reactants`,
`n_products`, `src_n_tokens`, `tgt_n_tokens`, `split`, `rxn_idx`, `ec_num`,
`organism`, `direction`, `quality`, `natural`, `steps`, **`uniprot_id`**,
**`sequence`** (amino-acid string), **`seq_len`**.

## Notes / limitations

- `sequence` comes from UniProtKB via the accession in `protein_refs`. Some
  accessions don't resolve (obsolete/demerged, or secondary accessions the
  `accessions` endpoint doesn't redirect); those pairs are dropped. See
  `build_manifest.json` → `n_accessions_resolved` / `n_dropped_no_seq`.
- No sequence-length filter is applied yet; use `seq_len` to filter if needed.
- Conditioning a model on the sequence (e.g. an ESM/ProtT5 encoder alongside the
  reaction) is the motivating use case — see the roadmap in
  [CLAUDE.md](../../CLAUDE.md).

## Load

```python
from epp_core.data import load_reactions

train = load_reactions("data/EnzymeMap_with_seq/processed", split="train")
train[["reactant_smiles", "product_smiles", "uniprot_id", "sequence"]].head()
```
