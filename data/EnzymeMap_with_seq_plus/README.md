# EnzymeMap_with_seq_plus

A **superset** of [`EnzymeMap_with_seq`](../EnzymeMap_with_seq/README.md): the same
sequence-conditioned (canonical reaction, UniProt enzyme) examples, **plus**
reactions whose enzyme had **no curated UniProt accession** in the raw CSV, recovered
by resolving each reaction's **(EC number, organism)** to candidate UniProtKB
accessions. Only ~31% of EnzymeMap rows carry an accession; this dataset adds back a
large fraction of the rest, roughly tripling the sequence-conditioned data.

An example is still a unique **(canonical reaction, UniProt ID)** pair with the
enzyme's amino-acid `sequence` attached.

## Provenance — the `accession_source` column

Every row is tagged:

| `accession_source` | meaning |
|--------------------|---------|
| `curated`  | accession came from the raw CSV's `protein_refs` — a direct BRENDA → UniProt link (identical to `EnzymeMap_with_seq`). |
| `resolved` | the row had no accession; its `(EC, organism)` was looked up in `resolved_accessions.csv` (built by `../EnzymeMap_with_seq/resolve_accessions.py` via `epp_core.data.search_accessions`) and the best-reviewed candidate attached. |

On a `(reaction, uniprot_id)` collision between the two sources, **curated wins** (the
builder feeds curated records first and dedup keeps the first-seen).

> [!IMPORTANT]
> **Resolved accessions are *inferred*, not verified.** A resolved accession is a
> candidate enzyme of the correct **(EC, organism)** — right catalytic activity and
> right species — but it is **not confirmed to catalyze the specific reaction**
> recorded. Reviewed/Swiss-Prot entries are preferred; TrEMBL is used only when no
> reviewed entry exists. For a **high-precision** evaluation, filter to
> `accession_source == "curated"`. Use `resolved` rows mainly to **expand training
> coverage**. For a product-prediction task this is usually safe — the EC number
> largely determines the chemistry, so any enzyme of the right (EC, organism) implies
> the same transformation.

## How it's built (`prepare.py`)

```
just data-prep EnzymeMap_with_seq_plus
```

Prereq: `../EnzymeMap_with_seq/processed/resolved_accessions.csv` must exist (run
`uv run python data/EnzymeMap_with_seq/resolve_accessions.py`).

1. Read EnzymeMap's raw CSV. Keep rows with `steps == "single"` and `quality >= 0.3`.
2. **Curated records**: rows with `protein_db ∈ {uniprot, swissprot}`, `protein_refs`
   exploded to one record per accession (`accession_source = "curated"`).
3. **Resolved records**: the complement (accession-less rows with a non-empty
   organism); attach accessions looked up by `(EC, organism)` from the resolved table
   (`accession_source = "resolved"`). When a pair maps to several candidates, the
   **`MULTI_ACCESSION = "pick_one"`** policy keeps only the first (reviewed-first)
   representative; set it to `"explode_all"` for one example per candidate (≤5).
4. Build canonical reactions and **dedupe on (reaction, UniProt ID)**.
5. **Fetch sequences** for every accession (UniProtKB, UniParc fallback), reusing
   `EnzymeMap_with_seq`'s caches; drop pairs whose accession didn't resolve.
6. Assign the grouped-random 80/10/10 `split` (seed 42) on the direction-collapsed
   reaction identity (forward/reverse twins stay together).
7. Cluster enzymes by sequence similarity (k-mer Jaccard) and assign a second grouped
   80/10/10 `enzyme_split` over whole clusters (homologous enzymes never straddle).
8. Write `processed/reactions.parquet` + `build_manifest.json` (both git-ignored).

## Processed schema (`reactions.parquet`)

Everything in [`EnzymeMap_with_seq`](../EnzymeMap_with_seq/README.md) plus one column:

`reaction_id`, `reactant_smiles`, `product_smiles`, `source`, `n_reactants`,
`n_products`, `src_n_tokens`, `tgt_n_tokens`, `split`, `rxn_idx`, `ec_num`,
`organism`, `direction`, `quality`, `natural`, `steps`, `uniprot_id`, `sequence`,
`seq_len`, `enzyme_cluster`, `enzyme_split`, **`accession_source`** (`curated` |
`resolved`).

The manifest records `accession_source_counts`, `multi_accession`, and a
`resolved_input` block (resolved-table sha256, pairs resolved, distinct accessions).

## Two splits — which to use

| column | held-out unit | leakage controlled | use it to measure |
|--------|---------------|--------------------|-------------------|
| `split` | undirected reaction | reaction identity | generalization to **new reactions** |
| `enzyme_split` | enzyme cluster (k-mer Jaccard) | enzyme + close homologs | generalization to **new enzymes** |

For a sequence-conditioned model, `enzyme_split` is the honest test. See the curated
dataset's README for the full rationale; `just data-report EnzymeMap_with_seq_plus`
renders a breakdown.

## Load

```python
from epp_core.data import load_reactions

# everything (curated + resolved)
train = load_reactions("data/EnzymeMap_with_seq_plus/processed", split="train")

# high-precision cut: curated accessions only
train = train[train["accession_source"] == "curated"]

# new-enzyme generalization test
test = load_reactions(
    "data/EnzymeMap_with_seq_plus/processed", split="test", split_col="enzyme_split"
)
```
