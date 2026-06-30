# EnzymeMap_with_seq_plus

A **superset** of [`EnzymeMap_with_seq`](../EnzymeMap_with_seq/README.md): the same
sequence-conditioned (canonical reaction, UniProt enzyme) examples, **plus**
reactions whose enzyme had **no curated UniProt accession** in the raw CSV, recovered
by resolving each reaction's **(EC number, organism)** to candidate UniProtKB
accessions. Only ~31% of EnzymeMap rows carry an accession; this dataset adds back a
large fraction of the rest, roughly tripling the sequence-conditioned data.

An example is still a unique **(canonical reaction, UniProt ID)** pair with the
enzyme's amino-acid `sequence` attached.

It additionally folds in curated **cytochrome-P450 (CYP) reactions across organisms**
(`accession_source == "cyp"`). **Non-human** CYP reactions are training augmentation;
**human** CYP reactions are **held out for testing** (forced into `split` /
`enzyme_split == "test"`). See [Cytochrome-P450 reactions](#cytochrome-p450-cyp-reactions).

## Provenance — the `accession_source` column

Every row is tagged:

| `accession_source` | meaning |
|--------------------|---------|
| `curated`  | accession came from the raw CSV's `protein_refs` — a direct BRENDA → UniProt link (identical to `EnzymeMap_with_seq`). |
| `resolved` | the row had no accession; its `(EC, organism)` was looked up in `resolved_accessions.csv` (built by `../EnzymeMap_with_seq/resolve_accessions.py` via `epp_core.data.search_accessions`) and the best-reviewed candidate attached. |
| `cyp`      | an all-organism cytochrome-P450 reaction from `raw/cyp_reactions.csv`; the enzyme `sequence` is **inline** in the source (no UniProt fetch), and `organism_class` tags human/animal/plant/microorganism. |

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
4. **CYP records**: read `raw/cyp_reactions.csv` (all-organism CYP), take each row's
   primary `Substrate1 → Product1`, attach the inline enzyme sequence (`blast`), and tag
   `accession_source = "cyp"` + `organism_class` (human/animal/plant/microorganism).
5. Build canonical reactions and **dedupe on (reaction, UniProt ID)**.
6. **Fetch sequences** only for rows lacking an inline one (curated/resolved); UniProtKB +
   UniParc fallback, reusing `EnzymeMap_with_seq`'s caches; drop pairs that don't resolve.
7. Assign the grouped-random 80/10/10 `split` (seed 42) on the direction-collapsed
   reaction identity (forward/reverse twins stay together).
8. Cluster enzymes by sequence similarity (k-mer Jaccard) and assign a second grouped
   80/10/10 `enzyme_split` over whole clusters (homologous enzymes never straddle).
9. **Reserve human CYP for test**: force every human CYP reaction's whole reaction-group
   and whole enzyme-cluster to `test` in both split columns (leakage-free whole-group moves).
10. Write `processed/reactions.parquet` + `build_manifest.json` (both git-ignored).

## Processed schema (`reactions.parquet`)

Everything in [`EnzymeMap_with_seq`](../EnzymeMap_with_seq/README.md) plus two columns:

`reaction_id`, `reactant_smiles`, `product_smiles`, `source`, `n_reactants`,
`n_products`, `src_n_tokens`, `tgt_n_tokens`, `split`, `rxn_idx`, `ec_num`,
`organism`, `direction`, `quality`, `natural`, `steps`, `uniprot_id`, `sequence`,
`seq_len`, `enzyme_cluster`, `enzyme_split`, **`accession_source`** (`curated` |
`resolved` | `cyp`), **`organism_class`** (`human` | `animal` | `plant` |
`microorganism` for CYP rows; `""` otherwise).

For CYP rows: `source = "allorganism-cyp"`, `ec_num` may be empty (`""`, ~28% of CYP
rows), `uniprot_id` is the row's UniProt accession (or the CYP `Symbol` when absent),
and `rxn_idx` is synthetic (≥ `100_000_000`).

The manifest records `accession_source_counts`, `multi_accession`, a `resolved_input`
block (resolved-table sha256, pairs resolved, distinct accessions), and a **`cyp`** block
(raw-file sha256, rows loaded/kept, non-human/human counts, `n_human_forced_test`, and
`n_rows_pulled_into_test_by_human_collision`).

## Two splits — which to use

| column | held-out unit | leakage controlled | use it to measure |
|--------|---------------|--------------------|-------------------|
| `split` | undirected reaction | reaction identity | generalization to **new reactions** |
| `enzyme_split` | enzyme cluster (k-mer Jaccard) | enzyme + close homologs | generalization to **new enzymes** |

For a sequence-conditioned model, `enzyme_split` is the honest test. See the curated
dataset's README for the full rationale; `just data-report EnzymeMap_with_seq_plus`
renders a breakdown.

## Cytochrome-P450 (CYP) reactions

`raw/cyp_reactions.csv` is the curated **all-organism CYP** reaction table (each row's
primary `Substrate1 → Product1`, enzyme sequence inline in `blast`), copied from the
sibling `metabolite-prediction` repo. `raw/` is git-ignored; regenerate it with:

```bash
cp <…>/metabolite-prediction/data/AllOrganism-CYP-v1/raw/reactions.csv \
   data/EnzymeMap_with_seq_plus/raw/cyp_reactions.csv
```

(SHA-256 `befcbbd3…fde23`, recorded in `build_manifest.json → cyp.raw_sha256`; the file is
latin-1 / cp1252 encoded.) Of **3,589** rows, **3,474** survive the build (canonicalize /
length / dedup): **2,920 non-human** (microorganism 1,576 · plant 1,025 · animal 316 · 3
unspecified) + **554 human**.

- **Non-human** CYP rows are **training augmentation** — they flow through the normal
  grouped `split` / `enzyme_split` like every other row.
- **Human** CYP rows are **reserved for testing**: each is forced into `split == "test"`
  *and* `enzyme_split == "test"`. The forcing moves whole reaction-groups / enzyme-clusters,
  so neither split leaks; a few non-human / EnzymeMap rows that share a reaction with a human
  CYP row are pulled along to test (`cyp.n_rows_pulled_into_test_by_human_collision`).

The held-out human CYP eval set is exactly `(accession_source == "cyp") &
(organism_class == "human")` (see [Load](#load) below).

> [!NOTE]
> CYP reactions are **off-distribution** vs general EnzymeMap: cofactor-free
> `substrate >> product` (no O₂ / NADPH), noisy family-level enzyme labels, and (for the
> human holdout) drug metabolism rather than biosynthesis. Read scores on the human CYP
> set as a **generalization signal**, not an apples-to-apples EnzymeMap metric.

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

# held-out human CYP reactions (test in both split columns)
test = load_reactions("data/EnzymeMap_with_seq_plus/processed", split="test")
human_cyp = test[(test["accession_source"] == "cyp") & (test["organism_class"] == "human")]
```
