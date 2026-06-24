# Victorien-CYP-metabolites

An **evaluation-only** benchmark (all rows `split="test"`) of human drug
parent→metabolite pairs, each conditioned on the cytochrome-P450 (CYP) isoform
that produced it. Built for the sequence-conditioned model
`ESM2-650M-frozen-ReactionT5`.

## Why

It is the **experimental-only test split** of the `Victorien-zaretzki-drugbank-v1`
dataset used to evaluate `Metatrans-ReactionT5` in the sibling
`metabolite-prediction` repo. Metatrans is enzyme-blind (it sees only
`REACTANT:<parent>REAGENT:` → `<metabolite>`); here we attach the metabolizing
enzyme so the same held-out drug-metabolism pairs can be scored by an
enzyme-conditioned model and the two approaches compared on the same molecules.

## Provenance

`data/*/raw/` is git-ignored (raw is regenerable; provenance lives here + in
`build_manifest.json`, which records the raw SHA-256 and the dataset content hash).

- **`raw/metabolite_test_pairs.csv`** (external input) — the 233 experimental test
  pairs exported from
  `metabolite-prediction/data/Victorien-zaretzki-drugbank-v1/processed/pairs.parquet`
  (`split == "test"`). Columns: `pair_id`, `parent_smiles`, `metabolite_smiles`,
  `cyp` (metabolizing isoform), `origin` (drugbank / chembl / drugbank+chembl).
  That split is experimental-only and leakage-free by construction (Butina
  clustering on parents; rule-generated pairs barred from test). Regenerate from
  the `metabolite-prediction` repo:

  ```python
  import pandas as pd
  df = pd.read_parquet("data/Victorien-zaretzki-drugbank-v1/processed/pairs.parquet")
  (df[df["split"] == "test"][["pair_id", "parent_smiles", "metabolite_smiles", "cyp", "origin"]]
      .to_csv(".../Victorien-CYP-metabolites/raw/metabolite_test_pairs.csv", index=False))
  ```

- **`raw/uniprot_sequences.json`** (fetched cache) — `prepare.py` maps each of the
  9 specific human CYP isoforms to its canonical reviewed (Swiss-Prot) UniProt
  accession (`CYP_TO_ACCESSION` in `prepare.py`) and fetches the sequence from the
  UniProt REST API (cached, resumable; regenerated automatically).

  | CYP | UniProt | len | | CYP | UniProt | len |
  |-----|---------|-----|-|-----|---------|-----|
  | CYP1A2 | P05177 | 516 | | CYP2C19 | P33261 | 490 |
  | CYP2A6 | P11509 | 494 | | CYP2D6 | P10635 | 497 |
  | CYP2B6 | P20813 | 491 | | CYP2E1 | P05181 | 493 |
  | CYP2C8 | P10632 | 490 | | CYP3A4 | P08684 | 503 |
  | CYP2C9 | P11712 | 490 | | | | |

  None of these 9 accessions appear in `EnzymeMap_with_seq` (the model's training
  data), so they are genuinely held-out enzymes.

## Build

`just data-prep Victorien-CYP-metabolites` runs `prepare.py`, which maps each
pair's `cyp` to its accession + sequence, drops pairs with no specific isoform
(`other` / `unknown` / `CYP_inferred` / `CYP_unspecified`), encodes each as a
`parent>>metabolite` reaction via `epp_core.data.build_reactions`, attaches the
enzyme `sequence`, and writes `processed/reactions.parquet` +
`processed/build_manifest.json`. `processed/` is git-ignored.

- **233** raw pairs → **21 dropped** (no mappable CYP) → **212** reactions over **9** enzymes.
- Kept by isoform: CYP3A4 101, CYP2D6 33, CYP2C9 21, CYP1A2 18, CYP2E1 18,
  CYP2C19 6, CYP2C8 6, CYP2A6 6, CYP2B6 3.

## Schema (`processed/reactions.parquet`)

Standard `REACTION_COLUMNS` (`reaction_id`, `reactant_smiles`, `product_smiles`,
`source`, `n_reactants`, `n_products`, `src_n_tokens`, `tgt_n_tokens`) plus:

| Column | Meaning |
|--------|---------|
| `uniprot_id` | CYP accession the model conditions on (embedding key) |
| `sequence` | enzyme amino-acid sequence |
| `seq_len` | sequence length |
| `cyp` | source CYP isoform label |
| `ec_num` | `1.14.14.1` (unspecific monooxygenase — all 9 CYPs) |
| `organism` | `Homo sapiens` |
| `origin` | source DB (drugbank / chembl / drugbank+chembl) |
| `source_pair_id` | `pair_id` in the originating Victorien-zaretzki-drugbank-v1 split |
| `split` | always `test` |

## Caveats

- **Evaluation-only**: there is no `train`/`valid` split, so the novelty metric
  (which compares against training products) is not meaningful here.
- **Off-distribution**: `EnzymeMap_with_seq` is general enzymology and contains no
  human drug-metabolism CYPs; inputs are cofactor-free (`parent>>metabolite`, no
  O₂/NADPH). Expect low absolute scores — read this as a generalization stress test.
- **Coarse labels**: `cyp` is a family-level label (CYP3A4 dominates at ~48% of
  rows), not a reaction-specific enzyme.

## Usage

```bash
just data-prep Victorien-CYP-metabolites
# correct-enzyme evaluation
just evaluate ESM2-650M-frozen-ReactionT5 --dataset-dir data/Victorien-CYP-metabolites/processed
# control: enzyme conditioning permuted across rows
just evaluate ESM2-650M-frozen-ReactionT5 --dataset-dir data/Victorien-CYP-metabolites/processed --shuffle-enzymes
```
