# Victorien-CYP-metabolites

Human drug parent→metabolite pairs, each conditioned on the cytochrome-P450 (CYP)
isoform that produced it. A train/valid/test dataset for the sequence-conditioned
model `ESM2-650M-frozen-ReactionT5` — used both to **fine-tune** it on drug
metabolism and to **evaluate** it against the enzyme-blind Metatrans baseline.

## Why

It mirrors the `Victorien-zaretzki-drugbank-v1` dataset that trains/evaluates
`Metatrans-ReactionT5` in the sibling `metabolite-prediction` repo, re-cast as
enzymatic reactions. Metatrans is enzyme-blind (it sees only
`REACTANT:<parent>REAGENT:` → `<metabolite>`); here we attach the metabolizing
enzyme so an enzyme-conditioned model can be trained/scored on the same molecules
and the two approaches compared directly. The splits are inherited from the source
(whole-parent-cluster, leakage-free), and the `test` split is experimental-only —
the identical 212 triples used for the enzyme-blind Metatrans test set.

## Provenance

`data/*/raw/` is git-ignored (raw is regenerable; provenance lives here + in
`build_manifest.json`, which records the raw SHA-256 and the dataset content hash).

- **`raw/metabolite_pairs.csv`** (external input) — all train/valid/test pairs
  exported from
  `metabolite-prediction/data/Victorien-zaretzki-drugbank-v1/processed/pairs.parquet`.
  Columns: `split`, `pair_id`, `parent_smiles`, `metabolite_smiles`,
  `cyp` (metabolizing isoform), `origin` (rule / drugbank / chembl / drugbank+chembl).
  Regenerate from the `metabolite-prediction` repo:

  ```python
  import pandas as pd
  df = pd.read_parquet("data/Victorien-zaretzki-drugbank-v1/processed/pairs.parquet")
  (df[["split", "pair_id", "parent_smiles", "metabolite_smiles", "cyp", "origin"]]
      .to_csv(".../Victorien-CYP-metabolites/raw/metabolite_pairs.csv", index=False))
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

  None of these 9 accessions appear in `EnzymeMap_with_seq` (the model's *original*
  training data), so they are held-out enzymes for the pretrained model.

## Build

`just data-prep Victorien-CYP-metabolites` runs `prepare.py`, which normalizes each
pair's `cyp` to a canonical isoform, maps it to its accession + sequence, drops
pairs with no specific isoform, encodes each as a `parent>>metabolite` reaction via
`epp_core.data.build_reactions`, carries the source `split`, and writes
`processed/reactions.parquet` + `processed/build_manifest.json` (git-ignored).

- The source `cyp` is **mixed-format**: experimental rows use `CYP3A4` while
  rule-generated rows use the bare `3A4` — both normalize to `CYP3A4`.
- **2335** raw pairs → **256 dropped** (no specific CYP: `other`/`unknown`/
  `CYP_inferred`/`CYP_unspecified`) → **2079** reactions over **9** enzymes.
- Per split: **train 1639, valid 228, test 212**.
- Kept by isoform: CYP3A4 822, CYP1A2 385, CYP2D6 261, CYP2C9 223, CYP2C8 115,
  CYP2C19 98, CYP2B6 61, CYP2A6 58, CYP2E1 56.

## Schema (`processed/reactions.parquet`)

Standard `REACTION_COLUMNS` (`reaction_id`, `reactant_smiles`, `product_smiles`,
`source`, `n_reactants`, `n_products`, `src_n_tokens`, `tgt_n_tokens`) plus:

| Column | Meaning |
|--------|---------|
| `uniprot_id` | CYP accession the model conditions on (embedding key) |
| `sequence` | enzyme amino-acid sequence |
| `seq_len` | sequence length |
| `cyp` | normalized CYP isoform (e.g. `CYP3A4`) |
| `ec_num` | `1.14.14.1` (unspecific monooxygenase — all 9 CYPs) |
| `organism` | `Homo sapiens` |
| `origin` | source DB (rule / drugbank / chembl / drugbank+chembl) |
| `source_pair_id` | `pair_id` in the originating Victorien-zaretzki-drugbank-v1 split |
| `split` | `train` / `valid` / `test` (inherited from the source) |

## Caveats

- **Off-distribution for the pretrained model**: `EnzymeMap_with_seq` is general
  enzymology with no human drug-metabolism CYPs; inputs are cofactor-free
  (`parent>>metabolite`, no O₂/NADPH). Fine-tuning on the `train` split is what
  adapts the model to this domain.
- **Rule-generated training data**: the `train`/`valid` splits include SMIRKS-rule
  metabolites (`origin == "rule"`), same as the data Metatrans trains on; `test` is
  experimental-only.
- **Coarse labels**: `cyp` is a family-level label (CYP3A4 dominates ~40% of rows),
  not a reaction-specific enzyme.

## Usage

```bash
just data-prep Victorien-CYP-metabolites
# fine-tune the (EnzymeMap-trained) model on the train split; ESM-2 stays frozen
just train ESM2-650M-frozen-ReactionT5 --finetune
# evaluate a checkpoint on the test split (beam 10)
just evaluate ESM2-650M-frozen-ReactionT5 \
  --model-dir models/ESM2-650M-frozen-ReactionT5/checkpoints-victorien-ft \
  --dataset-dir data/Victorien-CYP-metabolites/processed
# control: enzyme conditioning permuted across rows
just evaluate ESM2-650M-frozen-ReactionT5 \
  --model-dir models/ESM2-650M-frozen-ReactionT5/checkpoints-victorien-ft \
  --dataset-dir data/Victorien-CYP-metabolites/processed --shuffle-enzymes
```
