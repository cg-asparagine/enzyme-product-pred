# EnzymeMap (v2 / BRENDA 2023)

Enzymatic reactions for product prediction, one reaction per row as
`reactants>>products` SMILES with EC number, organism, and a quality score.

## Provenance

- **Source:** EnzymeMap (Heid et al., 2023) — a cleaned, atom-mapped set of
  enzymatic reactions derived from the **BRENDA 2023** enzyme database.
- **Raw file:** `raw/enzymemap_v2_brenda2023.csv` (~386 MB, 349,458 rows).
  **Git-ignored** (it exceeds GitHub's 100 MB limit). Its SHA-256 is recorded in
  `processed/build_manifest.json` at build time for reproducibility. Keep a local
  copy under `raw/`; if missing, `prepare.py` raises with this path.

### Raw columns (used by `prepare.py`)

| column | meaning |
| --- | --- |
| `rxn_idx` | EnzymeMap reaction index (forward + reverse twins and per-organism rows share it) |
| `unmapped` | reaction SMILES `reactants>>products` (the model input/target source) |
| `mapped` | atom-mapped reaction SMILES (not used yet) |
| `orig_rxn_text` | human-readable reaction (e.g. `acetaldehyde + NADH + H+ = ethanol + NAD+`) |
| `source` | `direct` / `direct reversed` / `suggested` / … — `reversed` ⇒ reverse direction |
| `steps` | `single` / `single from multi` / `multi` |
| `quality` | confidence score in `[0, 1]` (mean ≈ 0.95) |
| `natural` | whether the reaction is naturally occurring |
| `organism` | free-text organism name |
| `ec_num` | EC number, e.g. `1.1.1.1` |
| `protein_refs`, `protein_db`, `rule`, `rule_id` | not used yet (no amino-acid sequences here) |

## Build

```
just data-prep EnzymeMap
```

writes (both **git-ignored** under `processed/`):

- `reactions.parquet` — the processed dataset (schema below).
- `build_manifest.json` — dataset id, content hash, raw SHA-256, filters, split
  config, per-split counts, and `build_reactions` stats.

### Filters applied (defaults in `prepare.py`)

- `steps == "single"` (drop multi-step and single-from-multi reactions).
- `quality >= 0.3` (drop the low-confidence tail).
- per-side atom-token length `<= 300`.
- drop identity reactions (reactants == products) and any reaction with an
  unparseable molecule.
- dedupe on the canonical reaction key **plus `ec_num`**, so the same reaction
  catalyzed by different EC numbers survives as distinct rows.

### Split

Grouped-random 80/10/10 (seed 42), grouped on the **direction-collapsed reaction
identity** so a reaction's forward/reverse twins and its per-organism copies all
land in the same split. (Scaffold / sequence- & compound-similarity splits are on
the roadmap — see [CLAUDE.md](../../CLAUDE.md).)

### Processed schema (`reactions.parquet`)

`reaction_id`, `reactant_smiles`, `product_smiles`, `source`, `n_reactants`,
`n_products`, `src_n_tokens`, `tgt_n_tokens`, `split`, then the carried metadata:
`rxn_idx`, `ec_num`, `organism`, `direction`, `quality`, `natural`, `steps`.

`reactant_smiles` / `product_smiles` are dot-joined, individually-canonicalized,
sorted molecule sets. **Cofactors** (NAD+, NADH, ATP, H2O, …) are part of both
sides — relevant when reading `exact_set_match` (strict on the whole product set)
vs `top_k_accuracy` (any single correct product).

## Load

```python
from epp_core.data import load_reactions, train_product_smiles

train = load_reactions("data/EnzymeMap/processed", split="train")
products = train_product_smiles("data/EnzymeMap/processed")  # for the novelty metric
```
