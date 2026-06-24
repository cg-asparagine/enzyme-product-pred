"""Dataset-agnostic data helpers: deterministic reaction builder, splits, loaders, hashing."""

from epp_core.data.build import build_reactions
from epp_core.data.cluster import cluster_sequences, jaccard, kmer_set
from epp_core.data.loaders import load_reactions, train_product_smiles
from epp_core.data.registry import content_hash, sha256_file
from epp_core.data.schema import REACTION_COLUMNS, VALID_SPLITS, BuildStats
from epp_core.data.split import assign_splits, grouped_random_split
from epp_core.data.uniprot import (
    fetch_sequences,
    parse_fasta,
    search_accessions,
    uniparc_sequences,
)

__all__ = [
    "REACTION_COLUMNS",
    "VALID_SPLITS",
    "BuildStats",
    "assign_splits",
    "build_reactions",
    "cluster_sequences",
    "content_hash",
    "fetch_sequences",
    "grouped_random_split",
    "jaccard",
    "kmer_set",
    "load_reactions",
    "parse_fasta",
    "search_accessions",
    "sha256_file",
    "train_product_smiles",
    "uniparc_sequences",
]
