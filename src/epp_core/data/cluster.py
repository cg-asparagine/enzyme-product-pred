"""Enzyme (protein-sequence) clustering for sequence-aware dataset splits.

Groups UniProt accessions into clusters of *similar* enzymes so that whole
clusters — not individual accessions — can be assigned to train/valid/test (see
:func:`epp_core.data.split.assign_splits`). Holding out clusters keeps close
homologs out of training, giving an honest test of generalization to **new**
enzymes for sequence-conditioned models (the dominant leakage mode the v1
reaction-grouped split does *not* control).

The default method, ``"kmer"``, is greedy representative-based clustering
(CD-HIT / MMseqs ``easy-cluster`` style) over k-mer Jaccard similarity, in pure
Python so no external aligner is required:

* each sequence is reduced to the set of its length-``k`` substrings (k-mers);
* sequences are processed longest-first; each joins the **most similar** existing
  cluster whose representative has k-mer Jaccard >= ``threshold``, otherwise it
  starts a new cluster and becomes that cluster's representative.

An inverted index (k-mer -> representative cluster ids) restricts each comparison
to representatives that share at least one k-mer, and the shared-k-mer counts read
straight off the index give the exact intersection size — so Jaccard is exact and
the whole thing scales to tens of thousands of sequences without an all-pairs
matrix.

k-mer Jaccard is a *proxy* for sequence identity, not a calibrated percentage; a
higher ``threshold`` yields more, tighter clusters. Two cheaper methods are also
available: ``"exact"`` (byte-identical sequences share a cluster) and ``"uniprot"``
(each accession is its own cluster — the per-enzyme baseline).
"""

from __future__ import annotations

from collections import Counter, defaultdict

#: Skip k-mers occurring in more than this many cluster representatives when
#: gathering candidates. Such k-mers are near-ubiquitous (uninformative for
#: similarity) and dominate the inverted-index scan; ignoring them bounds cost
#: without changing which clusters clear the Jaccard threshold in practice.
_MAX_REP_DF = 4000


def kmer_set(seq: str, k: int) -> frozenset[str]:
    """The set of length-``k`` contiguous substrings of ``seq``.

    Sequences shorter than ``k`` fall back to a single token (the whole sequence)
    so they can still match an identical short sequence.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(seq) < k:
        return frozenset((seq,)) if seq else frozenset()
    return frozenset(seq[i : i + k] for i in range(len(seq) - k + 1))


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    """Jaccard similarity |a ∩ b| / |a ∪ b| (two empty sets are defined as 1.0)."""
    if not a and not b:
        return 1.0
    inter = len(a & b)
    if not inter:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def cluster_sequences(
    id_to_seq: dict[str, str],
    method: str = "kmer",
    k: int = 4,
    threshold: float = 0.4,
) -> dict[str, int]:
    """Map each id in ``id_to_seq`` to an integer cluster id.

    Deterministic: the result depends only on the sequences, ``method``, ``k`` and
    ``threshold`` (no randomness). ``threshold`` and ``k`` apply to ``method="kmer"``.
    """
    ids = sorted(id_to_seq)
    if method == "uniprot":
        return {u: i for i, u in enumerate(ids)}
    if method == "exact":
        seq_to_cluster: dict[str, int] = {}
        out: dict[str, int] = {}
        for u in ids:
            out[u] = seq_to_cluster.setdefault(id_to_seq[u], len(seq_to_cluster))
        return out
    if method != "kmer":
        raise ValueError(f"unknown method {method!r}; expected 'kmer', 'exact' or 'uniprot'")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    kmers = {u: kmer_set(seq, k) for u, seq in id_to_seq.items()}
    # Longest-first so the representative is the most informative member; id
    # tie-break keeps it deterministic.
    order = sorted(ids, key=lambda u: (-len(id_to_seq[u]), u))

    rep_kmers: list[frozenset[str]] = []  # cluster id -> representative k-mer set
    index: dict[str, list[int]] = defaultdict(list)  # k-mer -> representative cluster ids
    assignment: dict[str, int] = {}
    for u in order:
        ks = kmers[u]
        # |ks ∩ rep| for every representative sharing an (informative) k-mer.
        shared: Counter[int] = Counter()
        for km in ks:
            posting = index.get(km)
            if posting is not None and len(posting) <= _MAX_REP_DF:
                shared.update(posting)
        best_cid, best_sim = -1, -1.0
        for cid, inter in shared.items():
            union = len(ks) + len(rep_kmers[cid]) - inter
            sim = inter / union if union else 1.0
            # Highest similarity wins; smallest cluster id breaks ties (determinism).
            if sim > best_sim or (sim == best_sim and cid < best_cid):
                best_sim, best_cid = sim, cid
        if best_cid != -1 and best_sim >= threshold:
            assignment[u] = best_cid
        else:
            cid = len(rep_kmers)
            rep_kmers.append(ks)
            for km in ks:
                index[km].append(cid)
            assignment[u] = cid
    return assignment
