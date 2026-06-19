"""Fetch UniProtKB protein sequences by accession, with on-disk caching.

Used by datasets that condition on the enzyme sequence (e.g. EnzymeMap_with_seq).
Sequences are fetched in batches from the UniProt REST ``accessions`` endpoint
and cached to a JSON file so re-runs only fetch new accessions; the cache is
written after every batch, so an interrupted run resumes where it left off.
Unresolvable accessions (obsolete / demerged / secondary) are cached as misses
(``""``) and omitted from the returned mapping.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable
from pathlib import Path

UNIPROT_ACCESSIONS_URL = "https://rest.uniprot.org/uniprotkb/accessions"
UNIPARC_SEARCH_URL = "https://rest.uniprot.org/uniparc/search"
_USER_AGENT = "enzyme-product-pred/0.1 (https://github.com/cg-asparagine/enzyme-product-pred)"


def parse_fasta(text: str) -> dict[str, str]:
    """Parse UniProt FASTA text into ``{accession: sequence}``.

    The accession is the field between the first two ``|`` of a ``>db|ACC|name``
    header (falls back to the first whitespace token).
    """
    sequences: dict[str, str] = {}
    accession: str | None = None
    chunks: list[str] = []
    for line in text.splitlines():
        if line.startswith(">"):
            if accession is not None:
                sequences[accession] = "".join(chunks)
            header = line[1:]
            parts = header.split("|")
            accession = parts[1] if len(parts) >= 2 else header.split(maxsplit=1)[0]
            chunks = []
        elif accession is not None:
            chunks.append(line.strip())
    if accession is not None:
        sequences[accession] = "".join(chunks)
    return sequences


def _http_fetch(
    accessions: list[str], *, timeout: float = 60.0, retries: int = 3
) -> dict[str, str]:
    """Fetch one batch of accessions from UniProt as FASTA, with simple retries."""
    params = urllib.parse.urlencode({"accessions": ",".join(accessions), "format": "fasta"})
    request = urllib.request.Request(
        f"{UNIPROT_ACCESSIONS_URL}?{params}", headers={"User-Agent": _USER_AGENT}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return parse_fasta(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as error:  # network errors / 5xx
            last_error = error
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"UniProt fetch failed after {retries} attempts: {last_error}")


def _load_cache(path: Path) -> dict[str, str]:
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}


def fetch_sequences(
    accessions: Iterable[str],
    *,
    cache_path: str | Path,
    batch_size: int = 200,
    sleep: float = 0.2,
    fetch_batch: Callable[[list[str]], dict[str, str]] | None = None,
) -> dict[str, str]:
    """Return ``{accession: sequence}`` for every resolvable accession.

    Results (hits and misses) are cached to ``cache_path`` (JSON) so re-runs only
    fetch new accessions. ``fetch_batch`` is the per-batch fetcher (defaults to
    the UniProt REST call); inject a fake in tests. Missing accessions are cached
    as ``""`` and omitted from the returned mapping.
    """
    fetch = fetch_batch or _http_fetch
    cache_path = Path(cache_path)
    cache = _load_cache(cache_path)

    wanted = list(dict.fromkeys(accessions))  # de-dup, preserve order
    todo = [a for a in wanted if a not in cache]

    for start in range(0, len(todo), batch_size):
        batch = todo[start : start + batch_size]
        found = fetch(batch)
        for acc in batch:
            cache[acc] = found.get(acc, "")  # "" marks a known miss
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(cache, f)
        if sleep and start + batch_size < len(todo):
            time.sleep(sleep)

    return {a: cache[a] for a in wanted if cache.get(a)}


def _uniparc_fetch_one(accession: str, *, timeout: float = 60.0, retries: int = 3) -> str:
    """Look up the archived sequence for one accession via UniParc, or ``""``.

    UniParc keeps every sequence ever seen in any source database, so it resolves
    obsolete / secondary accessions that the live UniProtKB endpoint drops. One
    accession per request (UniParc search results don't echo the matched
    cross-reference, so batched queries can't be mapped back reliably).
    """
    params = urllib.parse.urlencode(
        {"query": accession, "fields": "upi,sequence", "format": "json", "size": "1"}
    )
    request = urllib.request.Request(
        f"{UNIPARC_SEARCH_URL}?{params}", headers={"User-Agent": _USER_AGENT}
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                results = json.loads(response.read().decode("utf-8")).get("results", [])
            if not results:
                return ""
            return (results[0].get("sequence") or {}).get("value") or ""
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(
        f"UniParc fetch failed for {accession} after {retries} attempts: {last_error}"
    )


def uniparc_sequences(
    accessions: Iterable[str],
    *,
    cache_path: str | Path,
    sleep: float = 0.1,
    save_every: int = 50,
    fetch_one: Callable[[str], str] | None = None,
) -> dict[str, str]:
    """Recover archived sequences (one UniParc lookup per accession) for accessions
    the live UniProtKB endpoint missed.

    Cached to ``cache_path`` (written every ``save_every`` lookups), so it's
    resumable and re-runs are instant. ``fetch_one`` is injectable for tests.
    Misses are cached as ``""`` and omitted from the returned mapping.
    """
    fetch = fetch_one or _uniparc_fetch_one
    cache_path = Path(cache_path)
    cache = _load_cache(cache_path)
    wanted = list(dict.fromkeys(accessions))
    todo = [a for a in wanted if a not in cache]

    def _save() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(cache, f)

    for i, accession in enumerate(todo):
        cache[accession] = fetch(accession)
        if (i + 1) % save_every == 0:
            _save()
        if sleep:
            time.sleep(sleep)
    _save()

    return {a: cache[a] for a in wanted if cache.get(a)}
