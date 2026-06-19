import json

import pytest

from epp_core.data.uniprot import fetch_sequences, parse_fasta, uniparc_sequences

FASTA = (
    ">sp|P0A9Q7|ENO_ECOLI Enolase OS=Escherichia coli\n"
    "MSKIVKII\n"
    "IEDGR\n"
    ">tr|H9ZGN0|H9ZGN0_OGAPD Alcohol dehydrogenase\n"
    "MAAAA\n"
)


def test_parse_fasta_extracts_accession_and_sequence():
    assert parse_fasta(FASTA) == {"P0A9Q7": "MSKIVKIIIEDGR", "H9ZGN0": "MAAAA"}


def test_parse_fasta_empty():
    assert parse_fasta("") == {}


def test_fetch_sequences_caches_and_skips_refetch(tmp_path):
    calls: list[list[str]] = []

    def fake(batch: list[str]) -> dict[str, str]:
        calls.append(list(batch))
        return {a: "SEQ_" + a for a in batch if a != "P2"}  # P2 is unresolvable

    cache = tmp_path / "cache.json"
    out = fetch_sequences(["P1", "P2", "P1"], cache_path=cache, fetch_batch=fake, sleep=0)
    assert out == {"P1": "SEQ_P1"}  # P2 (miss) omitted; P1 de-duped
    assert calls == [["P1", "P2"]]  # one batch, de-duped input

    cached = json.loads(cache.read_text())
    assert cached == {"P1": "SEQ_P1", "P2": ""}  # miss recorded as ""

    calls.clear()
    out2 = fetch_sequences(["P1", "P2"], cache_path=cache, fetch_batch=fake, sleep=0)
    assert out2 == {"P1": "SEQ_P1"}
    assert calls == []  # fully cached -> no fetch


def test_fetch_sequences_batches(tmp_path):
    seen: list[int] = []

    def fake(batch: list[str]) -> dict[str, str]:
        seen.append(len(batch))
        return {a: "S" for a in batch}

    out = fetch_sequences(
        [f"A{i}" for i in range(5)],
        cache_path=tmp_path / "c.json",
        fetch_batch=fake,
        batch_size=2,
        sleep=0,
    )
    assert len(out) == 5
    assert seen == [2, 2, 1]


def test_uniparc_sequences_recovers_and_caches(tmp_path):
    calls: list[str] = []

    def fake(acc: str) -> str:
        calls.append(acc)
        return {"P1": "SEQ1", "P3": "SEQ3"}.get(acc, "")  # P2 unrecoverable

    cache = tmp_path / "uniparc.json"
    out = uniparc_sequences(["P1", "P2", "P3", "P1"], cache_path=cache, fetch_one=fake, sleep=0)
    assert out == {"P1": "SEQ1", "P3": "SEQ3"}  # P2 omitted; P1 de-duped
    assert calls == ["P1", "P2", "P3"]

    cached = json.loads(cache.read_text())
    assert cached == {"P1": "SEQ1", "P2": "", "P3": "SEQ3"}

    calls.clear()
    out2 = uniparc_sequences(["P1", "P2", "P3"], cache_path=cache, fetch_one=fake, sleep=0)
    assert out2 == {"P1": "SEQ1", "P3": "SEQ3"}
    assert calls == []  # fully cached -> no fetch


@pytest.mark.slow
@pytest.mark.network
def test_fetch_sequences_real_uniprot(tmp_path):
    # Hits the live UniProt API; deselected from `just check`, run with `just test-slow`.
    out = fetch_sequences(["P0A9Q7"], cache_path=tmp_path / "c.json", sleep=0)
    assert out["P0A9Q7"].startswith("M")  # enolase, starts with Met


@pytest.mark.slow
@pytest.mark.network
def test_uniparc_recovers_obsolete_accession(tmp_path):
    # A0A023J8T6 is obsolete in UniProtKB but archived in UniParc.
    out = uniparc_sequences(["A0A023J8T6"], cache_path=tmp_path / "u.json", sleep=0)
    assert out["A0A023J8T6"].startswith("M")
