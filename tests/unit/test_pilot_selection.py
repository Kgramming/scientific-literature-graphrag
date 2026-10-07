"""Unit tests: pilot PMID selection + PubMedQA mapping (no network)."""
import pytest

from sgrag.data.pilot import (select_pilot_pmids, build_pubmedqa_mapping,
                              pmid_list_sha256)


def _pubids():
    return [1000 + i for i in range(10)]  # ints, like the HF dataset


def test_selection_first_n_in_order():
    pmids, info = select_pilot_pmids(_pubids(), n=4)
    assert pmids == ["1000", "1001", "1002", "1003"]
    assert info["n_selected"] == 4
    assert info["n_input"] == 10


def test_selection_deterministic_and_fingerprinted():
    a, info_a = select_pilot_pmids(_pubids(), n=5)
    b, info_b = select_pilot_pmids(_pubids(), n=5)
    assert a == b
    assert info_a["pmid_list_sha256"] == info_b["pmid_list_sha256"]
    assert info_a["pmid_list_sha256"] == pmid_list_sha256(a)


def test_selection_skips_non_numeric_and_dedupes():
    pmids, info = select_pilot_pmids(["abc", 1001, "1001", 1002], n=2)
    assert pmids == ["1001", "1002"]
    assert info["n_skipped_non_numeric"] == 1


def test_selection_too_few_raises():
    with pytest.raises(ValueError):
        select_pilot_pmids(_pubids(), n=999)


def test_mapping_by_pmid_equality():
    pubid_to_rows = {"1000": [0], "1001": [1], "1003": [3, 7]}
    mapping, stats = build_pubmedqa_mapping(pubid_to_rows, ["1000", "1003", "9999"])
    assert stats["n_mapped"] == 2
    assert stats["n_unmapped"] == 1
    assert stats["unmapped_pmids"] == ["9999"]
    assert stats["n_duplicate_mappings"] == 1
    assert stats["duplicate_mappings"] == {"1003": [3, 7]}
    assert mapping["1000"] == [0]
    assert stats["method"] == "pmid_equality_only"
