"""Unit tests: corpus auditing (synthetic records, no network)."""
from sgrag.data.audit import audit_records


def _rec(pmid, abstract="Some abstract text.", title="T", year=2020, journal="J"):
    return {"pmid": pmid, "title": title, "abstract": abstract,
            "journal": journal, "year": year, "mesh_terms": [],
            "parse_ok": True}


def test_audit_clean_records():
    recs = [_rec("1", abstract="Alpha abstract."),
            _rec("2", abstract="Beta abstract."),
            _rec("3", abstract="Gamma abstract.")]
    out = audit_records(recs)
    s = out["summary"]
    assert s["total"] == 3
    assert s["retained"] == 3
    assert s["pmid_coverage"] == 1.0
    assert s["n_duplicate_pmid_groups"] == 0
    assert s["n_duplicate_text_groups"] == 0
    assert all(f["status"] == "ok" for f in out["flagged"])


def test_audit_detects_problems():
    recs = [
        _rec("1", abstract="Shared text here."),
        _rec("1", abstract="Different text."),          # duplicate PMID
        _rec("2", abstract="Shared text here."),        # duplicate text
        _rec("3", abstract=None),                        # missing abstract
        _rec("4", abstract="   "),                       # empty abstract
        _rec("bad-id", abstract="Unique bad text."),    # invalid PMID
        _rec("5", abstract="Unique five text.", title=None),  # missing title, still ok
    ]
    out = audit_records(recs)
    s = out["summary"]
    assert s["total"] == 7
    assert s["missing_abstract"] == 1
    assert s["empty_abstract"] == 1
    assert s["invalid_pmid"] == 1
    assert s["n_duplicate_pmid_groups"] == 1
    assert s["duplicate_pmid_groups"][0]["pmid"] == "1"
    assert s["n_duplicate_text_groups"] == 1
    assert set(s["duplicate_text_groups"][0]["pmids"]) == {"1", "2"}
    statuses = {f["pmid"]: f["status"] for f in out["flagged"]}
    assert statuses["3"] == "missing_abstract"
    assert statuses["4"] == "empty_abstract"
    assert statuses["5"] == "ok_with_warnings"
    assert s["retained"] == 3  # pmids 1(x2) and 2 -> 'ok'


def test_audit_year_mismatch():
    recs = [_rec("10", year=2020)]
    out = audit_records(recs, pqal_years={"10": "2019"})
    s = out["summary"]
    assert len(s["year_mismatches"]) == 1
    assert s["year_mismatches"][0] == {"pmid": "10", "pqal_year": "2019",
                                      "pubmed_year": 2020}
    assert "year_mismatch" in out["flagged"][0]["flags"]
