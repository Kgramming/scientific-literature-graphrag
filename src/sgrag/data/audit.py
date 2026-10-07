"""Corpus auditing: duplicates, missing data, metadata consistency.

Pure functions over parsed records — no network, no side effects, fully
unit-testable. Nothing is dropped here; every record gets flags and the
manifest decides what "retained" means.
"""

import hashlib


def _norm_text(text):
    return " ".join(text.lower().split())


def _text_hash(text):
    return hashlib.sha256(_norm_text(text).encode("utf-8")).hexdigest()


def audit_records(records, pqal_years=None):
    """Audit parsed PubMed records.

    Args:
        records: list of dicts from parse_pubmed_xml.
        pqal_years: optional {pmid: year} from the QA file for cross-checking.

    Returns:
        {"summary": {...}, "flagged": [{"pmid":..., "flags":[...], "status":...}]}
    Status is "ok" only when the record has a valid PMID and a non-empty
    abstract; otherwise it names the blocking problem.
    """
    summary = {
        "total": len(records),
        "with_abstract": 0,
        "missing_abstract": 0,
        "empty_abstract": 0,
        "missing_title": 0,
        "missing_year": 0,
        "missing_journal": 0,
        "invalid_pmid": 0,
        "duplicate_pmid_groups": [],
        "duplicate_text_groups": [],
        "year_mismatches": [],
    }
    flagged = []

    seen_pmids = {}
    seen_texts = {}
    for rec in records:
        pmid = rec.get("pmid")
        flags = []

        if not pmid or not str(pmid).isdigit():
            flags.append("invalid_pmid")
            summary["invalid_pmid"] += 1
        else:
            seen_pmids.setdefault(str(pmid), []).append(rec)

        abstract = rec.get("abstract")
        if abstract is None:
            flags.append("missing_abstract")
            summary["missing_abstract"] += 1
        elif not abstract.strip():
            flags.append("empty_abstract")
            summary["empty_abstract"] += 1
        else:
            summary["with_abstract"] += 1
            h = _text_hash(abstract)
            seen_texts.setdefault(h, []).append(str(pmid))

        if not rec.get("title"):
            flags.append("missing_title")
            summary["missing_title"] += 1
        if rec.get("year") is None:
            flags.append("missing_year")
            summary["missing_year"] += 1
        if not rec.get("journal"):
            flags.append("missing_journal")
            summary["missing_journal"] += 1

        if pqal_years and pmid in pqal_years:
            try:
                if rec.get("year") is not None and int(pqal_years[pmid]) != rec["year"]:
                    flags.append("year_mismatch")
                    summary["year_mismatches"].append(
                        {"pmid": str(pmid), "pqal_year": pqal_years[pmid],
                         "pubmed_year": rec["year"]})
            except (ValueError, TypeError):
                pass

        if not flags:
            status = "ok"
        elif "invalid_pmid" in flags:
            status = "invalid_pmid"
        elif "missing_abstract" in flags:
            status = "missing_abstract"
        elif "empty_abstract" in flags:
            status = "empty_abstract"
        else:
            status = "ok_with_warnings"
        flagged.append({"pmid": pmid, "flags": flags, "status": status})

    for pmid, group in seen_pmids.items():
        if len(group) > 1:
            summary["duplicate_pmid_groups"].append(
                {"pmid": pmid, "count": len(group)})
    for h, pmids in seen_texts.items():
        if len(pmids) > 1:
            summary["duplicate_text_groups"].append(
                {"text_sha256": h[:16], "pmids": pmids, "count": len(pmids)})

    valid_pmids = sum(1 for r in records
                      if r.get("pmid") and str(r["pmid"]).isdigit())
    summary["pmid_coverage"] = valid_pmids / len(records) if records else 0.0
    summary["n_duplicate_pmid_groups"] = len(summary["duplicate_pmid_groups"])
    summary["n_duplicate_text_groups"] = len(summary["duplicate_text_groups"])
    summary["retained"] = sum(1 for f in flagged if f["status"] == "ok")
    summary["retained_with_warnings"] = sum(
        1 for f in flagged if f["status"] == "ok_with_warnings")
    return {"summary": summary, "flagged": flagged}
