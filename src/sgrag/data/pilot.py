"""Pilot corpus PMID selection and PubMedQA mapping.

The pilot PMID list is derived deterministically from the PubMedQA labeled
set (Hugging Face ``qiaojin/PubMedQA``, config ``pqa_labeled``): each question
carries the PubMed ID of its source article as ``pubid``. Taking the first N
pubids in split order gives a reproducible list where every abstract is, by
construction, the source article of at least one labeled PubMedQA question —
the property the full evaluation mapping (Phase 10) will rely on.
"""

import hashlib


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def pmid_list_sha256(pmids):
    """Stable fingerprint of an ordered PMID list."""
    return hashlib.sha256("\n".join(pmids).encode("utf-8")).hexdigest()


def select_pilot_pmids(pubids, n=200):
    """Deterministically select the first n numeric PubMed IDs.

    Args:
        pubids: pubids in dataset order (ints or strings).
        n: how many to select.

    Returns:
        (pmids, info) where pmids is the ordered list of PMID strings and
        info records input/unique/skipped counts. Non-numeric entries are
        skipped (counted, never silently kept); duplicates collapse to
        first-seen order. Raises if fewer than n usable IDs exist.
    """
    seen, out, skipped = set(), [], 0
    for p in pubids:
        s = str(p)
        if not s.isdigit():
            skipped += 1
            continue
        if s not in seen:
            seen.add(s)
            out.append(s)
        if len(out) >= n:
            break
    if len(out) < n:
        raise ValueError(f"only {len(out)} usable PubMed IDs, requested {n}")
    info = {
        "n_input": len(pubids),
        "n_unique_numeric": len(seen),
        "n_skipped_non_numeric": skipped,
        "n_selected": len(out),
        "selection_rule": "first-n numeric pubids in dataset split order, deduped",
        "pmid_list_sha256": pmid_list_sha256(out),
    }
    return out, info


def build_pubmedqa_mapping(pubid_to_rows, pilot_pmids):
    """Link pilot abstracts to PubMedQA questions by PubMed-ID equality only.

    Args:
        pubid_to_rows: {pubid_str: [dataset row indices]} for the QA split.
        pilot_pmids: ordered list of pilot PMID strings.

    No text matching, no forcing: a pilot PMID maps to the question(s) whose
    pubid equals it. Returns (mapping, stats).
    """
    mapping = {}
    for pmid in pilot_pmids:
        rows = pubid_to_rows.get(str(pmid), [])
        if rows:
            mapping[str(pmid)] = rows

    mapped = [p for p in pilot_pmids if str(p) in mapping]
    unmapped = [p for p in pilot_pmids if str(p) not in mapping]
    duplicate = {p: rows for p, rows in mapping.items() if len(rows) > 1}

    stats = {
        "n_pilot_pmids": len(pilot_pmids),
        "n_mapped": len(mapped),
        "n_unmapped": len(unmapped),
        "unmapped_pmids": unmapped,
        "n_duplicate_mappings": len(duplicate),
        "duplicate_mappings": duplicate,
        "method": "pmid_equality_only",
        "note": ("Pilot PMIDs were drawn from PubMedQA question pubids, so "
                 "mapping is by construction; duplicates would indicate a pubid "
                 "shared by multiple questions."),
    }
    return mapping, stats
