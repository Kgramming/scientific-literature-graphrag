"""Dataset manifest: the reproducibility record for a corpus build."""

import json
from datetime import datetime, timezone


REQUIRED_KEYS = [
    "project", "phase", "corpus_name", "source", "retrieval_date_utc",
    "requested", "retrieved", "retained", "dropped", "pmid_coverage",
    "filtering_criteria", "audit_summary", "pubmedqa_mapping",
    "provenance",
]


def build_manifest(project="scientific-literature-graphrag", phase="phase1_pilot",
                   corpus_name="pilot_200", source=None, requested=0,
                   retrieved=0, retained=0, dropped=None, pmid_coverage=0.0,
                   filtering_criteria="", audit_summary=None,
                   pubmedqa_mapping=None, provenance=None):
    """Assemble the manifest dict. All counts are explicit — no assumptions."""
    manifest = {
        "project": project,
        "phase": phase,
        "corpus_name": corpus_name,
        "source": source or {},
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "requested": requested,
        "retrieved": retrieved,
        "retained": retained,
        "dropped": dropped or {"count": 0, "reasons": {}, "pmids": []},
        "pmid_coverage": pmid_coverage,
        "filtering_criteria": filtering_criteria,
        "audit_summary": audit_summary or {},
        "pubmedqa_mapping": pubmedqa_mapping or {},
        "provenance": provenance or {},
    }
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(f"manifest missing required keys: {missing}")
    return manifest


def write_manifest(path, manifest):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return path
