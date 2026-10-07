#!/usr/bin/env python3
"""Audit the Phase 3A full corpus; write processed data + manifest + mapping.

Mirrors scripts/audit_pilot.py exactly (same audit_records, same manifest
schema, same PMID-equality mapping) with full-corpus paths. The pilot
manifest and results are never touched.

Additionally records year and abstract-length distributions in the manifest's
audit_summary (additive; required manifest keys unchanged).
"""
import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.utils.env import sanitize_proxy_env
from sgrag.data.pilot import select_pilot_pmids, build_pubmedqa_mapping
from sgrag.data.audit import audit_records
from sgrag.data.manifest import build_manifest, write_manifest


def distributions(records):
    years = [r["year"] for r in records
             if isinstance(r.get("year"), int)]
    lengths = [len(r.get("abstract") or "") for r in records]
    journals = Counter(r.get("journal") or "MISSING" for r in records)
    dist = {
        "n_records": len(records),
        "years": {
            "n_present": len(years),
            "min": min(years) if years else None,
            "max": max(years) if years else None,
            "median": statistics.median(years) if years else None,
            "by_year": dict(sorted(Counter(years).items())),
        },
        "abstract_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "median": statistics.median(lengths) if lengths else 0,
            "mean": round(statistics.mean(lengths), 1) if lengths else 0,
        },
        "top_journals": dict(journals.most_common(10)),
        "n_distinct_journals": len(journals),
    }
    return dist


def main():
    sanitize_proxy_env()
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser(description="Audit the full corpus.")
    ap.add_argument("--interim", default="data/interim/full_abstracts.jsonl")
    ap.add_argument("--qa-pubids", default="data/interim/full_qa_pubids.json")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--processed", default="data/processed/full_corpus.jsonl")
    ap.add_argument("--manifest", default="data/evaluation/full_corpus_manifest.json")
    ap.add_argument("--mapping", default="data/evaluation/full_pubmedqa_mapping.json")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.interim, encoding="utf-8") if l.strip()]
    qa = json.load(open(args.qa_pubids, encoding="utf-8"))
    qa_pubids = qa["pubids"]
    pmids, sel_info = select_pilot_pmids(qa_pubids, n=args.n)
    print(f"[audit-full] selection re-derived: {len(pmids)} PMIDs "
          f"(sha256={sel_info['pmid_list_sha256'][:16]}...)")

    audit = audit_records(records)
    s = audit["summary"]
    print(f"[audit-full] total={s['total']} with_abstract={s['with_abstract']} "
          f"missing_abstract={s['missing_abstract']} empty_abstract={s['empty_abstract']}")
    print(f"[audit-full] duplicate_pmid_groups={s['n_duplicate_pmid_groups']} "
          f"duplicate_text_groups={s['n_duplicate_text_groups']} "
          f"pmid_coverage={s['pmid_coverage']:.4f} retained={s['retained']}")

    flag_by_pmid = {f["pmid"]: f for f in audit["flagged"]}
    processed = Path(args.processed)
    processed.parent.mkdir(parents=True, exist_ok=True)
    with open(processed, "w", encoding="utf-8") as f:
        for rec in records:
            fl = flag_by_pmid.get(rec.get("pmid"), {"flags": [], "status": "unknown"})
            out = dict(rec)
            out["audit_flags"] = fl["flags"]
            out["status"] = fl["status"]
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"[audit-full] wrote {len(records)} flagged records -> {processed}")

    pubid_to_rows = {}
    for idx, p in enumerate(qa_pubids):
        pubid_to_rows.setdefault(str(p), []).append(idx)
    mapping, mapping_stats = build_pubmedqa_mapping(pubid_to_rows, pmids)
    mapping_path = Path(args.mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping, "stats": mapping_stats}, f, indent=2)
    print(f"[audit-full] PubMedQA mapping: {mapping_stats['n_mapped']} mapped, "
          f"{mapping_stats['n_unmapped']} unmapped, "
          f"{mapping_stats['n_duplicate_mappings']} duplicate mappings -> {mapping_path}")

    fetched_pmids = {str(r.get("pmid")) for r in records if r.get("pmid")}
    not_fetched = [p for p in pmids if p not in fetched_pmids]
    dropped_reasons, dropped_pmids = {}, []
    for fl in audit["flagged"]:
        if fl["status"] not in ("ok", "ok_with_warnings"):
            dropped_reasons[fl["status"]] = dropped_reasons.get(fl["status"], 0) + 1
            dropped_pmids.append(fl["pmid"])
    for p in not_fetched:
        dropped_reasons["fetch_failed"] = dropped_reasons.get("fetch_failed", 0) + 1
        dropped_pmids.append(p)

    dist = distributions(records)
    print(f"[audit-full] years: {dist['years']['min']}..{dist['years']['max']} "
          f"(n={dist['years']['n_present']}), abstract chars median={dist['abstract_chars']['median']}")

    audit_summary = dict(s)
    audit_summary["distributions"] = dist

    manifest = build_manifest(
        phase="phase3a_full_corpus",
        corpus_name="full_5000",
        source={
            "qa_dataset": "qiaojin/PubMedQA",
            "qa_configs": ["pqa_labeled", "pqa_unlabeled"],
            "qa_split": "train",
            "qa_questions_labeled": qa.get("n_labeled"),
            "qa_questions_unlabeled": qa.get("n_unlabeled"),
            "datasets_library": qa.get("datasets_version"),
            "pmid_selection": sel_info,
            "abstract_source": "NCBI Entrez efetch (db=pubmed, retmode=xml, 50/batch)",
            "raw_batches": "data/raw/full_efetch_batch_*.xml",
        },
        requested=args.n,
        retrieved=len(records),
        retained=s["retained"],
        dropped={"count": len(dropped_pmids), "reasons": dropped_reasons,
                 "pmids": dropped_pmids},
        pmid_coverage=s["pmid_coverage"],
        filtering_criteria=(
            "Deterministic: first 5000 numeric pubids of PubMedQA pqa_labeled/train "
            "then pqa_unlabeled/train in split order (deduped); the 200-abstract "
            "pilot is a strict subset. No content filtering before audit; records "
            "with missing/empty abstracts are flagged, not silently dropped."
        ),
        audit_summary=audit_summary,
        pubmedqa_mapping=mapping_stats,
        provenance={
            "scripts": ["scripts/build_full_corpus.py", "scripts/audit_full_corpus.py"],
            "modules": ["sgrag.data.pubmed_fetch", "sgrag.data.pilot",
                        "sgrag.data.audit", "sgrag.data.manifest"],
            "python": sys.version.split()[0],
            "audit_seconds": round(time.perf_counter() - t0, 1),
        },
    )
    write_manifest(Path(args.manifest), manifest)
    print(f"[audit-full] manifest -> {args.manifest} "
          f"({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
