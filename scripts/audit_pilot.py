#!/usr/bin/env python3
"""Audit the pilot corpus and write processed data + manifest + mapping report.

Steps:
  1. Load data/interim/pilot_abstracts.jsonl + data/interim/qa_pubids.json.
  2. Audit: duplicates, missing/empty abstracts, metadata consistency.
  3. Write data/processed/pilot_corpus.jsonl (every record + flags + status;
     nothing silently dropped).
  4. Write data/evaluation/pilot_corpus_manifest.json (committed provenance).
  5. Write data/evaluation/pilot_pubmedqa_mapping.json (PMID-equality mapping).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.utils.env import sanitize_proxy_env
from sgrag.data.pilot import select_pilot_pmids, build_pubmedqa_mapping
from sgrag.data.audit import audit_records
from sgrag.data.manifest import build_manifest, write_manifest


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Audit the pilot corpus.")
    ap.add_argument("--interim", default="data/interim/pilot_abstracts.jsonl")
    ap.add_argument("--qa-pubids", default="data/interim/qa_pubids.json")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--processed", default="data/processed/pilot_corpus.jsonl")
    ap.add_argument("--manifest", default="data/evaluation/pilot_corpus_manifest.json")
    ap.add_argument("--mapping", default="data/evaluation/pilot_pubmedqa_mapping.json")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.interim, encoding="utf-8") if l.strip()]
    qa = json.load(open(args.qa_pubids, encoding="utf-8"))
    qa_pubids = qa["pubids"]
    pmids, sel_info = select_pilot_pmids(qa_pubids, n=args.n)

    audit = audit_records(records)
    s = audit["summary"]
    print(f"[audit] total={s['total']} with_abstract={s['with_abstract']} "
          f"missing_abstract={s['missing_abstract']} empty_abstract={s['empty_abstract']}")
    print(f"[audit] duplicate_pmid_groups={s['n_duplicate_pmid_groups']} "
          f"duplicate_text_groups={s['n_duplicate_text_groups']} "
          f"pmid_coverage={s['pmid_coverage']:.3f} retained={s['retained']}")

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
    print(f"[audit] wrote {len(records)} flagged records -> {processed}")

    pubid_to_rows = {}
    for idx, p in enumerate(qa_pubids):
        pubid_to_rows.setdefault(str(p), []).append(idx)
    mapping, mapping_stats = build_pubmedqa_mapping(pubid_to_rows, pmids)
    mapping_path = Path(args.mapping)
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({"mapping": mapping, "stats": mapping_stats}, f, indent=2)
    print(f"[audit] PubMedQA mapping: {mapping_stats['n_mapped']} mapped, "
          f"{mapping_stats['n_unmapped']} unmapped, "
          f"{mapping_stats['n_duplicate_mappings']} duplicate mappings -> {mapping_path}")

    fetched_pmids = {str(r.get("pmid")) for r in records if r.get("pmid")}
    not_fetched = [p for p in pmids if p not in fetched_pmids]
    dropped_reasons, dropped_pmids = {}, []
    for f in audit["flagged"]:
        if f["status"] not in ("ok", "ok_with_warnings"):
            dropped_reasons[f["status"]] = dropped_reasons.get(f["status"], 0) + 1
            dropped_pmids.append(f["pmid"])
    for p in not_fetched:
        dropped_reasons["fetch_failed"] = dropped_reasons.get("fetch_failed", 0) + 1
        dropped_pmids.append(p)

    manifest = build_manifest(
        corpus_name="pilot_200",
        source={
            "qa_dataset": "qiaojin/PubMedQA",
            "qa_config": "pqa_labeled",
            "qa_split": "train",
            "qa_questions": len(qa_pubids),
            "datasets_library": qa.get("datasets_version"),
            "pmid_selection": sel_info,
            "abstract_source": "NCBI Entrez efetch (db=pubmed, retmode=xml, 50/batch)",
            "raw_batches": "data/raw/efetch_batch_*.xml",
        },
        requested=args.n,
        retrieved=len(records),
        retained=s["retained"],
        dropped={"count": len(dropped_pmids), "reasons": dropped_reasons,
                 "pmids": dropped_pmids},
        pmid_coverage=s["pmid_coverage"],
        filtering_criteria=(
            "Deterministic: first 200 numeric pubids of PubMedQA pqa_labeled/train "
            "in split order (deduped). No content filtering before audit; records "
            "with missing/empty abstracts are flagged, not silently dropped."
        ),
        audit_summary=s,
        pubmedqa_mapping=mapping_stats,
        provenance={
            "scripts": ["scripts/build_pilot_corpus.py", "scripts/audit_pilot.py"],
            "modules": ["sgrag.data.pubmed_fetch", "sgrag.data.pilot",
                        "sgrag.data.audit", "sgrag.data.manifest"],
            "python": sys.version.split()[0],
        },
    )
    manifest_path = Path(args.manifest)
    write_manifest(manifest_path, manifest)
    print(f"[audit] manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
