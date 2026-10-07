#!/usr/bin/env python3
"""Build the Phase 1 pilot corpus: deterministic 200-abstract PubMed sample.

Steps:
  1. Load PubMedQA pqa_labeled/train (Hugging Face) and take the first N
     question pubids in split order as the PMID list (deterministic).
  2. Fetch abstracts from NCBI Entrez efetch in polite batches -> data/raw/.
  3. Parse XML -> data/interim/pilot_abstracts.jsonl (all records).
  4. Cache the ordered pubid list -> data/interim/qa_pubids.json (for audit).

Re-running reproduces the same PMID list (pmid_list_sha256 is recorded);
re-fetching the same PMIDs reproduces the corpus barring PubMed updates.
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.utils.env import sanitize_proxy_env
from sgrag.data.pilot import select_pilot_pmids
from sgrag.data.pubmed_fetch import fetch_batches, parse_pubmed_xml

QA_DATASET = "qiaojin/PubMedQA"
QA_CONFIG = "pqa_labeled"
QA_SPLIT = "train"


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Build the 200-abstract pilot corpus.")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--interim", default="data/interim/pilot_abstracts.jsonl")
    ap.add_argument("--qa-pubids", default="data/interim/qa_pubids.json")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if raw batches are cached")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_path = Path(args.interim)
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    import datasets
    print(f"[pilot] loading {QA_DATASET}/{QA_CONFIG}/{QA_SPLIT} "
          f"(datasets {datasets.__version__})")
    ds = datasets.load_dataset(QA_DATASET, QA_CONFIG, split=QA_SPLIT)
    pubids_in_order = [str(ds[i]["pubid"]) for i in range(len(ds))]
    print(f"[pilot] QA questions: {len(pubids_in_order)}")

    pmids, sel_info = select_pilot_pmids(pubids_in_order, n=args.n)
    print(f"[pilot] selected {len(pmids)} PMIDs "
          f"(sha256={sel_info['pmid_list_sha256'][:16]}...): "
          f"{pmids[0]} .. {pmids[-1]}")

    qa_pubids_path = Path(args.qa_pubids)
    with open(qa_pubids_path, "w", encoding="utf-8") as f:
        json.dump({"pubids": pubids_in_order,
                   "datasets_version": datasets.__version__}, f)
    print(f"[pilot] cached {len(pubids_in_order)} QA pubids -> {qa_pubids_path}")

    email = os.environ.get("SGRAG_CONTACT_EMAIL")
    batches = fetch_batches(pmids, batch_size=args.batch_size, email=email)
    all_records, failed_pmids = [], []
    for i, b in enumerate(batches):
        out = raw_dir / f"efetch_batch_{i:02d}.xml"
        if b["xml"] is None:
            print(f"[pilot] batch {i}: FAILED after {b['attempts']} attempts: {b['error']}")
            failed_pmids.extend(b["pmids"])
            continue
        if args.force or not out.exists():
            out.write_bytes(b["xml"])
        recs = parse_pubmed_xml(b["xml"])
        print(f"[pilot] batch {i}: {len(b['pmids'])} requested -> "
              f"{len(recs)} parsed (http {b['http_status']})")
        all_records.extend(recs)

    with open(interim_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[pilot] wrote {len(all_records)} records -> {interim_path}")
    if failed_pmids:
        print(f"[pilot] WARNING: {len(failed_pmids)} PMIDs failed to fetch: "
              f"{failed_pmids[:10]}")

    meta = {
        "qa_dataset": QA_DATASET, "qa_config": QA_CONFIG, "qa_split": QA_SPLIT,
        "qa_questions": len(pubids_in_order),
        "datasets_version": datasets.__version__,
        "selection": sel_info,
        "n_failed_pmids": len(failed_pmids),
        "failed_pmids": failed_pmids,
        "n_parsed_records": len(all_records),
        "pmids": pmids,
    }
    meta_path = raw_dir / "pilot_build_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[pilot] build meta -> {meta_path}")


if __name__ == "__main__":
    main()
