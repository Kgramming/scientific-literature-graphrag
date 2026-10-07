#!/usr/bin/env python3
"""Build the Phase 3A full corpus: deterministic ~5,000-abstract PubMed sample.

Reuses the validated Phase 1 acquisition machinery (sgrag.data.*) unchanged.

PMID selection (deterministic, reproducible):
  PubMedQA pqa_labeled/train (1,000 questions) followed by pqa_unlabeled/train
  (~61k questions), in split order; the first N unique numeric pubids are
  taken. Because the pilot took the first 200 pubids of pqa_labeled/train,
  the pilot PMID set is a strict subset of this corpus (verified at runtime).

Steps:
  1. Load pubids (labeled, then unlabeled) and select N (default 5000).
  2. Fetch abstracts from NCBI Entrez efetch in polite batches -> data/raw/.
  3. Parse XML -> data/interim/full_abstracts.jsonl (all records).
  4. Cache the ordered pubid list -> data/interim/full_qa_pubids.json.
  5. Write build meta (timing, failures) -> data/raw/full_build_meta.json.

Nothing about the pilot corpus is touched.
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.utils.env import sanitize_proxy_env
from sgrag.data.pilot import select_pilot_pmids
from sgrag.data.pubmed_fetch import fetch_batches, parse_pubmed_xml

QA_DATASET = "qiaojin/PubMedQA"
QA_LABELED = "pqa_labeled"
QA_UNLABELED = "pqa_unlabeled"
QA_SPLIT = "train"


def load_pubids_ordered():
    """pubids: labeled/train rows first, then unlabeled/train rows (streamed)."""
    import datasets
    print(f"[full] loading {QA_DATASET}/{QA_LABELED}/{QA_SPLIT}")
    ds_lab = datasets.load_dataset(QA_DATASET, QA_LABELED, split=QA_SPLIT)
    pubids = [str(ds_lab[i]["pubid"]) for i in range(len(ds_lab))]
    n_lab = len(pubids)
    print(f"[full] labeled questions: {n_lab}")
    print(f"[full] streaming {QA_DATASET}/{QA_UNLABELED}/{QA_SPLIT}")
    ds_unlab = datasets.load_dataset(QA_DATASET, QA_UNLABELED, split=QA_SPLIT,
                                     streaming=True)
    n_unlab = 0
    for row in ds_unlab:
        pubids.append(str(row["pubid"]))
        n_unlab += 1
    print(f"[full] unlabeled questions: {n_unlab}")
    return pubids, n_lab, n_unlab, datasets.__version__


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Build the ~5000-abstract full corpus.")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--raw-prefix", default="full_efetch_batch")
    ap.add_argument("--interim", default="data/interim/full_abstracts.jsonl")
    ap.add_argument("--qa-pubids", default="data/interim/full_qa_pubids.json")
    ap.add_argument("--batch-size", type=int, default=50)
    ap.add_argument("--force", action="store_true",
                    help="re-fetch even if raw batches are cached")
    args = ap.parse_args()
    t0 = time.perf_counter()

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    interim_path = Path(args.interim)
    interim_path.parent.mkdir(parents=True, exist_ok=True)

    pubids_in_order, n_lab, n_unlab, ds_version = load_pubids_ordered()

    pmids, sel_info = select_pilot_pmids(pubids_in_order, n=args.n)
    print(f"[full] selected {len(pmids)} PMIDs "
          f"(sha256={sel_info['pmid_list_sha256'][:16]}...): "
          f"{pmids[0]} .. {pmids[-1]}")

    # Pilot-subset check: the pilot's PMID list must be a prefix of the full
    # selection (for n >= 200); for smaller smoke-test n, the selection must
    # be a prefix of the pilot list.
    pilot_check, pilot_info = select_pilot_pmids(pubids_in_order[:n_lab], n=200)
    k = min(200, len(pmids))
    assert pmids[:k] == pilot_check[:k], "pilot/full PMID selection mismatch!"
    print(f"[full] pilot-subset verified (first {k} PMIDs match; "
          f"pilot sha256={pilot_info['pmid_list_sha256'][:16]}...)")

    with open(args.qa_pubids, "w", encoding="utf-8") as f:
        json.dump({"pubids": pubids_in_order, "n_labeled": n_lab,
                   "n_unlabeled": n_unlab,
                   "datasets_version": ds_version}, f)
    print(f"[full] cached {len(pubids_in_order)} QA pubids -> {args.qa_pubids}")

    t_fetch0 = time.perf_counter()
    email = os.environ.get("SGRAG_CONTACT_EMAIL")
    batches = fetch_batches(pmids, batch_size=args.batch_size, email=email)
    all_records, failed_pmids, failed_batches = [], [], 0
    n_batches = 0
    for i, b in enumerate(batches):
        n_batches += 1
        out = raw_dir / f"{args.raw_prefix}_{i:03d}.xml"
        if b["xml"] is None:
            failed_batches += 1
            print(f"[full] batch {i}: FAILED after {b['attempts']} attempts: {b['error']}")
            failed_pmids.extend(b["pmids"])
            continue
        if args.force or not out.exists():
            out.write_bytes(b["xml"])
        recs = parse_pubmed_xml(b["xml"])
        if (i + 1) % 10 == 0 or len(recs) != len(b["pmids"]):
            print(f"[full] batch {i}: {len(b['pmids'])} requested -> "
                  f"{len(recs)} parsed (http {b['http_status']})")
        all_records.extend(recs)
    fetch_s = time.perf_counter() - t_fetch0

    with open(interim_path, "w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[full] wrote {len(all_records)} records -> {interim_path}")

    meta = {
        "qa_dataset": QA_DATASET,
        "qa_configs": [QA_LABELED, QA_UNLABELED],
        "qa_split": QA_SPLIT,
        "qa_questions_labeled": n_lab,
        "qa_questions_unlabeled": n_unlab,
        "datasets_version": ds_version,
        "selection": sel_info,
        "pilot_subset_sha256": pilot_info["pmid_list_sha256"],
        "requested": args.n,
        "n_batches": n_batches,
        "n_failed_batches": failed_batches,
        "n_failed_pmids": len(failed_pmids),
        "failed_pmids": failed_pmids,
        "n_parsed_records": len(all_records),
        "retrieval_date_utc": datetime.now(timezone.utc).isoformat(),
        "fetch_seconds": round(fetch_s, 1),
        "total_seconds": round(time.perf_counter() - t0, 1),
    }
    meta_path = raw_dir / "full_build_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[full] build meta -> {meta_path} "
          f"(fetch {fetch_s:.0f}s, total {time.perf_counter()-t0:.0f}s)")
    if failed_pmids:
        print(f"[full] WARNING: {len(failed_pmids)} PMIDs failed: {failed_pmids[:10]}")


if __name__ == "__main__":
    main()
