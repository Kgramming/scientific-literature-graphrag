#!/usr/bin/env python3
"""Validate the 5,000-abstract run against the 200-abstract pilot.

Compares chunk counts, chunks/article, size distributions, semantic
singleton/noise behavior, coherence statistics, and provenance coverage.
Purpose: detect pipeline regressions or distribution anomalies — NOT to prove
statistical identity. Flags relative deviations >30% for human review.

Note: the pilot's semantic run predates the max_tokens=256 enforcement fix,
so its max chunk size (350) vs the full run's (<=256) is an expected,
documented difference, not a regression.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sgrag.utils.env import sanitize_proxy_env

FLAG_THRESHOLD = 0.30


def rel_diff(a, b):
    if a == 0 and b == 0:
        return 0.0
    denom = abs(a) if a != 0 else abs(b)
    return (b - a) / denom


def main():
    sanitize_proxy_env()
    ap = argparse.ArgumentParser(description="Validate full-corpus scaling vs pilot.")
    ap.add_argument("--pilot-eval", default="results/phase2_chunking/chunking_eval.json")
    ap.add_argument("--full-eval", default="results/full_corpus/chunking_eval.json")
    ap.add_argument("--pilot-info", default="data/processed/chunks_semantic_cluster_info.json")
    ap.add_argument("--full-info", default="data/processed/full_chunks/chunks_semantic_cluster_info.json")
    ap.add_argument("--out", default="results/full_corpus/scaling_validation.json")
    args = ap.parse_args()

    pilot = json.load(open(args.pilot_eval, encoding="utf-8"))
    full = json.load(open(args.full_eval, encoding="utf-8"))
    pinfo = json.load(open(args.pilot_info, encoding="utf-8"))
    finfo = json.load(open(args.full_info, encoding="utf-8"))

    def singleton_rate(info):
        n_sent = sum(d.get("n_sentences", 0) for d in info)
        n_noise = sum(d.get("n_noise", 0) for d in info)
        return n_noise / n_sent if n_sent else 0.0

    rows, flags = [], []
    strategies = list(pilot["strategies"].keys())
    for s in strategies:
        p, f = pilot["strategies"][s], full["strategies"][s]
        pc, fc = p["coherence"], f["coherence"]
        comp = {
            "strategy": s,
            "n_chunks": (p["n_chunks"], f["n_chunks"]),
            "chunks_per_article": (p["chunks_per_article"]["mean"],
                                   f["chunks_per_article"]["mean"]),
            "token_mean": (p["token_length"]["mean"], f["token_length"]["mean"]),
            "token_median": (p["token_length"]["median"], f["token_length"]["median"]),
            "token_max": (p["token_length"]["max"], f["token_length"]["max"]),
            "integrity_violation_rate": (p["integrity_violation_rate"],
                                         f["integrity_violation_rate"]),
            "coherence_gap": (pc["gap_mean"], fc["gap_mean"]),
            "coherence_intra": (pc["intra_chunk_cosine_mean"],
                                fc["intra_chunk_cosine_mean"]),
            "coherence_inter": (pc["inter_chunk_cosine_mean"],
                                fc["inter_chunk_cosine_mean"]),
            "coverage_exact_once": (p["coverage_exact_once_articles"],
                                    f["coverage_exact_once_articles"]),
        }
        if s == "semantic_cluster":
            comp["singleton_rate"] = (singleton_rate(pinfo), singleton_rate(finfo))
        for metric, vals in comp.items():
            if metric in ("strategy", "coverage_exact_once"):
                continue
            pv, fv = vals
            if metric == "token_max" and s == "semantic_cluster":
                continue  # expected: pilot predates the max_tokens fix
            d = rel_diff(pv, fv)
            if abs(d) > FLAG_THRESHOLD:
                flags.append({"strategy": s, "metric": metric,
                              "pilot": pv, "full": fv,
                              "rel_diff": round(d, 3)})
        rows.append(comp)

    print(f"{'strategy':<18}{'metric':<26}{'pilot':>12}{'full':>12}{'rel.diff':>10}")
    for comp in rows:
        s = comp["strategy"]
        for metric, vals in comp.items():
            if metric == "strategy":
                continue
            pv, fv = vals
            d = rel_diff(pv, fv)
            mark = "  <-- REVIEW" if any(
                fl["strategy"] == s and fl["metric"] == metric for fl in flags) else ""
            pv_s = f"{pv:.4f}" if isinstance(pv, float) else str(pv)
            fv_s = f"{fv:.4f}" if isinstance(fv, float) else str(fv)
            print(f"{s:<18}{metric:<26}{pv_s:>12}{fv_s:>12}{d:>+10.1%}{mark}")

    out = {"rows": rows, "flags": flags,
           "flag_threshold": FLAG_THRESHOLD,
           "note": "token_max excluded for semantic_cluster: pilot predates "
                   "the max_tokens=256 enforcement fix (pilot max was 350)."}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2)
    print(f"\n[validate] {len(flags)} flags -> {out_path}")
    return 1 if flags else 0


if __name__ == "__main__":
    raise SystemExit(main())
