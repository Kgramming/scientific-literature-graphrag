# Data directory

- `raw/` — immutable downloads (PubMedQA, PubMed abstracts). **Git-ignored.**
- `interim/` — chunked / embedded / NER outputs. **Git-ignored** (regenerable).
- `processed/` — graph-load-ready artifacts. **Git-ignored** (regenerable).
- `evaluation/` — question manifests + provenance (`manifest.json`). **Committed** — small, and
  reproducibility depends on it.

Every artifact written by the pipeline carries `{producer_script, config_hash, seed, timestamp}`
provenance. See `docs/EXPERIMENT_PLAN.md` §2 for the deterministic eval-set construction.
