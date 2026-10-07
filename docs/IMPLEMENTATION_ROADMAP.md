# Implementation Roadmap

**Gate:** Phase 0 (this planning stage) ends with user approval. No Phase 1 work starts before that.
Each phase lists exit criteria — a phase is done only when its criteria are met and verified.

---

## Phase 0 — Architecture + environment ✅ (this stage)

- [x] Inspect reference architecture (read-only)
- [x] `REFERENCE_GAP_ANALYSIS.md`, `ARCHITECTURE_PROPOSAL.md`, `IMPLEMENTATION_ROADMAP.md`,
      `EXPERIMENT_PLAN.md`, `KNOWLEDGE_BYTES.md` (planned concepts)
- [x] Project skeleton, `.env.example` (`SGRAG_*`), `.gitignore`, `pyproject.toml`, `docker-compose.yml`
- [ ] **User approval** ← we are here
- Post-approval: `scripts/setup_env.sh` (venv + `pip install -e ".[dev]"`), verify imports, decide Neo4j host

## Phase 1 — Dataset + chunking

- Validate corpus source with real PMIDs; fetch PubMedQA gold abstracts (pilot: 200 abstracts)
- Implement 4 chunkers + chunk schema; cohesion analysis + plots on pilot
- **Exit:** all 4 strategies run on pilot; cohesion report written; chunker unit tests pass

## Phase 2 — Embeddings + entities

- Embedding provider (MiniLM baseline) + Parquet persistence; candidate biomedical model benchmarked
- scispaCy BC5CDR pipeline + normalization + canonical entity store
- **Exit:** embeddings cached for pilot; entity dedup report (raw vs canonical node counts); NER unit tests pass

## Phase 3 — Neo4j graph

- `docker-compose.yml` up on chosen host; constraints, indexes, vector index; APOC batch load
- Load pilot graph; verify counts (articles/chunks/entities/edges); Cypher sanity queries
- **Exit:** pilot graph queryable; vector search returns sane neighbors; integration tests pass

## Phase 4 — Vector retrieval

- Vector search module; top-k plumbing; per-query timing
- **Exit:** System A (baseline config) runs end-to-end on pilot questions; Recall@5/MRR computed

## Phase 5 — Graph retrieval

- Entity-aware retrieval; bounded expansion (hub guard); traversal tests on fixture graph
- **Exit:** expansion returns explainable paths; System C runs; no runaway traversals (assert caps)

## Phase 6 — Hybrid retrieval

- RRF + weighted fusion; YAML weights; weight-sensitivity script on pilot
- Query classifier + routing table
- **Exit:** System D runs; classifier unit tests pass; routing decisions logged

## Phase 7 — Reranking

- Cross-encoder integration (bounded pool); graph-aware score with normalized components
- Optional GDS PageRank/Louvain signals behind config flags
- **Exit:** System E runs; per-candidate score breakdowns verified; latency measured

## Phase 8 — Query decomposition

- LLM decomposer + deterministic validators + fallback; gated by query class
- **Exit:** System F runs; simple-question degradation ≈ 0; validators unit-tested

## Phase 9 — Generation + citations

- Provider-agnostic LLM client; evidence-first prompt; citation validation; claim→chunk mapping
- **Exit:** answers cite only retrieved chunk IDs; insufficient-evidence fallback demonstrated

## Phase 10 — Evaluation

- Deterministic eval-set builder + manifest; retrieval metrics (article + chunk level); generation
  metrics (ROUGE-L, BERTScore, faithfulness suite); latency tracker
- Scale corpus to ~5,000; full graph build
- **Exit:** eval manifest published; all metrics computed for System A on full eval set

## Phase 11 — Ablations

- Run systems A–G on identical questions via `run_experiment.py --config ...`
- CSVs, comparison tables, plots, % deltas, paired statistical tests, honest discussion
- **Exit:** `results/` contains reproducible runs; negative results reported as-is

## Phase 12 — UI

- Streamlit 6-tab research dashboard; PyVis graph view; eval tab reading run artifacts
- **Exit:** demo runs against the full system; screenshots in README

## Phase 13 — Testing + documentation

- Full test suite green (unit always; integration/e2e where services/keys exist)
- Finalize `docs/` (architecture, graph-schema, retrieval, evaluation, experiments, limitations),
  full `Knowledge Bytes/` grounded in the implementation, README with measured results
- **Exit:** README makes no unmeasured claim; repo pushed to new GitHub repo

---

## Milestones & rough sequencing

| Milestone | Phases | Depends on |
|---|---|---|
| M1: Pilot pipeline (200 abstracts, no Neo4j) | 0–2 | approval, env |
| M2: Graph-backed retrieval | 3–5 | Neo4j host available |
| M3: Full retrieval stack | 6–8 | M2 |
| M4: Grounded generation | 9 | M3 + LLM key |
| M5: Measured results | 10–11 | M4, full corpus |
| M6: Demo + docs | 12–13 | M5 |

Scale-up rule (from the assignment): 100 abstracts → smoke test, 500 → graph validation,
5,000 → final. Never build the expensive pipeline before the smaller one validates.
