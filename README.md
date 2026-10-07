# Scientific Literature GraphRAG — Advanced Biomedical Research QA

Evidence-backed question answering over biomedical scientific literature: ~5,000 PubMed abstracts are
chunked (4 strategies compared), embedded, entity-extracted, and stored in a Neo4j knowledge graph;
a hybrid vector + graph retrieval pipeline with cross-encoder and graph-aware reranking feeds a
grounded LLM that answers with validated citations. Every architectural choice is measured through
ablation experiments on a deterministically constructed PubMedQA evaluation set.

**Status:** planning phase (Phase 0). Architecture, roadmap, and experiment plan are in `docs/`.
No superiority claim is made until experiments demonstrate it.

## Research questions

Why is GraphRAG better than vector-only retrieval? When does graph expansion help, and when does it
hurt? Does semantic chunking actually improve retrieval? Does query decomposition help multi-hop
questions? Does graph-aware reranking improve Recall/MRR? Are the answers grounded? What is the
accuracy-vs-latency tradeoff? See `docs/EXPERIMENT_PLAN.md` for how each question maps to a measurement.

## Architecture

```text
Question → Query Understanding (classification + gated decomposition)
         → Multi-Strategy Retrieval (vector · entity-aware · graph expansion · [community])
         → Candidate Fusion (RRF / weighted, config-driven)
         → Cross-Encoder Reranking → Graph-Aware Reranking (explainable score blend)
         → Evidence Selection → Grounded LLM Answer (validated citations)
         → Streamlit UI (answer · retrieval · graph · explainability · evaluation · system)
```

Full design: `docs/ARCHITECTURE_PROPOSAL.md`. Reference comparison: `docs/REFERENCE_GAP_ANALYSIS.md`.

## Tech stack

`sentence-transformers` (all-MiniLM-L6-v2 baseline) · scispaCy (BC5CDR biomedical NER) · Neo4j 5 +
APOC + GDS (Docker) · cross-encoder reranker · provider-agnostic LLM (Groq/OpenAI/Ollama) ·
Streamlit + PyVis · ROUGE + BERTScore + faithfulness suite.

## Project layout

```text
app/            Streamlit research dashboard (6 tabs)
src/sgrag/      Python package: config, data, chunking, embeddings, entities, graph,
                retrieval, reranking, query, generation, evaluation, utils
configs/        Experiment configs (baseline → full); one command reproduces a run
data/           raw/ interim/ processed/ evaluation/ (only evaluation/ manifests are committed)
notebooks/      Analysis notebooks
tests/          unit/ integration/ e2e/
scripts/        setup_env.sh, run_experiment.sh, ...
docs/           architecture, roadmap, experiment plan, knowledge bytes, ...
```

## Quickstart (after Phase 0 approval)

```bash
cp .env.example .env          # fill in SGRAG_* values (fresh keys for this project)
./scripts/setup_env.sh        # creates .venv, installs the package
docker compose up -d          # Neo4j + APOC + GDS (needs Docker; not available on all hosts)
./scripts/run_experiment.sh --config configs/baseline.yaml
streamlit run app/streamlit_app.py
```

## Evaluation

Deterministic PubMedQA-based eval set with PMID-equality gold matching (manifest in
`data/evaluation/`). Retrieval: Recall@5/10, MRR, nDCG@10, Precision@5 (article + chunk level).
Generation: ROUGE-L, BERTScore, faithfulness suite (citation precision, evidence coverage,
unsupported-claim rate). Ablation systems A–G on identical questions; paired tests with 95% CIs;
negative results reported. Details: `docs/EXPERIMENT_PLAN.md`.

## Limitations (planning-stage, to be updated with measurements)

- PubMedQA gold granularity is the abstract; chunk-level labels are weak by construction.
- Some PubMedQA gold abstracts may be un-fetchable; drops are audited, never hidden.
- Biomedical NER and cross-encoders are CPU-feasible but add latency — measured per stage.

## License

MIT — see `LICENSE`.
