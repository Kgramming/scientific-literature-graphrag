# Demo Guide — Scientific Literature GraphRAG

How to run, demo, and explain the system during review.

## 1. Prerequisites

- Python 3.12 with the project virtualenv (`.venv/`)
- Neo4j 5.x running locally with the Phase 3B graph
  (5,000 articles / 53,091 chunks / 9,945 entities / 265,455 similarity edges)
- Environment variables (see below)
- Optional: `SGRAG_LLM_API_KEY` for live LLM answers (Groq).
  Without it the app runs with a clearly-labeled deterministic mock provider.

## 2. Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `SGRAG_NEO4J_PASSWORD` | Neo4j password (**required**) | — |
| `SGRAG_NEO4J_URI` | Neo4j bolt URI | `bolt://localhost:7687` |
| `SGRAG_LLM_PROVIDER` | `mock` \| `groq` | `mock` |
| `SGRAG_LLM_MODEL` | LLM model name | `mock-1` / `llama-3.3-70b-versatile` |
| `SGRAG_LLM_API_KEY` | LLM API key (never printed/logged) | — |
| `SGRAG_SCISPACY_MODEL` | path to extracted bc5cdr model dir | `en_ner_bc5cdr_md` |
| `HF_HUB_OFFLINE` | `1` — use local model snapshots | `1` recommended |

All secrets come from the environment. Nothing is hard-coded or committed.

## 3. Neo4j startup

```bash
export JAVA_HOME=$HOME/workspace/neo4j-dist/jdk-17.0.20.1+1
$HOME/workspace/neo4j-dist/neo4j-community-5.26.0/bin/neo4j start
# verify
no_proxy='*' NO_PROXY='*' SGRAG_NEO4J_PASSWORD=... \
  .venv/bin/python -c "
from neo4j import GraphDatabase
d = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j','...'))
with d.session() as s:
    print(s.run(\"SHOW INDEXES YIELD name, state WHERE name='chunk_embedding_idx' RETURN state\").data())"
```

## 4. Launch the app

```bash
cd ~/workspace/scientific-literature-graphrag
no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 \
  SGRAG_NEO4J_PASSWORD=... \
  SGRAG_SCISPACY_MODEL=$HOME/workspace/scispacy_models/en_ner_bc5cdr_md-0.5.4/en_ner_bc5cdr_md/en_ner_bc5cdr_md-0.5.4 \
  .venv/bin/streamlit run app.py
```

Non-UI fallback (always works):

```bash
no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1 \
  SGRAG_NEO4J_PASSWORD=... \
  SGRAG_SCISPACY_MODEL=... \
  .venv/bin/python scripts/ask.py --query "What are the risk factors associated with Alzheimer's disease?"
```

Retrieval-only smoke test:

```bash
.venv/bin/python scripts/retrieve.py --query "..." [--rerank] [--top 5]
```

## 5. Sample questions

1. What are the risk factors associated with Alzheimer's disease?
2. What drugs are associated with breast cancer treatment?
3. How does inflammation affect cardiovascular disease?

## 6. Expected workflow (what to narrate)

1. **Query classification** — factual / entity-focused / comparison /
   multi-part (bounded ≤3 subqueries) / unsupported. Shown in diagnostics.
2. **Hybrid retrieval** — MiniLM vector search (top-20) + entity→MENTIONS
   expansion (hub-capped) + SEMANTIC_SIMILAR 1-hop expansion.
3. **Fusion** — weighted 0.6/0.25/0.15 (or RRF), deterministic ranking.
4. **CrossEncoder rerank** — top-50 candidates reordered (toggle in sidebar).
5. **Evidence packing** — top-8 chunks with PMID/title/text/scores/sources.
6. **Grounded generation** — LLM answers ONLY from evidence, `[PMID:...]`
   citations; declines when evidence is insufficient.
7. **Citation validation** — every citation checked against the evidence;
   fabricated ones trigger one stricter regeneration, then removal + warning.

Point reviewers at: the **Retrieval transparency** panel (counts, entities,
latencies), the **Evidence** expanders (scores, sources, matched entities),
and the **Evidence graph** (query entity → chunks → similar chunks).

## 7. Architecture overview

```
USER QUERY
    ↓
Query Classification (factual / entity / comparison / multipart / unsupported)
    ↓ (multipart → ≤3 bounded subqueries, evidence fused)
Hybrid Retrieval
 ├── Vector Search (Neo4j chunk_embedding_idx, cosine, top-20)
 ├── Entity Retrieval (scispaCy bc5cdr → MENTIONS, hub_degree_cap=1000)
 └── Semantic Graph Expansion (SEMANTIC_SIMILAR 1-hop, top-3 per hit)
    ↓
Candidate Fusion (weighted 0.6/0.25/0.15, deterministic)
    ↓
CrossEncoder Reranking (ms-marco-MiniLM-L-6-v2, top-50 pool)
    ↓
Evidence Selection (top-8 packed with provenance)
    ↓
Grounded LLM Generation (Groq or mock; strict evidence-only prompt)
    ↓
Citation Validation ([PMID:...] checked vs evidence; repair-or-strip)
    ↓
Streamlit UI (answer, citations, evidence, diagnostics, evidence graph)
```

## 8. Measured evaluation (Phase 5A)

Question→source-abstract retrieval on 200 PubMedQA questions
(PMID-deduplicated ranking). **This is a retrieval benchmark, not
open-domain scientific QA accuracy.**

| metric | Hybrid | Hybrid + CrossEncoder | delta |
|---|---|---|---|
| Recall@5 | 0.9450 | 0.9850 | +0.0400 |
| Recall@10 | 0.9750 | 0.9850 | +0.0100 |
| MRR | 0.8623 | 0.9689 | +0.1066 |
| nDCG@10 | 0.8885 | 0.9729 | +0.0844 |

Full per-query results: `data/evaluation/retrieval_eval_5a.json`.

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| "Could not reach the literature database" | Start Neo4j (step 3); check `SGRAG_NEO4J_PASSWORD` |
| Vector index not ONLINE | Re-run `scripts/validate_graph.py`; index rebuilds on demand |
| Slow first query (~30s) | One-time model loads (MiniLM, scispaCy, CrossEncoder); cached after |
| "LLM unavailable" | Set `SGRAG_LLM_PROVIDER=groq` + `SGRAG_LLM_API_KEY`, or keep mock |
| HF hub download hangs | Export `HF_HUB_OFFLINE=1` and `no_proxy`/`NO_PROXY` as in step 4 |
| Empty answer / "insufficient evidence" | Query may be non-biomedical or too vague; try an example |

## 10. Limitations (state these honestly in review)

- Corpus is 5,000 abstracts (PubMedQA-derived), not all of PubMed.
- Entity extraction covers DISEASE/CHEMICAL only; noisy spans possible.
- The mock LLM provider is illustrative, not a real answerer.
- 77% of chunks are HDBSCAN singletons (small, fine-grained units).
- No multi-hop reasoning beyond 1-hop graph expansion.
- Neo4j runs natively on the VM (no Docker here); restart manually after
  VM replacement — data persists under `~/workspace/neo4j-dist`.
