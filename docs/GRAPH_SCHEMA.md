# Knowledge Graph — Schema & Runbook (Phase 3B)

## 1. Graph schema

### Nodes

**(:Article)** — one per PubMed abstract (5,000).

| property | source |
|---|---|
| `pmid` | corpus record (unique) |
| `title`, `journal`, `year`, `abstract` | corpus record |

**(:Chunk)** — one per chunk from the **semantic_cluster (S1)** strategy.

| property | source |
|---|---|
| `chunk_id` | chunker output, e.g. `21645374:semantic_cluster:3` (unique) |
| `pmid` | parent article |
| `strategy` | always `semantic_cluster` in this build |
| `text` | chunk text |
| `token_count`, `sentence_start`, `sentence_end` | chunker provenance |
| `embedding` | all-MiniLM-L6-v2, 384-dim float list |

**(:Entity)** — one per normalized biomedical entity mention.

| property | meaning |
|---|---|
| `normalized_key` | deterministic unique key, `TYPE:normalized-name` |
| `canonical_name` | most frequent surface form (lexicographic tie-break) |
| `entity_type` | `DISEASE` or `CHEMICAL` |
| `aliases` | sorted distinct surface forms observed |

### Relationships

- `(:Article)-[:HAS_CHUNK]->(:Chunk)` — chunk provenance.
- `(:Chunk)-[:MENTIONS {surface_form, char_start, char_end}]->(:Entity)` —
  entity provenance back to the exact chunk span.
- `(:Chunk)-[:SEMANTIC_SIMILAR {score, rank, model}]->(:Chunk)` — bounded
  similarity edges (see §5).

### Constraints & indexes (idempotent, `IF NOT EXISTS`)

- Uniqueness: `Article.pmid`, `Chunk.chunk_id`, `Entity.normalized_key`.
- Indexes: `Chunk.pmid`, `Entity.entity_type`.
- Vector index `chunk_embedding_idx` on `Chunk.embedding`
  (384 dims, cosine).

Because every write is a `MERGE`, re-running the build duplicates nothing.

## 2. Chunk strategy used

**semantic_cluster (S1)** — the validated/retained configuration from Phase 3:

- best coherence gap on the full corpus (+0.259 vs −0.420/−0.324),
- highest intra-chunk coherence (0.623),
- small coherent units (mean 35.5 tokens) giving precise entity provenance.

Strategies are never mixed within one build; `configs/graph_build.yaml`
pins `chunk_strategy: semantic_cluster` and the build script refuses anything else.

## 3. Entity extraction

- Model: scispaCy `en_ner_bc5cdr_md` (disease/chemical NER), via
  `src/sgrag/graph/entities.py` (`BiomedicalEntityExtractor`).
- Only `DISEASE` / `CHEMICAL` labels are kept; empty-after-normalization
  spans are dropped. The model emits spans only — nothing is hallucinated.
- Every mention keeps provenance: `chunk_id`, `surface_form`, character offsets.

## 4. Normalization (deterministic, conservative)

`normalized_key = TYPE:normalized-name`, where normalization is:

1. Unicode NFKC, 2. lowercase, 3. drop possessives (`'s`), 4. punctuation →
   space, 5. collapse whitespace.

So `Alzheimer's Disease`, `alzheimer disease`, `ALZHEIMER'S DISEASE` all map to
`DISEASE:alzheimer disease`. No alias dictionary and no fuzzy merging are
applied — synonym resolution beyond these rules is **not** claimed.

## 5. Semantic similarity edges (bounded)

- Rule: exactly **top_k = 5** outgoing `SEMANTIC_SIMILAR` edges per chunk,
  self-matches excluded, computed from L2-normalized MiniLM embeddings with
  cosine similarity (`src/sgrag/graph/similarity.py`).
- No all-pairs graph is ever materialized (blocked brute-force kNN).
- Edge properties: `score` (cosine), `rank` (1–5), `model`.

## 6. Vector index

- Name: `chunk_embedding_idx`, on `Chunk.embedding`.
- 384 dimensions, `cosine` similarity function, model `all-MiniLM-L6-v2`.
- Embeddings are stored raw (unnormalized); the index handles cosine.

## 7. Runbook

### 7a. Docker path (canonical architecture)

```bash
cd ~/workspace/scientific-literature-graphrag
export SGRAG_NEO4J_PASSWORD=<your-secure-password>   # or your own; never commit it
docker compose up -d
docker compose exec neo4j cypher-shell -u neo4j -p "$SGRAG_NEO4J_PASSWORD" "RETURN 1"
```

### 7b. Native path (used on this build VM — no Docker available)

```bash
source ~/workspace/neo4j-dist/env.sh   # NEO4J_HOME, JAVA_HOME
export SGRAG_NEO4J_PASSWORD=<your-secure-password>
$NEO4J_HOME/bin/neo4j start
$NEO4J_HOME/bin/cypher-shell -u neo4j -p "$SGRAG_NEO4J_PASSWORD" \
  "RETURN apoc.version() AS apoc, gds.version() AS gds"
```

Native install: Neo4j Community 5.26.0 + JDK 17, APOC core 5.26.0
(bundled `labs/` jar) and GDS 2.13.2 in `plugins/`. Binaries live in
`~/workspace/neo4j-dist/` (outside the repo, git-ignored by location).

#### 7c. Build-VM notes (this environment)

- Docker is unavailable on this VM, hence the native install above.
  `docker-compose.yml` remains the canonical architecture.
- The egress proxy breaks `huggingface_hub`'s HTTP client: the build resolves
  the MiniLM model from the local HF cache snapshot and runs with
  `no_proxy='*' NO_PROXY='*' HF_HUB_OFFLINE=1`.
- The legacy bc5cdr wheel does not build on Python 3.12; the model tarball was
  extracted manually to `~/workspace/scispacy_models/` and passed via
  `SGRAG_SCISPACY_MODEL` (the inner `en_ner_bc5cdr_md-0.5.4` data directory).

### Build the graph

```bash
export SGRAG_NEO4J_PASSWORD=<your-secure-password>
# if the bc5cdr model was manually extracted (wheel does not build on py3.12):
export SGRAG_SCISPACY_MODEL=$HOME/workspace/scispacy_models/en_ner_bc5cdr_md-0.5.4/en_ner_bc5cdr_md
.venv/bin/python scripts/build_graph.py --config configs/graph_build.yaml
# smoke test on 50 articles:  --limit 50
# wipe and rebuild:            --reset
```

### Validate

```bash
export SGRAG_NEO4J_PASSWORD=<your-secure-password>
.venv/bin/python scripts/validate_graph.py --config configs/graph_build.yaml
```

### Reset / rebuild safely

```bash
.venv/bin/python scripts/build_graph.py --config configs/graph_build.yaml --reset
```

`--reset` deletes all nodes/relationships but keeps constraints/indexes;
without it the build is idempotent (MERGE-only writes).

### Stop Neo4j (native)

```bash
source ~/workspace/neo4j-dist/env.sh && $NEO4J_HOME/bin/neo4j stop
```

## 8. Module map

- `src/sgrag/graph/config.py` — `GraphConfig`, env-based secrets, validation.
- `src/sgrag/graph/schema.py` — node/rel definitions, idempotent DDL, param builders.
- `src/sgrag/graph/entities.py` — scispaCy adapter + deterministic normalization.
- `src/sgrag/graph/similarity.py` — bounded top-k edge construction (pure NumPy).
- `src/sgrag/graph/ingest.py` — `GraphStore` (batched UNWIND+MERGE writes).
- `scripts/build_graph.py`, `scripts/validate_graph.py` — build + validation CLIs.
- `configs/graph_build.yaml` — build configuration (no secrets).
