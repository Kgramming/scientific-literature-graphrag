# 04 — Entity Extraction and Normalization

## 1. What this component does

Finds biomedical entities (diseases, chemicals/drugs) in chunk text and normalizes them to canonical keys, so that "breast cancer", "Breast Cancer", and "mammary carcinoma" can resolve to the same graph node.

## 2. Why it exists

Entities are the *joints* of the knowledge graph. Without them, chunks are isolated islands connected only by vector similarity. With them, a query mentioning "Alzheimer's" can walk `MENTIONS` edges to every chunk discussing it — even when the wording differs entirely. This is the "graph" in GraphRAG.

## 3. Where it lives in the repository

- `src/sgrag/graph/entities.py` — `ExtractedEntity`, `normalize_entity_name()`, `normalized_key()`, `normalize_entity()`
- `src/sgrag/retrieval/query.py` — `extract_query_entities()`: runs the extractor on the *query* at retrieval time
- Model: scispaCy `en_ner_bc5cdr_md` (BC5CDR: diseases + chemicals)
- `scripts/build_graph.py` — runs extraction over all chunks during graph build

## 4. Important classes/functions

- **`normalize_entity_name(name)`** — lowercases, strips punctuation/whitespace. `"Breast Cancer"` → `"breast cancer"`.
- **`normalized_key(entity_type, name)`** — combines type + normalized name: `("DISEASE", "breast cancer")` → `"disease:breast cancer"`. This key is the `:Entity` node's identity (uniqueness constraint).
- **`normalize_entity(entity_type, surface_form)`** — full pipeline for one mention → `{normalized_key, canonical_name, entity_type, surface_form, char_start/end}`.
- **`ExtractedEntity`** — dataclass carrying the normalized entity plus its character offsets in the chunk.
- **`extract_query_entities(query, model)`** (`retrieval/query.py`) — runs the same scispaCy pipeline on the user's query at retrieval time, returning normalized keys to match against graph `:Entity` nodes.

## 5. Step-by-step execution (entity lifecycle)

```text
chunk text: "...metformin treats type 2 diabetes..."
  → scispaCy en_ner_bc5cdr_md NER
    → ("metformin", CHEMICAL), ("type 2 diabetes", DISEASE)
  → normalize_entity_name() → "metformin", "type 2 diabetes"
  → normalized_key() → "chemical:metformin", "disease:type 2 diabetes"
  → MERGE (e:Entity {normalized_key})     [deduplicated node]
  → MERGE (c:Chunk)-[:MENTIONS {surface_form, char_start, char_end}]->(e)
```

Result in this project: **9,945 entities** (7,441 DISEASE, 2,504 CHEMICAL).

## 6. Data flow

`chunk.text` → `list[ExtractedEntity]` → `:Entity` nodes + `:MENTIONS` edges. At query time: `query string` → `extract_query_entities()` → `list[normalized_key]` → Cypher `MATCH (e:Entity) WHERE e.normalized_key IN $keys`.

## 7. How it connects to the rest of the system

Entities power the *entity expansion* retrieval path: `expand_entities()` in `retrieval/graph.py` walks `(:Entity)<-[:MENTIONS]-(:Chunk)` to find chunks mentioning query entities. The same normalized keys let query-time extraction match build-time nodes — normalization is the contract between the two.

## 8. Important design decisions

- **scispaCy BC5CDR, not generic NER:** biomedical text needs a biomedical model. BC5CDR covers exactly the two types that matter for this project's questions (diseases, chemicals/drugs).
- **Normalization before node creation:** without it, "Alzheimer's", "alzheimers", and "Alzheimer disease" become three nodes and the graph fragments. The `normalized_key` uniqueness constraint enforces it at the database level.
- **Surface form preserved on the edge:** `:MENTIONS {surface_form, char_start, char_end}` keeps the original text for display/provenance while the node stays canonical.
- **Same pipeline at build and query time:** `extract_query_entities()` uses the same model + normalization, so query entities actually match graph entities.

## 9. Example from the actual project

Query *"What drugs are associated with breast cancer treatment?"* → scispaCy finds `("breast cancer", DISEASE)` → normalized to `disease:breast cancer` → Cypher finds the `:Entity` node → `MENTIONS` expansion returns chunks discussing breast cancer → those chunks' own entities (drugs) become visible for the answer.

## 10. Common misunderstandings

- **"Entities are extracted by the LLM."** No — scispaCy NER, deterministic and local. The LLM never does extraction.
- **"Each mention becomes a node."** No — mentions become `:MENTIONS` *edges*; normalized entities become *nodes*. This is the deduplication.
- **"Entity types are unlimited."** Only DISEASE and CHEMICAL (BC5CDR). Genes, for example, are not extracted — a fair limitation to admit.

## 11. Professor Viva Questions

- What is an entity in your project?
- Which entity types? Why only those?
- What is normalization and why is it needed?
- How do entities connect chunks?
- What happens if the same disease is written three different ways?

## 12. Natural Spoken Answers

**"What is normalization and why?"**
> "Sir, NER finds surface mentions like 'Breast Cancer' or 'breast cancer' — different strings, same thing. We lowercase, strip punctuation, and combine with the entity type into a normalized key like `disease:breast cancer`. That key is the node's unique identity in Neo4j, enforced by a constraint. So all spelling variants collapse to one node, and the original wording is preserved on the MENTIONS edge for provenance."

## 13. Knowledge Bites

- scispaCy `en_ner_bc5cdr_md`: DISEASE + CHEMICAL only.
- 9,945 entities: 7,441 diseases, 2,504 chemicals.
- `normalized_key` = `type` + normalized name → the `:Entity` identity.
- Uniqueness constraint on `normalized_key` enforces dedup at the DB level.
- `:MENTIONS` edges carry `surface_form` + char offsets (provenance).
- Same extractor + normalization at build time and query time — that's the matching contract.
- Entities enable the graph expansion retrieval path (file 06).
- No LLM involved in extraction — deterministic local NER.
