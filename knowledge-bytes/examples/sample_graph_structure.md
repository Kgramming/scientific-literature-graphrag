# Example: Sample Graph Structure

**Type:** illustrative (node/edge *shapes* are real per `src/sgrag/graph/schema.py`; PMIDs/text are illustrative).

```text
(:Article {pmid: "12345678", title: "Tamoxifen in breast cancer therapy",
           journal: "J Clin Oncol", year: 2021})
  |
  | [:HAS_CHUNK]
  v
(:Chunk {chunk_id: "12345678:0003", pmid: "12345678",
         strategy: "fixed_token", token_count: 243,
         text: "Tamoxifen was administered to 240 patients...",
         embedding: [0.013, -0.042, ..., 0.087]  (384 dims)})
  |
  | [:MENTIONS {surface_form: "breast cancer", char_start: 41, char_end: 54}]
  v
(:Entity {normalized_key: "disease:breast cancer",
          canonical_name: "breast cancer", entity_type: "DISEASE"})
  ^
  | [:MENTIONS {surface_form: "mammary carcinoma", ...}]
  |
(:Chunk {chunk_id: "87654321:0007", ...})   ← different wording, same entity

(:Chunk {chunk_id: "12345678:0003"})
  |
  | [:SEMANTIC_SIMILAR {score: 0.91, rank: 1, model: "all-MiniLM-L6-v2"}]
  v
(:Chunk {chunk_id: "12345678:0004"})        ← neighboring chunk, similar content
```

**What this shows:**
- Articles contain chunks (`HAS_CHUNK`); chunks mention entities (`MENTIONS`); chunks link to similar chunks (`SEMANTIC_SIMILAR`).
- Two differently-worded chunks connect through one normalized `:Entity` node — the vocabulary-mismatch fix.
- `surface_form` preserves original text on the edge; `normalized_key` dedupes the node.

**Real counts** (via `GraphStore.counts()`): 5,000 Articles · 53,091 Chunks · 9,945 Entities · 265,455 SEMANTIC_SIMILAR edges.
