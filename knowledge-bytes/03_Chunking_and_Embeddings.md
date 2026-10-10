# 03 — Chunking and Embeddings

## 1. What this component does

Splits each article abstract into retrievable chunks and converts chunk text into 384-dimensional vectors. This is the bridge between "documents" and "searchable units."

## 2. Why it exists

You can't embed a whole abstract and expect precise retrieval — and you can't retrieve what you haven't chunked. Chunking strategy directly affects what the vector index can find. This project *measured* four strategies instead of guessing.

## 3. Where it lives in the repository

- `src/sgrag/chunking/base.py` — chunker interface
- `src/sgrag/chunking/fixed_token.py` — `chunk_record()`: 256 tokens, 32 overlap, sentence-safe (the **selected** strategy)
- `src/sgrag/chunking/sentence_boundary.py`, `sentences.py` — sentence-based baselines
- `src/sgrag/chunking/semantic_cluster.py` — `group_sentences()`: embedding-based semantic clustering
- `src/sgrag/chunking/coherence.py`, `evaluate.py`, `tokenizer.py` — coherence scoring, evaluation, tokenization
- `src/sgrag/embeddings/embed.py` — `load_model()`, `encode()`: sentence-transformers wrapper with disk caching
- `src/sgrag/embeddings/config.py` — embedding configuration
- `scripts/chunk_corpus.py`, `scripts/evaluate_chunking.py`, `scripts/compare_embeddings.py`

## 4. Important classes/functions

- **`chunk_record(record, config, tokenizer)`** (`fixed_token.py`) — splits one article into ~256-token chunks with 32-token overlap, cutting only at sentence boundaries (`_boundary_cuts`). *Input:* article dict. *Output:* list of chunk dicts with `chunk_id`, `text`, `token_count`, `sentence_start/end`.
- **`group_sentences(sentences, embeddings, config)`** (`semantic_cluster.py`) — clusters sentence embeddings, merges contiguous runs, enforces max tokens. The experimental alternative.
- **`encode(texts, config, use_cache=True)`** (`embed.py`) — batch-encodes via sentence-transformers with a disk cache keyed by model+text hash, so re-runs don't recompute.
- **`load_model(model_name, device="cpu")`** — loads `all-MiniLM-L6-v2` (384-dim).

## 5. Step-by-step execution

```text
scripts/chunk_corpus.py
  → for each article: chunk_record(article)     [256 tok / 32 overlap, sentence-safe]
  → 5,000 articles → 53,091 chunks
  → encode(chunk_texts)                          [all-MiniLM-L6-v2, 384-dim, cached]
  → chunks + embeddings → scripts/build_graph.py [stored as :Chunk nodes]
```

## 6. Data flow

`article {pmid, abstract}` → `list[chunk {chunk_id, pmid, text, token_count, ...}]` → `list[vector[384]]` → stored on `:Chunk.embedding` in Neo4j + vector index.

## 7. How it connects to the rest of the system

Chunks are the *unit of retrieval*: the Neo4j vector index is built on `Chunk.embedding`, `SEMANTIC_SIMILAR` edges connect similar chunks, and `RetrievalResult` objects carry chunk text to the reranker and evidence packer. Chunk IDs are the join key across vector hits, entity expansions, and fusion.

## 8. Important design decisions

- **Fixed-token won on measurement**, not taste. Four strategies were compared via `scripts/evaluate_chunking.py`; 256/32 sentence-safe was selected.
- **Sentence-safe cuts:** chunks never split mid-sentence (`_boundary_cuts`), so each chunk is coherent prose — critical for the cross-encoder, which scores (query, chunk_text) pairs.
- **all-MiniLM-L6-v2 kept; PubMedBERT rejected:** the candidate `S-PubMedBert-MS-MARCO` was measured at ~0.80 cosine on *unrelated* text (severe anisotropy), 6.2× slower, 2× dimensions — with no measured retrieval gain. Documented, not assumed.
- **Disk-cached embeddings:** `encode()` caches by content hash; rebuilding the graph doesn't recompute 53k vectors.
- **384 dimensions** is the MiniLM output size — small, fast, CPU-friendly.

## 9. Example from the actual project

5,000 abstracts → 53,091 chunks (avg ~10.6 chunks/article). Each chunk: `chunk_id` like `pmid:0003`, 256 tokens, 32-token overlap with neighbors, stored with its 384-dim vector in Neo4j.

## 10. Common misunderstandings

- **"Semantic chunking is what's used."** No — semantic clustering was *evaluated*; fixed-token was *selected* and is what the graph actually contains. Distinguish experiments from the production pipeline.
- **"Bigger embeddings are better."** The PubMedBERT candidate had 768 dims and performed worse (anisotropy). Dimensions ≠ quality.
- **"Chunking is just splitting text."** The sentence-safety, overlap, and token-count metadata all serve downstream: coherent reranker inputs and precise evidence spans.

## 11. Professor Viva Questions

- Why chunk documents at all?
- Which chunking strategy did you use and why?
- What is an embedding in your project?
- Why 384 dimensions? Why MiniLM over PubMedBERT?
- Where are embeddings stored? How are they searched?

## 12. Natural Spoken Answers

**"Which chunking strategy and why?"**
> "Sir, we compared four — fixed-token, sentence-boundary, sentence-based, and semantic clustering — using our own evaluation script. Fixed-token at 256 tokens with 32 overlap, cut only at sentence boundaries, won on the measurements. The sentence-safety matters because our cross-encoder reranker scores query-chunk pairs, and it needs coherent prose, not fragments."

**"Why MiniLM instead of a biomedical model?"**
> "We actually tested a PubMedBERT variant, sir. It was severely anisotropic — unrelated texts scored 0.80 cosine similarity — six times slower, twice the dimensions, and gave no measured retrieval gain. So we kept all-MiniLM-L6-v2: 384 dimensions, fast on CPU, and empirically better for our task."

## 13. Knowledge Bites

- 5,000 articles → 53,091 chunks via fixed-token (256/32, sentence-safe).
- Four strategies evaluated; fixed-token selected on measurements.
- Chunks never split mid-sentence — coherence for the reranker.
- Embeddings: all-MiniLM-L6-v2, 384-dim, CPU-friendly.
- PubMedBERT candidate rejected: anisotropy (~0.80 unrelated cosine), 6.2× slower, no gain.
- `encode()` disk-caches by content hash — no recompute on rebuilds.
- Chunk IDs join vector hits, entity expansions, and fusion.
- Embeddings stored on `:Chunk.embedding`, indexed by Neo4j vector index.
