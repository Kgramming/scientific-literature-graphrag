# Knowledge Bytes — Study Guide (planning-stage)

> This is the **planned** concept map for viva preparation. As each phase lands, entries will be
> rewritten to describe **our actual implementation** (file paths, parameters, measured numbers) —
> never generic definitions.

## 1. Why GraphRAG for scientific literature?
Vector search finds *similar text*; scientific questions need *connected knowledge* (a drug → its
mechanism → the disease it treats often live in different abstracts). The graph stores those
connections explicitly (shared entities, co-occurrence, MeSH topics) so retrieval can hop between
them. Our experiments measure exactly when the hops help (multi-hop/relationship queries) and when
they add noise (narrow single-fact queries).

## 2. Chunking strategies (what / why / ours)
- **Fixed-token:** slices text every N tokens. Simple, reproducible, but cuts mid-sentence and
  mid-idea — the baseline everything must beat.
- **Sentence-boundary:** never splits sentences; packs to a token budget. Respects linguistic units.
- **Semantic clustering (reference-style):** embeds sentences, HDBSCAN-clusters them; clusters become
  chunks. Captures topical grouping but can scatter narrative order and emit singleton noise chunks.
- **Semantic sections (ours):** adjacent-sentence cosine similarity → boundaries where similarity
  drops (topic shifts) → min/max size guards. Preserves *sections*, the unit a scientist would cite.
- **Cohesion:** intra-chunk similarity should exceed inter-chunk similarity; we plot both
  distributions per strategy as evidence the chunker does something real.

## 3. Embeddings & vector search
`all-MiniLM-L6-v2` maps text → 384D vectors where cosine similarity ≈ semantic similarity
(required baseline). Neo4j's vector index uses HNSW (approximate nearest neighbor): a navigable
graph over vectors giving ~log(N) search instead of scanning all 20k chunks. Chunking quality
directly bounds retrieval quality — a perfect vector index cannot find an idea split across chunks.

## 4. Neo4j: property graph, Cypher, APOC, GDS
- **Property graph:** nodes + relationships, each carrying key/value properties (our schema:
  Article/Chunk/Entity/Concept; see ARCHITECTURE_PROPOSAL §2.5).
- **Cypher:** declarative graph query language (`MATCH (c:Chunk)-[:MENTIONS]->(e:Entity) ...`).
- **APOC:** procedure library — we use it for batch inserts (`apoc.periodic.iterate`) so 100k+
  entity mentions load without one giant transaction.
- **GDS:** graph algorithms as transient in-memory projections. **PageRank** scores node importance
  by link structure (damping 0.85, few iterations on the *retrieved subgraph* — cheap and query-
  specific). **Louvain/Leiden** finds communities (research topics); we test it only as an optional
  retrieval signal.

## 5. Entity extraction & normalization
scispaCy `en_ner_bc5cdr_md` finds DISEASE/CHEMICAL spans (trained on the BC5CDR biomedical corpus —
the right tool for PubMed text). Raw spans are noisy ("TNF", "tnf", "tumor necrosis factor (TNF)"),
so we normalize to **canonical nodes** preserving surface forms and provenance. Bad normalization
merges distinct concepts (hurts precision); no normalization fragments the graph (hurts recall) —
we measure the tradeoff.

## 6. Hybrid retrieval & fusion
- **Vector:** semantic similarity. **Entity-aware:** query entities → chunks mentioning them
  (exact concept match, robust to paraphrase). **Graph expansion:** walk MENTIONS/RELATED_TO edges
  to find indirectly connected evidence (bounded depth, hub-guarded).
- **RRF:** `score = Σ 1/(k + rank)` — fuses ranked lists without calibrating raw scores.
- **Weighted sum:** normalize each signal to [0,1], then blend with config weights. Weights are
  *not* assumed optimal — we grid-search them and report sensitivity.

## 7. Cross-encoder reranking
Bi-encoders (retrieval) embed query and chunk independently — fast but shallow. A **cross-encoder**
(`cross-encoder/ms-marco-MiniLM-L-6-v2`) reads query+chunk *together* through full attention —
slower, more accurate; hence second-stage on a bounded pool (top-50 → 10). Graph-aware reranking
then blends reranker + graph proximity + entity overlap into one explainable score.

## 8. Query understanding & decomposition
Rule-based classifier → query class → routing table (explainable: "relationship question → entity +
graph retrieval"). Decomposition (LLM) splits multi-hop questions into sub-questions, each retrieved
independently, evidence merged. Validators (distinctness, drift guard, non-empty fallback) ensure a
bad split degrades to the original query instead of destroying results.

## 9. Grounded generation & citations
The LLM sees only retrieved chunks, must cite `[C<chunk_id>]`, distinguish evidence from inference,
and say when evidence is insufficient. Citations are *validated* against retrieved IDs. Claim→chunk
mapping lets the UI show exactly which text supports each claim.

## 10. Metrics — what each actually means
- **Recall@k:** fraction of gold articles appearing in top-k (coverage).
- **MRR:** 1/rank of the first gold hit, averaged (how high the first answer ranks).
- **nDCG@k:** rank-discounted gain, rewards putting gold higher.
- **Precision@k:** fraction of top-k that is gold.
- **ROUGE-L:** longest-common-subsequence overlap with the reference answer (lexical, not factual).
- **BERTScore:** embedding-based token similarity (semantic overlap, still not factuality).
- **Faithfulness suite:** citation precision, evidence coverage, unsupported-claim rate — our
  honesty layer *about* the answer, alongside the lexical metrics.

## 11. Likely viva questions (with our answers — to be finalized post-implementation)
1. *Why not just use a bigger embedding model?* → Domain fit and chunking quality dominate; we
   ablate the embedding choice and report the delta.
2. *Doesn't graph expansion add noise?* → Yes, on single-fact queries — measured in C-vs-A
   per-class breakdown; routing mitigates it.
3. *Why is your eval set trustworthy?* → Deterministic PMID-equality matching, manifest with drops
   audited, same questions for all systems, no fabricated mappings.
4. *What happens if the LLM decomposes badly?* → Validators + fallback; worst case equals no
   decomposition (measured).
5. *Why rule-based classification instead of an LLM?* → Routing decisions must be explainable and
   deterministic; the LLM is used where generation is needed (decomposition), guarded by rules.
6. *What did you do that the reference didn't?* → PMID-grounded eval at scale, un-merged semantic
   chunking, biomedical NER + canonicalization, cross-encoder rerank, validated decomposition,
   faithfulness + latency measurement, config-driven reproducible experiments.
