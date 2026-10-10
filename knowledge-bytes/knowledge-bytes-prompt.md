# Knowledge Bytes — Methodology Prompt

This file documents the methodology used to generate and maintain the `knowledge-bytes/` documentation system for the Scientific Literature GraphRAG repository. Follow it when creating or updating any knowledge byte.

## 1. Source-of-truth rules

1. **Code first.** Every technical claim must trace to an actual file, function, or class in the repository. Name them.
2. **Read before writing.** Open the source file and read the implementation before describing it. Never describe from memory of a summary.
3. **No invented details.** If you haven't seen it in the code, don't write it. "The retriever probably caches embeddings" is forbidden; "`encode()` disk-caches by content hash (`src/sgrag/embeddings/embed.py`)" is required.
4. **Actual vs planned.** If docs describe something the code doesn't implement, label it **"Planned / Future Scope"** explicitly. If code implements something docs don't mention, describe the code.
5. **Verify references.** After writing, check: every referenced file exists, every named class/function exists, no claim lacks a source.

## 2. How to trace

- **Entry points first:** `app.py` → `ask_pipeline()` → `GroundedGenerator.answer()` → `HybridRetriever.retrieve()`. Follow the call chain.
- **Config second:** `configs/*.yaml` + `src/sgrag/*/config.py` reveal the actual tuned values (top-k, weights, caps). Quote them.
- **Tests third:** `tests/` shows intended behavior and edge cases the authors cared about.
- **Docs fourth:** `docs/` provides context and *reported* results — but code wins on conflicts.

## 3. File structure (mandatory per byte)

Each numbered file follows the 13-section format: what/why/where → classes/functions → step-by-step → data flow → connections → design decisions → example → misunderstandings → viva questions → spoken answers → knowledge bites.

## 4. Viva-oriented explanation

- Every byte ends with realistic professor questions and **natural spoken answers** — student voice, not textbook.
- Include "Common misunderstandings": things a student might *incorrectly* claim (e.g. "GDS runs retrieval", "there's a graph-aware reranker"). Correct them explicitly.
- The end-to-end file (13) must be narratable in ~2 minutes.

## 5. What to avoid

- Generic textbook explanations untied to this project ("Neo4j stores graph data.").
- Presenting planned features as implemented.
- Copying content from other projects' knowledge bytes (style may be referenced; content must be generated from this repo).
- Secrets, API keys, or credential values — never include them.
- Fake outputs — label illustrative examples as illustrative; cite real outputs with their source file.

## 6. Maintenance

When code changes:
1. Re-read the changed files.
2. Update the affected byte(s) — especially step-by-step flows, config values, and function names.
3. Re-verify all cross-references.
4. Keep the examples/ directory's "illustrative vs actual" labels accurate.

## 7. Quality bar

A byte is done when a student can open it, read it once, and answer "what happens here, why, and how" about that subsystem without opening the code.
