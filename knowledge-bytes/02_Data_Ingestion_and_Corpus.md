# 02 — Data Ingestion and Corpus

## 1. What this component does

Fetches ~5,000 PubMed abstracts, audits them, and produces the versioned corpus manifests that everything downstream depends on. This is the "raw material" stage — no chunking, no embeddings, no graph yet.

## 2. Why it exists

Every claim in the project ("5,000 articles", "Recall@5 0.9850") traces back to a specific, auditable corpus. Without manifests carrying `{producer_script, config_hash, seed, timestamp}` provenance, results wouldn't be reproducible or trustworthy.

## 3. Where it lives in the repository

- `src/sgrag/data/pubmed_fetch.py` — `fetch_batches()`, `parse_pubmed_xml()`: batched PubMed E-utilities fetching with retries
- `src/sgrag/data/manifest.py` — `build_manifest()`, `write_manifest()`: provenance manifest construction
- `src/sgrag/data/audit.py` — corpus auditing (drops are audited, never hidden)
- `src/sgrag/data/pilot.py` — the audited 200-abstract pilot subset
- `scripts/build_full_corpus.py`, `scripts/build_pilot_corpus.py` — the runnable procedures
- `scripts/audit_full_corpus.py`, `scripts/audit_pilot.py` — audit procedures
- `data/evaluation/` — committed manifests (`full_corpus_manifest.json`, `pilot_corpus_manifest.json`)

## 4. Important classes/functions

- **`fetch_batches(pmids, ...)`** (`pubmed_fetch.py`) — fetches abstracts in batches from NCBI E-utilities with delay/retry logic. *Input:* list of PMIDs. *Output:* raw XML bytes. *Why batched:* NCBI rate-limits; hammering it gets you blocked.
- **`parse_pubmed_xml(xml_bytes)`** — parses XML into article records (pmid, title, journal, year, abstract).
- **`build_manifest(...)`** (`manifest.py`) — creates the provenance record: which script, which config hash, which seed, when. *This* is what makes the corpus a scientific artifact rather than "some files I downloaded."
- **`Article`** — the project's core data unit: `{pmid, title, journal, year, abstract}`. PMID is the natural primary key (hence the `article_pmid` uniqueness constraint in Neo4j).

## 5. Step-by-step execution

```text
scripts/build_full_corpus.py
  → fetch_batches(pmids)            [NCBI E-utilities, polite delays]
    → parse_pubmed_xml()            [XML → article dicts]
  → audit (audit_full_corpus.py)    [unfetchable PMIDs logged, not hidden]
  → build_manifest()                [provenance: script, config hash, seed, time]
  → write_manifest(data/evaluation/full_corpus_manifest.json)
  → raw JSONL to data/raw/          [git-ignored]
```

## 6. Data flow

`list[PMID]` → `XML bytes` → `list[article dicts]` → `manifest.json` + `data/raw/*.jsonl`. Downstream, `scripts/chunk_corpus.py` reads the raw articles; `scripts/build_graph.py` reads chunks + entities.

## 7. How it connects to the rest of the system

The corpus is the foundation everything stands on: chunking reads articles, the graph stores them as `:Article` nodes, retrieval returns their chunks, evaluation matches against their PMIDs. The manifest is the "birth certificate" reviewers can inspect.

## 8. Important design decisions

- **PMID as identity:** PubMed IDs are globally unique and stable — perfect natural keys, and they make PMID-equality gold matching in evaluation trivially correct.
- **Audit, don't hide:** unfetchable abstracts are logged in the audit, never silently dropped. A professor *will* ask "what about missing data" — this is the answer.
- **Manifests committed, raw data ignored:** `data/evaluation/*.json` is small and reproducibility-critical; `data/raw/` is large and refetchable. GitHub gets the provenance, not the gigabytes.

## 9. Example from the actual project

The full corpus: 5,000 abstracts → 5,000 `:Article` nodes in Neo4j (verified live via `get_graph_stats()`). The pilot: 200 audited abstracts used for embedding-model benchmarking and the 200-question eval set.

## 10. Common misunderstandings

- **"The data is in the repo."** No — only manifests. Raw/interim/processed data is git-ignored and rebuilt via scripts.
- **"PubMedQA is the corpus."** No — PubMedQA provides the *evaluation questions*; the corpus is PubMed abstracts. Related but distinct.

## 11. Professor Viva Questions

- Where did your data come from?
- How do you know your corpus is complete/correct?
- What happens to abstracts you couldn't fetch?
- What is a manifest and why does it matter?
- Why aren't the raw datasets on GitHub?

## 12. Natural Spoken Answers

**"Where did your data come from?"**
> "Sir, from PubMed via NCBI's E-utilities API. We fetch about 5,000 abstracts in polite batches, parse the XML into article records with PMID, title, journal, year, and abstract. Every fetch is audited — if an abstract can't be fetched, it's logged, never silently dropped — and we write a manifest recording exactly which script, config, and seed produced the corpus, so it's fully reproducible."

## 13. Knowledge Bites

- Corpus = ~5,000 PubMed abstracts; pilot = 200 audited abstracts.
- `fetch_batches()` respects NCBI rate limits with delays and retries.
- PMID is the natural primary key everywhere (articles, evaluation gold matching).
- Manifests carry `{producer_script, config_hash, seed, timestamp}` provenance.
- Audit logs drops; nothing is silently hidden.
- Raw data is git-ignored; manifests are committed — provenance over gigabytes.
- `parse_pubmed_xml()` turns NCBI XML into clean article dicts.
- The corpus feeds chunking → entities → graph → retrieval → evaluation.
