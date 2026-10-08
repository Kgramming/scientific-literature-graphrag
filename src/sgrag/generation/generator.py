"""Grounded generation orchestrator (Phase 5B).

Full pipeline:
    query -> classify -> (decompose | direct) -> hybrid retrieval (+rerank)
    -> evidence packing -> LLM -> citation validation
    -> (one stricter retry | strip+warn) -> GenerationResult

Insufficient-evidence and unsupported questions never reach the LLM.
"""

from __future__ import annotations

import time

from sgrag.generation import citations as cite_mod
from sgrag.generation.config import GenerationConfig, load_generation_config
from sgrag.generation.evidence import pack_evidence
from sgrag.generation.llm import LLMError, LLMProvider, make_provider
from sgrag.generation.models import CitationValidation, GenerationResult
from sgrag.generation.prompts import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    UNSUPPORTED_QUESTION_ANSWER,
    build_messages,
)
from sgrag.query import QueryClass, classify_query
from sgrag.retrieval import HybridRetriever


class GroundedGenerator:
    def __init__(
        self,
        gen_config: GenerationConfig | None = None,
        retriever: HybridRetriever | None = None,
        provider: LLMProvider | None = None,
        retrieval_config_path: str = "configs/retrieval.yaml",
    ) -> None:
        self.gen_config = gen_config or load_generation_config()
        self._retriever = retriever
        self._retrieval_config_path = retrieval_config_path
        self._provider = provider

    @property
    def retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever(self._retrieval_config_path)
        return self._retriever

    @property
    def llm(self) -> LLMProvider:
        if self._provider is None:
            gc = self.gen_config
            self._provider = make_provider(
                gc.provider, gc.model, api_key=gc.api_key,
                api_base=gc.api_base, timeout_s=gc.request_timeout_s)
        return self._provider

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()

    # -- pipeline ------------------------------------------------------
    def _retrieve_all(self, classified) -> tuple[list, dict]:
        """Run the retrieval pipeline per (sub)query; fuse evidence."""
        queries = classified.subqueries or [classified.text]
        merged: dict[str, object] = {}
        diag: dict = {"subqueries": queries,
                      "query_class": classified.query_class.value}
        n_vec = n_ent = n_sem = 0
        for sq in queries:
            results, d = self.retriever.retrieve(sq)
            n_vec += d.get("n_vector", 0)
            n_ent += d.get("n_entity", 0)
            n_sem += d.get("n_semantic", 0)
            for r in results:
                cur = merged.get(r.chunk_id)
                if cur is None or r.fused_score > cur.fused_score:
                    merged[r.chunk_id] = r
            # keep first query's entity list for the report
            diag.setdefault("query_entities", d.get("query_entities"))
            for k in ("embed_ms", "vector_ms", "entity_ms", "fusion_ms",
                      "rerank_ms"):
                diag[k] = diag.get(k, 0) + d.get(k, 0)
        fused = sorted(merged.values(),
                       key=lambda r: (-(r.final_score if r.final_score is not None
                                         else r.fused_score), r.chunk_id))
        for i, r in enumerate(fused, start=1):
            r.rank = i
        diag.update({"n_vector": n_vec, "n_entity": n_ent, "n_semantic": n_sem,
                     "n_fused": len(fused)})
        return fused, diag

    def answer(self, query: str) -> GenerationResult:
        t0 = time.perf_counter()
        classified = classify_query(query)
        result = GenerationResult(query=query, model=self.gen_config.model,
                                  provider=self.gen_config.provider)

        if classified.query_class == QueryClass.UNSUPPORTED:
            result.answer = UNSUPPORTED_QUESTION_ANSWER
            result.warnings.append(f"unsupported question: {classified.reason}")
            result.retrieval_metadata = {"query_class": "unsupported"}
            return result

        t = time.perf_counter()
        try:
            results, diag = self._retrieve_all(classified)
        except Exception as exc:  # noqa: BLE001 - retrieval errors -> message
            result.answer = (
                "Retrieval failed: the literature database could not be "
                f"reached ({exc}). Please try again later.")
            result.warnings.append(f"retrieval error: {exc}")
            return result
        result.retrieval_metadata = diag
        diag["retrieval_ms"] = round((time.perf_counter() - t) * 1000, 1)

        evidence = pack_evidence(results, top_k=self.gen_config.evidence_top_k)
        result.evidence = evidence
        if not evidence:
            result.answer = INSUFFICIENT_EVIDENCE_ANSWER
            result.warnings.append("no evidence retrieved; LLM not called")
            return result

        pmids = {e.pmid for e in evidence}
        chunk_ids = {e.chunk_id for e in evidence}

        def _generate(strict: bool) -> str:
            messages = build_messages(query, evidence, strict=strict)
            t_g = time.perf_counter()
            try:
                text = self.llm.generate(
                    messages, max_tokens=self.gen_config.max_tokens,
                    temperature=self.gen_config.temperature)
            except LLMError as exc:
                raise exc
            result.generation_metadata["generation_ms"] = round(
                (time.perf_counter() - t_g) * 1000, 1)
            result.generation_metadata["strict_retry"] = strict
            return text

        try:
            answer = _generate(strict=False)
        except LLMError as exc:
            result.answer = (
                "The language model is currently unavailable "
                f"({exc}). Evidence was retrieved but no answer could be "
                "generated.")
            result.warnings.append(f"LLM error: {exc}")
            return result

        validation = cite_mod.validate_citations(answer, pmids, chunk_ids)
        if not validation.valid and self.gen_config.max_citation_retries > 0:
            try:
                answer2 = _generate(strict=True)
            except LLMError:
                answer2 = None
            if answer2 is not None:
                v2 = cite_mod.validate_citations(answer2, pmids, chunk_ids)
                v2.regenerated = True
                if v2.valid:
                    answer, validation = answer2, v2
                else:
                    validation = v2
        if not validation.valid:
            answer = cite_mod.strip_invalid_citations(answer, validation.invalid)
            bad = ", ".join(
                f"PMID:{c.pmid}" + (f"|CHUNK:{c.chunk_id}" if c.chunk_id else "")
                for c in validation.invalid)
            result.warnings.append(
                f"removed {len(validation.invalid)} unsupported citation(s) "
                f"({bad}); remaining claims are evidence-grounded")

        result.answer = answer
        result.citations = validation.citations
        result.citation_validation = validation
        result.generation_metadata["total_ms"] = round(
            (time.perf_counter() - t0) * 1000, 1)
        return result
