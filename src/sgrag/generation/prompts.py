"""Strict evidence-grounded prompts (Phase 5B).

The LLM answers ONLY from supplied evidence, cites with [PMID:...] markers,
and declines when evidence is insufficient. A stricter variant is used for
the single citation-repair regeneration.
"""

SYSTEM_PROMPT = """\
You are a scientific literature assistant. You answer ONLY from the evidence \
provided below. Rules:

1. Every factual claim must be supported by the evidence. Cite it inline as \
[PMID:<pmid>] or [PMID:<pmid> | CHUNK:<chunk_id>] using EXACTLY the PMIDs and \
chunk IDs from the evidence. Never invent a PMID or chunk ID.
2. If the evidence is insufficient to answer, say so explicitly: \
"Insufficient evidence was retrieved from the indexed scientific literature \
to answer this question reliably." Do not guess.
3. Distinguish what the evidence REPORTS from your own inference. Do not \
claim causality when the evidence only reports an association.
4. Preserve scientific uncertainty: use "suggests", "associated with", \
"reported" where the evidence warrants it.
5. Be concise and structured. No markdown tables unless asked.
"""

STRICT_SYSTEM_PROMPT = SYSTEM_PROMPT + """\
6. CRITICAL: your previous answer contained citations that do not match the \
evidence. This time, cite ONLY PMIDs and chunk IDs that appear verbatim in \
the evidence list. If you cannot support a claim, omit the claim.
"""


def build_evidence_block(items) -> str:
    """Render the bounded evidence context for the prompt."""
    parts = []
    for i, it in enumerate(items, start=1):
        parts.append(
            f"[E{i}] PMID:{it.pmid} | CHUNK:{it.chunk_id}\n"
            f"Title: {it.title}\n"
            f"Text: {it.text}"
        )
    return "\n\n".join(parts)


def build_messages(query: str, items, strict: bool = False) -> list[dict]:
    system = STRICT_SYSTEM_PROMPT if strict else SYSTEM_PROMPT
    user = (
        f"EVIDENCE:\n{build_evidence_block(items)}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer the question using only the evidence above, with inline "
        "citations in the form [PMID:<pmid>] or [PMID:<pmid> | CHUNK:<chunk_id>]."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


INSUFFICIENT_EVIDENCE_ANSWER = (
    "Insufficient evidence was retrieved from the indexed scientific "
    "literature to answer this question reliably."
)

UNSUPPORTED_QUESTION_ANSWER = (
    "This question does not appear to be a scientific literature question "
    "that this system can answer from its indexed corpus of biomedical "
    "abstracts. Please ask about biomedical topics covered by the literature."
)
