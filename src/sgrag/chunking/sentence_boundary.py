"""Strategy B — sentence-boundary baseline.

Sentences are never split: they are packed greedily so each chunk holds as
many complete sentences as fit within max_tokens. Chunk boundaries therefore
always fall on sentence boundaries (cuts_sentences is always False).
"""

from dataclasses import asdict

from .base import Chunk, SentenceBoundaryConfig, make_chunk_id
from .sentences import split_sentences
from .tokenizer import count_tokens, get_tokenizer


def chunk_record(record, config=None, tokenizer=None):
    """Chunk one pilot-corpus record on sentence boundaries."""
    config = config or SentenceBoundaryConfig()
    if isinstance(config, dict):
        config = SentenceBoundaryConfig(**config)
    tokenizer = tokenizer or get_tokenizer(config.tokenizer_model)

    pmid = str(record["pmid"])
    text = record.get("abstract") or ""
    source = {"title": record.get("title"), "journal": record.get("journal"),
              "year": record.get("year")}
    sentences = split_sentences(text)
    if not sentences:
        return []

    # Greedy packing: start a new chunk when the next sentence would overflow.
    groups = []
    current, current_tokens = [], 0
    for i, sent in enumerate(sentences):
        t = count_tokens(sent, tokenizer)
        if current and current_tokens + t > config.max_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += t
    if current:
        groups.append(current)

    chunks = []
    for ordinal, idxs in enumerate(groups):
        chunk_text = " ".join(sentences[i] for i in idxs)
        chunks.append(Chunk(
            chunk_id=make_chunk_id(pmid, "sentence_boundary", ordinal),
            pmid=pmid, article_id=pmid, strategy="sentence_boundary",
            ordinal=ordinal, text=chunk_text,
            token_count=count_tokens(chunk_text, tokenizer),
            char_count=len(chunk_text), n_sentences=len(idxs),
            sentence_start=idxs[0], sentence_end=idxs[-1],
            cuts_sentences=False,
            config=asdict(config), source=source,
        ))
    return chunks
