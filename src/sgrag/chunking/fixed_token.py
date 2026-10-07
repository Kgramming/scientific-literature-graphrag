"""Strategy A — fixed-token baseline.

Deterministic sliding windows of N WordPiece tokens with M-token overlap.
Windows are cut on token boundaries, so they MAY split sentences; the
cuts_sentences flag records exactly where that happens (the baseline's
known weakness, measured rather than hidden).
"""

from dataclasses import asdict

from .base import Chunk, FixedTokenConfig, make_chunk_id
from .sentences import sentence_char_spans
from .tokenizer import get_tokenizer


def _boundary_cuts(char_pos, sent_spans, text_len):
    """True if char_pos falls strictly inside a sentence (not at an edge)."""
    if char_pos <= 0 or char_pos >= text_len:
        return False
    return any(s < char_pos < e for s, e in sent_spans)


def _overlapping_sentences(char_start, char_end, sent_spans):
    idx = [i for i, (s, e) in enumerate(sent_spans) if s < char_end and e > char_start]
    if not idx:
        return None, None
    return idx[0], idx[-1]


def chunk_record(record, config=None, tokenizer=None):
    """Chunk one pilot-corpus record into fixed-token windows."""
    config = config or FixedTokenConfig()
    if isinstance(config, dict):
        config = FixedTokenConfig(**config)
    tokenizer = tokenizer or get_tokenizer(config.tokenizer_model)

    pmid = str(record["pmid"])
    text = record.get("abstract") or ""
    source = {"title": record.get("title"), "journal": record.get("journal"),
              "year": record.get("year")}
    if not text.strip():
        return []

    from .tokenizer import token_char_spans
    tok_spans = token_char_spans(text, tokenizer)
    n = len(tok_spans)
    sent_spans = sentence_char_spans(text)
    text_len = len(text)

    chunks = []
    start = 0
    ordinal = 0
    while start < n:
        end = min(start + config.token_limit, n)
        char_start = tok_spans[start][0]
        char_end = tok_spans[end - 1][1]
        chunk_text = text[char_start:char_end]
        s_start, s_end = _overlapping_sentences(char_start, char_end, sent_spans)
        cuts = (_boundary_cuts(char_start, sent_spans, text_len)
                or _boundary_cuts(char_end, sent_spans, text_len))
        chunks.append(Chunk(
            chunk_id=make_chunk_id(pmid, "fixed_token", ordinal),
            pmid=pmid, article_id=pmid, strategy="fixed_token", ordinal=ordinal,
            text=chunk_text, token_count=end - start, char_count=len(chunk_text),
            n_sentences=(s_end - s_start + 1) if s_start is not None else 0,
            sentence_start=s_start, sentence_end=s_end,
            cuts_sentences=cuts,
            config=asdict(config), source=source,
        ))
        ordinal += 1
        if end >= n:
            break
        start = end - config.overlap
    return chunks
