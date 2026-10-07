"""Tokenizer utilities.

All strategies count tokens with the same WordPiece tokenizer used by the
required baseline embedding model (all-MiniLM-L6-v2), so "256 tokens" means
the same thing everywhere. Documented here once, referenced by every chunker.
"""

from functools import lru_cache

DEFAULT_TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def get_tokenizer(model_name=DEFAULT_TOKENIZER_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text, tokenizer=None):
    tokenizer = tokenizer or get_tokenizer()
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_char_spans(text, tokenizer=None):
    """Return [(char_start, char_end)] per token for slicing original text."""
    tokenizer = tokenizer or get_tokenizer()
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return enc["offset_mapping"]
