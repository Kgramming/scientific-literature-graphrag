"""Sentence segmentation.

Rule-based spaCy sentencizer (no model download, fully deterministic).
Used by the sentence-boundary and semantic strategies, and to measure
sentence-integrity violations of the fixed-token baseline.
"""

import spacy

_nlp = None


def get_sentencizer():
    global _nlp
    if _nlp is None:
        _nlp = spacy.blank("en")
        _nlp.add_pipe("sentencizer")
    return _nlp


def split_sentences(text):
    """Split text into stripped, non-empty sentences (order preserved)."""
    if not text or not text.strip():
        return []
    doc = get_sentencizer()(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def sentence_char_spans(text):
    """Return [(char_start, char_end)] per sentence for boundary checks."""
    if not text or not text.strip():
        return []
    doc = get_sentencizer()(text)
    spans = []
    for s in doc.sents:
        t = s.text.strip()
        if t:
            # re-anchor the stripped span inside the original text
            start = text.index(t, s.start_char)
            spans.append((start, start + len(t)))
    return spans
