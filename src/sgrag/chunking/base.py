"""Chunk schema and per-strategy configuration.

Every chunk carries full provenance: PMID/article, sentence range, chunk ID,
strategy, and the exact configuration that produced it — so any chunk can be
traced back to its source and the experiment repeated.
"""

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    pmid: str
    article_id: str
    strategy: str
    ordinal: int
    text: str
    token_count: int
    char_count: int
    n_sentences: int
    # Inclusive sentence indices overlapping this chunk (None for pure-token
    # windows that align to no sentence).
    sentence_start: int | None
    sentence_end: int | None
    # True when a chunk boundary falls strictly inside a sentence.
    cuts_sentences: bool
    config: dict = field(default_factory=dict)
    source: dict = field(default_factory=dict)  # title, journal, year

    def to_dict(self):
        return asdict(self)

    @classmethod
    def required_fields(cls):
        return ["chunk_id", "pmid", "article_id", "strategy", "ordinal",
                "text", "token_count", "char_count", "n_sentences",
                "sentence_start", "sentence_end", "cuts_sentences",
                "config", "source"]

    def is_valid(self):
        return (bool(self.chunk_id) and bool(self.pmid) and bool(self.text.strip())
                and self.token_count > 0)


@dataclass
class FixedTokenConfig:
    token_limit: int = 256
    overlap: int = 32
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SentenceBoundaryConfig:
    max_tokens: int = 256
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SemanticClusterConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    clustering: str = "hdbscan"
    min_cluster_size: int = 2
    metric: str = "euclidean"
    min_sentences: int = 3  # fewer -> single-chunk fallback
    # Hard cap: a cluster's contiguous run is greedily split on sentence
    # boundaries so no chunk exceeds this (single over-long sentences are
    # kept whole to preserve sentence integrity).
    max_tokens: int = 256
    # L2-normalize embeddings before clustering. For unit vectors, euclidean
    # distance is a monotonic transform of cosine distance
    # (||a-b||^2 = 2(1-cos)), so normalize=True clusters on cosine
    # (angular) geometry — the natural comparison for sentence-transformer
    # embeddings. The installed HDBSCAN build rejects metric="cosine".
    normalize_embeddings: bool = False
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2"


def make_chunk_id(pmid, strategy, ordinal):
    return f"{pmid}_{strategy}_{ordinal:03d}"
