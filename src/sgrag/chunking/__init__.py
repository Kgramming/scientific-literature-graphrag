"""Chunking strategy registry.

Phase 2 implements three strategies (A/B/C). The registry keeps the door
open for the improved semantic-sections chunker (strategy D) planned in the
architecture proposal, without changing any existing code.
"""

from . import fixed_token, sentence_boundary, semantic_cluster
from .base import (Chunk, FixedTokenConfig, SentenceBoundaryConfig,
                   SemanticClusterConfig)

STRATEGIES = {
    "fixed_token": fixed_token.chunk_record,
    "sentence_boundary": sentence_boundary.chunk_record,
    "semantic_cluster": semantic_cluster.chunk_record,
}

CONFIGS = {
    "fixed_token": FixedTokenConfig,
    "sentence_boundary": SentenceBoundaryConfig,
    "semantic_cluster": SemanticClusterConfig,
}

__all__ = ["STRATEGIES", "CONFIGS", "Chunk",
           "FixedTokenConfig", "SentenceBoundaryConfig", "SemanticClusterConfig"]
