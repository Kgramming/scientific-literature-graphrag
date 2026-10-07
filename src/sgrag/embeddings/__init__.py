"""Embedding generation: configuration, model loading, cached encoding."""

from .config import EmbeddingConfig
from .embed import encode, load_model

__all__ = ["EmbeddingConfig", "encode", "load_model"]
