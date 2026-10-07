"""Embedding configuration.

One dataclass per embedding model under test. The cache key binds the model
name, normalization choice, and the exact input texts, so a cached matrix
is only reused when every experimental condition matches.
"""

import hashlib
from dataclasses import asdict, dataclass


@dataclass
class EmbeddingConfig:
    model_name: str
    normalize: bool = False
    device: str = "cpu"
    batch_size: int = 32
    cache_dir: str = "data/processed/embeddings_cache"

    def cache_key(self, texts):
        h = hashlib.sha256()
        h.update(self.model_name.encode())
        h.update(b"|" + str(self.normalize).encode())
        for t in texts:
            h.update(b"\x00" + t.encode())
        return h.hexdigest()[:16]

    def cache_slug(self):
        # Filesystem-safe model identifier.
        return self.model_name.replace("/", "--")

    def to_dict(self):
        return asdict(self)
