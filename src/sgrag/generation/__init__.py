"""Evidence-grounded generation layer (Phase 5B).

Pipeline: retrieval results -> evidence packing -> grounded prompt ->
LLM generation -> citation validation -> GenerationResult.
"""

from sgrag.generation.config import GenerationConfig, load_generation_config
from sgrag.generation.generator import GroundedGenerator
from sgrag.generation.models import EvidenceItem, GenerationResult

__all__ = [
    "EvidenceItem",
    "GenerationConfig",
    "GenerationResult",
    "GroundedGenerator",
    "load_generation_config",
]
