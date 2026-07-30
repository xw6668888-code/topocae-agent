"""P4-A1 natural-language CAE parsing and deterministic grounding.

The package is intentionally independent from recognizer inference, STEP readers,
solver execution, and P5 pointer code.
"""

from .dataset import (
    DatasetIsolationError,
    DecodeAudit,
    iter_canonical_records,
    iter_rendering_records,
)
from .memory import (
    FeatureObjectMemoryError,
    build_feature_object_memory,
    canonical_memory_bytes,
    validate_feature_object_memory,
)
from .grounder import GroundingError, ground_intent_result, validate_grounding_result


__all__ = [
    "DatasetIsolationError",
    "DecodeAudit",
    "FeatureObjectMemoryError",
    "GroundingError",
    "build_feature_object_memory",
    "canonical_memory_bytes",
    "ground_intent_result",
    "iter_canonical_records",
    "iter_rendering_records",
    "validate_feature_object_memory",
    "validate_grounding_result",
]
