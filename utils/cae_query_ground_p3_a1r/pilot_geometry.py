"""Stable public API for P3-A1R pilot geometry revision 3."""

from __future__ import annotations

from . import pilot_geometry_public_base as _revision1


# The audited continuous-slot kernel fixes the sweep path to the host span.
# Within one part, profile width is the real varying size quantity.
for _slot in ("slot_rect", "slot_u", "slot_v", "slot_o"):
    _revision1.COMPARISON_METRIC[_slot] = "width"
_revision1.COMPARISON_METRIC["rib"] = "thickness"
_revision1.COMPARISON_METRIC["fillet"] = "radius"
_revision1.COMPARISON_METRIC["chamfer"] = "length"


CANONICAL_FEATURES = _revision1.CANONICAL_FEATURES
COMPARISON_METRIC = _revision1.COMPARISON_METRIC
COMPARISON_SEMANTICS = _revision1.COMPARISON_SEMANTICS
COMPLETE_DIMENSION_KEYS = _revision1.COMPLETE_DIMENSION_KEYS
FEATURE_FAMILY = _revision1.FEATURE_FAMILY
build_graph_with_metric_evidence = _revision1.build_graph_with_metric_evidence
build_multi_instance_result = _revision1.build_multi_instance_result
metric_table = _revision1.metric_table


__all__ = [
    "CANONICAL_FEATURES",
    "COMPARISON_METRIC",
    "COMPARISON_SEMANTICS",
    "COMPLETE_DIMENSION_KEYS",
    "FEATURE_FAMILY",
    "build_graph_with_metric_evidence",
    "build_multi_instance_result",
    "metric_table",
]
