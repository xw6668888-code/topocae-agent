from __future__ import annotations

from collections import Counter, defaultdict

from .constants import CANONICAL_FEATURES, CROSSWALK_PATH, LEVELS, OPERATIONS, STATUSES
from .formal_planner import make_formal_record
from .formal_planner_parallel import generate_formal_plans_parallel
from .formal_renderer_v3 import render_formal_all
from .formal_validation import split_leakage_audit
from .io import canonical_sha256, read_yaml
from . import validation as base


def diagnose() -> dict:
    plans = generate_formal_plans_parallel(read_yaml(CROSSWALK_PATH))
    canonical = []
    renderings = []
    for plan in plans:
        query, derived = render_formal_all(plan)
        canonical.append(make_formal_record(plan, record_kind="canonical", query=query))
        renderings.extend(
            make_formal_record(plan, record_kind="rendering", query=item) for item in derived
        )
    by_id = defaultdict(list)
    for record in canonical + renderings:
        by_id[record["source_ref"]["canonical_record_id"]].append(record)
    identity_failures = 0
    cardinality_failures = 0
    for records in by_id.values():
        if len({base.semantic_identity(record) for record in records}) != 1:
            identity_failures += 1
        kinds = Counter(record["record_kind"] for record in records)
        languages = Counter(
            record["query"]["language"] for record in records if record["record_kind"] == "rendering"
        )
        if kinds != {"canonical": 1, "rendering": 3} or languages != {"en-US": 2, "zh-CN": 1}:
            cardinality_failures += 1
    source_cardinality = Counter(record["source_ref"]["sample_id"] for record in canonical)
    duplicate_payloads = len(canonical) - len({canonical_sha256({
        "source_graph_sha256": record["source_ref"]["graph_sha256"],
        "query": record["query"],
        "intent_result": record["intent_result"],
        "ground_truth": record["ground_truth"],
    }) for record in canonical})
    counts = {
        "features": dict(Counter(plan["verbalization"]["feature_type"] for plan in plans)),
        "operations": dict(Counter(
            operation["operation_type"]
            for plan in plans
            for operation in plan["verbalization"]["operations"]
        )),
        "levels": dict(Counter(plan["level"] for plan in plans)),
        "statuses": dict(Counter(plan["status"] for plan in plans)),
        "languages": dict(Counter(record["query"]["language"] for record in renderings)),
    }
    coverage = {
        "features": set(counts["features"]) == set(CANONICAL_FEATURES),
        "operations": set(counts["operations"]) == set(OPERATIONS),
        "levels": set(counts["levels"]) == set(LEVELS),
        "statuses": set(counts["statuses"]) == set(STATUSES),
        "languages": set(counts["languages"]) == {"en-US", "zh-CN"},
    }
    leakage = split_leakage_audit(canonical, renderings)
    return {
        "canonical": len(canonical),
        "renderings": len(renderings),
        "identity_failures": identity_failures,
        "cardinality_failures": cardinality_failures,
        "plans_per_source": dict(Counter(source_cardinality.values())),
        "duplicate_payloads": duplicate_payloads,
        "counts": counts,
        "coverage": coverage,
        "leakage": leakage,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(diagnose(), ensure_ascii=False, sort_keys=True))

