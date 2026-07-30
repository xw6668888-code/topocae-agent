from __future__ import annotations

"""Revision 2: deterministic supported-wall-safe pair/triple allocation."""

from collections import Counter
from datetime import datetime, timezone
import json

import generate_p1_c_synthetic_holdout_v3 as base
from p1_c.all_features import MACRO_CLASSES
from p1_c.formal_plan import PlannedSample
from p1_c.rib_web_v3 import RIB_HOST_FAMILIES, WEB_HOST_FAMILIES


def _secondary_groups(primary: str, pair_count: int, triple_count: int) -> tuple[list[tuple[str, ...]], list[tuple[str, ...]]]:
    compatible = [macro for macro in MACRO_CLASSES if macro != primary]
    if primary == "rib":
        compatible.remove("web")
    elif primary == "web":
        compatible.remove("rib")
    offset = (MACRO_CLASSES.index(primary) * 7 + 3) % len(compatible)
    stream = [compatible[(offset + index * 5) % len(compatible)] for index in range(2000)]
    cursor = 0

    def take(existing: tuple[str, ...]) -> str:
        nonlocal cursor
        for _ in range(len(stream)):
            candidate = stream[cursor]
            cursor += 1
            macros = (primary, *existing, candidate)
            if candidate in existing:
                continue
            if sum(macro in {"rib", "web"} for macro in macros) > 1:
                continue
            return candidate
        raise RuntimeError(f"cannot allocate compatible Holdout-v3 secondary for {primary}")

    pairs = [(take(()),) for _ in range(pair_count)]
    triples = []
    for _ in range(triple_count):
        first = take(())
        second = take((first,))
        triples.append((first, second))
    return pairs, triples


def build_plan_v2(config):
    contexts = config["context_quota_per_class"]
    general_hosts = tuple(config["general_holdout_host_templates"])
    axes = tuple(tuple(float(value) for value in axis) for axis in config["rotation_axes"])
    angles = tuple(float(value) for value in config["rotation_angles_radians"])
    general_scales = tuple(float(value) for value in config["general_uniform_scales"])
    supported_scales = tuple(float(value) for value in config["supported_wall_uniform_scales"])
    records = []
    for primary_index, primary in enumerate(MACRO_CLASSES):
        pairs, triples = _secondary_groups(primary, int(contexts["pair"]), int(contexts["triple"]))
        secondaries = {
            "target_only": [()] * int(contexts["target_only"]),
            "pair": pairs,
            "triple": triples,
        }
        class_index = 0
        for context in ("target_only", "pair", "triple"):
            for context_index, secondary in enumerate(secondaries[context]):
                macros = (primary, *secondary)
                walls = [macro for macro in macros if macro in {"rib", "web"}]
                if len(walls) > 1:
                    raise RuntimeError(f"forbidden supported-wall composition: {macros}")
                if walls:
                    families = RIB_HOST_FAMILIES if walls[0] == "rib" else WEB_HOST_FAMILIES
                    host = families[(primary_index + class_index) % len(families)]
                    host_kind = "holdout_supported_template_v1"
                    scale = supported_scales[(primary_index * 3 + class_index) % len(supported_scales)]
                else:
                    host = general_hosts[(primary_index + class_index) % len(general_hosts)]
                    host_kind = "holdout_general_template_v1"
                    scale = general_scales[(primary_index * 3 + class_index) % len(general_scales)]
                seed = int(config["seed_namespace"]) + primary_index * 10000 + class_index
                template_id = host if host_kind == "holdout_general_template_v1" else f"holdout_supported_scaled::{host}"
                records.append(PlannedSample(
                    sample_id=f"p1c_holdout_v3_{primary_index + 1:02d}_{primary}_{class_index:03d}_{context}",
                    primary_target=primary, primary_index=primary_index, class_sample_index=class_index,
                    context=context, context_index=context_index, secondary_features=secondary,
                    split="holdout_v3", seed=seed, host_family=host, host_family_kind=host_kind,
                    rotation_axis=axes[(primary_index + class_index) % len(axes)],
                    rotation_angle_radians=angles[(primary_index * 2 + class_index) % len(angles)],
                    uniform_scale=scale,
                    base_geometry_family_id=f"p1c-holdout-v3-base::{primary_index:02d}::{class_index:03d}::{template_id}",
                    parametric_family_id=f"p1c-holdout-v3-param::{primary}::{context}::{template_id}::{class_index:03d}",
                ))
                class_index += 1
    base.audit_plan(records, config)
    return records


def prepare_v2():
    config = base.read_json(base.CONFIG_PATH)
    selection_sha = base.verify_selection()
    existing_files = [path for path in base.OUTPUT.rglob("*") if path.is_file()] if base.OUTPUT.exists() else []
    if existing_files:
        raise FileExistsError(f"Holdout-v3 output contains files: {existing_files[:5]}")
    if base.SELECTION.stat().st_mtime_ns >= base.CONFIG_PATH.stat().st_mtime_ns:
        raise RuntimeError("preholdout selection must predate Holdout-v3 config")
    for name in ("steps", "graphs", "state/records", "manifests", "reports", "human_review_36"):
        (base.OUTPUT / name).mkdir(parents=True, exist_ok=True)
    plan = build_plan_v2(config)
    with base.PLAN_PATH.open("w", encoding="utf-8", newline="\n") as handle:
        for row in plan:
            payload = row.to_dict()
            payload["holdout_host_template_id"] = base.holdout_host_template(row)
            handle.write(json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n")
    audit = base.audit_plan(plan, config)
    audit["secondary_allocation_revision"] = "supported_wall_safe_v2"
    audit["secondary_counts"] = dict(Counter(
        secondary for row in plan for secondary in row.secondary_features
    ))
    audit_path = base.OUTPUT / "manifests" / "generation_plan_audit.json"
    base.write_json(audit_path, audit)
    freeze = {
        "schema_version": "topocae_p1_c_holdout_v3_plan_freeze_2",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_revision": base.PLAN_REVISION,
        "generator_revision": base.GENERATOR_REVISION,
        "secondary_allocation_revision": "supported_wall_safe_v2",
        "sample_count": len(plan),
        "generation_plan_sha256": base.digest(base.PLAN_PATH),
        "generation_plan_audit_sha256": base.digest(audit_path),
        "generation_config_sha256": base.digest(base.CONFIG_PATH),
        "preholdout_selection_sha256": selection_sha,
        "holdout_results_seen": False,
        "old_holdout_v2_fingerprint_manifest_only": str(base.HOLDOUT_V2_FINGERPRINT),
        "old_holdout_v2_fingerprint_sha256": base.digest(base.HOLDOUT_V2_FINGERPRINT),
        "contracts": config["contracts"],
    }
    base.write_json(base.PLAN_FREEZE, freeze)
    return {"passed": True, "output": str(base.OUTPUT), "plan_freeze": freeze}


base.build_plan = build_plan_v2
base.prepare = prepare_v2


if __name__ == "__main__":
    raise SystemExit(base.main())
