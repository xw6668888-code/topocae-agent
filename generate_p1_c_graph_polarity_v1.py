from __future__ import annotations

"""Augment all strict P1-C Graph records from STEP with cylinder polarity."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

from OCC.Extend.DataExchange import read_step_file

from p1_c.oriented_cylinder_polarity_v1 import FEATURE_NAME, REVISION, augment_graph_with_oriented_cylinder_polarity
from utils.p1_c_strict_run_paths import DATASET_ROOT, ROOT


SOURCE = DATASET_ROOT / "graphs"
OUTPUT = DATASET_ROOT / "graphs_polarity_v1"
MANIFEST = DATASET_ROOT / "graph_polarity_v1_manifest.json"
ITERATION_PLAN = ROOT / "configs" / "p1_c" / "stage6_iteration1_oriented_cylinder_polarity_v1.json"
EXPECTED_PLAN_SHA256 = "b60a57c55294ecf012ced1bc24edca6cffa73d728594f6e5b77ca3916eeae036"


def _hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def main() -> int:
    if _hash(ITERATION_PLAN) != EXPECTED_PLAN_SHA256:
        raise RuntimeError("Stage 6 iteration-1 plan drift")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = sorted(SOURCE.glob("*.json"))
    if len(sources) != 7200:
        raise RuntimeError(f"expected 7200 source graphs, found {len(sources)}")
    records = []
    by_macro: dict[str, list[float]] = defaultdict(list)
    sign_counts: Counter[str] = Counter()
    started = time.perf_counter()
    for position, source in enumerate(sources, 1):
        graph = json.loads(source.read_text(encoding="utf-8"))
        sample_id = str(graph["sample_id"])
        step_path = DATASET_ROOT / "steps" / f"{sample_id}.step"
        destination = OUTPUT / source.name
        shape = read_step_file(str(step_path), verbosity=False)
        augmented = augment_graph_with_oriented_cylinder_polarity(shape, graph)
        destination.write_text(json.dumps(augmented, indent=2, ensure_ascii=False), encoding="utf-8")
        for node in augmented["nodes"]:
            if node.get("primitive_surface") != "cylinder":
                continue
            value = float(node["attributes"]["recognition_features_v2"][FEATURE_NAME])
            by_macro[str(node.get("macro_feature", "stock"))].append(value)
            sign_counts["negative" if value < -0.5 else "positive" if value > 0.5 else "zero"] += 1
        records.append({
            "sample_id": sample_id,
            "source_graph_sha256": _hash(source),
            "step_sha256": _hash(step_path),
            "graph_polarity_sha256": _hash(destination),
        })
        if position % 500 == 0:
            print(json.dumps({"processed": position, "total": len(sources)}), flush=True)
    class_audit = {
        macro: {
            "count": len(values),
            "mean": sum(values) / len(values),
            "negative_fraction": sum(value < -0.5 for value in values) / len(values),
            "positive_fraction": sum(value > 0.5 for value in values) / len(values),
        }
        for macro, values in sorted(by_macro.items()) if values
    }
    result = {
        "schema_version": "topocae_p1_c_graph_oriented_cylinder_polarity_manifest_1",
        "revision": REVISION,
        "iteration_plan_sha256": EXPECTED_PLAN_SHA256,
        "source_dataset": str(DATASET_ROOT),
        "graph_count": len(records),
        "feature_name": FEATURE_NAME,
        "sign_counts": dict(sign_counts),
        "class_audit": class_audit,
        "records": records,
        "build_seconds": time.perf_counter() - started,
        "contracts": {
            "step_geometry_modified": False,
            "labels_modified": False,
            "face_order_modified": False,
            "feature_uses_labels": False,
            "real_data_used": False,
            "holdout_used": False,
        },
    }
    MANIFEST.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "manifest": str(MANIFEST), "class_audit": {k: class_audit[k] for k in ("pocket_circular", "cylinder_boss")}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

