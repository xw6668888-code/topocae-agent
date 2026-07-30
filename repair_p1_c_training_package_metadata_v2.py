from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_c_clean_context_7200"
PACKAGES = DATASET / "training_packages"
P1_B_CONFIG = ROOT / "reports" / "p1_b_training_runs_v1" / "same_instance_seed_20260718" / "config.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    baseline = json.loads(P1_B_CONFIG.read_text(encoding="utf-8"))
    baseline_dims = baseline["model"]["dimensions"]
    expected_primitive = int(baseline_dims["primitive"])
    expected_semantic = int(baseline_dims["semantic"])
    max_primitive = 0
    max_semantic = 0
    for path in sorted((DATASET / "graphs").glob("*.json")):
        graph = json.loads(path.read_text(encoding="utf-8"))
        for node in graph["nodes"]:
            max_primitive = max(max_primitive, int(node.get("primitive_surface_id", 0)))
            max_semantic = max(max_semantic, int(node.get("semantic_label", 0)))
    if max_primitive >= expected_primitive or max_semantic >= expected_semantic:
        raise RuntimeError(
            f"P1-C targets exceed frozen P1-B head dimensions: primitive={max_primitive}, semantic={max_semantic}"
        )
    results = []
    for package in sorted(path for path in PACKAGES.iterdir() if path.is_dir()):
        metadata_path = package / "metadata.json"
        manifest_path = package / "manifest.json"
        history = package / "metadata_history"
        history.mkdir(exist_ok=True)
        archived_metadata = history / "metadata_r1_before_dimension_repair.json"
        archived_manifest = history / "manifest_r1_before_dimension_repair.json"
        if not archived_metadata.exists():
            shutil.copy2(metadata_path, archived_metadata)
        if not archived_manifest.exists():
            shutil.copy2(manifest_path, archived_manifest)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        old = {
            "primitive_surface_class_count": metadata["primitive_surface_class_count"],
            "semantic_class_count": metadata["semantic_class_count"],
        }
        metadata["schema_version"] = "topocae_p1_c_training_package_2"
        metadata["primitive_surface_class_count"] = expected_primitive
        metadata["semantic_class_count"] = expected_semantic
        metadata["architecture_compatibility"] = {
            "reference_config": str(P1_B_CONFIG.resolve()),
            "reference_config_sha256": sha256_file(P1_B_CONFIG),
            "frozen_dimensions": baseline_dims,
            "observed_max_primitive_target": max_primitive,
            "observed_max_semantic_target": max_semantic,
            "dimensions_fit_targets": True,
            "repair_scope": "metadata declarations only; packaged tensors and geometry unchanged",
            "repaired_before_model_results": True,
            "repaired_at_utc": datetime.now(timezone.utc).isoformat(),
            "previous_declarations": old,
        }
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        files = [package / "train.pt", package / "val.pt", package / "development_test.pt", metadata_path]
        manifest = {
            "schema_version": "topocae_p1_c_training_package_manifest_2",
            "variant": metadata["variant"],
            "files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
                for path in files
            ],
            "metadata_dimension_repair": {
                "primitive_surface_class_count": expected_primitive,
                "semantic_class_count": expected_semantic,
                "observed_max_primitive_target": max_primitive,
                "observed_max_semantic_target": max_semantic,
                "p1_b_config_sha256": sha256_file(P1_B_CONFIG),
            },
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        results.append(
            {
                "variant": metadata["variant"],
                "metadata_sha256": sha256_file(metadata_path),
                "manifest_sha256": sha256_file(manifest_path),
            }
        )
    payload = {
        "passed": True,
        "frozen_dimensions": baseline_dims,
        "observed_max_primitive_target": max_primitive,
        "observed_max_semantic_target": max_semantic,
        "packages": results,
    }
    report = DATASET / "reports" / "training_package_dimension_repair.json"
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
