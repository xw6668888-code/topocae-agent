from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "reports/p1_c_final_acceptance_v1/delivery_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    manifest = json.loads(PATH.read_text(encoding="utf-8"))
    by_path = {item["path"]: item for item in manifest["artifacts"]}
    additions = [
        "reports/p1_c_final_acceptance_v1/protected_inventory_final.json",
        "verify_p1_c_protected_inventory_v1.py",
        "verify_p1_c_final_acceptance_v1.py",
        "verify_p1_c_final_acceptance_v2.py",
        "run_paper_v1_p1_c_pipeline_v2.py",
        "p1_c/runtime_state_isolation_v1.py",
        "p1_c/runtime_state_isolation_v2.py",
        "p1_c/runtime_state_isolation_v3.py",
        "tests/test_p1_c_00000_runtime_state_guard_v3.py",
        "tests/test_p1_c_import_order_isolation.py",
        "tests/test_p1_c_protected_inventory.py",
    ]
    for relative in additions:
        by_path.setdefault(relative, {"path": relative})
    artifacts = []
    for relative in sorted(by_path):
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest["schema_version"] = "topocae_p1_c_delivery_manifest_2"
    manifest["manifest_builder_revision"] = "finalize_p1_c_delivery_manifest_v2"
    manifest["artifact_count"] = len(artifacts)
    manifest["artifacts"] = artifacts
    PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "accepted": manifest["accepted"],
        "artifact_count": len(artifacts),
        "p2_entered": manifest["p2_entered"],
    }, ensure_ascii=False, indent=2))
    return 0 if manifest["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
