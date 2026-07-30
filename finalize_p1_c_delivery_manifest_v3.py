from __future__ import annotations

import hashlib
import json
from pathlib import Path

import finalize_p1_c_delivery_manifest_v2 as v2


ROOT = Path(__file__).resolve().parent
PATH = ROOT / "reports/p1_c_final_acceptance_v1/delivery_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    v2.main()
    manifest = json.loads(PATH.read_text(encoding="utf-8"))
    by_path = {item["path"]: item for item in manifest["artifacts"]}
    for relative in (
        "verify_p1_c_final_acceptance_v3.py",
        "run_paper_v1_p1_c_pipeline_v3.py",
        "finalize_p1_c_delivery_manifest_v2.py",
        "finalize_p1_c_delivery_manifest_v3.py",
    ):
        by_path.setdefault(relative, {"path": relative})
    artifacts = []
    for relative in sorted(by_path):
        path = ROOT / relative
        artifacts.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)}
        )
    manifest["schema_version"] = "topocae_p1_c_delivery_manifest_3"
    manifest["manifest_builder_revision"] = "finalize_p1_c_delivery_manifest_v3"
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
