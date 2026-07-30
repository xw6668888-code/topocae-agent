"""Recompute P1/P2/P3 frozen-asset hashes without mutating frozen assets."""

from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from utils.workspace_paths import resolve_workspace_path


ROOT = Path(__file__).resolve().parent
MANIFESTS = (
    ("P1", ROOT / "reports/p1_d_lite_v1/freeze/recognizer_dev_manifest.json"),
    ("P2", ROOT / "reports/p2_a1_cae_ir_v1/freeze/freeze_manifest.json"),
    ("P3", ROOT / "reports/p3_a1r_multi_instance_v1/freeze/freeze_manifest.json"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict[str, Any]:
    rows = []
    for stage, manifest_path in MANIFESTS:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changes = []
        matching = 0
        for asset in manifest["assets"]:
            path = resolve_workspace_path(str(asset["path"]))
            expected_hash = str(asset["sha256"])
            expected_bytes = int(asset["bytes"])
            if not path.is_file():
                changes.append(
                    {
                        "path": str(asset["path"]),
                        "change": "missing",
                    }
                )
                continue
            actual_bytes = path.stat().st_size
            actual_hash = sha256_file(path)
            if actual_bytes != expected_bytes or actual_hash != expected_hash:
                changes.append(
                    {
                        "path": str(asset["path"]),
                        "change": "content_mismatch",
                        "expected_bytes": expected_bytes,
                        "actual_bytes": actual_bytes,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                    }
                )
            else:
                matching += 1
        rows.append(
            {
                "stage": stage,
                "path": str(manifest_path.relative_to(ROOT)).replace("\\", "/"),
                "manifest_sha256": sha256_file(manifest_path),
                "asset_count": len(manifest["assets"]),
                "matching_count": matching,
                "change_count": len(changes),
                "changes": changes,
            }
        )
    status = "PASS" if not any(row["changes"] for row in rows) else "FAIL"
    return {
        "schema_version": "topocae-p4-a1-frozen-integrity/1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "h4_pre_pause_recheck",
        "status": status,
        "manifests": rows,
        "scope_guards": {
            "test_query_or_ground_truth_printed": False,
            "test_query_or_ground_truth_manually_read": False,
            "llm_called": False,
            "model_trained": False,
            "solver_executed": False,
            "p5_entered": False,
            "forbidden_asset_content_accessed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "output_p4_a1_language_grounding_v1"
            / "integrity"
            / "frozen_assets_h4_pre_pause.json"
        ),
    )
    args = parser.parse_args()
    result = verify()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output.resolve()),
                "manifests": [
                    {
                        "stage": row["stage"],
                        "matching": row["matching_count"],
                        "total": row["asset_count"],
                        "changes": row["change_count"],
                    }
                    for row in result["manifests"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
