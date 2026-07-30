"""R0.1 full frozen-asset SHA-256 verification and scope guards."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

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


def _asset_path(value: str, *, root: Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() and candidate.is_file():
        return candidate
    rooted = root / candidate
    if rooted.is_file():
        return rooted
    return resolve_workspace_path(value)


def verify_full_freeze(
    manifests: Iterable[tuple[str, Path]] = MANIFESTS,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    rows = []
    for stage, manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        changes = []
        matching = 0
        for asset in manifest["assets"]:
            path = _asset_path(str(asset["path"]), root=root)
            expected_bytes = int(asset["bytes"])
            expected_sha256 = str(asset["sha256"]).lower()
            if not path.is_file():
                changes.append({
                    "path": str(asset["path"]),
                    "error_code": "E_FROZEN_ASSET_MISSING",
                })
                continue
            actual_bytes = path.stat().st_size
            actual_sha256 = sha256_file(path)
            if actual_bytes != expected_bytes:
                changes.append({
                    "path": str(asset["path"]),
                    "error_code": "E_FROZEN_ASSET_BYTES",
                    "expected_bytes": expected_bytes,
                    "actual_bytes": actual_bytes,
                })
            elif actual_sha256 != expected_sha256:
                changes.append({
                    "path": str(asset["path"]),
                    "error_code": "E_FROZEN_ASSET_SHA256",
                    "expected_sha256": expected_sha256,
                    "actual_sha256": actual_sha256,
                })
            else:
                matching += 1
        rows.append({
            "stage": stage,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "matching_count": matching,
            "total_count": len(manifest["assets"]),
            "change_count": len(changes),
            "changes": changes,
        })
    passed = all(row["change_count"] == 0 for row in rows)
    return {
        "schema_version": "topocae-p4-api-r0-1-full-freeze-audit/1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL",
        "error_code": None if passed else "E_FROZEN_ASSET_CHANGE",
        "stages": rows,
        "scope_guards": {
            "asset_access_mode": "binary_sha256_only",
            "formal_test_payload_decoded": False,
            "formal_test_payload_rendered": False,
            "formal_test_payload_semantically_inspected": False,
            "real_api_called": False,
            "train_val_run": False,
            "p5_entered": False,
            "abaqus_executed": False,
        },
    }


__all__ = ["MANIFESTS", "sha256_file", "verify_full_freeze"]
