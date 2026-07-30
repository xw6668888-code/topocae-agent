from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


def verify_freeze_manifest(repo_root: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assets = manifest.get("assets", [])
    results = []
    for asset in assets:
        path = repo_root / Path(asset["path"])
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = sha256_file(path) if exists else None
        matches = (
            exists
            and actual_bytes == int(asset["bytes"])
            and actual_sha256 == str(asset["sha256"]).lower()
        )
        results.append({
            "path": asset["path"],
            "exists": exists,
            "expected_bytes": int(asset["bytes"]),
            "actual_bytes": actual_bytes,
            "expected_sha256": str(asset["sha256"]).lower(),
            "actual_sha256": actual_sha256,
            "matches": matches,
        })
    mismatches = [item for item in results if not item["matches"]]
    return {
        "schema_version": "p3-a1r-frozen-asset-verification/1",
        "manifest_path": str(manifest_path.resolve()),
        "declared_asset_count": int(manifest.get("asset_count", -1)),
        "checked_asset_count": len(results),
        "matching_asset_count": len(results) - len(mismatches),
        "hash_changes": len(mismatches),
        "automatic_pass": len(results) == 101 and not mismatches,
        "mismatches": mismatches,
    }
