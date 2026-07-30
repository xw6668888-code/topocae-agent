from __future__ import annotations

import hashlib
import json
from pathlib import Path

import finalize_paper_v1_clean_candidate as base


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "paper_v1_clean_candidate_freeze_manifest.json"
SUPPLEMENTAL_ARTIFACTS = [
    "evaluate_paper_v1_recognition_v4.py",
    "finalize_paper_v1_clean_candidate_v2.py",
    "verify_paper_v1_p1_security.py",
    "verify_paper_v1_p1_security_v2.py",
    "reports/paper_v1_p1/security_scope_scan.json",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def complete_manifest() -> dict:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    by_path = {item["path"]: item for item in manifest["artifacts"]}
    for relative in SUPPLEMENTAL_ARTIFACTS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        by_path[relative] = {
            "path": relative,
            "sha256": digest(path),
            "bytes": path.stat().st_size,
        }
    manifest["artifacts"] = [by_path[key] for key in sorted(by_path)]
    manifest["finalizer"] = "finalize_paper_v1_clean_candidate_v2.py"
    manifest["security_scan"] = "reports/paper_v1_p1/security_scope_scan.json"
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    base.main()
    manifest = complete_manifest()
    print(json.dumps({"freeze_manifest_artifact_count": len(manifest["artifacts"])}, indent=2))


if __name__ == "__main__":
    main()
