from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1b_schema_diagnosis_v1"
NAMES = (
    "preflight.json",
    "diagnosis.json",
    "security_audit.json",
    "final_diagnosis.zh-CN.md",
)


def main() -> None:
    manifest = {
        "schema_version": "topocae-p4-api-r1b-delivery-manifest/1",
        "files": [
            {
                "path": name,
                "sha256": hashlib.sha256((REPORT / name).read_bytes()).hexdigest(),
                "bytes": (REPORT / name).stat().st_size,
            }
            for name in NAMES
        ],
    }
    (REPORT / "delivery_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    main()
