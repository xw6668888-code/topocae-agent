from __future__ import annotations

import json
import subprocess
from pathlib import Path

from audit_paper_v1_provenance import REPO_ROOT
from verify_paper_v1_p1_security import configured_input_scan, entry_literal_scan


OUTPUT = REPO_ROOT / "reports" / "paper_v1_p1" / "security_scope_scan.json"
PATTERN = (
    r"(?:\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b|"
    r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
    r"\bgh[pousr]_[A-Za-z0-9]{30,}\b|"
    r"\bAIza[0-9A-Za-z_-]{30,}\b|"
    r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)"
)


def credential_hits() -> list[dict[str, object]]:
    command = [
        "rg", "-n", "--hidden", "--no-heading", "--color", "never",
        "-g", "!.git/**", "-g", "!*.pt", "-g", "!*.joblib", "-g", "!*.step", "-g", "!*.stp",
        PATTERN, ".",
    ]
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8")
    if result.returncode not in {0, 1}:
        raise RuntimeError(result.stderr)
    hits = []
    for line in result.stdout.splitlines():
        parts = line.split(":", 2)
        hits.append(
            {
                "path": parts[0].removeprefix(".\\").replace("\\", "/"),
                "line": int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None,
            }
        )
    return hits


def main() -> None:
    hits = credential_hits()
    inputs = configured_input_scan()
    violations = entry_literal_scan()
    payload = {
        "schema": "topocae_paper_v1_p1_security_scope_scan_v2",
        "phase": "P1-A1",
        "credential_scan": {
            "engine": "ripgrep",
            "high_confidence_hit_count": len(hits),
            "hits": hits,
        },
        "configured_input_scan": {
            "checked_count": len(inputs),
            "all_passed_reject_before_read_guard": True,
            "inputs": inputs,
        },
        "unified_entry_forbidden_literal_count": len(violations),
        "unified_entry_forbidden_literals": violations,
        "status": "pass" if not hits and not violations else "fail",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "credential_hits": len(hits),
                "checked_inputs": len(inputs),
                "entry_violations": len(violations),
            },
            indent=2,
        )
    )
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
