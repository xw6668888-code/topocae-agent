from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

from audit_paper_v1_provenance import REPO_ROOT, assert_allowed_path


CONFIG_ROOT = REPO_ROOT / "configs" / "paper_v1_clean_candidate"
OUTPUT = REPO_ROOT / "reports" / "paper_v1_p1" / "security_scope_scan.json"
TEXT_SUFFIXES = {
    ".py", ".json", ".md", ".yaml", ".yml", ".toml", ".ini", ".env", ".txt", ".ps1"
}
SKIP_PARTS = {".git", "__pycache__", ".pytest_cache"}
HIGH_CONFIDENCE_PATTERNS = {
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def walk_strings(value: Any, prefix: str = "") -> list[tuple[str, str]]:
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            rows.extend(walk_strings(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(walk_strings(item, f"{prefix}[{index}]"))
    elif isinstance(value, str):
        rows.append((prefix, value))
    return rows


def credential_scan() -> list[dict[str, Any]]:
    hits = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if any(part in SKIP_PARTS for part in path.parts) or path.stat().st_size > 5_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in HIGH_CONFIDENCE_PATTERNS.items():
            for match in pattern.finditer(text):
                hits.append(
                    {
                        "pattern": name,
                        "path": path.relative_to(REPO_ROOT).as_posix(),
                        "line": text.count("\n", 0, match.start()) + 1,
                    }
                )
    return hits


def configured_input_scan() -> list[dict[str, str]]:
    datasets = load(CONFIG_ROOT / "datasets.json")
    models = load(CONFIG_ROOT / "models.json")
    candidates = []
    for key, value in walk_strings(datasets):
        if any(token in key for token in ("graph_root", "step_root", "dataset_meta", "training_metadata", "splits", "truth")):
            if value not in {"not_computable", "area_truth_only"}:
                candidates.append((f"datasets.{key}", value))
    for key, value in walk_strings(models):
        if key.endswith(".path") or key.startswith("clean_components") or key.startswith("training_data"):
            candidates.append((f"models.{key}", value))
    rows = []
    for key, value in candidates:
        assert_allowed_path(REPO_ROOT / value, datasets)
        rows.append({"key": key, "path": value})
    return rows


def entry_literal_scan() -> list[str]:
    entry = REPO_ROOT / "evaluate_paper_v1_recognition_v5.py"
    tree = ast.parse(entry.read_text(encoding="utf-8"), filename=str(entry))
    forbidden = tuple(
        str(item).casefold() for item in load(CONFIG_ROOT / "datasets.json")["excluded_tokens"]
    )
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            lowered = node.value.casefold()
            if any(token in lowered for token in forbidden):
                violations.append(node.value)
    return violations


def main() -> None:
    credential_hits = credential_scan()
    inputs = configured_input_scan()
    entry_violations = entry_literal_scan()
    payload = {
        "schema": "topocae_paper_v1_p1_security_scope_scan_v1",
        "phase": "P1-A1",
        "credential_scan": {
            "pattern_set": sorted(HIGH_CONFIDENCE_PATTERNS),
            "hit_count": len(credential_hits),
            "hits": credential_hits,
        },
        "configured_input_scan": {
            "checked_count": len(inputs),
            "all_passed_reject_before_read_guard": True,
            "inputs": inputs,
        },
        "unified_entry_forbidden_literal_count": len(entry_violations),
        "unified_entry_forbidden_literals": entry_violations,
        "status": "pass" if not credential_hits and not entry_violations else "fail",
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "credential_hits": len(credential_hits), "checked_inputs": len(inputs), "entry_violations": len(entry_violations)}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
