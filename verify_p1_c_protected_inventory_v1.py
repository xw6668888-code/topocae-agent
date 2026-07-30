from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from p1_c.preflight import protected_inventory


ROOT = Path(__file__).resolve().parent
BASELINE_PATH = ROOT / "reports" / "p1_c_preflight" / "protected_inventory.json"
OUTPUT_DIR = ROOT / "reports" / "p1_c_final_acceptance_v1"


def _records(inventory: dict[str, Any]) -> dict[str, tuple[int, str]]:
    result: dict[str, tuple[int, str]] = {}
    for directory in inventory["directories"]:
        for item in directory["files"]:
            result[item["path"]] = (int(item["bytes"]), str(item["sha256"]))
    return result


def compare(baseline: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    before = _records(baseline)
    after = _records(current)
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))
    changed = sorted(
        path for path in set(before) & set(after) if before[path] != after[path]
    )
    directory_diffs: list[dict[str, Any]] = []
    before_dirs = {item["path"]: item for item in baseline["directories"]}
    after_dirs = {item["path"]: item for item in current["directories"]}
    for path in sorted(set(before_dirs) | set(after_dirs)):
        lhs = before_dirs.get(path, {})
        rhs = after_dirs.get(path, {})
        directory_diffs.append(
            {
                "path": path,
                "baseline_exists": lhs.get("exists", False),
                "current_exists": rhs.get("exists", False),
                "baseline_file_count": lhs.get("file_count", 0),
                "current_file_count": rhs.get("file_count", 0),
                "baseline_byte_count": lhs.get("byte_count", 0),
                "current_byte_count": rhs.get("byte_count", 0),
                "unchanged": (
                    lhs.get("exists") == rhs.get("exists")
                    and lhs.get("file_count") == rhs.get("file_count")
                    and lhs.get("byte_count") == rhs.get("byte_count")
                ),
            }
        )
    passed = (
        baseline["aggregate_sha256"] == current["aggregate_sha256"]
        and baseline["total_file_count"] == current["total_file_count"]
        and baseline["total_byte_count"] == current["total_byte_count"]
        and not added
        and not removed
        and not changed
        and all(item["unchanged"] for item in directory_diffs)
    )
    return {
        "schema_version": "topocae_p1_c_protected_inventory_diff_1",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "baseline_aggregate_sha256": baseline["aggregate_sha256"],
        "current_aggregate_sha256": current["aggregate_sha256"],
        "baseline_total_file_count": baseline["total_file_count"],
        "current_total_file_count": current["total_file_count"],
        "baseline_total_byte_count": baseline["total_byte_count"],
        "current_total_byte_count": current["total_byte_count"],
        "added_count": len(added),
        "removed_count": len(removed),
        "changed_count": len(changed),
        "added": added,
        "removed": removed,
        "changed": [
            {
                "path": path,
                "baseline_bytes": before[path][0],
                "current_bytes": after[path][0],
                "baseline_sha256": before[path][1],
                "current_sha256": after[path][1],
            }
            for path in changed
        ],
        "directories": directory_diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline_path = root / BASELINE_PATH.relative_to(ROOT)
    output_dir = root / OUTPUT_DIR.relative_to(ROOT)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    current = protected_inventory(root)
    diff = compare(baseline, current)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "protected_inventory_final.json").write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "protected_inventory_diff.json").write_text(
        json.dumps(diff, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: diff[key] for key in (
        "passed", "current_aggregate_sha256", "current_total_file_count",
        "current_total_byte_count", "added_count", "removed_count", "changed_count"
    )}, ensure_ascii=False, indent=2))
    return 0 if diff["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
