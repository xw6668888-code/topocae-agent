from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_paper_v1_provenance import REPO_ROOT, load_json, stable_hash, write_json
from evaluate_paper_v1_recognition import canonicalize


DEFAULT_RUN_A = REPO_ROOT / "reports" / "paper_v1_p1" / "evaluations" / "repeat_1"
DEFAULT_RUN_B = REPO_ROOT / "reports" / "paper_v1_p1" / "evaluations" / "repeat_2"
DEFAULT_OUTPUT = REPO_ROOT / "reports" / "paper_v1_p1" / "repeatability.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two selected Clean-Core evaluation runs.")
    parser.add_argument("--run-a", type=Path, default=DEFAULT_RUN_A)
    parser.add_argument("--run-b", type=Path, default=DEFAULT_RUN_B)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def comparable_files(root: Path) -> dict[str, Path]:
    selected = root / "clean_core"
    return {
        path.relative_to(selected).as_posix(): path
        for path in selected.rglob("*.json")
        if path.parent.name in {"predictions", "feature_objects"}
    }


def main() -> None:
    args = parse_args()
    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()
    files_a = comparable_files(run_a)
    files_b = comparable_files(run_b)
    missing_in_a = sorted(set(files_b) - set(files_a))
    missing_in_b = sorted(set(files_a) - set(files_b))
    mismatches = []
    combined_hashes_a = []
    combined_hashes_b = []
    for relative_path in sorted(set(files_a) & set(files_b)):
        hash_a = stable_hash(canonicalize(load_json(files_a[relative_path])))
        hash_b = stable_hash(canonicalize(load_json(files_b[relative_path])))
        combined_hashes_a.append([relative_path, hash_a])
        combined_hashes_b.append([relative_path, hash_b])
        if hash_a != hash_b:
            mismatches.append({"path": relative_path, "hash_a": hash_a, "hash_b": hash_b})
    metrics_a = canonicalize(load_json(run_a / "metrics.json")["metrics"]["clean_core"])
    metrics_b = canonicalize(load_json(run_b / "metrics.json")["metrics"]["clean_core"])
    payload = {
        "schema": "topocae_paper_v1_selected_clean_repeatability_v1",
        "selected_variant": "clean_core",
        "run_a": str(run_a),
        "run_b": str(run_b),
        "compared_file_count": len(set(files_a) & set(files_b)),
        "combined_canonical_sha256_a": stable_hash(combined_hashes_a),
        "combined_canonical_sha256_b": stable_hash(combined_hashes_b),
        "missing_in_a": missing_in_a,
        "missing_in_b": missing_in_b,
        "mismatches": mismatches,
        "metrics_exact_match": metrics_a == metrics_b,
        "metrics_canonical_sha256_a": stable_hash(metrics_a),
        "metrics_canonical_sha256_b": stable_hash(metrics_b),
        "status": (
            "pass"
            if not missing_in_a and not missing_in_b and not mismatches and metrics_a == metrics_b
            else "fail"
        ),
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["status"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
