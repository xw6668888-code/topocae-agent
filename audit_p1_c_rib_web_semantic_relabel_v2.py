from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "output_generation_p1_c_clean_context_7200"
RIB_HOSTS = {"l_corner", "rect_cavity", "circular_cavity"}
WEB_HOSTS = {"two_supports"}


def main() -> int:
    graph_audit_path = ROOT / "reports" / "p1_c_iterations_v1" / "rib_web_semantic_relabel_audit.json"
    graph_audit = _read_json(graph_audit_path)
    split_counts: Counter[str] = Counter()
    split_target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_host_counts: dict[str, Counter[str]] = defaultdict(Counter)
    incompatible_ids = []
    wall_total = 0
    for split in ("train", "val", "development_test"):
        path = DATASET / "splits" / f"{split}.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            macros = [
                macro for macro in record.get("macro_instance_counts", {})
                if macro in {"rib", "web"}
            ]
            if not macros:
                continue
            if len(macros) != 1:
                raise AssertionError(f"expected one supported wall: {record['sample_id']}")
            wall_total += 1
            target = macros[0]
            host = str(record["host_family"])
            compatible = host in (RIB_HOSTS if target == "rib" else WEB_HOSTS)
            if not compatible:
                split_counts[split] += 1
                split_target_counts[split][target] += 1
                split_host_counts[split][host] += 1
                incompatible_ids.append(str(record["sample_id"]))
    graph_incompatible = sum(graph_audit["goal_concept_incompatible_count"].values())
    if wall_total != int(graph_audit["supported_wall_sample_count"]):
        raise AssertionError(f"wall total mismatch: {wall_total}")
    if len(incompatible_ids) != graph_incompatible:
        raise AssertionError(
            f"incompatible total mismatch: split={len(incompatible_ids)} graph={graph_incompatible}"
        )
    result: dict[str, Any] = {
        **graph_audit,
        "schema_version": "topocae_p1_c_rib_web_semantic_relabel_audit_2",
        "supersedes": str(graph_audit_path.resolve()),
        "incompatible_split_distribution": dict(split_counts),
        "incompatible_target_by_split": {
            split: dict(counts) for split, counts in sorted(split_target_counts.items())
        },
        "incompatible_host_by_split": {
            split: dict(counts) for split, counts in sorted(split_host_counts.items())
        },
        "incompatible_sample_ids": incompatible_ids,
        "split_manifest_cross_check_passed": True,
    }
    destination = ROOT / "reports" / "p1_c_iterations_v1" / "rib_web_semantic_relabel_audit_v2.json"
    destination.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "output": str(destination),
        "incompatible_total": len(incompatible_ids),
        "split_distribution": dict(split_counts),
        "target_by_split": result["incompatible_target_by_split"],
        "cross_check_passed": True,
    }, ensure_ascii=False))
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
