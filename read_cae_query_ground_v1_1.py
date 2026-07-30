from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "output_cae_query_ground_v1_1" / "manifests" / "dataset_manifest.json"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def source_paths(kind: str) -> list[Path]:
    manifest = load_manifest()
    if kind == "canonical":
        roles = {
            "upstream_v1_canonical_train",
            "upstream_v1_canonical_val",
            "upstream_v1_canonical_test",
            "p3_a1r_supplement_canonical",
        }
    else:
        roles = {
            "upstream_v1_renderings_train",
            "upstream_v1_renderings_val",
            "upstream_v1_renderings_test",
            "p3_a1r_supplement_renderings",
        }
    return [Path(entry["path"]) for entry in manifest["files"] if entry["role"] in roles]


def iter_records(kind: str, split: str | None = None) -> Iterable[dict[str, Any]]:
    canonical_split: dict[str, str] = {}
    if kind == "renderings" and split is not None:
        for record in iter_records("canonical", split):
            canonical_split[str(record["source_ref"]["canonical_record_id"])] = split
    for path in source_paths(kind):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if split is None:
                    yield record
                    continue
                if kind == "canonical":
                    if str(record["source_ref"]["dataset_split"]) == split:
                        yield record
                else:
                    record_split = record.get("source_ref", {}).get("dataset_split")
                    if record_split is None:
                        canonical_id = str(
                            record.get("canonical_record_id")
                            or record.get("source_ref", {}).get("canonical_record_id")
                        )
                        record_split = canonical_split.get(canonical_id)
                    if record_split == split:
                        yield record


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only CAE-QueryGround 1.1 entrypoint")
    parser.add_argument("--kind", choices=("canonical", "renderings"), default="canonical")
    parser.add_argument("--split", choices=("train", "val", "test"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        manifest = load_manifest()
        print(json.dumps({
            "dataset_version": manifest["dataset_version"],
            "counts": manifest["counts"],
            "semantic_strata": manifest["semantic_strata"],
        }, ensure_ascii=False, indent=2))
        return 0
    emitted = 0
    for record in iter_records(args.kind, args.split):
        print(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
        emitted += 1
        if args.limit and emitted >= args.limit:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
