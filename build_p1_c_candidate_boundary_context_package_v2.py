from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from utils.p1_c_run_paths import p1_c_run_dir


SPLITS = ("train", "val", "development_test")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_dir = p1_c_run_dir(args.seed)
    source = run_dir / "candidate_training_v1"
    output = run_dir / "candidate_training_v2_boundary_context"
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "metadata": str(metadata_path)}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    predictions = torch.load(run_dir / "predictions.pt", map_location="cpu", weights_only=False)
    source_metadata = _read_json(source / "metadata.json")
    audits: dict[str, Any] = {}
    started = time.perf_counter()
    for split in SPLITS:
        records = torch.load(source / f"{split}.pt", map_location="cpu", weights_only=False)
        by_sample = {str(record["sample_id"]): record for record in predictions[split]}
        graph_cache: dict[str, dict[str, Any]] = {}
        histogram: Counter[int] = Counter()
        augmented = []
        for position, record in enumerate(records):
            graph_path = str(record["source_graph_path"])
            if graph_path not in graph_cache:
                graph_cache[graph_path] = _read_json(Path(graph_path))
            face_set = set(int(index) for index in record["face_indices"])
            neighbors = _boundary_neighbors(graph_cache[graph_path], face_set)
            embeddings = by_sample[str(record["sample_id"])]["node_embeddings"]
            context = (
                embeddings[neighbors].detach().cpu().to(torch.float16).contiguous()
                if neighbors
                else torch.empty((0, int(embeddings.shape[1])), dtype=torch.float16)
            )
            revised = dict(record)
            revised["boundary_context_face_indices"] = neighbors
            revised["boundary_context_face_embeddings"] = context
            augmented.append(revised)
            histogram[len(neighbors)] += 1
            if (position + 1) % 20000 == 0:
                print(f"seed={args.seed} split={split} candidate={position + 1}/{len(records)}", flush=True)
        destination = output / f"{split}.pt"
        torch.save(augmented, destination)
        audits[split] = {
            "candidate_count": len(augmented),
            "sample_count": len(by_sample),
            "boundary_context_count_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "empty_boundary_context_count": int(histogram[0]),
            "context_labels_added": False,
            "sha256": _sha256(destination),
            "bytes": destination.stat().st_size,
        }
    metadata = {
        **source_metadata,
        "schema_version": "topocae_p1_c_candidate_training_package_boundary_context_1",
        "source_candidate_package": str(source.resolve()),
        "source_candidate_manifest_sha256": _sha256(source / "manifest.json"),
        "boundary_context": {
            "kind": "one_hop_brep_adjacent_faces_outside_candidate",
            "embedding_source": "frozen_face_gnn_node_embeddings",
            "embedding_dim": int(source_metadata["dimensions"]["face_embedding_dim"]),
            "context_or_host_labels_used": False,
        },
        "splits": audits,
        "build_seconds": time.perf_counter() - started,
        "contracts": {
            **source_metadata.get("contracts", {}),
            "geometry_or_split_changed": False,
            "proposal_generation_changed": False,
            "truth_used_for_boundary_context": False,
            "real_data_used": False,
            "holdout_used": False,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [output / f"{split}.pt" for split in SPLITS] + [metadata_path]
    manifest = {
        "schema_version": "topocae_p1_c_candidate_boundary_context_manifest_1",
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = all(
        audit["candidate_count"] == source_metadata["splits"][split]["candidate_count"]
        and audit["sample_count"] == source_metadata["splits"][split]["graph_count"]
        for split, audit in audits.items()
    )
    print(json.dumps({"automatic_pass": passed, "seed": args.seed, "audits": audits, "output": str(output)}, ensure_ascii=False))
    return 0 if passed else 1


def _boundary_neighbors(graph: dict[str, Any], candidate_faces: set[int]) -> list[int]:
    neighbors = set()
    for edge in graph.get("edges", []):
        if not isinstance(edge, (list, tuple)) or len(edge) < 2:
            continue
        left, right = int(edge[0]), int(edge[1])
        if left in candidate_faces and right not in candidate_faces:
            neighbors.add(right)
        elif right in candidate_faces and left not in candidate_faces:
            neighbors.add(left)
    return sorted(neighbors)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
