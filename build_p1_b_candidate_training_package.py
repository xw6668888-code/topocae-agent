from __future__ import annotations

"""Build candidate-level training tensors from a frozen P1-B GNN run."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable

import torch

from utils.p1_b_candidate_features import ENGINEERED_FEATURE_SCHEMA, MACRO_CLASSES, build_candidate_tensor_record
from utils.p1_b_candidate_hard_negatives import generate_candidate_hard_negatives
from utils.p1_b_candidate_proposer import PROPOSAL_SOURCES, generate_candidate_proposals
from utils.p1_b_instance_assembly import assemble_same_instance_candidates


ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "reports" / "p1_b_training_runs_v1"
TRAINING_METADATA = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2" / "metadata.json"
EDGE_DIM = 20
SEMANTIC_ROLE_COUNT = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = RUN_ROOT / f"same_instance_seed_{args.seed}"
    output = run_dir / "candidate_training_v1"
    metadata_path = output / "metadata.json"
    if metadata_path.exists() and not args.force:
        print(json.dumps({"status": "already_complete", "metadata": str(metadata_path)}, ensure_ascii=False))
        return 0
    output.mkdir(parents=True, exist_ok=True)
    source_predictions_path = run_dir / "predictions.pt"
    predictions = torch.load(source_predictions_path, map_location="cpu", weights_only=False)
    training_metadata = _read_json(TRAINING_METADATA)
    compact_macros = list(training_metadata["macro_classes"])
    threshold_selection = _read_json(run_dir / "same_instance_threshold_selection.json")
    same_threshold = float(threshold_selection["global_threshold"])
    threshold_hash = str(threshold_selection["selection_sha256"])
    split_payloads: dict[str, list[dict[str, Any]]] = {}
    audits: dict[str, Any] = {}
    started = time.perf_counter()
    for split, prediction_records in predictions.items():
        candidate_records: list[dict[str, Any]] = []
        truth_total = truth_covered = truth_exact = 0
        source_counts: Counter[str] = Counter()
        origin_counts: Counter[str] = Counter()
        validity_counts: Counter[int] = Counter()
        hard_negative_kinds: Counter[str] = Counter()
        for position, prediction_record in enumerate(prediction_records):
            graph = _read_json(Path(prediction_record["source_graph_path"]))
            prediction = _prediction_data(prediction_record, compact_macros)
            assembly = assemble_same_instance_candidates(
                graph, prediction, global_threshold=same_threshold,
                threshold_selection_sha256=threshold_hash,
            )
            proposal_result = generate_candidate_proposals(graph, prediction, assembly)
            proposals = list(proposal_result["proposals"])
            source_counts.update(proposal_result.get("source_counts", {}))
            truths = _truth_instances(graph)
            recall = _proposal_recall(proposals, truths)
            truth_total += recall["truth_count"]
            truth_covered += recall["covered_at_iou_0_5"]
            truth_exact += recall["covered_exact"]
            for proposal in proposals:
                candidate_records.append(_build_record(
                    graph, prediction, proposal, truths, prediction_record,
                    origin="generated_proposal", force_invalid=False,
                ))
                origin_counts["generated_proposal"] += 1
            if split == "train":
                for truth_index, truth in enumerate(truths):
                    proposal = {
                        "candidate_id": f"teacher_truth_{truth_index:04d}",
                        "proposal_sources": ["same_instance_strong_component"],
                        "face_indices": truth["face_indices"],
                        "macro_hints": [truth["macro_feature"]],
                        "topology_evidence": {"teacher_forced_synthetic_truth": True},
                    }
                    candidate_records.append(_build_record(
                        graph, prediction, proposal, truths, prediction_record,
                        origin="teacher_truth", force_invalid=False,
                    ))
                    origin_counts["teacher_truth"] += 1
                hard = generate_candidate_hard_negatives(graph, truths)["records"]
                for negative in _balanced_hard_negative_subset(hard):
                    proposal = {
                        "candidate_id": negative["hard_negative_id"],
                        "proposal_sources": [],
                        "face_indices": negative["face_indices"],
                        "macro_hints": [negative["proposed_macro"]],
                        "topology_evidence": negative["topology_evidence"],
                    }
                    built = _build_record(
                        graph, prediction, proposal, truths, prediction_record,
                        origin=f"hard_negative:{negative['negative_kind']}", force_invalid=True,
                    )
                    built["negative_kind"] = negative["negative_kind"]
                    candidate_records.append(built)
                    hard_negative_kinds[negative["negative_kind"]] += 1
                    origin_counts["hard_negative"] += 1
            if (position + 1) % 500 == 0:
                print(f"seed={args.seed} split={split} graph={position + 1}/{len(prediction_records)} candidates={len(candidate_records)}", flush=True)
        validity_counts.update(int(record["validity_target"].item()) for record in candidate_records)
        destination = output / f"{split}.pt"
        torch.save(candidate_records, destination)
        split_payloads[split] = candidate_records
        audits[split] = {
            "graph_count": len(prediction_records), "candidate_count": len(candidate_records),
            "truth_instance_count": truth_total,
            "proposal_oracle_recall_iou_0_5": truth_covered / max(truth_total, 1),
            "proposal_oracle_exact_recall": truth_exact / max(truth_total, 1),
            "proposal_source_counts": dict(sorted(source_counts.items())),
            "all_seven_sources_present": all(source_counts[source] > 0 for source in PROPOSAL_SOURCES),
            "origin_counts": dict(origin_counts), "validity_counts": dict(validity_counts),
            "hard_negative_kind_counts": dict(sorted(hard_negative_kinds.items())),
            "sha256": _sha256(destination), "bytes": destination.stat().st_size,
        }
    face_dims = {int(record["face_embeddings"].shape[1]) for records in split_payloads.values() for record in records}
    edge_dims = {
        int(tensor.shape[1])
        for records in split_payloads.values() for record in records
        for tensor in (record["internal_edge_embeddings"], record["boundary_edge_embeddings"])
    }
    metadata = {
        "schema_version": "topocae_p1_b_candidate_training_package_1",
        "seed": args.seed, "source_gnn_run": str(run_dir.resolve()),
        "source_predictions_sha256": _sha256(source_predictions_path),
        "same_instance_threshold_selection_sha256": threshold_hash,
        "dimensions": {
            "face_embedding_dim": next(iter(face_dims)), "edge_embedding_dim": EDGE_DIM,
            "engineered_feature_dim": len(ENGINEERED_FEATURE_SCHEMA), "semantic_role_count": SEMANTIC_ROLE_COUNT,
        },
        "macro_classes": MACRO_CLASSES, "engineered_feature_schema": ENGINEERED_FEATURE_SCHEMA,
        "splits": audits, "build_seconds": time.perf_counter() - started,
        "contracts": {
            "candidate_generation_uses_truth": False,
            "teacher_truth_used_train_only": True,
            "hard_negatives_use_synthetic_truth_and_generic_topology_only": True,
            "threshold_selected_synthetic_val_only": True,
            "real_data_used": False, "holdout_used": False,
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "schema_version": "topocae_p1_b_candidate_training_manifest_1",
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in [output / "train.pt", output / "val.pt", output / "development_test.pt", metadata_path]
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = (
        face_dims == {128} and edge_dims <= {EDGE_DIM}
        and audits["val"]["proposal_oracle_recall_iou_0_5"] >= 0.95
        and audits["val"]["all_seven_sources_present"]
        and len(audits["train"]["hard_negative_kind_counts"]) >= 4
    )
    print(json.dumps({
        "automatic_pass": passed, "seed": args.seed,
        "candidate_counts": {split: audit["candidate_count"] for split, audit in audits.items()},
        "val_proposal_oracle_recall": audits["val"]["proposal_oracle_recall_iou_0_5"],
        "val_proposal_exact_recall": audits["val"]["proposal_oracle_exact_recall"],
        "output": str(output),
    }, ensure_ascii=False), flush=True)
    return 0 if passed else 1


def _prediction_data(record: dict[str, Any], compact_macros: list[str]) -> dict[str, Any]:
    return {
        "faces": [
            {"idx": index, "macro_feature": compact_macros[int(macro)], "embedding": record["node_embeddings"][index].float().tolist()}
            for index, macro in enumerate(record["macro_prediction"])
        ],
        "node_embeddings": record["node_embeddings"].float().tolist(),
        "edge_same_instance_probabilities": record["edge_probability"].float().tolist(),
    }


def _truth_instances(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "instance_id": int(record.get("instance_id", position + 1)),
            "macro_feature": str(record["macro_feature"]),
            "face_indices": sorted(set(int(index) for index in record.get("node_indices", record.get("face_indices", [])))),
        }
        for position, record in enumerate(graph.get("feature_sets", []))
        if record.get("node_indices", record.get("face_indices", []))
    ]


def _build_record(
    graph: dict[str, Any], prediction: dict[str, Any], proposal: dict[str, Any],
    truths: list[dict[str, Any]], source: dict[str, Any], *, origin: str, force_invalid: bool,
) -> dict[str, Any]:
    tensor = build_candidate_tensor_record(graph, prediction, proposal)
    candidate_faces = set(int(index) for index in proposal["face_indices"])
    matched, iou = _best_truth(candidate_faces, truths)
    validity = int(not force_invalid and matched is not None and iou >= 0.5)
    truth_faces = set(matched["face_indices"]) if matched is not None else set()
    macro_target = MACRO_CLASSES.index(matched["macro_feature"]) if validity else 0
    membership = [float(index in truth_faces) for index in tensor["face_indices"]]
    boundary_truth = _boundary_faces(graph, truth_faces)
    boundary = [float(index in boundary_truth) for index in tensor["face_indices"]]
    node_by_index = {int(node.get("idx", position)): node for position, node in enumerate(graph.get("nodes", []))}
    semantic_role = [_semantic_role(int(node_by_index[index].get("semantic_label", 0) or 0)) for index in tensor["face_indices"]]
    internal = torch.tensor(tensor["internal_edge_embeddings"], dtype=torch.float16)
    boundary_edges = torch.tensor(tensor["boundary_edge_embeddings"], dtype=torch.float16)
    if internal.numel() == 0:
        internal = torch.empty((0, EDGE_DIM), dtype=torch.float16)
    if boundary_edges.numel() == 0:
        boundary_edges = torch.empty((0, EDGE_DIM), dtype=torch.float16)
    return {
        "sample_id": source["sample_id"], "source_graph_path": source["source_graph_path"],
        "base_geometry_family_id": source["base_geometry_family_id"], "derivation_kind": source["derivation_kind"],
        "candidate_id": str(proposal.get("candidate_id", "")), "origin": origin,
        "proposal": proposal, "face_indices": tensor["face_indices"],
        "face_embeddings": torch.tensor(tensor["face_embeddings"], dtype=torch.float16),
        "internal_edge_embeddings": internal, "boundary_edge_embeddings": boundary_edges,
        "engineered_features": torch.tensor(tensor["engineered_features"], dtype=torch.float32),
        "validity_target": torch.tensor(float(validity)), "macro_target": torch.tensor(macro_target, dtype=torch.long),
        "membership_target": torch.tensor(membership, dtype=torch.float32),
        "boundary_target": torch.tensor(boundary, dtype=torch.float32),
        "semantic_role_target": torch.tensor(semantic_role, dtype=torch.long),
        "matched_truth_instance_id": matched["instance_id"] if matched is not None else None,
        "matched_truth_macro": matched["macro_feature"] if matched is not None else None,
        "matched_truth_face_indices": sorted(truth_faces), "matched_truth_iou": iou,
    }


def _best_truth(candidate_faces: set[int], truths: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, float]:
    if not candidate_faces or not truths:
        return None, 0.0
    scored = []
    for truth in truths:
        truth_faces = set(truth["face_indices"])
        union = candidate_faces | truth_faces
        scored.append((len(candidate_faces & truth_faces) / max(len(union), 1), truth))
    iou, truth = max(scored, key=lambda item: (item[0], -int(item[1]["instance_id"])))
    return truth, float(iou)


def _boundary_faces(graph: dict[str, Any], truth_faces: set[int]) -> set[int]:
    boundary = set()
    for edge in graph.get("edges", []):
        left, right = int(edge[0]), int(edge[1])
        if left in truth_faces and right not in truth_faces:
            boundary.add(left)
        if right in truth_faces and left not in truth_faces:
            boundary.add(right)
    return boundary


def _proposal_recall(proposals: list[dict[str, Any]], truths: list[dict[str, Any]]) -> dict[str, int]:
    covered = exact = 0
    proposal_sets = [set(record["face_indices"]) for record in proposals]
    for truth in truths:
        truth_faces = set(truth["face_indices"])
        ious = [len(truth_faces & faces) / max(len(truth_faces | faces), 1) for faces in proposal_sets]
        best = max(ious, default=0.0)
        covered += int(best >= 0.5)
        exact += int(best >= 1.0)
    return {"truth_count": len(truths), "covered_at_iou_0_5": covered, "covered_exact": exact}


def _balanced_hard_negative_subset(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_kind[str(record["negative_kind"])].append(record)
    selected = []
    for kind in sorted(by_kind):
        selected.extend(by_kind[kind][:2])
    return selected


def _semantic_role(label: int) -> int:
    if label in {4, 7, 25, 27, 32, 41, 58, 62}:
        return 1  # bottom
    if label in {13, 29, 45, 51}:
        return 2  # top
    if label in {2, 36}:
        return 3  # step/shoulder
    if label in {1, 3, 5, 6, 9, 12, 31, 35, 37, 39, 40}:
        return 4  # cylindrical/curved wall
    if label in {8, 24, 26, 28, 30, 44, 50, 57, 61, 63}:
        return 5  # planar side wall
    if label in {34, 38}:
        return 6  # cone/bevel
    if label == 33:
        return 7  # transition
    return 0


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
