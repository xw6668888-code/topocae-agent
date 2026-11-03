"""One fixed E3 Bayesian R+Q recognition service for arbitrary STEP files.

The engineering cache is never read by infer(); an optional, separate parity
audit compares fresh results only after the truth-free inference is complete.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch

from candidate_pool import capture_legacy_stages
from e_v09_adapter import infer_cached, load_route
from g09_common import (ARTIFACT, G07_ARTIFACT, G07_REPORT, ROOT, checked,
                        descriptor, digest, now, read, write_json)
from infer_step import _legacy_inference_module, _role_and_parameter_objects
from v09_predict_topo import _frozen_imports, _identity


SEED = 20260719
OUT = ARTIFACT / "inference_e3"


def _keys(objects):
    return Counter((obj.get("class", obj.get("macro_feature")),
                    tuple(sorted(obj.get("faces", obj.get("face_indices", [])))))
                   for obj in objects)


class E3StepService:
    """One shared E3 decision for every part, independent of filename and truth."""

    def __init__(self, device: str = "cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError("Only CPU or CUDA frozen inference is supported")
        self.legacy = _legacy_inference_module()
        self.geometry_only, Predictor, _ = _frozen_imports()
        from step_entry import numeric_guard
        self.numeric_guard = numeric_guard
        self.r = Predictor("D0", "old", SEED, device=device)
        self.q = Predictor("D0", "new", SEED, device=device,
                           configuration="dense_instance")
        self.model, self.mean, self.std, self.prior, self.temperature, self.score_identity = load_route("E3")
        self.identity = {"R": _identity(self.r, "R", SEED),
                         "Q": _identity(self.q, "Q", SEED),
                         "E3": self.score_identity,
                         "geometry_entry": descriptor(ROOT / "tools/aei_paper_goals_v1/g07/step_entry.py"),
                         "service": descriptor(__file__)}
        self.device = device

    def infer(self, step: str | Path) -> dict:
        step = Path(step).resolve()
        if not step.is_file() or step.suffix.lower() not in (".stp", ".step"):
            raise ValueError("An existing STEP file is required")
        started = time.perf_counter()
        shape = self.legacy.read_step_shape(step)
        raw = self.legacy.extract_graph_v2_2_1(shape, build_patches=False)
        graph, changes = self.numeric_guard(raw, enabled=True)
        graph = self.geometry_only(graph)
        if not graph["nodes"] or [row["idx"] for row in graph["nodes"]] != list(range(len(graph["nodes"]))):
            raise ValueError("Extracted STEP face indices are not contiguous")
        r_pred = self.r.forward(self.r.inputs([graph]))[0]
        q_pred = self.q.forward(self.q.inputs([graph]))[0]
        r_stages = capture_legacy_stages(graph, r_pred, self.r.same,
                                          self.r.candidate, self.r.global_decode)
        q_stages = capture_legacy_stages(graph, q_pred, self.q.same,
                                          self.q.candidate, self.q.global_decode)
        result = infer_cached(graph, r_pred, q_pred, r_stages, q_stages,
                              self.model, self.mean, self.std, route="E3",
                              prior=self.prior, temperature=self.temperature)
        if result["face_macro_classes"] != [face["macro_feature"] for face in r_pred["faces"]]:
            raise ValueError("R face labels changed during E3 object selection")
        if len(r_pred["faces"]) != len(graph["nodes"]) or len(q_pred["faces"]) != len(graph["nodes"]):
            raise ValueError("R/Q frozen model faces differ from STEP geometry")
        objects = [{"macro_feature": row["class"], "face_indices": row["faces"]}
                   for row in result["objects"]]
        roles, spacing = _role_and_parameter_objects(step, graph, r_pred, objects)
        if _keys(objects) != _keys([{"class": obj["class"], "faces": obj["graph_node_indices"]}
                                   for obj in roles]):
            raise ValueError("Delivered roles/parameters changed E3 object identity")
        return {"schema": "g09_e3_bayesian_step_service_v1", "at_utc": now(),
                "input_step": descriptor(step), "system": "frozen_D0_R_Q_seed_20260719_E3_bayesian_joint",
                "device": self.device, "source_identity": self.identity,
                "face_macro_classes": result["face_macro_classes"],
                "objects": result["objects"], "feature_objects": roles,
                "spacing_if_three_counterbores": spacing,
                "evidence_trace": {"graph_digest": digest(graph), "guard_change_count": len(changes),
                                   "R_prediction": r_pred, "Q_prediction": q_pred,
                                   "R_stages": r_stages, "Q_stages": q_stages,
                                   "candidate_count": result["candidate_count"],
                                   "selector": result["selection"]},
                "inference_seconds": time.perf_counter() - started,
                "truth_used": False}


def compare_engineering_cache(output: dict, reference: dict, service: E3StepService) -> dict:
    """Source-only cache audit performed after inference; no target access."""
    step = output["input_step"]
    if step != reference["source_step"]:
        raise ValueError("Engineering STEP source identity differs from cached reference")
    folder = checked(reference["prediction"]).parent
    cache_graph = service.geometry_only(read(folder / "guard_input_graph.json"))
    if output["evidence_trace"]["graph_digest"] != digest(cache_graph):
        raise ValueError("Fresh STEP guarded geometry differs from R/Q cached geometry")
    r_cache = read(folder / "probabilities.json")["guard"]
    qpath = G07_ARTIFACT / "engineering_predictions/D0/seed_20260719" / (folder.name + ".json")
    q_cache_row = read(qpath)
    if q_cache_row["source_step"] != step or q_cache_row["source_input"] != descriptor(folder / "step_input_graph.json"):
        raise ValueError("Q engineering prediction is not bound to the same STEP graph")
    q_cache = q_cache_row["guarded"]["prediction"]
    discrepancies = {}
    for branch, left, right in (("R", output["evidence_trace"]["R_prediction"], r_cache),
                                 ("Q", output["evidence_trace"]["Q_prediction"], q_cache)):
        if len(left["faces"]) != len(right["faces"]) or [f["macro_feature"] for f in left["faces"]] != [
            f["macro_feature"] for f in right["faces"]]:
            raise ValueError(f"Fresh {branch} face labels differ from cache")
        for name in ("edge_probabilities", "relation_probabilities"):
            if name not in right:
                continue
            a, b = left[name], right[name]
            if len(a) != len(b):
                raise ValueError(f"Fresh {branch} {name} cardinality differs from cache")
            delta = max((abs(x - y) for x, y in zip(a, b)), default=0.)
            discrepancies[f"{branch}_{name}_max_abs_delta"] = delta
            if delta > 1e-5:
                raise ValueError(f"Fresh {branch} {name} differs from cached probabilities")
        name = "macro_probabilities"
        for f, g in zip(left["faces"], right["faces"]):
            if set(f[name]) != set(g[name]):
                raise ValueError(f"Fresh {branch} face probability class keys differ")
        delta = max((abs(f[name][cls] - g[name][cls])
                     for f, g in zip(left["faces"], right["faces"])
                     for cls in f[name]), default=0.)
        discrepancies[f"{branch}_{name}_max_abs_delta"] = delta
        if delta > 1e-5:
            raise ValueError(f"Fresh {branch} face probabilities differ from cache")
    cache_r = capture_legacy_stages(cache_graph, r_cache, service.r.same,
                                     service.r.candidate, service.r.global_decode)
    cache_q = capture_legacy_stages(cache_graph, q_cache, service.q.same,
                                     service.q.candidate, service.q.global_decode)
    if _keys(cache_r["final"]) != _keys(output["evidence_trace"]["R_stages"]["final"]) or \
       _keys(cache_q["final"]) != _keys(output["evidence_trace"]["Q_stages"]["final"]):
        raise ValueError("Fresh R/Q candidate finals differ from cached stages")
    cached_result = infer_cached(cache_graph, r_cache, q_cache, cache_r, cache_q,
                                 service.model, service.mean, service.std, route="E3",
                                 prior=service.prior, temperature=service.temperature)
    if _keys(cached_result["objects"]) != _keys(output["objects"]):
        raise ValueError("Fresh STEP E3 objects differ from cached E3 scoring")
    return {"step": step, "R_cache": descriptor(folder / "probabilities.json"),
            "Q_cache": descriptor(qpath), "probability_delta": discrepancies,
            "object_count": len(output["objects"]), "same_graph": True,
            "same_selected_objects": True}


def seal_existing_engineering(destination: Path, references: list[dict],
                              service: E3StepService) -> dict:
    """Validate and seal 13 already completed truth-free outputs without rerunning."""
    seal_path = destination / "COMPLETE.json"
    if seal_path.exists():
        raise FileExistsError("Retain the existing E3 STEP batch seal")
    rows = []
    for reference in references:
        step = reference["source_step"]
        path = destination / f"{Path(step['path']).stem}_{step['sha256'][:12]}.json"
        parity_path = path.with_name(path.stem + "_PARITY.json")
        output, parity = read(path), read(parity_path)
        if output.get("schema") != "g09_e3_bayesian_step_service_v1" \
                or output.get("input_step") != step or output.get("source_identity") != service.identity \
                or output.get("truth_used") is not False or parity.get("step") != step \
                or parity.get("same_graph") is not True or parity.get("same_selected_objects") is not True:
            raise ValueError("Existing E3 STEP output or source-only parity changed")
        rows.append({"part": reference["part"], "input_step": step,
                     "prediction": descriptor(path), "parity": descriptor(parity_path),
                     "objects": len(output["objects"]), "seconds": output["inference_seconds"]})
    if len(rows) != 13 or len({row["part"] for row in rows}) != 13:
        raise ValueError("E3 STEP batch denominator changed")
    status = {"schema": "g09_e3_bayesian_step_batch_v1", "at_utc": now(),
              "system": "frozen_D0_R_Q_seed_20260719_E3_bayesian_joint",
              "source_identity": service.identity, "parts": 13,
              "source_only_cached_parity_parts": 13, "rows": rows,
              "inference_seconds_sum": sum(row["seconds"] for row in rows),
              "truth_used": False, "engineering_development_only": True}
    write_json(seal_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("steps", nargs="*", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--engineering-parity", action="store_true")
    parser.add_argument("--seal-existing-engineering", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    if not args.steps and not args.engineering_parity and not args.seal_existing_engineering:
        parser.error("Provide STEP files or run the 13-step engineering parity audit")
    destination = args.output_dir.resolve()
    if not destination.is_relative_to(ARTIFACT.resolve()):
        parser.error("Inference evidence must remain under G09 artifacts")
    references = []
    if args.engineering_parity or args.seal_existing_engineering:
        baseline = read(G07_REPORT / "input_audit/ENGINEERING_BASELINE_R.json")
        if baseline["parts"] != 13 or baseline["instances"] != 182:
            raise ValueError("Cached engineering source denominator changed")
        references = baseline["per_part"]
    if args.steps and references:
        parser.error("Explicit STEP list and full engineering parity are separate modes")
    if args.engineering_parity and args.seal_existing_engineering:
        parser.error("Run inference or seal an existing batch, not both")
    service = E3StepService(args.device)
    if args.seal_existing_engineering:
        status = seal_existing_engineering(destination, references, service)
        print(json.dumps({"seal": descriptor(destination / "COMPLETE.json"),
                          "parts": status["parts"], "truth_used": False}, ensure_ascii=False), flush=True)
        return
    for step, reference in ((Path(item["source_step"]["path"]), item) for item in references) if references \
            else ((step, None) for step in args.steps):
        result = service.infer(step)
        path = destination / f"{step.stem}_{result['input_step']['sha256'][:12]}.json"
        if path.exists():
            raise FileExistsError(f"Preserve existing E3 inference: {path}")
        write_json(path, result)
        parity = compare_engineering_cache(result, reference, service) if reference else None
        if parity is not None:
            write_json(path.with_name(path.stem + "_PARITY.json"), parity)
        print(json.dumps({"STEP": str(step), "output": descriptor(path),
                          "objects": len(result["objects"]), "seconds": result["inference_seconds"],
                          "cache_parity": parity is not None, "truth_used": False},
                         ensure_ascii=False), flush=True)
    if references:
        status = seal_existing_engineering(destination, references, service)
        print(json.dumps({"batch_seal": descriptor(destination / "COMPLETE.json"),
                          "parts": status["parts"], "truth_used": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
