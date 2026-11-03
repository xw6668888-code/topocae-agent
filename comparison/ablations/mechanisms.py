"""Paper-only views of the fixed R/Q candidate-pool ablation mechanism."""
from __future__ import annotations
import time
from typing import Any, Iterable
import torch
A0="R"
A1="G10_M3_A1_Q_FINAL"
A3="G10_M3_A3_GENERIC_SCORE_EXACT_PACK"
A4="G10_M3_A4_E1_LEARNED_SCORE_EXACT_PACK"
A6="E3"
VARIANTS=(A0,A1,A3,A4,A6)

def _key(row: dict[str, Any]) -> tuple[str, tuple[int, ...]]:
    macro = row.get("class", row.get("macro_feature"))
    faces = row.get("faces", row.get("face_indices"))
    if not isinstance(macro, str) or not macro or not isinstance(faces, list):
        raise ValueError("Malformed complete-object record")
    normalized = tuple(sorted(set(faces)))
    if not normalized or len(normalized) != len(faces) or any(
        isinstance(face, bool) or not isinstance(face, int) or face < 0 for face in faces
    ):
        raise ValueError("Complete-object faces must be distinct nonnegative integers")
    return macro, normalized


def _objects(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for row in rows:
        key = _key(row)
        if key in seen:
            continue
        seen.add(key)
        result.append({"class": key[0], "faces": list(key[1])})
    return result


def infer_variants(graph: dict[str, Any], r_prediction: dict[str, Any],
                   q_prediction: dict[str, Any], r_stages: dict[str, Any],
                   q_stages: dict[str, Any], assets: dict[str, Any]) -> dict[str, Any]:
    """Infer only the paper component ablations without identity, path, or truth input."""
    modules = assets["modules"]
    graph = modules["g07_decoder"].clean_graph(graph)
    r_prediction = modules["g07_decoder"].clean_prediction(r_prediction)
    q_prediction = modules["g07_decoder"].clean_prediction(q_prediction)
    started = time.perf_counter()
    pool = modules["build_candidate_pool"](
        {"R": r_stages, "Q": q_stages}, face_count=len(graph["nodes"])
    )
    generic_rows = []
    vectors = []
    for candidate in pool:
        branches = {origin["branch"] for origin in candidate["provenance"]}
        if not branches or not branches <= {"R", "Q"}:
            raise ValueError("Unexpected branch provenance in joint candidate")
        r_features = modules["extract_object_features"](candidate, r_prediction, graph)
        q_features = modules["extract_object_features"](candidate, q_prediction, graph)
        vector = (tuple(r_features["vector"]) + tuple(q_features["vector"])
                  + (float("R" in branches), float("Q" in branches)))
        generic_rows.append({
            **candidate,
            "score": (modules["deterministic_score"](r_features)
                      + modules["deterministic_score"](q_features)) / 2.0,
            "features": {"vector": vector},
        })
        vectors.append(vector)
    feature_seconds = time.perf_counter() - started

    started = time.perf_counter()
    if vectors:
        x = torch.tensor(vectors, dtype=torch.float32)
        with torch.no_grad():
            e1_scores = assets["e1"](((x - assets["e1_mean"]) / assets["e1_std"]).clamp(-8, 8)).tolist()
    else:
        e1_scores = []
    neural_score_seconds = time.perf_counter() - started

    scored = {A3: generic_rows, A4: [], A6: []}
    for candidate, vector, e1_raw in zip(pool, vectors, e1_scores, strict=True):
        common = {**candidate, "features": {"vector": vector}}
        scored[A4].append({**common, "score": float(e1_raw)})
        scored[A6].append({**common, "score": modules["calibrated_score"](
            float(e1_raw), candidate, assets["prior"], assets["temperature"])})

    selections = {}
    selection_seconds = {}
    for variant in (A3, A4, A6):
        began = time.perf_counter()
        selections[variant] = modules["select_objects"](
            scored[variant], exact_limit=32, beam_width=128
        )
        selection_seconds[variant] = time.perf_counter() - began
    outputs = {
        A0: _objects(r_stages["final"]),
        A1: _objects(q_stages["final"]),
    }
    for variant, selection in selections.items():
        outputs[variant] = _objects(selection["selected_objects"])
    if set(outputs) != set(VARIANTS):
        raise ValueError("Ablation output set differs from frozen protocol")
    return {
        "objects": outputs,
        "candidate_objects": _objects(pool),
        "candidate_count": len(pool),
        "selection": {variant: selections[variant]["diagnostics"] for variant in selections},
        "timing_seconds": {
            "joint_candidate_and_features": feature_seconds,
            "object_neural_scores": neural_score_seconds,
            "selection": selection_seconds,
        },
    }

