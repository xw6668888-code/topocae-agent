"""v5.11 Object Proposal Recall Diagnostics.

Diagnostic-only tool — traces each area-truth target through four pipeline stages:
  1. proposal          — raw GNN candidate exists
  2. classified        — candidate survives classifier without rejected label
  3. accepted          — candidate action is accepted_remap
  4. final_survival    — candidate faces still carry the classified macro in final predictions

Does NOT modify model architecture, training, generation, or inference.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from utils.area_truth_normalization import normalize_truth_macro
from utils.instance_candidate_assembly import assemble_instance_candidates
from utils.instance_candidate_postprocess import classify_instance_candidate_proposals
from utils.object_candidate_proposer import generate_hybrid_object_proposals

REJECT_LABELS = {"reject", "stock", "background", "none", "mixed"}

FAILURE_STAGES = (
    "survives_final_prediction",
    "classifier_accepts_but_final_changes_or_removes",
    "proposal_exists_classifier_wrong_or_rejected",
    "no_area_compatible_proposal",
)


def normalize_macro_label(
    value: object,
    *,
    sample_id: str | None = None,
    area: float | None = None,
    face_count: int | None = None,
    truth_source_name: str | None = None,
) -> str:
    """Normalize macro labels into the project prediction vocabulary."""
    text = str(value).strip()
    if not text:
        return "unknown"
    return normalize_truth_macro(
        text,
        sample_id=sample_id,
        area=area,
        face_count=face_count,
        truth_source_name=truth_source_name,
    )


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_json(path: Path) -> object:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Area helpers
# ---------------------------------------------------------------------------


def node_area(node: dict) -> float:
    """Read area from a graph node, preferring attributes.physical.area."""
    try:
        physical = node.get("attributes", {}).get("physical", {})
        if isinstance(physical, dict) and "area" in physical:
            val = physical["area"]
            if val is not None:
                return float(val)
    except (TypeError, AttributeError):
        pass
    try:
        val = node.get("area")
        if val is not None:
            return float(val)
    except (TypeError, ValueError):
        pass
    return 0.0


def candidate_area(candidate: dict, graph: dict | None = None) -> float:
    """Return candidate area, falling back to graph node area sum."""
    area = candidate.get("area")
    if area is not None:
        try:
            return float(area)
        except (TypeError, ValueError):
            pass

    if graph is not None:
        face_ids = _face_ids(candidate)
        nodes = _node_list(graph)
        node_by_idx = {}
        for node in nodes:
            idx = _node_idx(node)
            node_by_idx[idx] = node
        total = 0.0
        for fid in face_ids:
            if fid in node_by_idx:
                total += node_area(node_by_idx[fid])
        return total

    return 0.0


# ---------------------------------------------------------------------------
# Face id extraction
# ---------------------------------------------------------------------------


def _face_ids(candidate: dict) -> list[int]:
    for key in ("face_indices", "faces", "face_ids"):
        val = candidate.get(key)
        if val is not None:
            try:
                return sorted(int(v) for v in val)
            except (TypeError, ValueError):
                pass
    return []


def _node_idx(node: dict) -> int:
    for key in ("idx", "face_index", "index", "id"):
        val = node.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return -1


def _node_list(graph: dict) -> list[dict]:
    raw = graph.get("faces", graph.get("nodes", []))
    if isinstance(raw, list):
        return raw
    return []


# ---------------------------------------------------------------------------
# Truth normalization
# ---------------------------------------------------------------------------


def normalize_truth_entries(payload: object) -> dict[str, list[dict]]:
    """Convert area truth JSON into {sample_name: [normalized_entry, ...]}.

    Accepts common key aliases: sample/part/name, macro/Macro/feature/feature_type,
    area/Area/area_sum, face_count/number_of_faces/num_faces.

    Macros are normalized to the project prediction vocabulary via
    utils.area_truth_normalization.normalize_truth_macro.
    """
    result: dict[str, list[dict]] = {}

    if not isinstance(payload, dict):
        return result

    truth_source_name = None
    if isinstance(payload, dict):
        truth_source_name = payload.get("source_xlsx") or payload.get("truth_source_name") or payload.get("source")
        if truth_source_name:
            truth_source_name = Path(str(truth_source_name)).name

    samples = payload.get("samples", payload)
    if not isinstance(samples, dict):
        return result

    for sample_name, entries in samples.items():
        if not isinstance(entries, list):
            continue
        normalized = []
        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            area = _first_float(entry, ("area", "Area", "area_sum"), 0.0)
            face_count = _first_int(entry, ("face_count", "number_of_faces", "number of faces", "num_faces"))
            raw_macro = _first_str(entry, ("macro", "Macro", "feature", "feature_type"), "unknown")
            normalized.append(
                {
                    "sample": str(sample_name),
                    "macro": normalize_macro_label(
                        raw_macro,
                        sample_id=str(sample_name),
                        area=area,
                        face_count=face_count,
                        truth_source_name=truth_source_name,
                    ),
                    "raw_macro": str(raw_macro),
                    "area": area,
                    "face_count": face_count,
                    "truth_index": idx,
                }
            )
        result[str(sample_name)] = normalized

    return result


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------


def candidate_macro_for_stage(candidate: dict, stage: str) -> str | None:
    """Return the normalized macro label to use for a given pipeline stage."""
    if stage == "proposal":
        val = candidate.get("proposal_macro")
        return normalize_macro_label(val) if val is not None else None
    if stage in ("classified", "accepted", "final_survival"):
        val = candidate.get("classified_macro")
        if val is not None:
            label = normalize_macro_label(val)
            if label.lower() in (r.lower() for r in REJECT_LABELS):
                return None
            return label
        return None
    return None


def final_survives(candidate: dict, final_prediction: dict, macro: str) -> bool:
    """Check whether candidate faces still carry *macro* in the final prediction."""
    face_ids = _face_ids(candidate)
    if not face_ids:
        return False
    faces = final_prediction.get("faces", [])
    face_macro_by_idx: dict[int, str] = {}
    for face in faces:
        if not isinstance(face, dict):
            continue
        idx = _node_idx(face)
        if idx < 0:
            continue
        face_macro = _first_str(face, ("macro_feature", "predicted_macro_feature", "predicted_macro", "macro"), "")
        face_macro_by_idx[idx] = normalize_macro_label(face_macro)

    for fid in face_ids:
        actual = face_macro_by_idx.get(fid)
        if actual != macro:
            return False
    return True


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def candidate_matches_truth(
    candidate: dict,
    truth: dict,
    stage: str,
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
    final_prediction: dict | None = None,
) -> bool:
    """Check whether a candidate matches a truth entry at a given pipeline stage."""
    stage_macro = candidate_macro_for_stage(candidate, stage)
    if stage_macro is None:
        return False
    truth_macro = str(truth.get("macro", ""))
    if stage_macro != truth_macro:
        return False

    cand_area = candidate_area(candidate)
    truth_area = float(truth.get("area", 0))
    tolerance = max(absolute_area_tolerance, relative_area_tolerance * max(abs(truth_area), 1.0))
    if abs(cand_area - truth_area) > tolerance:
        return False

    truth_fc = truth.get("face_count")
    if truth_fc is not None:
        cand_fc = len(_face_ids(candidate))
        if cand_fc != int(truth_fc):
            return False

    if stage == "accepted":
        action = candidate.get("action", "")
        if action != "accepted_remap":
            return False

    if stage == "final_survival":
        if final_prediction is None:
            return False
        if not final_survives(candidate, final_prediction, stage_macro):
            return False

    return True


def _truth_key(truth: dict, position: int) -> int:
    """Return a stable integer key for a truth entry."""
    tid = truth.get("truth_index")
    if tid is not None:
        try:
            return int(tid)
        except (TypeError, ValueError):
            pass
    return position


def matched_truth_indices_for_stage(
    truth_entries: list[dict],
    candidates: list[dict],
    stage: str,
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
    final_prediction: dict | None = None,
) -> set[int]:
    """Return truth indices that have at least one matching candidate at *stage*."""
    matched: set[int] = set()
    for pos, truth in enumerate(truth_entries):
        for candidate in candidates:
            if candidate_matches_truth(
                candidate,
                truth,
                stage,
                absolute_area_tolerance,
                relative_area_tolerance,
                final_prediction=final_prediction,
            ):
                matched.add(_truth_key(truth, pos))
                break
    return matched


def summarize_stage_matches(truth_entries: list[dict], matched_indices: set[int]) -> dict:
    """Build a stage summary dict with per-macro breakdowns."""
    matched_by_macro: dict[str, int] = defaultdict(int)
    truth_by_macro: dict[str, int] = defaultdict(int)
    missed_by_macro: dict[str, int] = defaultdict(int)

    for pos, truth in enumerate(truth_entries):
        macro = str(truth.get("macro", "unknown"))
        idx = _truth_key(truth, pos)
        truth_by_macro[macro] += 1
        if idx in matched_indices:
            matched_by_macro[macro] += 1
        else:
            missed_by_macro[macro] += 1

    return {
        "matched_truth_count": len(matched_indices),
        "recall": len(matched_indices) / len(truth_entries) if truth_entries else 0.0,
        "matched_by_macro": dict(matched_by_macro),
        "truth_by_macro": dict(truth_by_macro),
        "missed_by_macro": dict(missed_by_macro),
    }


def classify_truth_failure_stage(
    truth: dict,
    candidates: list[dict],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
    final_prediction: dict | None = None,
) -> str:
    """Assign exactly one failure/survival stage for a truth entry."""
    # Stage 4: final_survival
    final_matched = matched_truth_indices_for_stage(
        [truth], candidates, "final_survival", absolute_area_tolerance, relative_area_tolerance, final_prediction
    )
    if final_matched:
        return "survives_final_prediction"

    # Stage 3: accepted
    accepted_matched = matched_truth_indices_for_stage(
        [truth], candidates, "accepted", absolute_area_tolerance, relative_area_tolerance
    )
    if accepted_matched:
        return "classifier_accepts_but_final_changes_or_removes"

    # Stage 2: classified/proposal
    proposal_matched = matched_truth_indices_for_stage(
        [truth], candidates, "proposal", absolute_area_tolerance, relative_area_tolerance
    )
    if proposal_matched:
        return "proposal_exists_classifier_wrong_or_rejected"

    return "no_area_compatible_proposal"


def match_truth_by_stage(
    truth_entries: list[dict],
    candidates: list[dict],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
    final_prediction: dict | None = None,
) -> dict:
    """Compute per-stage matched indices and aggregate summaries."""
    stages = {
        "proposal": matched_truth_indices_for_stage(
            truth_entries, candidates, "proposal", absolute_area_tolerance, relative_area_tolerance
        ),
        "classified": matched_truth_indices_for_stage(
            truth_entries,
            candidates,
            "classified",
            absolute_area_tolerance,
            relative_area_tolerance,
        ),
        "accepted": matched_truth_indices_for_stage(
            truth_entries, candidates, "accepted", absolute_area_tolerance, relative_area_tolerance
        ),
        "final_survival": matched_truth_indices_for_stage(
            truth_entries,
            candidates,
            "final_survival",
            absolute_area_tolerance,
            relative_area_tolerance,
            final_prediction=final_prediction,
        ),
    }

    result = {}
    for stage_name, matched in stages.items():
        result[stage_name + "_stage" if not stage_name.endswith("_stage") else stage_name] = summarize_stage_matches(
            truth_entries, matched
        )

    # Failure stage counts
    failure_counts: Counter = Counter()
    truth_failures = []
    for pos, truth in enumerate(truth_entries):
        failure_stage = classify_truth_failure_stage(
            truth, candidates, absolute_area_tolerance, relative_area_tolerance, final_prediction
        )
        failure_counts[failure_stage] += 1
        truth_failures.append(
            {
                "truth_index": _truth_key(truth, pos),
                "macro": truth["macro"],
                "area": truth["area"],
                "face_count": truth.get("face_count"),
                "failure_stage": failure_stage,
            }
        )

    result["failure_stage_counts"] = dict(failure_counts)
    result["truth_failures"] = truth_failures

    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _first_str(d: dict, keys: tuple[str, ...], default: str) -> str:
    for key in keys:
        val = d.get(key)
        if val is not None:
            return str(val)
    return default


def _first_float(d: dict, keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        val = d.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return default


def _first_int(d: dict, keys: tuple[str, ...]) -> int | None:
    for key in keys:
        val = d.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Source coverage helpers
# ---------------------------------------------------------------------------


def proposal_source_counts(candidates: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for candidate in candidates:
        for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]):
            counts[str(source)] += 1
    return dict(counts)


def proposal_source_macro_counts(candidates: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for candidate in candidates:
        macro = candidate_macro_for_stage(candidate, "proposal") or "unknown"
        for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]):
            counts[str(source)][macro] += 1
    return {source: dict(counter) for source, counter in counts.items()}


def matched_truth_count_by_source_macro(
    truth_entries: list[dict],
    candidates: list[dict],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for truth in truth_entries:
        matched_sources: set[str] = set()
        for candidate in candidates:
            if candidate_matches_truth(
                candidate,
                truth,
                "proposal",
                absolute_area_tolerance=absolute_area_tolerance,
                relative_area_tolerance=relative_area_tolerance,
            ):
                matched_sources.update(str(source) for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]))
        for source in matched_sources:
            counts[source][str(truth.get("macro", "unknown"))] += 1
    return {source: dict(counter) for source, counter in counts.items()}


def matched_truth_count_by_source(
    truth_entries: list[dict],
    candidates: list[dict],
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
) -> dict[str, int]:
    counts: Counter = Counter()
    for truth in truth_entries:
        matched_sources: set[str] = set()
        for candidate in candidates:
            if candidate_matches_truth(
                candidate,
                truth,
                "proposal",
                absolute_area_tolerance=absolute_area_tolerance,
                relative_area_tolerance=relative_area_tolerance,
            ):
                matched_sources.update(str(source) for source in candidate.get("proposal_sources", [candidate.get("source", "unknown")]))
        for source in matched_sources:
            counts[source] += 1
    return dict(counts)


# ---------------------------------------------------------------------------
# Candidate assembly replay
# ---------------------------------------------------------------------------


def proposal_key(record: dict) -> tuple[str, tuple[int, ...]]:
    """Return (proposal_macro, sorted_face_tuple) for stable matching."""
    macro = str(record.get("proposal_macro", ""))
    faces = tuple(sorted(int(f) for f in _face_ids(record)))
    return (macro, faces)


def build_candidate_summaries(
    proposals: list[dict],
    classifier_records: list[dict],
    graph: dict | None = None,
) -> list[dict]:
    """Merge proposal records with classifier results by face set + proposal macro.

    Action mapping:
      - classified macro in REJECT_LABELS → rejected_label
      - classifier record has accepted==True or action=="accepted_remap" → accepted_remap
      - classifier record has action → reuse it
      - no classifier record → unclassified
    """
    # Index classifier records by proposal_key
    classifier_by_key: dict[tuple[str, tuple[int, ...]], dict] = {}
    for rec in classifier_records:
        key = proposal_key(rec)
        classifier_by_key[key] = rec

    candidates = []
    for proposal in proposals:
        key = proposal_key(proposal)
        cls_record = classifier_by_key.get(key)

        macro = str(proposal.get("proposal_macro", ""))
        classified_macro = macro
        action = "unclassified"
        confidence = None
        proposal_sources = list(proposal.get("proposal_sources", [proposal.get("source", "unknown")]))

        if cls_record is not None:
            predicted_label = str(cls_record.get("predicted_label", ""))
            classified_macro = predicted_label
            confidence = cls_record.get("confidence")

            if classified_macro.lower() in (r.lower() for r in REJECT_LABELS):
                action = "rejected_label"
            elif cls_record.get("accepted") is True or cls_record.get("action") == "accepted_remap":
                action = "accepted_remap"
            elif cls_record.get("action"):
                action = str(cls_record.get("action"))
            else:
                action = "classified_no_action"

        area = candidate_area(proposal, graph)
        face_ids = _face_ids(proposal)

        candidate = {
            "source": proposal.get("source", "assembled_instance_candidate"),
            "proposal_macro": macro,
            "classified_macro": classified_macro,
            "proposal_sources": proposal_sources,
            "action": action,
            "face_indices": face_ids,
            "face_count": len(face_ids),
            "area": area,
            "mean_same_instance_probability": proposal.get("mean_same_instance_probability"),
            "min_same_instance_probability": proposal.get("min_same_instance_probability"),
        }
        if confidence is not None:
            candidate["confidence"] = confidence
        candidates.append(candidate)

    # Sort by: macro name, min face id, face count
    candidates.sort(key=lambda c: (c["proposal_macro"], min(c["face_indices"]) if c["face_indices"] else 0, c["face_count"]))
    return candidates


def replay_candidates_for_sample(
    graph: dict,
    raw_prediction: dict,
    classifier_bundle: object | None,
    strong_threshold: float,
    weak_threshold: float,
    min_confidence: float,
    proposal_mode: str = "same_instance",
) -> list[dict]:
    """Replay candidate assembly and classification for one sample.

    Does NOT call topology postprocess — this is diagnostic-only.
    """
    same_instance_proposals = assemble_instance_candidates(
        graph,
        raw_prediction,
        strong_threshold=strong_threshold,
        weak_threshold=weak_threshold,
    )

    if proposal_mode == "hybrid":
        proposals = generate_hybrid_object_proposals(
            graph,
            raw_prediction,
            same_instance_proposals=same_instance_proposals,
        )
    else:
        proposals = same_instance_proposals

    if classifier_bundle is not None:
        classifier_records = classify_instance_candidate_proposals(
            graph,
            raw_prediction,
            proposals,
            classifier_bundle,
        )
        # Apply min_confidence and action logic matching apply_instance_candidate_postprocess
        occupied_faces: set[int] = set()
        for record in sorted(classifier_records, key=lambda item: float(item.get("confidence", 0)), reverse=True):
            face_set = set(record.get("face_indices", []))
            if str(record.get("predicted_label", "")) == "reject":
                record["action"] = "rejected_label"
                record["accepted"] = False
            elif float(record.get("confidence", 0)) < float(min_confidence):
                record["action"] = "skipped_low_confidence"
                record["accepted"] = False
            elif face_set & occupied_faces:
                record["action"] = "skipped_overlap"
                record["accepted"] = False
            else:
                occupied_faces.update(face_set)
                record["action"] = "accepted_remap"
                record["accepted"] = True
    else:
        classifier_records = []

    return build_candidate_summaries(proposals, classifier_records, graph)


# ---------------------------------------------------------------------------
# Dataset-level result builder
# ---------------------------------------------------------------------------


def merge_stage_summary(accumulator: dict, sample_summary: dict) -> None:
    """Merge per-sample stage summary into the accumulator."""
    for field in ("matched_truth_count", "recall"):
        # recall is recomputed later
        pass
    for macro_field in ("matched_by_macro", "truth_by_macro", "missed_by_macro"):
        if macro_field not in accumulator:
            accumulator[macro_field] = defaultdict(int)
        for macro, count in sample_summary.get(macro_field, {}).items():
            accumulator[macro_field][macro] += count


def merge_failure_counts(accumulator: Counter, sample_failures: list[dict]) -> None:
    """Merge per-sample failure counts into the accumulator."""
    for failure in sample_failures:
        accumulator[failure["failure_stage"]] += 1


def build_threshold_result(
    graph_by_sample: dict[str, dict],
    raw_prediction_by_sample: dict[str, dict],
    truth_by_sample: dict[str, list[dict]],
    final_prediction_by_sample: dict[str, dict],
    classifier_bundle: object | None,
    strong_threshold: float,
    weak_threshold: float,
    min_confidence: float,
    absolute_area_tolerance: float,
    relative_area_tolerance: float,
    proposal_mode: str = "same_instance",
) -> dict:
    """Build a full per-threshold diagnostic result across all samples."""
    all_truth_entries: list[dict] = []
    sample_results = []

    # Aggregate per-macro accumulators
    proposal_matched_by_macro: dict[str, int] = defaultdict(int)
    proposal_truth_by_macro: dict[str, int] = defaultdict(int)
    classified_matched_by_macro: dict[str, int] = defaultdict(int)
    classified_truth_by_macro: dict[str, int] = defaultdict(int)
    accepted_matched_by_macro: dict[str, int] = defaultdict(int)
    accepted_truth_by_macro: dict[str, int] = defaultdict(int)
    final_matched_by_macro: dict[str, int] = defaultdict(int)
    final_truth_by_macro: dict[str, int] = defaultdict(int)
    all_failure_counts: Counter = Counter()
    candidate_count_by_source: Counter = Counter()
    matched_count_by_source: Counter = Counter()
    candidate_count_by_source_macro: dict[str, Counter] = defaultdict(Counter)
    matched_count_by_source_macro: dict[str, Counter] = defaultdict(Counter)

    for sample_name in sorted(graph_by_sample.keys()):
        graph = graph_by_sample.get(sample_name)
        raw_prediction = raw_prediction_by_sample.get(sample_name)
        final_prediction = final_prediction_by_sample.get(sample_name)
        truth_entries = truth_by_sample.get(sample_name, [])

        if graph is None or raw_prediction is None:
            continue

        candidates = replay_candidates_for_sample(
            graph, raw_prediction, classifier_bundle,
            strong_threshold, weak_threshold, min_confidence,
            proposal_mode=proposal_mode,
        )

        stage_result = match_truth_by_stage(
            truth_entries, candidates,
            absolute_area_tolerance, relative_area_tolerance,
            final_prediction=final_prediction,
        )

        # Build candidate summaries for this sample
        cand_summaries = []
        for c in candidates:
            cand_summaries.append({
                "source": c.get("source", ""),
                "proposal_macro": c.get("proposal_macro", ""),
                "classified_macro": c.get("classified_macro", ""),
                "action": c.get("action", ""),
                "confidence": c.get("confidence"),
                "face_indices": c.get("face_indices", []),
                "face_count": c.get("face_count", 0),
                "area": c.get("area", 0.0),
                "mean_same_instance_probability": c.get("mean_same_instance_probability"),
                "min_same_instance_probability": c.get("min_same_instance_probability"),
            })

        sample_result = {
            "sample": sample_name,
            "truth_count": len(truth_entries),
            "candidate_count": len(candidates),
            "proposal_stage": stage_result.get("proposal_stage", {}),
            "classified_stage": stage_result.get("classified_stage", {}),
            "accepted_stage": stage_result.get("accepted_stage", {}),
            "final_survival_stage": stage_result.get("final_survival_stage", {}),
            "truth_failures": stage_result.get("truth_failures", []),
            "candidates": cand_summaries,
        }
        sample_results.append(sample_result)

        # Aggregate per-macro stats
        macro_accumulators = {
            "proposal": (proposal_matched_by_macro, proposal_truth_by_macro),
            "classified": (classified_matched_by_macro, classified_truth_by_macro),
            "accepted": (accepted_matched_by_macro, accepted_truth_by_macro),
            "final_survival": (final_matched_by_macro, final_truth_by_macro),
        }
        stage_sources = {
            "proposal": "proposal_stage",
            "classified": "classified_stage",
            "accepted": "accepted_stage",
            "final_survival": "final_survival_stage",
        }
        for prefix, (matched_acc, truth_acc) in macro_accumulators.items():
            stage = stage_result.get(stage_sources[prefix], {})
            for macro, count in stage.get("matched_by_macro", {}).items():
                matched_acc[macro] += count
            for macro, count in stage.get("truth_by_macro", {}).items():
                truth_acc[macro] += count

        for failure in stage_result.get("truth_failures", []):
            all_failure_counts[failure["failure_stage"]] += 1

        candidate_count_by_source.update(proposal_source_counts(candidates))
        matched_count_by_source.update(
            matched_truth_count_by_source(
                truth_entries,
                candidates,
                absolute_area_tolerance,
                relative_area_tolerance,
            )
        )

        for source, macro_counts in proposal_source_macro_counts(candidates).items():
            candidate_count_by_source_macro[source].update(macro_counts)
        for source, macro_counts in matched_truth_count_by_source_macro(
            truth_entries,
            candidates,
            absolute_area_tolerance,
            relative_area_tolerance,
        ).items():
            matched_count_by_source_macro[source].update(macro_counts)

        all_truth_entries.extend(truth_entries)

    total_truth = len(all_truth_entries)

    def _stage_result(matched_by_macro, truth_by_macro):
        matched_count = sum(matched_by_macro.values())
        missed_by_macro = {m: truth_by_macro.get(m, 0) - matched_by_macro.get(m, 0)
                           for m in truth_by_macro}
        return {
            "matched_truth_count": matched_count,
            "recall": matched_count / total_truth if total_truth > 0 else 0.0,
            "matched_by_macro": dict(matched_by_macro),
            "truth_by_macro": dict(truth_by_macro),
            "missed_by_macro": {m: c for m, c in missed_by_macro.items() if c > 0},
        }

    return {
        "strong_threshold": strong_threshold,
        "weak_threshold": weak_threshold,
        "sample_count": len(sample_results),
        "truth_count": total_truth,
        "proposal_candidate_count": sum(s["candidate_count"] for s in sample_results),
        "proposal_stage": _stage_result(proposal_matched_by_macro, proposal_truth_by_macro),
        "classified_stage": _stage_result(classified_matched_by_macro, classified_truth_by_macro),
        "accepted_stage": _stage_result(accepted_matched_by_macro, accepted_truth_by_macro),
        "final_survival_stage": _stage_result(final_matched_by_macro, final_truth_by_macro),
        "failure_stage_counts": dict(all_failure_counts),
        "proposal_candidate_count_by_source": dict(candidate_count_by_source),
        "proposal_matched_truth_count_by_source": dict(matched_count_by_source),
        "proposal_candidate_count_by_source_macro": {source: dict(counter) for source, counter in candidate_count_by_source_macro.items()},
        "proposal_matched_truth_count_by_source_macro": {source: dict(counter) for source, counter in matched_count_by_source_macro.items()},
        "samples": sample_results,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_threshold_pairs(
    strong_thresholds: list[float],
    weak_thresholds: list[float],
) -> tuple[list[dict], list[dict]]:
    """Return valid threshold pairs and skipped invalid pairs."""
    pairs: list[dict] = []
    skipped: list[dict] = []
    for strong in strong_thresholds:
        for weak in weak_thresholds:
            record = {"strong_threshold": strong, "weak_threshold": weak}
            if weak >= strong:
                skipped.append(
                    {
                        **record,
                        "reason": "weak_threshold_must_be_less_than_strong_threshold",
                    }
                )
                continue
            pairs.append(record)
    return pairs, skipped


def parse_thresholds(value: str) -> list[float]:
    """Parse comma-separated threshold values."""
    result = []
    for part in value.split(","):
        part = part.strip()
        if part:
            result.append(float(part))
    return result


def collect_json_by_stem(directory: Path) -> dict[str, dict]:
    """Collect JSON files by stem name, skipping manifest files."""
    result = {}
    for path in sorted(directory.glob("*.json")):
        stem = path.stem
        if stem.endswith("_manifest") or stem == "manifest":
            continue
        if "manifest" in stem.lower():
            continue
        result[stem] = load_json(path)
    return result


def load_truth_by_sample(path: Path) -> dict[str, list[dict]]:
    """Load and normalize area truth JSON."""
    payload = load_json(path)
    return normalize_truth_entries(payload)


def best_result(results: list[dict], stage_name: str) -> dict:
    """Return the threshold result with the best recall for a given stage."""
    best = None
    best_recall = -1.0
    for r in results:
        stage = r.get(stage_name, {})
        recall = stage.get("recall", 0.0)
        if recall > best_recall:
            best_recall = recall
            best = r
    return best if best is not None else {}


def build_report(args: argparse.Namespace) -> dict:
    """Build the full diagnostic report across all threshold combinations."""
    graph_by_sample = collect_json_by_stem(args.graph_dir)
    raw_prediction_by_sample = collect_json_by_stem(args.raw_prediction_dir)

    final_prediction_by_sample = {}
    if args.final_prediction_dir:
        final_prediction_by_sample = collect_json_by_stem(args.final_prediction_dir)

    truth_by_sample = load_truth_by_sample(args.truth_json)

    classifier_bundle = None
    if args.classifier:
        from utils.instance_candidate_postprocess import load_instance_candidate_classifier_bundle
        classifier_bundle = load_instance_candidate_classifier_bundle(args.classifier)

    strong_thresholds = parse_thresholds(args.strong_thresholds)
    weak_thresholds = parse_thresholds(args.weak_thresholds)

    threshold_pairs, skipped_threshold_pairs = build_threshold_pairs(strong_thresholds, weak_thresholds)

    threshold_results = []
    for pair in threshold_pairs:
        strong = pair["strong_threshold"]
        weak = pair["weak_threshold"]
        result = build_threshold_result(
            graph_by_sample=graph_by_sample,
            raw_prediction_by_sample=raw_prediction_by_sample,
            truth_by_sample=truth_by_sample,
            final_prediction_by_sample=final_prediction_by_sample,
            classifier_bundle=classifier_bundle,
            strong_threshold=strong,
            weak_threshold=weak,
            min_confidence=args.min_confidence,
            absolute_area_tolerance=args.absolute_area_tolerance,
            relative_area_tolerance=args.relative_area_tolerance,
            proposal_mode=args.proposal_mode,
        )
        threshold_results.append(result)

    report = {
        "schema": "cae_object_proposal_recall_report_v1",
        "graph_dir": str(args.graph_dir),
        "raw_prediction_dir": str(args.raw_prediction_dir),
        "truth_json": str(args.truth_json),
        "classifier": str(args.classifier) if args.classifier else None,
        "final_prediction_dir": str(args.final_prediction_dir) if args.final_prediction_dir else None,
        "absolute_area_tolerance": args.absolute_area_tolerance,
        "relative_area_tolerance": args.relative_area_tolerance,
        "proposal_mode": args.proposal_mode,
        "requested_threshold_pair_count": len(strong_thresholds) * len(weak_thresholds),
        "evaluated_threshold_pair_count": len(threshold_results),
        "skipped_threshold_pairs": skipped_threshold_pairs,
        "threshold_results": threshold_results,
        "best_by_proposal_recall": best_result(threshold_results, "proposal_stage"),
        "best_by_final_survival_recall": best_result(threshold_results, "final_survival_stage"),
    }
    return report


def write_markdown(path: Path, report: dict) -> None:
    """Write the human-readable markdown report."""
    lines = []
    proposal_mode = str(report.get("proposal_mode", "same_instance"))
    lines.append("# v5.13 Object Proposal Recall Diagnostic Report")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Graph dir: `{report['graph_dir']}`")
    lines.append(f"- Raw prediction dir: `{report['raw_prediction_dir']}`")
    lines.append(f"- Truth JSON: `{report['truth_json']}`")
    lines.append(f"- Classifier: `{report.get('classifier', 'N/A')}`")
    lines.append(f"- Final prediction dir: `{report.get('final_prediction_dir', 'N/A')}`")
    lines.append(f"- Proposal mode: `{proposal_mode}`")
    lines.append("")
    lines.append("## Threshold Pair Accounting")
    lines.append("")
    lines.append(f"- Requested threshold pairs: {report.get('requested_threshold_pair_count', 0)}")
    lines.append(f"- Evaluated valid threshold pairs: {report.get('evaluated_threshold_pair_count', len(report.get('threshold_results', [])))}")
    lines.append(f"- Skipped invalid threshold pairs: {len(report.get('skipped_threshold_pairs', []))}")
    lines.append("")

    lines.append("## Tolerance")
    lines.append("")
    lines.append("- Absolute area tolerance: " + str(report.get("absolute_area_tolerance", "N/A")) + " mm^2")
    lines.append(f"- Relative area tolerance: {report['relative_area_tolerance']}")
    lines.append("")
    lines.append("> Important: This validation is area-level only, not face IoU. Area truth cannot prove face-level correspondence.")
    lines.append("")

    best_proposal = report.get("best_by_proposal_recall", {})
    best_final = report.get("best_by_final_survival_recall", {})

    lines.append("## Best Thresholds")
    lines.append("")
    if best_proposal:
        bp = best_proposal
        lines.append(
            f"- Best by proposal recall: strong={bp.get('strong_threshold')}, weak={bp.get('weak_threshold')} -> proposal recall={bp.get('proposal_stage', {}).get('recall', 0.0):.4f}"
        )
    if best_final:
        bf = best_final
        lines.append(
            f"- Best by final-survival recall: strong={bf.get('strong_threshold')}, weak={bf.get('weak_threshold')} -> final-survival recall={bf.get('final_survival_stage', {}).get('recall', 0.0):.4f}"
        )
    lines.append("")

    # Threshold grid table
    lines.append("## Threshold Grid")
    lines.append("")
    header = "| strong | weak | samples | truth | proposal recall | classified recall | accepted recall | final-survival recall |"
    sep = "|--------|------|---------|-------|-----------------|-------------------|-----------------|----------------------|"
    lines.append(header)
    lines.append(sep)
    for tr in report.get("threshold_results", []):
        ps = tr.get("proposal_stage", {})
        cs = tr.get("classified_stage", {})
        as_ = tr.get("accepted_stage", {})
        fs = tr.get("final_survival_stage", {})
        lines.append(
            f"| {tr['strong_threshold']} | {tr['weak_threshold']} | {tr['sample_count']} | {tr['truth_count']} | "
            f"{ps.get('recall', 0):.4f} | {cs.get('recall', 0):.4f} | {as_.get('recall', 0):.4f} | {fs.get('recall', 0):.4f} |"
        )
    lines.append("")

    # Macro coverage for best proposal-recall threshold
    if best_proposal:
        lines.append("## Macro-Level Coverage (Best Proposal-Recall Threshold)")
        lines.append("")
        bp_ps = best_proposal.get("proposal_stage", {})
        lines.append("### Proposal Stage")
        lines.append("")
        lines.append("| Macro | Truth Count | Matched | Missed |")
        lines.append("|-------|-------------|---------|--------|")
        truth_by = bp_ps.get("truth_by_macro", {})
        matched_by = bp_ps.get("matched_by_macro", {})
        for macro in sorted(truth_by.keys()):
            t = truth_by[macro]
            m = matched_by.get(macro, 0)
            missed = t - m
            lines.append(f"| {macro} | {t} | {m} | {missed} |")
        lines.append("")

        bp_fs = best_proposal.get("final_survival_stage", {})
        lines.append("### Final-Survival Stage")
        lines.append("")
        lines.append("| Macro | Truth Count | Matched | Missed |")
        lines.append("|-------|-------------|---------|--------|")
        fs_truth_by = bp_fs.get("truth_by_macro", {})
        fs_matched_by = bp_fs.get("matched_by_macro", {})
        for macro in sorted(fs_truth_by.keys()):
            t = fs_truth_by[macro]
            m = fs_matched_by.get(macro, 0)
            missed = t - m
            lines.append(f"| {macro} | {t} | {m} | {missed} |")
        lines.append("")

    # Failure stage counts
    if best_proposal:
        lines.append("## Failure-Stage Counts (Best Proposal-Recall Threshold)")
        lines.append("")
        fsc = best_proposal.get("failure_stage_counts", {})
        lines.append("| Failure Stage | Count |")
        lines.append("|---------------|-------|")
        for stage in FAILURE_STAGES:
            count = fsc.get(stage, 0)
            lines.append(f"| {stage} | {count} |")
        lines.append("")

    # Proposal Source Coverage
    if best_proposal and best_proposal.get("proposal_candidate_count_by_source") is not None:
        lines.append("## Proposal Source Coverage")
        lines.append("")
        lines.append("| Source | Candidate Count | Matched Truth Count |")
        lines.append("|--------|-----------------|---------------------|")
        source_candidates = best_proposal.get("proposal_candidate_count_by_source", {})
        source_matches = best_proposal.get("proposal_matched_truth_count_by_source", {})
        for source in sorted(set(source_candidates) | set(source_matches)):
            lines.append(f"| {source} | {source_candidates.get(source, 0)} | {source_matches.get(source, 0)} |")
        lines.append("")
        lines.append("Source matched counts are non-exclusive because one deduplicated candidate can have multiple proposal sources.")
        lines.append("")

    source_macro_candidates = best_proposal.get("proposal_candidate_count_by_source_macro", {})
    source_macro_matches = best_proposal.get("proposal_matched_truth_count_by_source_macro", {})
    if source_macro_candidates or source_macro_matches:
        lines.append("### Proposal Source Coverage by Macro")
        lines.append("")
        lines.append("| Source | Macro | Candidate Count | Matched Truth Count |")
        lines.append("|--------|-------|-----------------|---------------------|")
        all_sources = sorted(set(source_macro_candidates) | set(source_macro_matches))
        for source in all_sources:
            macros = sorted(set(source_macro_candidates.get(source, {})) | set(source_macro_matches.get(source, {})))
            for macro in macros:
                lines.append(
                    f"| {source} | {macro} | {source_macro_candidates.get(source, {}).get(macro, 0)} | "
                    f"{source_macro_matches.get(source, {}).get(macro, 0)} |"
                )
        lines.append("")

    # Top missed truth entries
    if best_proposal:
        lines.append("## Top Missed Truth Entries (Best Proposal-Recall Threshold)")
        lines.append("")
        lines.append("| Sample | Macro | Area | Face Count | Failure Stage |")
        lines.append("|--------|-------|------|------------|---------------|")
        missed_entries = []
        for sample in best_proposal.get("samples", []):
            for failure in sample.get("truth_failures", []):
                if failure["failure_stage"] != "survives_final_prediction":
                    missed_entries.append((sample["sample"], failure))
        for sample_name, failure in sorted(missed_entries, key=lambda x: (x[1]["failure_stage"], x[1]["macro"]))[:50]:
            lines.append(
                f"| {sample_name} | {failure['macro']} | {failure['area']:.2f} | {failure.get('face_count', '-')} | "
                f"{failure['failure_stage']} |"
            )
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_decision_memo(path: Path, report: dict) -> None:
    """Write the decision memo recommending exactly one next action."""
    best_proposal = report.get("best_by_proposal_recall", {})
    failure_counts = best_proposal.get("failure_stage_counts", {})

    no_proposal = failure_counts.get("no_area_compatible_proposal", 0)
    classifier_wrong = failure_counts.get("proposal_exists_classifier_wrong_or_rejected", 0)
    final_changes = failure_counts.get("classifier_accepts_but_final_changes_or_removes", 0)
    survives = failure_counts.get("survives_final_prediction", 0)
    total_non_survival = no_proposal + classifier_wrong + final_changes

    recommendation = "defer_v5_10_and_keep_v5_9_default"
    reasoning = ""

    if total_non_survival > 0:
        dominant_count = max(no_proposal, classifier_wrong, final_changes)
        dominant_ratio = dominant_count / total_non_survival if total_non_survival > 0 else 0

        if no_proposal == dominant_count and no_proposal > 0:
            if dominant_ratio >= 0.4:
                recommendation = "proposal_generation_fix"
                reasoning = f"no_area_compatible_proposal ({no_proposal}) dominates non-survival failures "
                reasoning += f"({dominant_ratio:.1%} of {total_non_survival})"
            else:
                recommendation = "defer_v5_10_and_keep_v5_9_default"
                reasoning = f"no single failure dominates (max ratio {dominant_ratio:.1%}), "
                reasoning += f"and no_area_compatible_proposal={no_proposal}"

        elif classifier_wrong == dominant_count and classifier_wrong > 0:
            if dominant_ratio >= 0.4:
                recommendation = "classifier_recalibration_or_retraining"
                reasoning = f"proposal_exists_classifier_wrong_or_rejected ({classifier_wrong}) dominates "
                reasoning += f"non-survival failures ({dominant_ratio:.1%} of {total_non_survival})"
            else:
                recommendation = "defer_v5_10_and_keep_v5_9_default"
                reasoning = f"no single failure dominates (max ratio {dominant_ratio:.1%}), "
                reasoning += f"and classifier_wrong={classifier_wrong}"

        elif final_changes == dominant_count and final_changes > 0:
            if dominant_ratio >= 0.4:
                recommendation = "topology_export_interaction_fix"
                reasoning = f"classifier_accepts_but_final_changes_or_removes ({final_changes}) dominates "
                reasoning += f"non-survival failures ({dominant_ratio:.1%} of {total_non_survival})"
            else:
                recommendation = "defer_v5_10_and_keep_v5_9_default"
                reasoning = f"no single failure dominates (max ratio {dominant_ratio:.1%}), "
                reasoning += f"and final_changes={final_changes}"

    # Check if final-survival recall is below v5.9-C recall
    fs_recall = best_proposal.get("final_survival_stage", {}).get("recall", 0.0)
    if recommendation == "defer_v5_10_and_keep_v5_9_default":
        reasoning = f"final-survival recall ({fs_recall:.4f}) below v5.9-C recall (0.6716) "
        reasoning += "and no failure stage dominates. Keep v5.9-C as current best."

    lines = []
    lines.append("# v5.11.1 Proposal Recall Decision Memo (Macrofix)")
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(f"**{recommendation}**")
    lines.append("")
    lines.append("## Reasoning")
    lines.append("")
    lines.append(reasoning)
    lines.append("")
    lines.append("## Supporting Data")
    lines.append("")
    lines.append(f"- Final-survival recall: {fs_recall:.4f}")
    lines.append(f"- v5.9-C recall: 0.6716")
    lines.append(f"- Survives final: {survives}")
    lines.append(f"- No compatible proposal: {no_proposal}")
    lines.append(f"- Classifier wrong/rejected: {classifier_wrong}")
    lines.append(f"- Accepted but final removes: {final_changes}")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    lines.append(f"- Graph dir: `{report['graph_dir']}`")
    lines.append(f"- Truth JSON: `{report['truth_json']}`")
    lines.append(f"- Classifier: `{report.get('classifier', 'N/A')}`")
    lines.append(f"- Final prediction dir: `{report.get('final_prediction_dir', 'N/A')}`")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="v5.11 Object Proposal Recall Diagnostics")
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--raw-prediction-dir", type=Path, required=True)
    parser.add_argument("--truth-json", type=Path, required=True)
    parser.add_argument("--classifier", type=Path, default=None)
    parser.add_argument("--final-prediction-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--strong-thresholds", type=str, default="0.55")
    parser.add_argument("--weak-thresholds", type=str, default="0.35")
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--proposal-mode", choices=("same_instance", "hybrid"), default="same_instance")
    parser.add_argument("--absolute-area-tolerance", type=float, default=10.0)
    parser.add_argument("--relative-area-tolerance", type=float, default=0.02)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    report = build_report(args)

    write_json(args.output_json, report)
    write_markdown(args.output_md, report)

    decision_path = args.output_md.parent / "v5_11_1_macrofix_decision.md"
    write_decision_memo(decision_path, report)

    print(f"Report written to {args.output_json}")
    print(f"Markdown written to {args.output_md}")
    print(f"Decision memo written to {decision_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
