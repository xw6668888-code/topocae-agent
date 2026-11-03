"""Pure-data E3 relocation checks before any model/geometry import.

The actual protected checkpoint dictionaries must later be extracted with an
approved isolated PyTorch interpreter and compared here. Synthetic values do
not establish that those protected binaries were inspected or can run.
"""
from __future__ import annotations

from pathlib import Path

from g10_common import digest, read_json, sha256
from m8_portable_rebase_v4 import ClosureError, RUNTIME_BINARY, RUNTIME_JSON, _path


def _anchors(plan: dict) -> dict:
    anchors = plan.get("binary_bound_original_identity", {})
    required = ("G03_configuration_freeze_sha256", "E1_protocol_json_digest",
                "E1_train_seal_json_digest", "E1_scorer_checkpoint_sha256")
    if any(not isinstance(anchors.get(field), str) or len(anchors[field]) != 64
           for field in required):
        raise ClosureError("original checkpoint attestation anchors incomplete")
    return anchors


def verify_q_binary_bound(q_saved: dict, plan: dict) -> dict:
    anchors = _anchors(plan)
    if not isinstance(q_saved, dict) or q_saved.get("configuration_freeze_sha256") != \
            anchors["G03_configuration_freeze_sha256"]:
        raise ClosureError("Q embedded original configuration freeze identity changed")
    if not isinstance(q_saved.get("model_state_dict"), dict):
        raise ClosureError("embedded Q model state differs")
    return {"original_freeze_sha256": anchors["G03_configuration_freeze_sha256"],
            "model_state_present": True}


def verify_scorer_binary_bound(scorer_saved: dict, plan: dict) -> dict:
    anchors = _anchors(plan)
    if not isinstance(scorer_saved, dict) or not isinstance(scorer_saved.get("protocol"), dict) or \
            not isinstance(scorer_saved.get("train_seal"), dict) or \
            digest(scorer_saved["protocol"]) != anchors["E1_protocol_json_digest"] or \
            digest(scorer_saved["train_seal"]) != anchors["E1_train_seal_json_digest"]:
        raise ClosureError("E1 embedded original protocol or D0 train seal identity changed")
    features = plan["templates"]["scorer_protocol"]["feature_names"]
    if scorer_saved.get("feature_names") != features or \
            not isinstance(scorer_saved.get("model_state_dict"), dict) or \
            not isinstance(scorer_saved.get("feature_mean"), list) or \
            not isinstance(scorer_saved.get("feature_std"), list) or \
            len(scorer_saved["feature_mean"]) != len(features) or \
            len(scorer_saved["feature_std"]) != len(features):
        raise ClosureError("embedded E1/Q model or feature dimensionality differs")
    return {"scorer_checkpoint_sha256": anchors["E1_scorer_checkpoint_sha256"],
            "features": len(features), "model_state_present": True}


def verify_binary_bound(q_saved: dict, scorer_saved: dict, plan: dict) -> dict:
    q = verify_q_binary_bound(q_saved, plan)
    scorer = verify_scorer_binary_bound(scorer_saved, plan)
    return {"status": "EMBEDDED_ORIGINAL_IDENTITIES_MATCH_SUPPLIED_DICTS_NOT_MODEL_REPLAY",
            "original_freeze_sha256": q["original_freeze_sha256"],
            "scorer_checkpoint_sha256": scorer["scorer_checkpoint_sha256"],
            "features": scorer["features"], "actual_checkpoint_load_verified": False,
            "reproducibility_pass": False}


def verify_materialized_call_sites(bundle: Path, plan: dict, mapping: dict) -> dict:
    """Assert the descriptor comparisons used by protected G03/G09 loaders.

    This checks their JSON preconditions, not their checkpoint or OCC execution.
    """
    root = bundle.resolve()
    if mapping.get("schema") != "g10_m8_relocated_e3_runtime_sha_mapping_v4" or \
            mapping.get("source_only_plan_digest") != digest(plan) or \
            mapping.get("status") != "STRUCTURED_RUNTIME_JSON_REBASED_BINARY_BOUND_NOT_REPLAYED":
        raise ClosureError("relocated manifest differs from exact source-only plan")
    rows = mapping.get("sha_mapping")
    if not isinstance(rows, list) or len(rows) != len(RUNTIME_BINARY) + len(RUNTIME_JSON):
        raise ClosureError("relocation SHA mapping incomplete")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"relative_path", "original_sha256",
                                                    "relocated_sha256", "relocated_bytes"}:
            raise ClosureError("relocation SHA mapping malformed")
        name = row["relative_path"]
        if name in indexed:
            raise ClosureError("duplicate relocated asset")
        indexed[name] = row
        path = _path(root, name)
        if not path.is_file() or path.stat().st_size != row["relocated_bytes"] or \
                sha256(path) != row["relocated_sha256"]:
            raise ClosureError("materialized asset hash changed")
    originals = plan["original_assets"]
    if set(indexed) != {originals[key]["relative_path"] for key in RUNTIME_BINARY + RUNTIME_JSON}:
        raise ClosureError("unlisted or missing relocated runtime asset")
    for key in RUNTIME_BINARY + RUNTIME_JSON:
        if indexed[originals[key]["relative_path"]]["original_sha256"] != originals[key]["sha256"]:
            raise ClosureError("relocated asset original hash mapping differs")
    read = lambda key: read_json(_path(root, originals[key]["relative_path"]))
    desc = lambda key: {"path": _path(root, originals[key]["relative_path"]).resolve().as_posix(),
                        "bytes": indexed[originals[key]["relative_path"]]["relocated_bytes"],
                        "sha256": indexed[originals[key]["relative_path"]]["relocated_sha256"]}
    freeze, q, scorer, protocol = (read(key) for key in
                                    ("freeze", "q_metrics", "scorer_complete", "e3_protocol"))
    if q["checkpoint"] != desc("q_checkpoint") or \
            read("q_status")["status"] != "completed" or \
            freeze["frozen_files"] != [desc("metadata")] or \
            freeze["baseline_runs"][0]["files"] != [desc("r_checkpoint")]:
        raise ClosureError("G03 R/Q frozen JSON loader comparisons differ")
    if scorer["checkpoint"] != desc("scorer_checkpoint") or \
            scorer["protocol"] != desc("scorer_protocol") or \
            scorer["train_seal"] != desc("train_seal") or \
            read("scorer_protocol")["train_seal"] != desc("train_seal"):
        raise ClosureError("G09 E1 scorer JSON descriptors differ")
    if protocol["prior"] != desc("prior") or protocol["E1_scorer"] != desc("scorer_complete") \
            or protocol["fixed_rule"]["T"] != 3:
        raise ClosureError("G09 E3 route JSON comparisons differ")
    return {"status": "PROTECTED_JSON_CALL_SITE_PRECONDITIONS_MATCH_NOT_INFERENCE",
            "G03_R_Q_descriptor_checks": True, "G09_E1_E3_descriptor_checks": True,
            "Q_original_freeze_binary_check_pending": True,
            "E1_original_protocol_and_train_seal_binary_check_pending": True,
            "isolated_model_or_OCC_imports_verified": False,
            "actual_inferences": 0, "reproducibility_pass": False}
