"""G10-only, target-free E3 runtime descriptor closure and relocation templates.

This does not modify protected G03/G09 files or inspect training labels. A
source-only audit cannot authorize copying model weights or licensed software.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from g10_common import ARTIFACT, REPORT, ROOT, digest, read_json, relative_public_path, sha256, write_json


class ClosureError(ValueError):
    pass


@dataclass(frozen=True)
class Source:
    relative_path: str
    sha256: str
    role: str


# These are protected source bytes, not hashes of the portable projections.
# A change to any original requires a successor and a fresh provenance audit.
SOURCES = {
    "freeze": Source("reports/aei_paper_goals_v1/g03/CONFIGURATION_FREEZE.json", "8e4bdc1338ab2c375ae74a1930187d6913d09454b4ce81e8138a731a18bd3922", "binary_bound_json"),
    "q_metrics": Source("artifacts/aei_paper_goals_v1/g03/training_runs/dense_instance/seed_20260719/metrics.json", "8718246756eea5140a74b48a5da18c163185af35505d1d85518d1d2fe3ad30f2", "runtime_json"),
    "q_status": Source("artifacts/aei_paper_goals_v1/g03/training_runs/dense_instance/seed_20260719/STATUS.json", "410da6f9a6567531e7def08ab548179d0ff1b1ac62050aae17025ba1ab7c78a6", "runtime_json"),
    "q_config": Source("artifacts/aei_paper_goals_v1/g03/training_runs/dense_instance/seed_20260719/config.json", "6d62fa741f9444c006183311a4a97f68d77b53d94fdff7ef20bde025a0ebe1a1", "provenance_only"),
    "e3_protocol": Source("reports/aei_paper_goals_v1/g09/E3_BAYESIAN_CALIBRATION_PROTOCOL.json", "0e280a6444da83508fdbc8165ee168e08b2fd903b363b9b8bb4b1f3c627f0b76", "runtime_json"),
    "prior": Source("artifacts/aei_paper_goals_v1/g09/e_joint_source_priors/D0_train_seed_20260719.json", "76fe71d59527ec26738f0090ffe8ddf8116e5eda19d923b1939adaee4463cd6c", "runtime_json"),
    "scorer_complete": Source("artifacts/aei_paper_goals_v1/g09/e_joint_scorer/seed_20260719/COMPLETE.json", "e5ad625bd45eb5a066222b177125e8adb34818dd65d425e19d1a20eae2532808", "runtime_json"),
    "scorer_protocol": Source("reports/aei_paper_goals_v1/g09/E_JOINT_SCORER_PROTOCOL.json", "f4cb0a7dc552ee6a5ad795e561746d08426083adc06600458d5c978490274fef", "binary_bound_json"),
    "train_seal": Source("artifacts/aei_paper_goals_v1/g09/e_joint_train/full/COMPLETE.json", "580f91d335bf60b267dcbbf295c594d3270bdbf50f72a78825dad13d0987375b", "binary_bound_json"),
    "e2_protocol": Source("reports/aei_paper_goals_v1/g09/E2_RETENTION_PROTOCOL.json", "d9b4e49b9203243de074f391100fc30e260ae4d5af84d47c1b67e9673d5b0715", "provenance_only"),
    "metadata": Source("paper_release/topocae_p1_legacy_baseline_v1/output_training_p1_e6r2_r3_graph_v2_2_1_v1/metadata.json", "a450467fcefad201130d35bba2fe907ca2ab49a6d504bf85a7d794b5724597ba", "model_metadata"),
    "r_checkpoint": Source("paper_release/topocae_p1_legacy_baseline_v1/reports/p1_e6r2_r3_m2_m5_controlled_training_v1/training_runs/final_8370_seed_20260719/best.pt", "491655bfec5591c5a6a1b7f6300294f4818c29bf60b4c1b8dd5ff6078a057054", "model_checkpoint"),
    "r_threshold": Source("paper_release/topocae_p1_legacy_baseline_v1/reports/p1_e6r2_r3_m2_m5_controlled_training_v1/training_runs/final_8370_seed_20260719/same_instance_threshold_selection.json", "9a94f60111ff77fd8c80885188e6886e476ad82140f011a8e44af1761403e02d", "decoder_parameter"),
    "q_checkpoint": Source("artifacts/aei_paper_goals_v1/g03/training_runs/dense_instance/seed_20260719/best.pt", "f6ca9bc84d88bc540134b598353b2a2a0f9b83e1b2779c986b2ecc6577bfb70e", "model_checkpoint"),
    "q_threshold": Source("artifacts/aei_paper_goals_v1/g03/training_runs/dense_instance/seed_20260719/same_instance_threshold_selection.json", "96ee358d24dcb1946531c123281826ad53f0e0580d7ef0c17bc729dc40e8d3c8", "decoder_parameter"),
    "candidate_threshold": Source("paper_release/topocae_p1_legacy_baseline_v1/reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718/candidate_model/candidate_threshold_selection.json", "448659d94ac868703ca6dfb74ff42095bb2205eac8cc92dc16d5b2f2d4208c8d", "decoder_parameter"),
    "global_decode": Source("paper_release/topocae_p1_legacy_baseline_v1/reports/p1_e3c_seed_matched_candidate_v1/pipelines/old_same_instance_seed_20260718/global_decode_selection.json", "38d27f363f87df929c5648f27f55b6fb83a1cad7fbeb1458b960339ada844879", "decoder_parameter"),
    "scorer_checkpoint": Source("artifacts/aei_paper_goals_v1/g09/e_joint_scorer/seed_20260719/scorer.pt", "28668becd3ce4adf6f6f666f1ec1b35192c38da15df45fecc99df6a47061fd05", "model_checkpoint"),
}
RUNTIME_JSON = ("freeze", "q_metrics", "q_status", "prior", "train_seal",
                "scorer_protocol", "scorer_complete", "e3_protocol")
RUNTIME_BINARY = ("metadata", "r_checkpoint", "r_threshold", "q_checkpoint",
                  "q_threshold", "candidate_threshold", "global_decode", "scorer_checkpoint")
IMPORT_ROOTS = (
    "tools/aei_paper_goals_v1/g10/m8_relocated_infer_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_child_entry_v4.py",
    "tools/aei_paper_goals_v1/g10/m6_runner.py",
    "tools/aei_paper_goals_v1/g09/infer_step_e3.py",
    "tools/aei_paper_goals_v1/g07/topo_predictor.py",
)
IMPORT_DECLARED_SEED = set(IMPORT_ROOTS) | {
    "tools/aei_paper_goals_v1/g03/inference.py",
    "tools/aei_paper_goals_v1/g03/common.py",
    "tools/aei_paper_goals_v1/g09/g09_common.py",
    "tools/aei_paper_goals_v1/g09/e_joint_scorer.py",
    "tools/aei_paper_goals_v1/g09/e_v09_adapter.py",
    "tools/aei_paper_goals_v1/g10/g10_common.py",
    "tools/aei_paper_goals_v1/g10/m8_e3_runtime_adapter_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_embedded_identity_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_portable_rebase_v4.py",
}
SCOPE_POLICY = {
    "receipt": Source("reports/aei_paper_goals_v1/g10/literature-search-20260915-aei-cad-cae-source-policy-v3/SCREENING_RECEIPT.json", "890318a4b7ba601639ed3f9969c8b8b4631a181d97ca9c848e586ac494b29eb6", "citable_policy_receipt"),
    "matrix": Source("reports/aei_paper_goals_v1/g10/literature-search-20260915-aei-cad-cae-source-policy-v3/NOVELTY_AND_SCOPE_MATRIX.md", "ba9495fa21f445bd1845cac1cae2c648e4b5d93635b53ca5a07ce145147d9189", "primary_citable_matrix"),
    "papers": Source("reports/aei_paper_goals_v1/g10/literature-search-20260915-aei-cad-cae-source-policy-v3/papers.md", "e5910ed91e13fadc944bcc387dda007ad023196757589e678d0111d8bac9814a", "primary_citable_closest_work"),
    "historical_matrix": Source("reports/aei_paper_goals_v1/g10/NOVELTY_AND_SCOPE_MATRIX.md", "a895cab5526294fda3246077822620b0b63cccc404745bc0450bf15c1762903a", "historical_screening_only_not_primary_citation"),
}
AUDIT_FILES = {
    "protected_E3_closure": ARTIFACT / "m8_portability/v4/source_only_audits/protected-e3-runtime-20260915/CLOSURE.json",
    "future_entry_imports": ARTIFACT / "m8_portability/v4/source_only_audits/protected-e3-runtime-imports-v2-20260915/IMPORTS.json",
    "publication_scope": ARTIFACT / "m8_portability/v4/source_only_audits/publication-source-policy-20260915/PUBLICATION_SCOPE.json",
}
V4_SOURCES = (
    "tools/aei_paper_goals_v1/g10/m8_portable_rebase_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_embedded_identity_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_e3_runtime_adapter_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_relocated_infer_v4.py",
    "tools/aei_paper_goals_v1/g10/m8_child_entry_v4.py",
    "tools/aei_paper_goals_v1/g10/M8_PORTABLE_REPLAY_V4.md",
    "tests/aei_paper_goals_v1/test_g10_m8_rebase_v4.py",
)
ABSOLUTE = re.compile(r"(?:[A-Za-z]:[/\\]|/Users/|/home/|\\\\[^\\\s]+\\)")


def _path(root: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path or "\\" in relative_path or \
            relative_path.startswith("/") or any(part in ("", ".", "..") for part in relative_path.split("/")):
        raise ClosureError("unsafe POSIX asset path")
    path = root / relative_path
    if not path.resolve().is_relative_to(root.resolve()):
        raise ClosureError("asset escapes relocation root")
    return path


def _originals(root: Path, sources: dict[str, Source]) -> tuple[dict, dict]:
    assets, values = {}, {}
    for key, source in sources.items():
        path = _path(root, source.relative_path)
        if not path.is_file() or sha256(path) != source.sha256:
            raise ClosureError("protected source missing or changed: " + key)
        assets[key] = {"relative_path": source.relative_path, "bytes": path.stat().st_size,
                       "sha256": source.sha256, "role": source.role}
        if source.role.endswith("json") or source.role == "provenance_only":
            values[key] = read_json(path)
    return assets, values


def _edge(source_root: Path, original: Any, target: dict, *, context: str) -> dict:
    if not isinstance(original, dict) or set(original) != {"path", "bytes", "sha256"}:
        raise ClosureError("invalid original descriptor at " + context)
    path = _path(source_root, target["relative_path"])
    if original != {"path": path.resolve().as_posix(), "bytes": target["bytes"],
                    "sha256": target["sha256"]}:
        raise ClosureError("protected runtime edge changed: " + context)
    return {key: target[key] for key in ("relative_path", "bytes", "sha256")}


def _no_descriptors(value: Any) -> None:
    if isinstance(value, dict):
        if {"path", "bytes", "sha256"} <= set(value):
            raise ClosureError("unmapped absolute descriptor in runtime projection")
        for child in value.values():
            _no_descriptors(child)
    elif isinstance(value, list):
        for child in value:
            _no_descriptors(child)


def _fields(value: dict, *names: str) -> dict:
    if not isinstance(value, dict) or any(name not in value for name in names):
        raise ClosureError("missing protected projection field")
    return {name: deepcopy(value[name]) for name in names}


def _projections(root: Path, originals: dict, values: dict) -> dict:
    edge = lambda desc, name, context: _edge(root, desc, originals[name], context=context)
    freeze = values["freeze"]
    if freeze.get("seeds") is None or 20260719 not in freeze["seeds"] or \
            not {"full", "dense_instance"} <= set(freeze["configs"]):
        raise ClosureError("protected G03 route or seed differs")
    metadata = [item for item in freeze["frozen_files"] if item.get("path") ==
                _path(root, originals["metadata"]["relative_path"]).resolve().as_posix()]
    r_runs = [item for item in freeze["baseline_runs"] if item.get("seed") == 20260719]
    if len(metadata) != 1 or len(r_runs) != 1:
        raise ClosureError("G03 frozen metadata or R seed is ambiguous")
    r_binary = [item for item in r_runs[0]["files"] if Path(item["path"]).name == "best.pt"]
    if len(r_binary) != 1:
        raise ClosureError("R frozen checkpoint is ambiguous")
    freeze_view = {"configs": _fields(freeze["configs"], "full", "dense_instance"),
                   "seeds": [20260719],
                   "frozen_files": [edge(metadata[0], "metadata", "freeze.metadata")],
                   "baseline_runs": [{"seed": 20260719,
                                      "files": [edge(r_binary[0], "r_checkpoint", "freeze.R_best")]}]}
    q = values["q_metrics"]
    if q.get("config") != "dense_instance" or q.get("seed") != 20260719 or \
            values["q_status"].get("status") != "completed":
        raise ClosureError("Q frozen run differs")
    q_view = _fields(q, "config", "seed", "best_epoch", "completed_epochs", "new_test_read_count")
    q_view["checkpoint"] = edge(q["checkpoint"], "q_checkpoint", "q.metrics.checkpoint")
    edge(q.get("config_file"), "q_config", "Q.historical_training_config")
    scorer = values["scorer_complete"]
    if scorer.get("schema") != "g09_e_joint_scorer_v1" or scorer.get("parts") != 7440:
        raise ClosureError("E1 scorer schema or train denominator differs")
    scorer_view = _fields(scorer, "schema", "parts", "candidates", "positive", "negative")
    for field, target in (("checkpoint", "scorer_checkpoint"),
                          ("protocol", "scorer_protocol"), ("train_seal", "train_seal")):
        scorer_view[field] = edge(scorer[field], target, "E1." + field)
    scorer_protocol = values["scorer_protocol"]
    scorer_protocol_view = _fields(scorer_protocol, "schema", "feature_names", "fixed_rule")
    scorer_protocol_view["train_seal"] = edge(scorer_protocol["train_seal"], "train_seal", "E1.protocol.train_seal")
    seal = values["train_seal"]
    if seal.get("schema") != "g09_e_joint_train_data_v1" or seal["stats"].get("parts") != 7440:
        raise ClosureError("E1 protected train seal differs")
    for name in ("candidates.jsonl", "labels.jsonl"):
        if not any(Path(item["path"]).name == name for item in seal["outputs"]):
            raise ClosureError("protected train sidecar provenance missing")
    edge(seal["identity"]["Q_config"], "q_config", "E1.train.Q_config")
    seal_view = {"schema": seal["schema"], "stats": _fields(seal["stats"], "parts", "candidates", "positive"),
                 "identity": _fields(seal["identity"], "schema", "dataset", "split", "seed",
                                     "sample_count", "feature_names")}
    prior = values["prior"]
    if prior.get("schema") != "g09_e_joint_train_source_priors_v1" or \
            prior.get("no_validation_or_engineering_labels") is not True:
        raise ClosureError("E3 train-only prior differs")
    edge(prior["train_seal"], "train_seal", "E3.prior.train_seal")
    prior_view = _fields(prior, "schema", "group_rule", "overall", "by_source", "by_class",
                         "no_validation_or_engineering_labels")
    protocol = values["e3_protocol"]
    if protocol.get("schema") != "g09_e3_bayesian_calibration_protocol_v1" or \
            protocol["fixed_rule"].get("T") != 3 or protocol.get("final_test_exposed") is not False:
        raise ClosureError("frozen E3 calibration differs")
    edge(protocol["train_q25_source"], "e2_protocol", "E3.historical_Q25")
    e3_view = _fields(protocol, "schema", "fixed_rule", "final_test_exposed")
    e3_view["prior"] = edge(protocol["prior"], "prior", "E3.prior")
    e3_view["E1_scorer"] = edge(protocol["E1_scorer"], "scorer_complete", "E3.E1_scorer")
    projections = {"freeze": freeze_view, "q_metrics": q_view,
                   "q_status": _fields(values["q_status"], "status"), "prior": prior_view,
                   "train_seal": seal_view, "scorer_protocol": scorer_protocol_view,
                   "scorer_complete": scorer_view, "e3_protocol": e3_view}
    for key, value in projections.items():
        _no_descriptors(value)
        if ABSOLUTE.search(json.dumps(value, ensure_ascii=False)):
            raise ClosureError("machine path remains in " + key)
    return projections


def source_only_plan(source_root: Path = ROOT, sources: dict[str, Source] = SOURCES) -> dict:
    """Audit protected source identities and runtime edges, without copying assets."""
    root = source_root.resolve()
    assets, values = _originals(root, sources)
    projections = _projections(root, assets, values)
    sidecars = values["train_seal"]["outputs"]
    provenance = [{"name": Path(item["path"]).name, "bytes": item["bytes"],
                   "sha256": item["sha256"], "copied": False, "contents_opened": False}
                  for item in sidecars]
    return {"schema": "g10_m8_protected_e3_runtime_closure_v4",
            "status": "SOURCE_ONLY_HASHED_RUNTIME_EDGE_AUDIT_NOT_RELEASE_OR_REPLAY",
            "original_assets": assets, "templates": projections,
            "binary_bound_original_identity": {
                "G03_configuration_freeze_sha256": assets["freeze"]["sha256"],
                "E1_protocol_json_digest": digest(values["scorer_protocol"]),
                "E1_train_seal_json_digest": digest(values["train_seal"]),
                "E1_scorer_checkpoint_sha256": assets["scorer_checkpoint"]["sha256"]},
            "training_sidecars_provenance_only": provenance,
            "historical_Q25_protocol_sha256": assets["e2_protocol"]["sha256"],
            "historical_Q_training_config_sha256": assets["q_config"]["sha256"],
            "runtime_binary_keys": list(RUNTIME_BINARY), "runtime_json_keys": list(RUNTIME_JSON),
            "protected_source_files_modified": False, "T10_truth_opened": False,
            "training_labels_opened": False, "actual_inferences": 0,
            "actual_CAD_edits": 0, "actual_CAE_solves": 0,
            "license_and_full_import_closure_pass": False,
            "binary_embedded_identity_verified": False, "reproducibility_pass": False}


def publication_scope(source_root: Path = ROOT,
                      sources: dict[str, Source] = SCOPE_POLICY) -> dict:
    root = source_root.resolve()
    assets, _ = _originals(root, sources)
    receipt = read_json(_path(root, assets["receipt"]["relative_path"]))
    if receipt.get("schema") != "g10_literature_citable_screen_v3" or \
            receipt.get("status") != "source_quality_exclusion_corrected_no_new_metrics" or \
            receipt.get("excluded_count_internal") != 1 or \
            receipt.get("new_T10_or_H10_accessed") is not False:
        raise ClosureError("citable v3 literature policy receipt differs")
    for field, key in (("citable_scope", "matrix"), ("citable_closest_work", "papers"),
                       ("historical_scope_retained", "historical_matrix")):
        _edge(root, receipt[field], assets[key], context="literature." + field)
    citable_text = "\n".join(_path(root, assets[key]["relative_path"]).read_text(
        encoding="utf-8-sig") for key in ("matrix", "papers"))
    historical = _path(root, assets["historical_matrix"]["relative_path"]).read_text(
        encoding="utf-8-sig")
    if "mdpi.com" in citable_text.lower() or "mdpi.com" not in historical.lower():
        raise ClosureError("citable/historical source-policy boundary differs")
    return {"schema": "g10_m8_publication_scope_selection_v4",
            "status": "CITABLE_SOURCE_POLICY_V3_SELECTED_NO_PERFORMANCE_CLAIM",
            "primary_citable_assets": [assets[key] for key in ("receipt", "matrix", "papers")],
            "historical_screening_only": assets["historical_matrix"],
            "historical_matrix_may_be_retained_internally": True,
            "historical_matrix_must_not_be_primary_citation": True,
            "old_evidence_readiness_snapshot_not_current_M6_evidence": True,
            "current_M6_13_of_13_truth_claim_source": "separate_current_G10_receipts_required",
            "T10_or_H10_accessed": False, "method_or_test_performance_claim": False,
            "reproducibility_pass": False}


def _relative_asset(path: Path) -> dict:
    resolved = path.resolve(strict=True)
    return {"relative_path": relative_public_path(resolved), "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved)}


def register_source_only_preflight() -> dict:
    expected = {"protected_E3_closure": source_only_plan(),
                "future_entry_imports": local_import_inventory(
                    ROOT, list(IMPORT_ROOTS), IMPORT_DECLARED_SEED),
                "publication_scope": publication_scope()}
    audits = {}
    for name, path in AUDIT_FILES.items():
        if not path.is_file() or read_json(path) != expected[name]:
            raise ClosureError("registered source-only audit absent or stale: " + name)
        audits[name] = _relative_asset(path)
    imports = expected["future_entry_imports"]
    if imports["status"] != "MISSING_LOCAL_RUNTIME_SOURCE" or \
            len(imports["missing_local_modules_or_paths"]) != 21:
        raise ClosureError("expected fail-closed 21-local import gap changed")
    sources = [_relative_asset(_path(ROOT, name)) for name in V4_SOURCES]
    required = {"m8_relocated_infer_v4.py", "m8_child_entry_v4.py",
                "m8_e3_runtime_adapter_v4.py", "m8_embedded_identity_v4.py",
                "m8_portable_rebase_v4.py"}
    if not required <= {Path(row["relative_path"]).name for row in sources}:
        raise ClosureError("future v4 release sources are not registered")
    return {"schema": "g10_m8_portability_source_only_preflight_v4",
            "status": "PASS_SOURCE_ONLY_REBASE_PREFLIGHT_RELEASE_AND_REPLAY_BLOCKED",
            "audits": audits, "v4_source_and_test_files": sources,
            "protected_original_assets_pinned": 18,
            "structured_runtime_descriptor_edges_verified": 9,
            "static_missing_local_count": 21,
            "static_missing_local_modules_or_paths": imports["missing_local_modules_or_paths"],
            "future_release_v4_sources_in_inventory": True,
            "future_release_transitive_import_closure_complete": False,
            "synthetic_tests_last_run": 10,
            "synthetic_test_command": "python -B -m pytest -q tests/aei_paper_goals_v1/test_g10_m8_rebase_v4.py -p no:cacheprovider",
            "runtime_requirements": {
                "CPython": "new exact-distribution venv outside source and bundle; no system site packages; Torch NumPy SciPy PythonOCC/OCC 7.9.0 imports pending",
                "Abaqus": "external licensed Abaqus 2020 launcher and Python 2 ODB runtime; not bundled; exact release/seat/solve/ODB audit pending"},
            "license_gates": {
                "project_source_model_metadata_decoder_STEP_local_copy": "not_supplied",
                "public_redistribution": "not_authorized",
                "Python_dependency_notices_and_lock": "not_supplied",
                "Abaqus_binary_included": False},
            "publication_scope": "versioned source-policy-v3 citable; root matrix historical screening only",
            "old_AEI_readiness_snapshot_used_as_current_M6_truth": False,
            "historical_negative_receipts_modified": False,
            "actual_checkpoint_loads": 0, "actual_STEP_inferences": 0,
            "actual_CAD_edits": 0, "actual_CAE_solves": 0,
            "actual_full_offline_recounts": 0, "T10_truth_opened": False,
            "formal_portable_candidate_present": False,
            "distinct_environment_real_run": False, "reproducibility_pass": False,
            "remaining_real_gates": [
                "21_local_and_dynamic_import_source_closure",
                "OCC_native_dependency_and_exact_venv_lock",
                "source_model_metadata_decoder_STEP_license_copy_decisions",
                "real_R_Q_E1_checkpoint_load_and_embedded_identity_attestation",
                "final_method_T10_freezes_one_time_score_and_M5_CI",
                "materialized_candidate_and_two_fresh_STEP_inferences",
                "one_actual_CAD_edit_and_independent_STEP_reimport_audit",
                "legal_Abaqus_2020_seat_one_solve_and_independent_Python2_ODB_audit",
                "full_postscore_offline_main_table_and_CI_recount"]}


def _references(value: Any) -> list[dict]:
    result = []
    if isinstance(value, dict):
        if set(value) == {"relative_path", "bytes", "sha256"}:
            result.append(value)
        else:
            for child in value.values():
                result.extend(_references(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_references(child))
    return result


def local_import_inventory(source_root: Path, entrypoints: list[str],
                           declared_local: set[str]) -> dict:
    """Parse reachable local import statements, with an explicit missing set.

    Dynamic imports and environment package versions need separate validation;
    a static PASS is never a licensed or isolated runtime PASS.
    """
    root = source_root.resolve()
    if not isinstance(declared_local, set) or not declared_local:
        raise ClosureError("explicit runtime local source inventory absent")
    local_module = {}
    for rel in declared_local:
        path = _path(root, rel)
        if not path.is_file() or path.suffix != ".py":
            raise ClosureError("declared local source missing: " + rel)
        local_module.setdefault(path.stem, set()).add(rel)
    queued = list(entrypoints)
    visited, missing, external = set(), set(), set()
    while queued:
        rel = queued.pop()
        if rel in visited:
            continue
        if rel not in declared_local:
            missing.add(rel)
            continue
        visited.add(rel)
        tree = ast.parse(_path(root, rel).read_text(encoding="utf-8-sig"), filename=rel)
        for node in ast.walk(tree):
            names = ([(name.name.split(".")[0]) for name in node.names]
                     if isinstance(node, ast.Import) else
                     [node.module.split(".")[0]] if isinstance(node, ast.ImportFrom) and node.module
                     else [])
            for name in names:
                if name in local_module:
                    paths = local_module[name]
                    if len(paths) != 1:
                        raise ClosureError("ambiguous local module inventory: " + name)
                    queued.extend(paths)
                else:
                    # Resolve the possible workspace module before accepting it
                    # as a third-party/runtime package.
                    candidates = list(root.glob("tools/aei_paper_goals_v1/*/" + name + ".py"))
                    candidates += list(root.glob("tools/aei_paper_goals_v1/*/" + name + "/__init__.py"))
                    candidates += [root / (name + ".py"), root / name / "__init__.py"]
                    if any(path.is_file() for path in candidates):
                        missing.add(name)
                    else:
                        external.add(name)
    return {"schema": "g10_m8_static_local_import_inventory_v4",
            "status": "MISSING_LOCAL_RUNTIME_SOURCE" if missing else
                      "STATIC_LOCAL_IMPORTS_DECLARED_DYNAMIC_RUNTIME_UNVERIFIED",
            "entrypoints": sorted(entrypoints), "visited_paths": sorted(visited),
            "missing_local_modules_or_paths": sorted(missing),
            "external_names_need_exact_venv_lock": sorted(external),
            "dynamic_imports_verified": False, "isolated_runtime_imports_verified": False,
            "reproducibility_pass": False}


def _materialized(value: Any, resolved: dict[str, dict]) -> Any:
    if isinstance(value, dict):
        if set(value) == {"relative_path", "bytes", "sha256"}:
            record = resolved.get(value["relative_path"])
            if record is None or record["original_sha256"] != value["sha256"] or \
                    record["original_bytes"] != value["bytes"]:
                raise ClosureError("unlisted or changed template dependency")
            return {key: record[key] for key in ("path", "bytes", "sha256")}
        return {key: _materialized(child, resolved) for key, child in value.items()}
    if isinstance(value, list):
        return [_materialized(child, resolved) for child in value]
    return value


def materialize(bundle: Path, plan: dict, *, expected_plan_digest: str, write) -> dict:
    """Relocate verified, locally cleared assets into a different bundle root.

    ``write`` is the caller's structured JSON writer; never pass a protected
    source root. The caller owns copy/license and environment validation.
    """
    root = bundle.resolve()
    if root == ROOT.resolve() or digest(plan) != expected_plan_digest:
        raise ClosureError("protected root or changed closure manifest")
    if plan.get("schema") != "g10_m8_protected_e3_runtime_closure_v4" or \
            plan.get("status") != "SOURCE_ONLY_HASHED_RUNTIME_EDGE_AUDIT_NOT_RELEASE_OR_REPLAY" or \
            plan.get("license_and_full_import_closure_pass") is not False:
        raise ClosureError("unsupported source-only closure manifest")
    original = plan["original_assets"]
    templates = plan["templates"]
    if set(templates) != set(RUNTIME_JSON) or not set(RUNTIME_BINARY) <= set(original):
        raise ClosureError("incomplete protected E3 runtime roots")
    resolved = {}
    for key in RUNTIME_BINARY:
        asset = original[key]
        path = _path(root, asset["relative_path"])
        if not path.is_file() or path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ClosureError("uncopied or changed licensed binary/model asset: " + key)
        resolved[asset["relative_path"]] = {"path": path.resolve().as_posix(),
             "bytes": asset["bytes"], "sha256": asset["sha256"],
             "original_bytes": asset["bytes"], "original_sha256": asset["sha256"]}
    pending = dict(templates)
    while pending:
        advanced = False
        for key, value in list(pending.items()):
            missing = [desc["relative_path"] for desc in _references(value) if
                       desc["relative_path"] not in resolved]
            if missing:
                if any(name not in {original[item]["relative_path"] for item in pending} for name in missing):
                    raise ClosureError("template needs an unlisted runtime dependency")
                continue
            asset = original[key]
            path = _path(root, asset["relative_path"])
            if path.exists():
                raise ClosureError("refuse to replace existing relocated runtime JSON: " + key)
            relocated = _materialized(value, resolved)
            write(path, relocated)
            resolved[asset["relative_path"]] = {"path": path.resolve().as_posix(),
                 "bytes": path.stat().st_size, "sha256": sha256(path),
                 "original_bytes": asset["bytes"], "original_sha256": asset["sha256"]}
            del pending[key]
            advanced = True
        if not advanced:
            raise ClosureError("cyclic runtime descriptor graph")
    return {"schema": "g10_m8_relocated_e3_runtime_sha_mapping_v4",
            "status": "STRUCTURED_RUNTIME_JSON_REBASED_BINARY_BOUND_NOT_REPLAYED",
            "source_only_plan_digest": digest(plan),
            "sha_mapping": [{"relative_path": name, "original_sha256": item["original_sha256"],
                             "relocated_sha256": item["sha256"], "relocated_bytes": item["bytes"]}
                            for name, item in sorted(resolved.items())],
            "binary_bound_original_identity": plan["binary_bound_original_identity"],
            "training_labels_copied": False, "binary_embedded_identity_verified": False,
            "actual_inferences": 0, "reproducibility_pass": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "record-audit", "audit-imports", "record-imports",
                                         "audit-scope", "record-scope", "materialize",
                                         "register-preflight"))
    parser.add_argument("--audit-id", default="source-only-20260915")
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--closure", type=Path)
    parser.add_argument("--expected-closure-sha256")
    parser.add_argument("--mapping-output", type=Path)
    args = parser.parse_args()
    if args.mode == "register-preflight":
        result = register_source_only_preflight()
        target = REPORT / "M8_PORTABILITY_SOURCE_ONLY_PREFLIGHT_V4.json"
        write_json(target, result, overwrite=False)
        print(json.dumps({"status": result["status"], "receipt_sha256": sha256(target),
                          "sources": len(result["v4_source_and_test_files"]),
                          "actual_checkpoint_loads": 0, "actual_inferences": 0,
                          "actual_CAD_edits": 0, "actual_CAE_solves": 0,
                          "actual_full_offline_recounts": 0,
                          "reproducibility_pass": False}, indent=2))
        return
    if args.mode == "materialize":
        if not all((args.bundle_root, args.closure, args.expected_closure_sha256,
                    args.mapping_output)):
            parser.error("materialize requires bundle, exact closure identity and mapping output")
        bundle, closure, output = (path.resolve() for path in
                                   (args.bundle_root, args.closure, args.mapping_output))
        if not bundle.is_relative_to(ARTIFACT.resolve()) or not closure.is_relative_to(bundle) or \
                not output.is_relative_to(bundle) or output.exists() or len(args.expected_closure_sha256) != 64 or \
                not closure.is_file() or sha256(closure) != args.expected_closure_sha256:
            parser.error("materialization paths or exact source-only closure identity invalid")
        plan = read_json(closure)
        result = materialize(bundle, plan, expected_plan_digest=digest(plan), write=write_json)
        write_json(output, result, overwrite=False)
        print(json.dumps({"status": result["status"], "mapping_sha256": sha256(output),
                          "mapped_assets": len(result["sha_mapping"]),
                          "reproducibility_pass": False}, indent=2))
        return
    if args.mode in ("audit-scope", "record-scope"):
        result = publication_scope()
        if args.mode == "record-scope":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.audit_id):
                parser.error("unsafe audit ID")
            target = ARTIFACT / "m8_portability/v4/source_only_audits" / args.audit_id / "PUBLICATION_SCOPE.json"
            write_json(target, result, overwrite=False)
            print(json.dumps({"status": result["status"], "audit_sha256": sha256(target),
                              "primary_asset_count": len(result["primary_citable_assets"]),
                              "reproducibility_pass": False}, indent=2))
        else:
            print(json.dumps(result, indent=2))
        return
    if args.mode in ("audit-imports", "record-imports"):
        result = local_import_inventory(ROOT, list(IMPORT_ROOTS), IMPORT_DECLARED_SEED)
        if args.mode == "record-imports":
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.audit_id):
                parser.error("unsafe audit ID")
            target = ARTIFACT / "m8_portability/v4/source_only_audits" / args.audit_id / "IMPORTS.json"
            write_json(target, result, overwrite=False)
            print(json.dumps({"status": result["status"], "audit_sha256": sha256(target),
                              "missing_local_count": len(result["missing_local_modules_or_paths"]),
                              "reproducibility_pass": False}, indent=2))
        else:
            print(json.dumps(result, indent=2))
        return
    plan = source_only_plan()
    if args.mode == "record-audit":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.audit_id):
            parser.error("unsafe audit ID")
        target = ARTIFACT / "m8_portability/v4/source_only_audits" / args.audit_id / "CLOSURE.json"
        write_json(target, plan, overwrite=False)
        print(json.dumps({"status": plan["status"], "audit_sha256": sha256(target),
                          "asset_count": len(plan["original_assets"]),
                          "reproducibility_pass": False}, indent=2))
    else:
        print(json.dumps({"status": plan["status"], "asset_count": len(plan["original_assets"]),
                          "runtime_descriptor_edges": sum(len(_references(t)) for t in plan["templates"].values()),
                          "training_sidecars_copied": False, "reproducibility_pass": False}, indent=2))


if __name__ == "__main__":
    main()
