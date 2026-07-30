from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "paper_v1" / "recognition.json"


class ExcludedResearchArtifactError(ValueError):
    """Raised before an excluded research artifact can be read or written."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen TopoCAE-Recognizer paper_v1 pipeline.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify hashes, label schema, and model loading without reading an inference input.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _is_same_or_descendant(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def assert_path_allowed(
    path: Path,
    policy: dict[str, Any],
    *,
    purpose: str,
    recursive: bool = False,
) -> Path:
    resolved = path.resolve()
    normalized = resolved.as_posix().casefold()
    for token in policy.get("deny_path_substrings", []):
        if str(token).casefold() in normalized:
            raise ExcludedResearchArtifactError(
                f"paper_v1 rejected {purpose}: path contains excluded token {token!r}: {resolved}"
            )

    for record in policy.get("excluded_artifacts", []):
        excluded = resolve_repo_path(record["path"])
        if _is_same_or_descendant(resolved, excluded):
            raise ExcludedResearchArtifactError(
                f"paper_v1 rejected {purpose}: excluded artifact boundary: {resolved}"
            )
        if recursive and _is_same_or_descendant(excluded, resolved):
            raise ExcludedResearchArtifactError(
                f"paper_v1 rejected recursive {purpose}: it could include excluded artifact {excluded}"
            )
    return resolved


def load_runtime_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = read_json(config_path.resolve())
    if config.get("schema") != "topocae_paper_recognition_config_v1":
        raise ValueError(f"Unsupported paper recognition config schema: {config.get('schema')!r}")
    if config.get("freeze_id") != "paper_v1":
        raise ValueError(f"Expected freeze_id='paper_v1', got {config.get('freeze_id')!r}")
    policy = read_json(resolve_repo_path(config["dataset_policy_path"]))
    return config, policy


def verify_label_schema(config: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    schema_path = assert_path_allowed(
        resolve_repo_path(config["label_schema_path"]), policy, purpose="label schema"
    )
    schema = read_json(schema_path)
    source_path = assert_path_allowed(
        resolve_repo_path(schema["source_config_path"]), policy, purpose="label schema source"
    )
    source_hash = sha256_file(source_path)
    if source_hash != schema["source_config_sha256"]:
        raise RuntimeError(
            f"Label schema source hash mismatch for {source_path}: {source_hash} != {schema['source_config_sha256']}"
        )
    source = read_json(source_path)
    label_hash = canonical_json_sha256(source[schema["label_spaces_location"]])
    if label_hash != schema["label_spaces_canonical_sha256"]:
        raise RuntimeError(
            "Canonical checkpoint label-space hash mismatch: "
            f"{label_hash} != {schema['label_spaces_canonical_sha256']}"
        )
    return {
        "path": str(schema_path),
        "source_path": str(source_path),
        "source_sha256": source_hash,
        "label_spaces_canonical_sha256": label_hash,
        "counts": schema["counts"],
    }


def verify_freeze_manifest(config: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    manifest_path = assert_path_allowed(
        resolve_repo_path(config["freeze_manifest_path"]), policy, purpose="freeze manifest"
    )
    manifest = read_json(manifest_path)
    if manifest.get("freeze_id") != "paper_v1":
        raise RuntimeError(f"Unexpected freeze manifest id: {manifest.get('freeze_id')!r}")
    verified = []
    for record in manifest.get("files", []):
        path = assert_path_allowed(resolve_repo_path(record["path"]), policy, purpose="frozen file")
        if not path.is_file():
            raise FileNotFoundError(f"Frozen file is missing: {path}")
        actual_hash = sha256_file(path)
        if actual_hash != record["sha256"]:
            raise RuntimeError(
                f"Frozen file hash mismatch for {path}: {actual_hash} != {record['sha256']}"
            )
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Frozen file size mismatch for {path}")
        verified.append(record["path"])
    return {
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "verified_file_count": len(verified),
    }


def _load_checkpoint(path: Path):
    import torch

    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def _bundle_summary(name: str, bundle: dict[str, Any], path: Path) -> dict[str, Any]:
    metadata = bundle.get("metadata", {})
    estimator = bundle.get("estimator") or bundle.get("model")
    feature_schema = metadata.get("feature_schema") or bundle.get("feature_schema") or []
    return {
        "name": name,
        "path": str(path),
        "bundle_type": type(bundle).__name__,
        "estimator_type": type(estimator).__name__ if estimator is not None else None,
        "feature_count": len(feature_schema),
        "label_count": len(metadata.get("id_to_label", {})),
    }


def verify_model_loading(config: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    import torch

    from run_gnn_inference import build_model_from_checkpoint
    from utils.candidate_classifier_postprocess import load_candidate_classifier_bundle
    from utils.rib_boundary_classifier_postprocess import load_rib_boundary_classifier_bundle
    from utils.rib_candidate_proposer_postprocess import load_rib_candidate_proposer_bundle

    model_config = config["models"]
    checkpoint_path = assert_path_allowed(
        resolve_repo_path(model_config["gnn_checkpoint"]["path"]), policy, purpose="GNN checkpoint"
    )
    checkpoint = _load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint, torch.device("cpu"))
    result: dict[str, Any] = {
        "gnn_checkpoint": {
            "path": str(checkpoint_path),
            "model_type": type(model).__name__,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "edge_feature_dim": checkpoint["config"].get("edge_feature_dim"),
            "node_feature_count": len(checkpoint["config"].get("node_feature_schema", [])),
        },
        "auxiliary_models": [],
    }

    loaders = {
        "candidate_classifier": load_candidate_classifier_bundle,
        "rib_candidate_proposer": load_rib_candidate_proposer_bundle,
        "rib_boundary_classifier": load_rib_boundary_classifier_bundle,
    }
    for name, loader in loaders.items():
        record = model_config[name]
        if not record.get("enabled"):
            continue
        path = assert_path_allowed(resolve_repo_path(record["path"]), policy, purpose=name)
        result["auxiliary_models"].append(_bundle_summary(name, loader(path), path))
    return result


def verify_paper_v1(config_path: Path) -> dict[str, Any]:
    config, policy = load_runtime_config(config_path)
    return {
        "freeze_manifest": verify_freeze_manifest(config, policy),
        "label_schema": verify_label_schema(config, policy),
        "model_loading": verify_model_loading(config, policy),
    }


def _configured_model_path(config: dict[str, Any], name: str) -> Path | None:
    record = config["models"][name]
    if not record.get("enabled"):
        return None
    return resolve_repo_path(record["path"])


def _ensure_new_or_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"paper_v1 requires a new or empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_commands(
    config: dict[str, Any],
    input_path: Path,
    output_dir: Path,
) -> tuple[list[str], list[str]]:
    inference = config["inference"]
    predictions_dir = output_dir / "predictions"
    feature_objects_dir = output_dir / "feature_objects"
    gnn_command = [
        sys.executable,
        str(REPO_ROOT / "run_gnn_inference.py"),
        "--checkpoint",
        str(_configured_model_path(config, "gnn_checkpoint")),
        "--input-path",
        str(input_path),
        "--output-dir",
        str(predictions_dir),
        "--device",
        str(inference["device"]),
        "--feature-round-ndigits",
        str(inference["feature_round_ndigits"]),
        "--same-instance-threshold",
        str(inference["same_instance_threshold"]),
        "--candidate-classifier-min-confidence",
        str(inference["candidate_classifier_min_confidence"]),
        "--instance-candidate-min-confidence",
        str(inference["instance_candidate_min_confidence"]),
        "--same-instance-assembly-threshold",
        str(inference["same_instance_assembly_threshold"]),
        "--same-instance-weak-threshold",
        str(inference["same_instance_weak_threshold"]),
        "--rib-candidate-proposer-min-confidence",
        str(inference["rib_candidate_proposer_min_confidence"]),
        "--rib-boundary-classifier-min-confidence",
        str(inference["rib_boundary_classifier_min_confidence"]),
    ]
    if inference["recursive"]:
        gnn_command.append("--recursive")
    if not inference["topology_postprocess"]:
        gnn_command.append("--disable-topology-postprocess")
    for name, argument in (
        ("candidate_classifier", "--candidate-classifier"),
        ("instance_candidate_classifier", "--instance-candidate-classifier"),
        ("rib_candidate_proposer", "--rib-candidate-proposer"),
        ("rib_boundary_classifier", "--rib-boundary-classifier"),
    ):
        path = _configured_model_path(config, name)
        if path is not None:
            gnn_command.extend([argument, str(path)])

    graph_dir = input_path if input_path.is_dir() else input_path.parent
    feature_command = [
        sys.executable,
        str(REPO_ROOT / "prediction_to_feature_objects.py"),
        "--prediction-path",
        str(predictions_dir),
        "--graph-dir",
        str(graph_dir),
        "--output-dir",
        str(feature_objects_dir),
        "--recursive",
        "--fingerprint-round-ndigits",
        str(inference["fingerprint_round_ndigits"]),
        "--grouping-mode",
        str(inference["feature_object_grouping"]),
    ]
    return gnn_command, feature_command


def _output_hashes(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(output_dir.rglob("*.json")):
        if path.name == "paper_v1_run_manifest.json":
            continue
        records.append(
            {
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def run_pipeline(config_path: Path, input_path: Path, output_dir: Path) -> dict[str, Any]:
    verification = verify_paper_v1(config_path)
    config, policy = load_runtime_config(config_path)
    safe_input = assert_path_allowed(
        input_path,
        policy,
        purpose="recognition input",
        recursive=bool(config["inference"]["recursive"]),
    )
    if not safe_input.exists():
        raise FileNotFoundError(f"Recognition input does not exist: {safe_input}")
    safe_output = assert_path_allowed(output_dir, policy, purpose="recognition output")
    _ensure_new_or_empty_directory(safe_output)

    gnn_command, feature_command = build_commands(config, safe_input, safe_output)
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(config["inference"]["pythonhashseed"])
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    subprocess.run(gnn_command, cwd=REPO_ROOT, env=environment, check=True)
    subprocess.run(feature_command, cwd=REPO_ROOT, env=environment, check=True)

    run_manifest = {
        "schema": "topocae_paper_recognition_run_v1",
        "freeze_id": "paper_v1",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
        "freeze_manifest_sha256": verification["freeze_manifest"]["sha256"],
        "input_path": str(safe_input),
        "output_dir": str(safe_output),
        "commands": [gnn_command, feature_command],
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "pythonhashseed": environment["PYTHONHASHSEED"],
            "device": config["inference"]["device"],
        },
        "outputs": _output_hashes(safe_output),
    }
    write_json(safe_output / "paper_v1_run_manifest.json", run_manifest)
    return {"verification": verification, "run_manifest": run_manifest}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    if args.verify_only:
        print(json.dumps(verify_paper_v1(config_path), indent=2, ensure_ascii=False))
        return
    if args.input_path is None or args.output_dir is None:
        raise SystemExit("--input-path and --output-dir are required unless --verify-only is used")
    result = run_pipeline(config_path, args.input_path, args.output_dir)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

