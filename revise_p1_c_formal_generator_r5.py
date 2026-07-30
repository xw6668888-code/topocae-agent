from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from p1_c.formal_generator_v5 import GENERATOR_REVISION


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_generation_p1_c_clean_context_7200"
ARCHIVE = OUTPUT / "review_history" / "generator_r4_partial_failure"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    output = OUTPUT.resolve()
    archive = ARCHIVE.resolve()
    if output not in archive.parents:
        raise RuntimeError("revision archive escaped the formal dataset root")
    if archive.exists():
        raise RuntimeError(f"revision archive already exists: {archive}")
    training_paths = (
        ROOT / "reports" / "p1_c_training_runs_v1",
        ROOT / "reports" / "p1_c_training_ablation_summary_v1",
        ROOT / "reports" / "p1_c_freeze_v1",
    )
    existing_training = [str(path) for path in training_paths if path.exists() and any(path.iterdir())]
    if existing_training:
        raise RuntimeError(f"cannot revise generator after P1-C model results exist: {existing_training}")
    failure_path = output / "state" / "p1c_02_blind_hole_round_0366_triple.failure.json"
    if not failure_path.exists():
        raise RuntimeError("expected r4 systematic failure evidence is missing")
    record_dir = output / "state" / "records"
    completed_count = len(list(record_dir.glob("*.json")))
    archive.mkdir(parents=True)
    moved: list[Path] = []
    for relative in ("steps", "graphs", "state/records"):
        source = output / relative
        destination = archive / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.exists():
            shutil.move(str(source), str(destination))
            moved.append(destination)
        source.mkdir(parents=True, exist_ok=True)
    archived_state = archive / "state"
    archived_state.mkdir(parents=True, exist_ok=True)
    for source in sorted((output / "state").glob("*.json")):
        destination = archived_state / source.name
        shutil.move(str(source), str(destination))
        moved.append(destination)
    config_path = output / "manifests" / "generation_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    old_config_sha = sha256_file(config_path)
    old_revision = config["generator_revision"]
    config["generator_revision"] = GENERATOR_REVISION
    config["revision_history"] = [
        {
            "from": old_revision,
            "to": GENERATOR_REVISION,
            "reason": "r4 restricted chamfer/fillet candidates to host top edges; two-support composites have other valid outer linear edges",
            "model_results_seen": False,
            "frozen_generation_plan_changed": False,
            "partial_output_policy": "archived_and_regenerated_from_sample_1",
        }
    ]
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    file_records: list[dict[str, Any]] = []
    for path in sorted(item for item in archive.rglob("*") if item.is_file()):
        file_records.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    revision = {
        "schema_version": "topocae_p1_c_formal_generator_revision_1",
        "revised_at_utc": datetime.now(timezone.utc).isoformat(),
        "from_generator_revision": old_revision,
        "to_generator_revision": GENERATOR_REVISION,
        "completed_record_count_archived": completed_count,
        "failure_sample_id": "p1c_02_blind_hole_round_0366_triple",
        "failure_evidence_sha256": sha256_file(archived_state / failure_path.name),
        "old_generation_config_sha256": old_config_sha,
        "new_generation_config_sha256": sha256_file(config_path),
        "generation_plan_changed": False,
        "model_results_seen": False,
        "root_cause": "valid transition candidates existed outside the top-edge-only candidate set",
        "corrective_action": "enumerate geometry-derived outer linear edges and accept the first cleanup-idempotent one-face transition",
        "archived_files": file_records,
    }
    revision_path = output / "manifests" / "generator_revision_r5.json"
    revision_path.write_text(json.dumps(revision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, **revision}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
