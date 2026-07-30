from __future__ import annotations

"""Revision 3: route the frozen no-canonical-patch artifact directory explicitly."""

from hashlib import sha256
import json
from pathlib import Path

import finalize_p1_c_stage6_summary_v1 as implementation
import finalize_p1_c_stage6_summary_v2 as revision2


def actual_path(path: Path) -> Path:
    text = str(path)
    if "no_canonical_patch_seed_20260718" not in text:
        return path
    text = text.replace("synthetic_pipeline_v1", "synthetic_pipeline_no_canonical_patch_v1")
    text = text.replace("candidate_model_v1", "candidate_model_no_canonical_patch_v1")
    return Path(text)


_base_read_json = implementation.read_json


def mapped_read_json(path: Path):
    return _base_read_json(actual_path(path))


def mapped_file_sha(path: Path) -> str:
    digest = sha256()
    with actual_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().lower()


implementation.read_json = mapped_read_json
implementation.file_sha = mapped_file_sha
implementation.pipeline_row = revision2._pipeline_row
revision2._file_sha = mapped_file_sha


def main() -> int:
    code = implementation.main()
    summary_path = implementation.OUTPUT / "summary.json"
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for row in payload["ablations"]:
        if row["variant"] != "no_canonical_patch":
            continue
        row["metrics_path"] = str(actual_path(Path(row["metrics_path"])))
        row["candidate_metrics_path"] = str(actual_path(Path(row["candidate_metrics_path"])))
    payload["builder_revision"] = "finalize_p1_c_stage6_summary_v3"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "stage6_pass": payload["automatic_acceptance"]["stage6_pass"],
        "summary": str(summary_path),
        "summary_sha256": mapped_file_sha(summary_path),
        "builder_revision": payload["builder_revision"],
    }, ensure_ascii=False))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
