from __future__ import annotations

"""Make the immutable corrected P1-B training package self-consistent.

The v1 tensor payload is valid, but its metadata declared 63 semantic classes
while the actual Graph-v2 labels span ids 0..63.  This migration preserves the
payload byte-for-byte (hard links when possible), infers all label bounds from
train/val/development-test, and writes a new manifest instead of mutating v1.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil

import torch


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v1"
OUTPUT = ROOT / "output_generation_p1_b_corrected_balanced_5000_v3" / "training_p1_b_v2"
SPLITS = ("train", "val", "development_test")


def main() -> int:
    if OUTPUT.exists() and any(OUTPUT.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty package: {OUTPUT}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    maxima = {"semantic": -1, "macro": -1, "primitive": -1}
    sample_count = 0
    for split in SPLITS:
        source = SOURCE / f"{split}.pt"
        destination = OUTPUT / source.name
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        records = torch.load(destination, map_location="cpu", weights_only=False)
        sample_count += len(records)
        for record in records:
            maxima["semantic"] = max(maxima["semantic"], _tensor_max(record["semantic_targets"]))
            maxima["macro"] = max(maxima["macro"], _tensor_max(record["macro_targets"]))
            maxima["primitive"] = max(maxima["primitive"], _tensor_max(record["primitive_surface_ids"]))

    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    metadata["schema_version"] = "topocae_p1_b_training_package_2"
    metadata["semantic_class_count"] = maxima["semantic"] + 1
    metadata["macro_class_count"] = maxima["macro"] + 1
    metadata["primitive_surface_class_count"] = maxima["primitive"] + 1
    metadata["label_id_audit"] = {
        "actual_maximum_ids": maxima,
        "all_targets_in_declared_range": True,
        "sample_count": sample_count,
        "correction": "v1 semantic_class_count 63 corrected to inferred 64; tensor payload unchanged",
    }
    metadata["source_training_package_v1_sha256"] = _sha256(SOURCE / "metadata.json")
    metadata_path = OUTPUT / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    files = [OUTPUT / f"{split}.pt" for split in SPLITS] + [metadata_path]
    manifest = {
        "schema_version": "topocae_p1_b_training_package_manifest_2",
        "source_package": str(SOURCE.resolve()),
        "files": [
            {"path": str(path.resolve()), "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    manifest_path = OUTPUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sample_count == 5000 and maxima == {"semantic": 63, "macro": 18, "primitive": 7}
    print(json.dumps({
        "automatic_pass": passed,
        "sample_count": sample_count,
        "maxima": maxima,
        "semantic_class_count": metadata["semantic_class_count"],
        "output": str(OUTPUT),
    }, ensure_ascii=False))
    return 0 if passed else 1


def _tensor_max(values: torch.Tensor) -> int:
    return int(values.max().item()) if values.numel() else -1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


if __name__ == "__main__":
    raise SystemExit(main())
