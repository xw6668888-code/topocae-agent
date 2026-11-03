from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "p1_e6r_rotation_invariant_decoder_repair_v1"
PACKAGE = ROOT / "output_generation_p1_e6r_graph_v2_mixed_9300" / "training_package"
SOURCE = ROOT / "output_generation_p1_e5_web_v2_mixed_9300"
LEGACY_REPORT = ROOT / "reports" / "p1_e6_controlled_retraining_v1"
SEEDS = (20260718, 20260719, 20260720)

MACRO_CLASSES = (
    "stock",
    "through_hole_round",
    "blind_hole_round",
    "stepped_hole_round",
    "counterbore_hole",
    "countersink_hole",
    "obround_hole",
    "slot_u",
    "slot_rect",
    "pocket_rect",
    "pocket_circular",
    "cylinder_boss",
    "rib",
    "web",
    "rectangular_boss",
    "fillet",
    "chamfer",
    "slot_v",
    "slot_o",
)
TARGET_CLASSES = MACRO_CLASSES[1:]
RAW_MACRO_IDS = {
    "stock": 0,
    "through_hole_round": 1,
    "blind_hole_round": 2,
    "stepped_hole_round": 3,
    "counterbore_hole": 4,
    "countersink_hole": 5,
    "obround_hole": 6,
    "slot_u": 9,
    "slot_rect": 10,
    "pocket_rect": 11,
    "pocket_circular": 12,
    "cylinder_boss": 14,
    "rib": 20,
    "web": 21,
    "rectangular_boss": 24,
    "fillet": 25,
    "chamfer": 26,
    "slot_v": 33,
    "slot_o": 35,
}
RAW_TO_CONTIGUOUS = {RAW_MACRO_IDS[name]: index for index, name in enumerate(MACRO_CLASSES)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def hashed(path: Path) -> dict[str, Any]:
    return {
        "relative_path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
