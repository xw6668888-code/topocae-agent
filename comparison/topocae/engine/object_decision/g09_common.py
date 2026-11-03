"""G09 paths, immutable-source descriptors, and owned atomic outputs."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/aei_paper_goals_v1/g09"
REPORT = ROOT / "reports/aei_paper_goals_v1/g09"
ARTIFACT = ROOT / "artifacts/aei_paper_goals_v1/g09"
TESTS = ROOT / "tests/aei_paper_goals_v1"

G03_TOOLS = ROOT / "tools/aei_paper_goals_v1/g03"
G07_TOOLS = ROOT / "tools/aei_paper_goals_v1/g07"
G08_TOOLS = ROOT / "tools/aei_paper_goals_v1/g08"
G03_ARTIFACT = ROOT / "artifacts/aei_paper_goals_v1/g03"
G07_ARTIFACT = ROOT / "artifacts/aei_paper_goals_v1/g07"
G08_ARTIFACT = ROOT / "artifacts/aei_paper_goals_v1/g08"
G07_REPORT = ROOT / "reports/aei_paper_goals_v1/g07"
G08_REPORT = ROOT / "reports/aei_paper_goals_v1/g08"
FROZEN = ROOT / "paper_release/topocae_p1_legacy_baseline_v1"
SEEDS = (20260718, 20260719, 20260720)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def descriptor(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def checked(value: dict[str, Any]) -> Path:
    path = Path(value["path"])
    actual = descriptor(path)
    expected = {key: value[key] for key in ("path", "bytes", "sha256")}
    if actual != expected:
        raise ValueError(f"Immutable source changed: {path}")
    return path


def digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _owned(path: str | Path) -> Path:
    path = Path(path).resolve()
    roots = (TOOLS.resolve(), REPORT.resolve(), ARTIFACT.resolve())
    valid = any(path.is_relative_to(root) for root in roots)
    valid |= path.parent == TESTS.resolve() and path.name.startswith("test_g09_")
    if not valid:
        raise ValueError(f"G09 output outside authorized directories: {path}")
    return path


def write_json(path: str | Path, value: Any) -> None:
    path = _owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_text(path: str | Path, value: str) -> None:
    path = _owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def append_jsonl(path: str | Path, value: Any) -> None:
    path = _owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def runtime_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONIOENCODING="utf-8",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
        NUMEXPR_NUM_THREADS="4",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        DGLBACKEND="pytorch",
    )
    for key, folder in (
        ("DGLDEFAULTDIR", "dgl"),
        ("TORCH_HOME", "torch"),
        ("HF_HOME", "huggingface"),
        ("MPLCONFIGDIR", "matplotlib"),
    ):
        env[key] = str(ARTIFACT / "runtime_cache" / folder)
    return env
