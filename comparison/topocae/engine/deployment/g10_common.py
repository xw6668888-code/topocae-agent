"""Owned paths, immutable descriptors, atomic writes, and resource locks for G10."""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/aei_paper_goals_v1/g10"
REPORT = ROOT / "reports/aei_paper_goals_v1/g10"
ARTIFACT = ROOT / "artifacts/aei_paper_goals_v1/g10"
TESTS = ROOT / "tests/aei_paper_goals_v1"
LOCK_ROOT = ROOT / "resource_locks"
SEEDS = (20260718, 20260719, 20260720)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def descriptor(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256(resolved),
    }


def checked(value: dict[str, Any]) -> Path:
    path = Path(value["path"])
    expected = {key: value[key] for key in ("path", "bytes", "sha256")}
    if descriptor(path) != expected:
        raise ValueError(f"Immutable source changed: {path}")
    return path


def digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _owned(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    roots = (TOOLS.resolve(), REPORT.resolve(), ARTIFACT.resolve())
    valid = any(resolved.is_relative_to(root) for root in roots)
    valid |= resolved.parent == TESTS.resolve() and resolved.name.startswith("test_g10_")
    if not valid:
        raise ValueError(f"G10 output outside authorized directories: {resolved}")
    return resolved


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json(path: str | Path, value: Any, *, overwrite: bool = True) -> None:
    owned = _owned(path)
    if owned.exists() and not overwrite:
        raise FileExistsError(f"Preserve existing G10 evidence: {owned}")
    payload = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_replace(owned, payload)


def write_text(path: str | Path, value: str, *, overwrite: bool = True) -> None:
    owned = _owned(path)
    if owned.exists() and not overwrite:
        raise FileExistsError(f"Preserve existing G10 evidence: {owned}")
    _atomic_replace(owned, value.encode("utf-8"))


def append_jsonl(path: str | Path, value: Any) -> None:
    owned = _owned(path)
    owned.parent.mkdir(parents=True, exist_ok=True)
    with owned.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def relative_public_path(path: str | Path) -> str:
    """Return a repo-relative path suitable for release-candidate documentation."""
    resolved = Path(path).resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise ValueError(f"Public path is outside repository: {resolved}")
    return resolved.relative_to(ROOT.resolve()).as_posix()


def process_creation_time(pid: int | None = None) -> int | None:
    """Return a Windows process creation FILETIME for PID-reuse checks."""
    if os.name != "nt":
        return None
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    values = [wintypes.FILETIME() for _ in range(4)]
    handle = kernel.GetCurrentProcess() if pid is None else kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return None
    try:
        if not kernel.GetProcessTimes(handle, *(ctypes.byref(value) for value in values)):
            return None
    finally:
        if pid is not None:
            kernel.CloseHandle(handle)
    return (values[0].dwHighDateTime << 32) + values[0].dwLowDateTime


def _live_owner(record: dict[str, Any]) -> bool:
    if record.get("released") is True:
        return False
    pid = int(record.get("pid", -1))
    if pid <= 0 or pid == os.getpid():
        return False
    actual = process_creation_time(pid)
    expected = record.get("process_creation_time")
    if expected and expected < 1e12:
        expected = int((expected + 11644473600) * 1e7)
    return bool(actual and (not expected or abs(actual - int(expected)) < 20_000_000))


def _write_shared_json(path: Path, value: Any) -> None:
    """Write only G10-owned metadata in the explicitly shared lock directory."""
    if path.parent.resolve() != LOCK_ROOT.resolve() or not path.name.startswith("g10_"):
        raise ValueError(f"G10 cannot write foreign lock metadata: {path}")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


@contextmanager
def resource_lock(resource: str, experiment_id: str) -> Iterator[Path]:
    """Acquire the repository's OS lock set without altering foreign metadata."""
    if resource not in {"gpu", "occ", "abaqus", "cpu_geometry"}:
        raise ValueError(f"Unknown shared resource: {resource}")
    LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    lock = LOCK_ROOT / f"g10_{resource}.lock"
    owner_path = lock.with_suffix(".owner.json")
    record = {
        "schema": "g10_resource_lock_v1",
        "resource": resource,
        "experiment_id": experiment_id,
        "pid": os.getpid(),
        "process_creation_time": process_creation_time(),
        "acquired_at_utc": now(),
        "host": socket.gethostname(),
        "lock_path": lock.as_posix(),
        "os_lock_authoritative": os.name == "nt",
        "released": False,
    }
    for path in LOCK_ROOT.glob("*.owner.json"):
        try:
            existing = read_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if _live_owner(existing):
            raise RuntimeError(f"Live shared resource owner: {path}: {existing}")
    lock.touch(exist_ok=True)
    if lock.stat().st_size == 0:
        lock.write_bytes(b"0")
    streams = []
    try:
        if os.name != "nt":
            raise RuntimeError("G10 shared resource locking is currently validated only on Windows")
        import msvcrt

        for path in sorted(LOCK_ROOT.glob("*.lock")):
            stream = path.open("r+b")
            try:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                stream.close()
                raise RuntimeError(f"Shared resource lock held by another task: {path}") from error
            streams.append(stream)
        _write_shared_json(owner_path, record)
        yield lock
    finally:
        if streams:
            import msvcrt

            for stream in reversed(streams):
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                stream.close()
        if owner_path.exists():
            current = read_json(owner_path)
            if current.get("pid") != os.getpid() or current.get("experiment_id") != experiment_id:
                raise RuntimeError(f"G10 lock ownership changed while held: {owner_path}")
            current.update(released=True, released_at_utc=now())
            _write_shared_json(owner_path, current)
