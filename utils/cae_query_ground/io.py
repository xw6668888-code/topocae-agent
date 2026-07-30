from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            records.append(value)
    return records


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value


def _ensure_new_or_forced(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, value: Any, *, force: bool = False, pretty: bool = True) -> None:
    _ensure_new_or_forced(path, force)
    if pretty:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    else:
        payload = canonical_json_bytes(value).decode("utf-8") + "\n"
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]], *, force: bool = False) -> int:
    _ensure_new_or_forced(path, force)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(canonical_json_bytes(record).decode("utf-8") + "\n")
            count += 1
    return count


def write_yaml(path: Path, value: Any, *, force: bool = False) -> None:
    _ensure_new_or_forced(path, force)
    payload = yaml.safe_dump(value, allow_unicode=True, sort_keys=False, width=120)
    path.write_text(payload, encoding="utf-8", newline="\n")


def write_text(path: Path, text: str, *, force: bool = False) -> None:
    _ensure_new_or_forced(path, force)
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")
