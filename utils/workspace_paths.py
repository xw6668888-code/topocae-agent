"""Resolve frozen workspace paths without rewriting frozen artifacts."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ORIGINAL_WORKSPACE_ROOT = PureWindowsPath(
    r"W:\python\FEA\dataset_generator"
)


def _workspace_root() -> Path:
    configured = os.environ.get("TOPOCAE_WORKSPACE_ROOT")
    root = Path(configured) if configured else PROJECT_ROOT
    if not root.is_absolute():
        raise ValueError("TOPOCAE_WORKSPACE_ROOT must be an absolute path")
    return root.resolve()


def _ensure_within_workspace(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes workspace root: {path}") from error
    return resolved


def _original_windows_root() -> PureWindowsPath:
    configured = os.environ.get("TOPOCAE_ORIGINAL_WORKSPACE_ROOT")
    root = PureWindowsPath(configured) if configured else DEFAULT_ORIGINAL_WORKSPACE_ROOT
    if not root.is_absolute() or str(root.drive).startswith("\\\\"):
        raise ValueError(
            "TOPOCAE_ORIGINAL_WORKSPACE_ROOT must be an absolute drive path"
        )
    if any(part in {".", ".."} for part in root.parts[1:]):
        raise ValueError("TOPOCAE_ORIGINAL_WORKSPACE_ROOT contains traversal")
    return root


def _mapped_windows_relative(value: str) -> tuple[str, ...] | None:
    windows_path = PureWindowsPath(value)
    if not windows_path.is_absolute():
        if windows_path.drive or windows_path.root:
            raise ValueError(f"Incomplete Windows absolute path: {value}")
        return None
    if str(windows_path.drive).startswith("\\\\"):
        raise ValueError(f"UNC paths are not allowed: {value}")

    original_root = _original_windows_root()
    candidate_parts = windows_path.parts
    root_parts = original_root.parts
    if len(candidate_parts) < len(root_parts) or any(
        candidate.casefold() != expected.casefold()
        for candidate, expected in zip(candidate_parts, root_parts)
    ):
        raise ValueError(f"Windows path is outside the original workspace: {value}")
    relative = candidate_parts[len(root_parts):]
    if any(part in {".", ".."} for part in relative):
        raise ValueError(f"Windows path contains traversal: {value}")
    return relative


def resolve_workspace_path(value: str | Path) -> Path:
    """Resolve a native, frozen Windows, or relative path inside the workspace."""

    raw = os.fspath(value)
    if not raw:
        raise ValueError("Workspace path must not be empty")
    root = _workspace_root()
    native = Path(raw)

    if native.is_absolute():
        try:
            return _ensure_within_workspace(native, root)
        except ValueError:
            pass

    windows_relative = _mapped_windows_relative(raw)
    if windows_relative is not None:
        return _ensure_within_workspace(root.joinpath(*windows_relative), root)

    if native.is_absolute():
        raise ValueError(f"Absolute path is outside the workspace: {raw}")

    relative_parts = PureWindowsPath(raw).parts if "\\" in raw else native.parts
    if any(part in {".", ".."} for part in relative_parts):
        raise ValueError(f"Relative path contains traversal: {raw}")
    return _ensure_within_workspace(root.joinpath(*relative_parts), root)


__all__ = ["resolve_workspace_path"]
