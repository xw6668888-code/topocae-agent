from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_PYTHON = (ROOT.parent.parent / "envs" / "feagpt" / "python.exe").resolve()


def _require_runtime() -> None:
    actual = Path(sys.executable).resolve()
    if actual != REQUIRED_PYTHON:
        raise SystemExit(
            "P1-C requires the frozen OCC/PyTorch runtime. Run:\n"
            f"  {REQUIRED_PYTHON} {Path(__file__).name} <command>\n"
            f"Current interpreter: {actual}"
        )


def _preflight(_args: argparse.Namespace) -> int:
    from p1_c.preflight import write_preflight

    report = write_preflight(ROOT)
    print(
        json.dumps(
            {
                "output": str(ROOT / "reports" / "p1_c_preflight"),
                "protected_file_count": report["protected_summary"]["total_file_count"],
                "protected_byte_count": report["protected_summary"]["total_byte_count"],
                "aggregate_sha256": report["protected_summary"]["aggregate_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1-C clean-geometry full pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight", help="freeze environment and P1-B protected inventory")
    preflight.set_defaults(handler=_preflight)
    return parser


def main() -> int:
    _require_runtime()
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
