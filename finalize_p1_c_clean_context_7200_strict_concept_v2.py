"""Finalize the strict formal dataset with the balanced matched-5000 selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import generate_p1_c_clean_context_7200_strict_concept as strict_protocol
from p1_c.matched_5000_v3 import write_matched_5000


def _write_matched_5000(
    output: Path, records: list[dict[str, Any]]
) -> dict[str, Any]:
    return write_matched_5000(
        output,
        records,
        sha256_file=strict_protocol.protocol.sha256_file,
    )


strict_protocol.protocol._write_matched_5000 = _write_matched_5000


def main() -> int:
    payload = strict_protocol.finalize(strict_protocol.OUTPUT.resolve())
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
