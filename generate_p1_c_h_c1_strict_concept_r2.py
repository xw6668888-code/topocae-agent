from __future__ import annotations

import json
from pathlib import Path

from p1_c.pilot_h_c1_v4 import generate_h_c1


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_generation_p1_c1_strict_concept_hosts_pilot_96_r2"


def main() -> int:
    payload = generate_h_c1(ROOT, OUTPUT)
    print(
        json.dumps(
            {
                "passed": payload["automatic_pass"],
                "pilot_count": payload["pilot_count"],
                "canonical_count": payload["canonical_count"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
