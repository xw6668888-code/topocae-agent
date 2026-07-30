from __future__ import annotations

import json
from pathlib import Path

from p1_c.pilot_h_c2_v2 import generate_h_c2


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output_generation_p1_c2_all_class_canonical_36_strict_concept"


def main() -> int:
    payload = generate_h_c2(ROOT, OUTPUT)
    print(
        json.dumps(
            {
                "passed": payload["automatic_pass"],
                "sample_count": payload["sample_count"],
                "output": str(OUTPUT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if payload["automatic_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
