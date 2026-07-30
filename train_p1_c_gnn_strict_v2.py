"""Use the post-preflight frozen strict training plan v3."""

from __future__ import annotations

from pathlib import Path

import train_p1_c_gnn_strict as strict_protocol


PLAN = Path(__file__).resolve().parent / "configs" / "p1_c" / "training_plan_strict_v3.json"
strict_protocol.protocol.PLAN = PLAN


def main() -> int:
    return strict_protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
