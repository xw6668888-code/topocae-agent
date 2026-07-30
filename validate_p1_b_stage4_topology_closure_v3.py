"""Official stage-4 validator using the legal-slot verifier revision."""

from __future__ import annotations

import utils.p1_b_topology_verifier_v2
import validate_p1_b_stage4_topology_closure_v2 as revised
import validate_p1_b_stage4_topology_closure as base


if __name__ == "__main__":
    raise SystemExit(base.main())
