"""Build the deterministic, no-LLM CAE-QueryGround v1 pilot.

The entry point is intentionally limited to source audit and H3-01 pilot generation.
It has no formal-scale, model-training, LLM, solver, or P4 mode.
"""

from __future__ import annotations

import argparse
import json

from utils.cae_query_ground.audit import audit_sources, render_audit_markdown
from utils.cae_query_ground.constants import (
    CROSSWALK_PATH,
    SOURCE_AUDIT_JSON,
    SOURCE_AUDIT_MD,
)
from utils.cae_query_ground.io import write_json, write_text, write_yaml


def run_audit(*, force: bool) -> dict:
    audit, crosswalk = audit_sources()
    write_json(SOURCE_AUDIT_JSON, audit, force=force)
    write_text(SOURCE_AUDIT_MD, render_audit_markdown(audit), force=force)
    if not audit["automatic_pass"]:
        raise RuntimeError(
            "Source/role audit failed; generation is prohibited. "
            f"See {SOURCE_AUDIT_JSON}"
        )
    write_yaml(CROSSWALK_PATH, crosswalk, force=force)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "pilot"), default="pilot")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    audit = run_audit(force=args.force)
    if args.stage == "audit":
        print(json.dumps({"stage": "audit", "automatic_pass": True, "source": audit["source"]}, ensure_ascii=False))
        return 0
    raise NotImplementedError("Pilot stage is not wired yet; audit completed successfully.")


if __name__ == "__main__":
    raise SystemExit(main())
