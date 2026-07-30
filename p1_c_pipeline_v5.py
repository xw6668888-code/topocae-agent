from __future__ import annotations

import argparse
import json
from pathlib import Path

from p1_c_pipeline_v4 import ROOT, _pilot_h_c1, _preflight, _require_runtime, _verify_h_c1


def _pilot_h_c2(args: argparse.Namespace) -> int:
    from p1_c.pilot_h_c2_v3 import generate_h_c2

    output = args.output_dir.resolve()
    payload = generate_h_c2(ROOT, output)
    print(
        json.dumps(
            {
                "automatic_pass": payload["automatic_pass"],
                "sample_count": payload["sample_count"],
                "generator_revision": payload["generator_revision"],
                "prior_failed_review_archive": payload.get("prior_failed_review_archive"),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="P1-C clean-geometry full pipeline, revision 5")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.set_defaults(handler=_preflight)
    h_c1 = subparsers.add_parser("pilot-h-c1")
    h_c1.add_argument("--output-dir", type=Path, default=ROOT / "output_generation_p1_c1_clean_rib_web_pilot_48")
    h_c1.set_defaults(handler=_pilot_h_c1)
    verify = subparsers.add_parser("verify-h-c1")
    verify.set_defaults(handler=_verify_h_c1)
    h_c2 = subparsers.add_parser("pilot-h-c2")
    h_c2.add_argument("--output-dir", type=Path, default=ROOT / "output_generation_p1_c2_all_class_canonical_36")
    h_c2.set_defaults(handler=_pilot_h_c2)
    return parser


def main() -> int:
    _require_runtime()
    args = _parser().parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
