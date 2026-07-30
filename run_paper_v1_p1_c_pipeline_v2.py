"""Final P1-C single entry with the corrected acceptance verifier."""

from __future__ import annotations

import sys

import run_paper_v1_p1_c_pipeline as v1


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "verify":
        v1._run("verify_p1_c_final_acceptance_v2.py")
        return 0
    return v1.main()


if __name__ == "__main__":
    raise SystemExit(main())
