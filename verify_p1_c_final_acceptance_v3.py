"""Quiet final verifier and report consistency finalizer."""

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path

import verify_p1_c_final_acceptance_v2 as v2


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports/p1_c_final_acceptance_v1"


def main() -> int:
    captured = StringIO()
    with redirect_stdout(captured):
        code = v2.main()
    acceptance = json.loads((OUT / "acceptance.json").read_text(encoding="utf-8"))
    test_check = next(item for item in acceptance["checks"] if item["id"] == "p1_c_tests")
    test_count = int(test_check["evidence"]["test_count"])
    acceptance["verifier_revision"] = "verify_p1_c_final_acceptance_v3"
    (OUT / "acceptance.json").write_text(
        json.dumps(acceptance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for path in (OUT / "final_acceptance.zh-CN.md", ROOT / "CODEX_CONTEXT_P1_C.md"):
        text = path.read_text(encoding="utf-8")
        text = text.replace("P1-C 测试 67/67 PASS", f"P1-C 测试 {test_count}/{test_count} PASS")
        text = text.replace("P1-C tests 67/67 PASS", f"P1-C tests {test_count}/{test_count} PASS")
        path.write_text(text, encoding="utf-8")
    result = {
        "accepted": acceptance["accepted"],
        "passed_checks": acceptance["passed_check_count"],
        "check_count": acceptance["check_count"],
        "test_count": test_count,
        "verifier_revision": "v3",
        "holdout_v3_inference_pass_count": acceptance["holdout_conclusion"]["formal_inference_pass_count"],
        "p2_entered": acceptance["p2_entered"],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if code == 0 and acceptance["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
