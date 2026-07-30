"""H-C2 review package with the extended H-C1 denominator recorded correctly."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from p1_c import pilot_h_c2_v2 as revision2


CANONICAL_SEED_BASE = 2026091000


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    records = payload["records"]
    h_c1 = payload["h_c1_review"]
    lines = [
        "# P1-C H-C2 严格概念宿主自动审计摘要",
        "",
        f"- H-C1 r2：{h_c1['pass_count']}/{h_c1['row_count']} PASS；SHA-256 `{h_c1['sha256']}`。",
        f"- H-C2：{sum(record['audit']['passed'] for record in records)}/{len(records)} 自动 PASS。",
        "- 18类各1个 target-only 和1个 pair，合计36件。",
        "- 全部检查单实体、单壳、有效B-Rep、cleanup幂等、零冗余face/edge、标签100%、实例集合、STEP重读和Graph v2 finite。",
        "- rib 使用人工通过的 T-junction 宿主；web 使用人工通过且无裸露端面的 buttressed two-support 宿主。",
        "- rib/web 的 pair 孔位于基体并与薄壁保持净距，不穿过 target 薄壁。",
        "",
        "自动审计通过后进入 Human Gate H-C2；36/36 人工 PASS 前不得生成 Clean-Context-7200。",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


revision2.protocol.CANONICAL_SEED_BASE = CANONICAL_SEED_BASE
revision2.protocol._write_summary = _write_summary

generate_h_c2 = revision2.generate_h_c2
verify_h_c1 = revision2.verify_h_c1


__all__ = ["generate_h_c2", "verify_h_c1"]
