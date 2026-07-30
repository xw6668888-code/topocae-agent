from __future__ import annotations

from pathlib import Path

import generate_p1_b1_slot_pilot as protocol
import generate_p1_b1_slot_pilot_v2 as identity_safe_protocol


GUIDE_LINES = [
    "# P1-B1 槽几何 Human Gate H-B1：UG/NX 检查指南",
    "",
    "本审查只判断几何概念是否正确，不需要填写 face ID。请打开 `human_review_canonical_20/` 内全部 20 个 STEP。",
    "",
    "## 建议操作",
    "",
    "1. 在 NX 中逐个导入 STEP，确认 Part Navigator/模型树中只有一个有效实体。",
    "2. 使用剖切、测量或旋转视图检查截面与槽端；必要时隐藏基准或打开边显示。",
    "3. 对每个 STEP 检查下列七项；任一项不满足即填 FAIL。",
    "",
    "## 七项必检内容",
    "",
    "1. 模型树是否只有一个有效实体。",
    "2. 截面是否分别为矩形、U、V、O 的精确解析形状，而不是折线近似。",
    "3. 弯槽 guide 是否视觉连续、没有折角和分段面带。",
    "4. `open_open` 通槽是否两端都无残余实体。",
    "5. `open_closed` 半通槽是否恰好一端开放、另一端有完整终止壁。",
    "6. `closed_loop` 闭槽是否首尾连续，且整圈不与基体边界相交。",
    "7. 是否存在埋在实体内部、没有开放到表面的隐藏通道。",
    "",
    "## 填写 CSV",
    "",
    "打开同级 `human_review_canonical_20.csv`，只填写两列：",
    "",
    "- `review_result`：填写 `PASS` 或 `FAIL`。",
    "- `review_reason`：PASS 可简写“七项均通过”；FAIL 请写简短原因和可识别的几何位置。",
    "",
    "不要修改 STEP 文件、seed、哈希或其他 CSV 列。完成 20 行后保存 CSV，并把结论交回 Codex。",
]


def write_clean_nx_guide(path: Path) -> None:
    path.write_text("\n".join(GUIDE_LINES) + "\n", encoding="utf-8")


def main() -> int:
    protocol.write_nx_guide = write_clean_nx_guide
    return identity_safe_protocol.main()


if __name__ == "__main__":
    raise SystemExit(main())
