"""Mechanical correction: cover all allowed HTTP attempts in cost upper bound."""

from pathlib import Path


path = Path(__file__).resolve().parent / "build_p4_api_r1a_smoke_reports.py"
text = path.read_text(encoding="utf-8")
replacements = {
    '"actual_run_cost_upper_cny": 0.08293884,':
        '"actual_run_cost_upper_cny": 0.24830844000000002,',
    "真实请求总数可严格限定为 3 至 7，费用保守上界为 0.083 元，均未接近硬上限。":
        "真实请求总数可严格限定为 3 至 7；按两次 schema 尝试各自最多 3 个 HTTP attempts 计，费用保守上界为 0.249 元，均未接近硬上限。",
    "本次请求数严格位于 3 至 7 之间，费用保守上界为 0.083 元，未接近 201 次或 20 元硬上限。":
        "本次请求数严格位于 3 至 7 之间；按两次 schema 尝试各自最多 3 个 HTTP attempts 计，费用保守上界为 0.249 元，未接近 201 次或 20 元硬上限。",
    "Conservative cost upper: CNY 0.08293884.":
        "HTTP-retry-covered conservative cost upper: CNY 0.24830844.",
}
for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"missing expected text: {old}")
    text = text.replace(old, new)
path.write_text(text, encoding="utf-8", newline="\n")
