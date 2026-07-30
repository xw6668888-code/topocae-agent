"""Append the third resumed-block audit and refresh delivery hashes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1_smoke_v1"
CONTEXT = REPORT / "CODEX_CONTEXT_P4_API_R1A.md"
MARKER = "Third resumed-block audit"


text = CONTEXT.read_text(encoding="utf-8")
if MARKER not in text:
    now = datetime.now(timezone.utc).isoformat()
    entry = f"""## Update Log

### {now}

- {MARKER}: the same Structured schema failure remains the active blocker; no API call or configuration change occurred.
- Reverified 13 delivery files, 12/12 asset hashes, frozen assets PASS, formal test run=0/content_accessed=false, and zero secret-pattern hits.
- The resumed blocked-audit threshold is now satisfied; the persistent goal may be marked blocked pending supervisor action.
"""
    text = text.replace("## Update Log\n", entry, 1)
    CONTEXT.write_text(text, encoding="utf-8", newline="\n")

manifest_path = REPORT / "delivery_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
for row in manifest["assets"]:
    path = ROOT / row["path"]
    raw = path.read_bytes()
    row["bytes"] = len(raw)
    row["sha256"] = hashlib.sha256(raw).hexdigest()
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
    newline="\n",
)
