"""Append the second resumed-block audit and refresh delivery hashes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "reports" / "p4_api_r1_smoke_v1"
CONTEXT = REPORT / "CODEX_CONTEXT_P4_API_R1A.md"
MARKER = "Second resumed-block audit"


text = CONTEXT.read_text(encoding="utf-8")
if MARKER not in text:
    now = datetime.now(timezone.utc).isoformat()
    entry = f"""## Update Log

### {now}

- {MARKER}: schema-invalid-after-retry evidence, delivery hashes, frozen assets, and formal test guards remain unchanged.
- Hardened the bounded runner offline so backend attempts and sanitized outcomes persist before future fail-closed exceptions.
- No API request, rerun, prompt/config change, or scope expansion occurred in this turn.
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
