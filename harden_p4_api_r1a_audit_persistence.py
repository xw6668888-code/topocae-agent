"""Mechanically harden R1A audit persistence without changing inference policy."""

from pathlib import Path


path = Path(__file__).resolve().parent / "run_p4_api_r1a_smoke.py"
text = path.read_text(encoding="utf-8")
replacements = [
    (
        '''        self.stage = "unassigned"\n\n    def __call__(self, url, headers, payload, timeout):''',
        '''        self.stage = "unassigned"\n        self.audit_path = OUTPUT / "backend_request_counter.json"\n        self._persist()\n\n    def _persist(self) -> None:\n        write_json(\n            self.audit_path,\n            {\n                "schema_version": "topocae-p4-api-r1a-backend-counter/1",\n                "backend_request_count": self.total,\n                "backend_requests_by_stage": dict(self.by_stage),\n                "backend_request_cap": BACKEND_REQUEST_CAP,\n            },\n        )\n\n    def __call__(self, url, headers, payload, timeout):''',
    ),
    (
        '''        self.total += 1\n        self.by_stage[self.stage] += 1\n        return _default_transport(url, headers, payload, timeout)''',
        '''        self.total += 1\n        self.by_stage[self.stage] += 1\n        self._persist()\n        return _default_transport(url, headers, payload, timeout)''',
    ),
    (
        '''        self.audit_rows.append(row)\n        if outcome.retry_count > 1:''',
        '''        self.audit_rows.append(row)\n        audit_path = OUTPUT / "outcome_audit.jsonl"\n        audit_path.parent.mkdir(parents=True, exist_ok=True)\n        with audit_path.open("a", encoding="utf-8", newline="\\n") as handle:\n            handle.write(\n                json.dumps(row, ensure_ascii=False, separators=(",", ":"))\n                + "\\n"\n            )\n            handle.flush()\n        if outcome.retry_count > 1:''',
    ),
]
for old, new in replacements:
    if old not in text:
        raise RuntimeError("expected runner fragment not found")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8", newline="\n")
