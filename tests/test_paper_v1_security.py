from __future__ import annotations

import re
from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".cfg",
    ".env",
    ".ini",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_DIRECTORY_NAMES = {
    ".codex_pdf_text",
    ".codex_pdfdeps",
    ".git",
    "__pycache__",
    "_literature_reading_tmp",
}
SECRET_PATTERNS = {
    "openai_style_key": re.compile(("s" + "k-") + r"[A-Za-z0-9_-]{16,}"),
    "google_style_key": re.compile(("AI" + "za") + r"[0-9A-Za-z_-]{20,}"),
    "github_style_token": re.compile(("gh" + r"[pousr]_") + r"[A-Za-z0-9_]{20,}"),
    "aws_access_key": re.compile(("AK" + "IA") + r"[0-9A-Z]{16}"),
    "literal_secret_assignment": re.compile(
        r"(?i)(api[_-]?key|access[_-]?token|secret[_-]?key|password)"
        r"\s*[:=]\s*['\"][^'\"]{8,}['\"]"
    ),
    "literal_bearer_header": re.compile(
        r"(?i)authorization\s*[:=]\s*['\"]bearer\s+[^'\"]{8,}['\"]"
    ),
}


def iter_scannable_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRECTORY_NAMES for part in relative_parts):
            continue
        if any(part.startswith("output") for part in relative_parts[:-1]):
            continue
        if any(part.startswith("test_part") for part in relative_parts[:-1]):
            continue
        if path.name == ".env.example" or path.suffix.casefold() in TEXT_SUFFIXES:
            yield path


class PaperV1SecurityTest(unittest.TestCase):
    def test_repository_source_contains_no_plaintext_credentials(self):
        findings = []
        for path in iter_scannable_files(REPO_ROOT):
            try:
                text = path.read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                continue
            for name, pattern in SECRET_PATTERNS.items():
                if pattern.search(text):
                    findings.append((path.relative_to(REPO_ROOT).as_posix(), name))
        self.assertEqual(findings, [])

    def test_local_dotenv_files_are_ignored_but_example_is_retained(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".env", gitignore.splitlines())
        self.assertIn(".env.*", gitignore.splitlines())
        self.assertIn("!.env.example", gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()
