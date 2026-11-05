"""Scan git-tracked files for credential patterns; exit 1 on any hit.

Self-contained (no third-party dependency) so CI can run it as a plain
python step. Patterns cover the credential families this project actually
touches (LLM keys, GitHub tokens, cloud keys, private keys, webhook URLs)
plus generic assignments to secret-looking variable names.

  .venv/Scripts/python.exe scripts/scan_secrets.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

PATTERNS: tuple[tuple[str, str], ...] = (
    ("llm-key", r"sk-[a-zA-Z0-9]{20,}"),
    ("github-token", r"gh[pousr]_[A-Za-z0-9]{20,}"),
    ("github-pat", r"github_pat_[A-Za-z0-9_]{20,}"),
    ("aws-access-key", r"AKIA[0-9A-Z]{16}"),
    ("private-key", r"-----BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    ("slack-token", r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    ("google-api-key", r"AIza[0-9A-Za-z_\-]{30,}"),
    ("wecom-webhook", r"qyapi\.weixin\.qq\.com/cgi-bin/webhook/send\?key=[A-Za-z0-9-]{20,}"),
    ("telegram-bot", r"bot\d{8,}:[A-Za-z0-9_-]{30,}"),
    ("secret-assignment", r"(?i)(secret[_-]?key|admin[_-]?token|api[_-]?key|password)\s*[=:]\s*[\"']?[A-Za-z0-9+/_-]{16,}[\"']?"),
)

# known non-secrets that legitimately match a pattern (allowlist)
ALLOWLIST: tuple[str, ...] = (
    "change-me",
    "please-change-me",
    "test-admin-token",
    "deploy-admin-token",
    "placeholder",
    "your-key-here",
    "example.com",
)


def tracked_files() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line.strip()]


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for name, pattern in PATTERNS:
            for match in re.finditer(pattern, content):
                snippet = match.group(0)
                if any(allow.lower() in snippet.lower() for allow in ALLOWLIST):
                    continue
                line_no = content[: match.start()].count("\n") + 1
                findings.append(f"{path}:{line_no}: {name} ({snippet[:24]}...)")

    if findings:
        print("SECRET-LIKE STRINGS FOUND:")
        for finding in findings:
            print(f"  {finding}")
        print("If a finding is a false positive, extend ALLOWLIST in scripts/scan_secrets.py.")
        return 1
    print(f"secrets scan clean ({len(PATTERNS)} patterns).")
    return 0


if __name__ == "__main__":
    from pathlib import Path  # noqa: PLC0415  # keep the top import block pattern-free

    sys.exit(main())
