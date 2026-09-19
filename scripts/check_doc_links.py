#!/usr/bin/env python3
"""Fail if a relative Markdown link in the repository points at a missing file.

Keeping this in CI is cheap insurance against documentation that drifts away
from the code layout as phases land.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIXES = ("http://", "https://", "mailto:", "#")
ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    broken: list[str] = []
    for md in sorted(ROOT.rglob("*.md")):
        if any(part in {".venv", "build", "node_modules", ".git"} for part in md.parts):
            continue
        text = md.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            target = target.split(" ", 1)[0].split("#", 1)[0].strip()
            if not target or target.startswith(SKIP_PREFIXES):
                continue
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                broken.append(f"{md.relative_to(ROOT)} -> {target}")

    if broken:
        print("Broken relative links:")
        for item in broken:
            print(f"  {item}")
        return 1
    print("All relative Markdown links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
