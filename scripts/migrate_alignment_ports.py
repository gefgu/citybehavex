#!/usr/bin/env python
"""One-off codemod: point every scenario config at the consolidated aligner server.

Rewrites, in every configs/**/*.yaml:
  - schedules.alignment_base_url
  - activities.alignment_base_url
  - profiles.coherence_alignment_base_url
  - profiles.ownership_alignment_base_url
  - embedding.base_url
to the new consolidated scripts/serve_aligners.py port (default 8090). Every
*_alignment_model / embedding.model field is left untouched -- those are the
lazy-load registry keys and don't change. Line-based (not a YAML round-trip) so
comments/formatting/ordering in hand-maintained configs are preserved exactly;
`embedding.base_url` is matched by section (only rewritten while inside a
top-level `embedding:` block) since `base_url` alone is ambiguous with the
diary LLM's `llm.base_url`, which must NOT be touched.

Usage: .venv/bin/python scripts/migrate_alignment_ports.py [--port 8090] [--dry-run]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

_UNAMBIGUOUS_KEYS = (
    "alignment_base_url",
    "coherence_alignment_base_url",
    "ownership_alignment_base_url",
)
_TOP_LEVEL_RE = re.compile(r"^(\S+):\s*$")
_URL_LINE_RE = re.compile(r"^(\s*)([A-Za-z_]+):\s*(\S+)\s*$")


def migrate_file(path: Path, new_url: str) -> tuple[str, int]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    out: list[str] = []
    section: str | None = None
    n_changed = 0
    for line in lines:
        top_level = _TOP_LEVEL_RE.match(line)
        if top_level and not line.startswith((" ", "\t")):
            section = top_level.group(1)
            out.append(line)
            continue

        match = _URL_LINE_RE.match(line)
        if match:
            indent, key, _value = match.groups()
            is_target = key in _UNAMBIGUOUS_KEYS or (key == "base_url" and section == "embedding")
            if is_target:
                out.append(f"{indent}{key}: {new_url}\n")
                n_changed += 1
                continue
        out.append(line)
    return "".join(out), n_changed


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--configs-root", default="configs")
    args = parser.parse_args(argv)

    new_url = f"http://{args.host}:{args.port}"
    root = Path(args.configs_root)
    total_files = 0
    total_lines = 0
    for path in sorted(root.rglob("*.yaml")):
        new_text, n_changed = migrate_file(path, new_url)
        if n_changed == 0:
            continue
        total_files += 1
        total_lines += n_changed
        print(f"{path}: {n_changed} line(s)")
        if not args.dry_run:
            path.write_text(new_text, encoding="utf-8")
    verb = "would update" if args.dry_run else "updated"
    print(f"{verb} {total_lines} line(s) across {total_files} file(s) -> {new_url}")


if __name__ == "__main__":
    main()
