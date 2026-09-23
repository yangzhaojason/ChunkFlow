#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
import subprocess

DEFAULT_MAX_BYTES = 5 * 1024 * 1024

INTERNAL_PATTERNS = (
    re.compile(r"/" r"Users/"),
    re.compile(r"/" r"home/(?!\$(?:USER(?:/|$)|\{USER\}(?:/|$)))"),
    re.compile(r"/" r"mnt/pi-data(?:/|$)"),
    re.compile(r"tos" r"://"),
    re.compile(r"\byao" r"mingyuan\b"),
    re.compile(r"\blucy" r"shi\b"),
    re.compile(r"\bopenpi_" r"main\b"),
    re.compile(r"\bcyto" r"derm\d*_dataset"),
)

SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN " rb"(?:[A-Z0-9]+ )*PRIVATE KEY(?: BLOCK)?-----"),
    re.compile(rb"\bgh" rb"p_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub" rb"_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\b(?:AK" rb"IA|AS" rb"IA)[A-Z0-9]{16}\b"),
    re.compile(rb"\bx" rb"ox(?:a|b|p|r|s)-[A-Za-z0-9-]{10,}\b"),
)

LOCAL_WORKFLOW_DIRS = frozenset({".agents", ".claude", ".codex", ".cursor"})
LOCAL_WORKFLOW_FILES = frozenset({"AGENTS.md", "CLAUDE.md", ".cursorrules", ".mcp.json"})
ASSISTANT_INSTRUCTION_PATTERN = re.compile(
    r"^[ \t]*(?:[>#*][ \t]*)*"
    r"(?:For agentic workers:|REQUIRED SUB-SKILL:|"
    r"(?:Generated|Written|Created) (?:by|with) (?:Claude(?: Code)?|Codex|ChatGPT)\b|"
    r"Co-authored-by:[ \t]*(?:Claude|Codex)\b)",
    flags=re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class Finding:
    kind: str
    path: str


def scan_file(
    path: str | os.PathLike[str],
    display_name: str,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[Finding]:
    file_path = Path(path)
    try:
        metadata = file_path.lstat()
    except OSError:
        raise RuntimeError(f"file inspection failed: {display_name}") from None
    if stat.S_ISLNK(metadata.st_mode):
        return [Finding("symlink", display_name)]
    if not stat.S_ISREG(metadata.st_mode):
        return []
    if metadata.st_size > max_bytes:
        return [Finding("large-file", display_name)]

    try:
        contents = file_path.read_bytes()
    except OSError:
        raise RuntimeError(f"file read failed: {display_name}") from None
    decoded_contents = contents.decode("utf-8", errors="surrogateescape")

    findings = []
    relative_path = PurePosixPath(display_name)
    if (
        LOCAL_WORKFLOW_DIRS.intersection(relative_path.parts[:-1])
        or relative_path.name in LOCAL_WORKFLOW_FILES
        or ("docs", "superpowers") in zip(relative_path.parts, relative_path.parts[1:], strict=False)
    ):
        findings.append(Finding("local-workflow-file", display_name))
    if ASSISTANT_INSTRUCTION_PATTERN.search(decoded_contents):
        findings.append(Finding("assistant-instruction", display_name))
    if any(pattern.search(decoded_contents) for pattern in INTERNAL_PATTERNS):
        findings.append(Finding("internal-path", display_name))
    if any(pattern.search(contents) for pattern in SECRET_PATTERNS):
        findings.append(Finding("secret-pattern", display_name))
    return findings


def tracked_files(root: str | os.PathLike[str]) -> list[Path]:
    root_path = Path(root)
    try:
        result = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard", "-z"],
            cwd=root_path,
            check=False,
            capture_output=True,
        )
    except OSError:
        raise RuntimeError(f"git file discovery failed for {root_path}") from None
    if result.returncode != 0:
        raise RuntimeError(f"git file discovery failed for {root_path}") from None
    return sorted(Path(os.fsdecode(name)) for name in result.stdout.split(b"\0") if name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit release files for private or generated data.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        findings = [
            finding
            for relative_path in tracked_files(root)
            for finding in scan_file(root / relative_path, relative_path.as_posix())
        ]
    except RuntimeError as error:
        print(f"audit-error: {error}")
        return 2
    if findings:
        for finding in findings:
            print(f"{finding.kind}: {finding.path}")
        return 1

    print("release audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
