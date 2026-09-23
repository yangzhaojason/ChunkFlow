from pathlib import Path
import subprocess

import pytest

from scripts.audit_release import Finding
from scripts.audit_release import main
from scripts.audit_release import scan_file
from scripts.audit_release import tracked_files


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def test_reports_internal_home_path(tmp_path: Path) -> None:
    path = tmp_path / "config.py"
    private_path = "/" + "home/research/data"
    path.write_text(f"DATA_ROOT = '{private_path}'\n", encoding="utf-8")

    assert scan_file(path, "config.py") == [Finding("internal-path", "config.py")]


def test_reports_numbered_organization_dataset_prefix(tmp_path: Path) -> None:
    path = tmp_path / "config.py"
    dataset_name = "cyto" + "derm2_dataset_all_frame_idxs_add_path.pkl"
    path.write_text(dataset_name + "\n", encoding="utf-8")

    assert scan_file(path, "config.py") == [Finding("internal-path", "config.py")]


def test_reports_plain_organization_dataset_prefix(tmp_path: Path) -> None:
    path = tmp_path / "config.py"
    dataset_name = "cyto" + "derm_dataset"
    path.write_text(dataset_name + "\n", encoding="utf-8")

    assert scan_file(path, "config.py") == [Finding("internal-path", "config.py")]


def test_reports_github_personal_access_token(tmp_path: Path) -> None:
    path = tmp_path / "config.py"
    token = "github" + "_pat_" + "a" * 32
    path.write_text(token, encoding="utf-8")

    assert scan_file(path, "config.py") == [Finding("secret-pattern", "config.py")]


def test_reports_file_larger_than_limit(tmp_path: Path) -> None:
    path = tmp_path / "large.bin"
    path.write_bytes(b"a" * 129)

    assert scan_file(path, "large.bin", max_bytes=128) == [Finding("large-file", "large.bin")]


def test_scanner_source_is_not_exempt_from_internal_path_patterns(tmp_path: Path) -> None:
    path = tmp_path / "audit_release.py"
    private_path = "/" + "Users/researcher/data"
    path.write_text(f"example = '{private_path}'\n", encoding="utf-8")

    assert scan_file(path, "scripts/audit_release.py") == [Finding("internal-path", "scripts/audit_release.py")]


def test_scanner_source_still_reports_large_file(tmp_path: Path) -> None:
    path = tmp_path / "audit_release.py"
    path.write_bytes(b"a" * 129)

    assert scan_file(path, "scripts/audit_release.py", max_bytes=128) == [
        Finding("large-file", "scripts/audit_release.py")
    ]


def test_scanner_source_still_reports_secret_pattern(tmp_path: Path) -> None:
    path = tmp_path / "audit_release.py"
    token = "github" + "_pat_" + "a" * 32
    path.write_text(token, encoding="utf-8")

    assert scan_file(path, "scripts/audit_release.py") == [Finding("secret-pattern", "scripts/audit_release.py")]


def test_scanner_source_regex_definitions_do_not_self_trigger() -> None:
    path = Path(__file__).with_name("audit_release.py")

    assert scan_file(path, "scripts/audit_release.py") == []


def test_reports_symlink_without_reading_external_target(tmp_path: Path) -> None:
    external = tmp_path / "external.txt"
    external.write_text("/" + "Users/researcher/private\n", encoding="utf-8")
    scan_root = tmp_path / "scan-root"
    scan_root.mkdir()
    path = scan_root / "external-link"
    path.symlink_to(external)

    assert scan_file(path, "external-link") == [Finding("symlink", "external-link")]


def test_reports_broken_symlink_without_dereferencing(tmp_path: Path) -> None:
    path = tmp_path / "broken-link"
    path.symlink_to(tmp_path / "missing-target")

    assert scan_file(path, "broken-link") == [Finding("symlink", "broken-link")]


def test_skips_directory_without_reading_it(tmp_path: Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()

    assert scan_file(path, "directory") == []


def test_tracked_files_and_cli_use_real_git_repository(tmp_path: Path, capsys) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _run_git(root, "init", "--quiet")

    (root / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    tracked = root / "tracked.txt"
    tracked.write_text("/" + "Users/researcher/private\n", encoding="utf-8")
    (root / "untracked.txt").write_text("portable\n", encoding="utf-8")
    ignored_token = "github" + "_pat_" + "a" * 32
    (root / "ignored.txt").write_text(ignored_token, encoding="utf-8")
    _run_git(root, "add", ".gitignore", "tracked.txt")

    assert tracked_files(root) == [Path(".gitignore"), Path("tracked.txt"), Path("untracked.txt")]
    assert main(["--root", str(root)]) == 1
    assert capsys.readouterr().out == "internal-path: tracked.txt\n"

    tracked.write_text("portable\n", encoding="utf-8")
    assert main(["--root", str(root)]) == 0
    assert capsys.readouterr().out == "release audit passed\n"


def test_invalid_utf8_does_not_manufacture_internal_path_but_keeps_ascii_secret(tmp_path: Path) -> None:
    path = tmp_path / "mixed.bin"
    token = ("github" + "_pat_" + "a" * 32).encode("ascii")
    path.write_bytes(b"/Us\xffers/researcher/private\n" + token)

    assert scan_file(path, "mixed.bin") == [Finding("secret-pattern", "mixed.bin")]


def test_invalid_utf8_still_reports_literal_internal_path(tmp_path: Path) -> None:
    path = tmp_path / "mixed.bin"
    private_path = ("/" + "Users/researcher/private\n").encode("utf-8")
    path.write_bytes(private_path + b"\xff")

    assert scan_file(path, "mixed.bin") == [Finding("internal-path", "mixed.bin")]


def test_mixed_internal_and_secret_findings_have_stable_order(tmp_path: Path) -> None:
    path = tmp_path / "config.py"
    token = "github" + "_pat_" + "a" * 32
    path.write_text("/" + "Users/researcher/private\n" + token, encoding="utf-8")

    assert scan_file(path, "config.py") == [
        Finding("internal-path", "config.py"),
        Finding("secret-pattern", "config.py"),
    ]


def test_tracked_files_reports_concise_error_for_non_repository(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError) as error:
        tracked_files(tmp_path)

    assert str(error.value) == f"git file discovery failed for {tmp_path}"


def test_cli_reports_git_failure_without_traceback(tmp_path: Path, capsys) -> None:
    assert main(["--root", str(tmp_path)]) == 2

    captured = capsys.readouterr()
    assert captured.out == f"audit-error: git file discovery failed for {tmp_path}\n"
    assert captured.err == ""
    assert "Traceback" not in captured.out + captured.err


def test_scan_file_reports_concise_error_for_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(RuntimeError) as error:
        scan_file(path, "missing.txt")

    assert str(error.value) == "file inspection failed: missing.txt"


@pytest.mark.parametrize(
    "relative_path",
    [
        ".claude/settings.json",
        ".codex/session.jsonl",
        "src/.cursor/rules/review.mdc",
        ".agents/skills/review/SKILL.md",
        "docs/superpowers/plans/release.md",
        "src/AGENTS.md",
        "CLAUDE.md",
        ".cursorrules",
        ".mcp.json",
    ],
)
def test_reports_local_workflow_files_even_without_keywords(tmp_path: Path, relative_path: str) -> None:
    path = tmp_path / "artifact"
    path.write_text("local settings\n", encoding="utf-8")

    assert scan_file(path, relative_path) == [Finding("local-workflow-file", relative_path)]


@pytest.mark.parametrize(
    "instruction",
    [
        "> **For agentic workers:** execute the following plan.",
        "**REQUIRED SUB-SKILL:** use the implementation workflow.",
        "# Generated by Codex",
        "Co-authored-by: Claude <assistant@example.com>",
    ],
)
def test_reports_copied_assistant_instructions(tmp_path: Path, instruction: str) -> None:
    path = tmp_path / "notes.md"
    path.write_text(instruction + "\n", encoding="utf-8")

    assert scan_file(path, "notes.md") == [Finding("assistant-instruction", "notes.md")]


def test_research_terms_and_upstream_generated_headers_are_allowed(tmp_path: Path) -> None:
    path = tmp_path / "model.py"
    path.write_text(
        "# This file was automatically generated from modular code.\n"
        "# An AI agent uses a language model for control.\n"
        'BLOCKED_FILES = {"CLAUDE.md", "AGENTS.md"}\n',
        encoding="utf-8",
    )

    assert scan_file(path, "model.py") == []


def test_cli_checks_forced_tracked_workflow_file_despite_ignore(tmp_path: Path, capsys) -> None:
    _run_git(tmp_path, "init", "--quiet")
    (tmp_path / ".gitignore").write_text(".claude/\n", encoding="utf-8")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude/settings.json").write_text("{}\n", encoding="utf-8")
    _run_git(tmp_path, "add", "-f", ".claude/settings.json")

    assert main(["--root", str(tmp_path)]) == 1
    assert capsys.readouterr().out == "local-workflow-file: .claude/settings.json\n"
