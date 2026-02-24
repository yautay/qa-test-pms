from __future__ import annotations

import subprocess


def _run_git_command(args: list[str]) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    output = completed.stdout.strip()
    return output or None


def get_build_info() -> dict[str, str]:
    tag = _run_git_command(["describe", "--tags", "--exact-match", "HEAD"])
    if tag is None:
        tag = _run_git_command(["describe", "--tags", "--abbrev=0"])

    commit_hash = _run_git_command(["rev-parse", "--short", "HEAD"])
    commit_date = _run_git_command(["show", "-s", "--format=%cI", "HEAD"])
    branch = _run_git_command(["rev-parse", "--abbrev-ref", "HEAD"])

    if tag is None:
        tag = "unknown"
    if commit_hash is None:
        commit_hash = "unknown"
    if commit_date is None:
        commit_date = "unknown"
    if branch is None:
        branch = "unknown"

    version = tag

    return {
        "version": version,
        "tag": tag,
        "commit_hash": commit_hash,
        "commit_date": commit_date,
        "branch": branch,
    }
