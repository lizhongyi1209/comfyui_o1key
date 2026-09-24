"""Safely fast-forward a Git installation of this node package."""

import os
import subprocess
from pathlib import Path


PLUGIN_DIR = Path(__file__).resolve().parent.parent


class UpdateError(Exception):
    pass


def _git(*args, timeout=60, check=True):
    env = os.environ.copy()
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=PLUGIN_DIR,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise UpdateError("未找到 Git，请先安装 Git。") from exc
    except subprocess.TimeoutExpired as exc:
        raise UpdateError("Git 操作超时，请检查网络后重试。") from exc
    if check and result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise UpdateError(detail[-1] if detail else "Git 操作失败。")
    return result


def update_package():
    """Update origin/main without discarding local changes or switching branches."""
    if not (PLUGIN_DIR / ".git").exists():
        raise UpdateError("当前节点包不是 Git 安装。请通过 Git 安装后再使用界面更新。")

    branch = _git("symbolic-ref", "--quiet", "--short", "HEAD", check=False)
    if branch.returncode or branch.stdout.strip() != "main":
        raise UpdateError("当前不在 main 分支，请手动检查分支后更新。")

    if _git("status", "--porcelain", "--untracked-files=no").stdout.strip():
        raise UpdateError("节点包有本地修改，请先保存或处理修改后再更新。")

    old_commit = _git("rev-parse", "HEAD").stdout.strip()
    old_requirements = _git("show", "HEAD:requirements.txt", check=False).stdout
    _git("fetch", "origin", "main")
    new_commit = _git("rev-parse", "FETCH_HEAD").stdout.strip()
    if old_commit == new_commit:
        return {"updated": False, "version": old_commit[:7], "requirements_changed": False}

    if _git("merge-base", "--is-ancestor", "HEAD", "FETCH_HEAD", check=False).returncode:
        raise UpdateError("本地与 origin/main 已分叉，无法安全快进。请手动处理。")

    _git("merge", "--ff-only", "FETCH_HEAD")
    requirements_changed = old_requirements != (PLUGIN_DIR / "requirements.txt").read_text(encoding="utf-8")
    return {
        "updated": True,
        "version": new_commit[:7],
        "requirements_changed": requirements_changed,
    }
