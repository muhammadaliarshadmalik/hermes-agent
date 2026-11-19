"""Deterministic workspace validators. No LLM, no network.

A task's `check` is Python source run in a fresh subprocess AFTER the agent
finishes. It is written to a temp dir outside the workspace (the agent cannot
edit it), with PYTHONPATH=<workspace> and cwd=<workspace>. It must print
CHECK_OK as its last stdout line on success; any assertion/exception fails it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

CHECK_TIMEOUT_SECONDS = 60.0

PRELUDE = '''
import json, os, sqlite3, subprocess, sys
def run(*args, timeout=20):
    return subprocess.run([sys.executable, *args], capture_output=True, text=True, timeout=timeout)
def write(name, text):
    with open(name, "w", encoding="utf-8", newline="\\n") as f:
        f.write(text)
def read(name):
    with open(name, "r", encoding="utf-8") as f:
        return f.read()
'''


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: Optional[str]  # None when passed


def _last_line(text: str) -> str:
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    return lines[-1][:300] if lines else ""


def run_check(workspace: Path | str, check_src: str,
              timeout: float = CHECK_TIMEOUT_SECONDS) -> CheckResult:
    ws = Path(workspace).resolve()
    env = {k: v for k, v in os.environ.items() if k in ("SYSTEMROOT", "PATH", "TEMP", "TMP", "HOME", "USERPROFILE")}
    env.update(PYTHONPATH=str(ws), PYTHONHASHSEED="0", PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
    with tempfile.TemporaryDirectory(prefix="benchcheck_") as td:
        script = Path(td) / "check.py"
        script.write_text(PRELUDE + "\n" + check_src + "\nprint('CHECK_OK')\n", encoding="utf-8")
        try:
            p = subprocess.run([sys.executable, str(script)], cwd=str(ws), env=env,
                               capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return CheckResult(False, f"validator timed out after {timeout:.0f}s")
    out_lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
    if p.returncode == 0 and out_lines and out_lines[-1] == "CHECK_OK":
        return CheckResult(True, None)
    reason = _last_line(p.stderr) or _last_line(p.stdout) or f"validator exit code {p.returncode}"
    return CheckResult(False, reason)


def validate_task(task, workspace: Path | str) -> CheckResult:
    return run_check(workspace, task.check)
