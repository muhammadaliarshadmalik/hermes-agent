"""JSONL trajectory writer with secret redaction. Stdlib only.

One file per task run: bench/runs/<run_id>/trajectories/<task_id>[__rN].jsonl
Each line: {"seq": int, "ts": ISO-8601 UTC, "type": str, "data": {...}}
"""
from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

REDACTED = "[REDACTED]"

_SECRET_KEY = re.compile(r"(api[_-]?key|token|secret|password|passwd|authorization|credential)", re.I)
_SECRET_VALUE = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{16,}"),
    re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}"),
]
_KV_ASSIGN = re.compile(
    r"(?i)((?:api[_-]?key|token|secret|password|passwd)\s*[=:]\s*)(['\"]?)[^\s'\",;]{6,}\2"
)


def redact_text(s: str) -> str:
    for pat in _SECRET_VALUE:
        s = pat.sub(REDACTED, s)
    return _KV_ASSIGN.sub(lambda m: m.group(1) + REDACTED, s)


def redact(obj: Any) -> Any:
    """Recursively redact secrets in str/dict/list/tuple; other types pass through."""
    if isinstance(obj, str):
        return redact_text(obj)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if isinstance(k, str) and _SECRET_KEY.search(k) and isinstance(v, (str, bytes)):
                out[k] = REDACTED
            else:
                out[k] = redact(v)
        return out
    if isinstance(obj, (list, tuple)):
        return [redact(v) for v in obj]
    return obj


def trajectory_ref(run_id: str, task_id: str, repeat: Optional[int] = None) -> str:
    """POSIX relative ref stored in ScoreReport.trajectory_ref."""
    name = task_id if repeat is None else f"{task_id}__r{repeat}"
    return f"bench/runs/{run_id}/trajectories/{name}.jsonl"


class TrajectoryWriter:
    """Append-only JSONL writer. Use as a context manager or call close()."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(self.path, "a", encoding="utf-8", newline="\n")
        self._seq = 0
        self._lock = threading.Lock()

    def event(self, type_: str, **data: Any) -> None:
        with self._lock:
            rec = {
                "seq": self._seq,
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "type": type_,
                "data": redact(data),
            }
            self._f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            self._f.flush()
            self._seq += 1

    def close(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.close()

    def __enter__(self) -> "TrajectoryWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_trajectory(path: Path | str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
