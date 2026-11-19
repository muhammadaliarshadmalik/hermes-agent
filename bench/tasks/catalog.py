"""Benchmark task catalog: 12 train tasks. Held-out tasks live in bench/tasks/heldout.py
and are only loaded when HERMES_BENCH_ALLOW_HELDOUT=1 (sealed from the variation agent)."""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

HELDOUT_ENV = "HERMES_BENCH_ALLOW_HELDOUT"
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@dataclass(frozen=True)
class Task:
    task_id: str
    split: str
    prompt: str
    check: str                                   # validator source (see validators.py)
    files: Dict[str, str] = field(default_factory=dict)      # seed files
    reference: Dict[str, str] = field(default_factory=dict)  # known-good solution (self-tests only)
    setup: Optional[Callable[[Path], None]] = None


def prepare_workspace(task: Task, dest: Path | str) -> Path:
    """Create a fresh isolated workspace containing only the task's seed state."""
    d = Path(dest)
    d.mkdir(parents=True, exist_ok=True)
    for name, text in task.files.items():
        p = d / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
    if task.setup:
        task.setup(d)
    return d


def apply_reference(task: Task, workspace: Path | str) -> None:
    """Overlay the reference solution (used only by self-tests, never by the runner)."""
    for name, text in task.reference.items():
        p = Path(workspace) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")


def _seed_users_db(ws: Path) -> None:
    con = sqlite3.connect(str(ws / "app.db"))
    con.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL, email TEXT NOT NULL)")
    con.executemany("INSERT INTO users (id, name, email) VALUES (?,?,?)",
                    [(1, "ann", "ann@x.io"), (2, "bob", "bob@x.io"), (3, "cy", "cy@x.io")])
    con.commit()
    con.close()


TRAIN: List[Task] = [
    Task(
        "train-001-fix-off-by-one", "train",
        "File sumrange.py has two bugs. sum_range(a, b) must return the sum of integers from a to b INCLUSIVE "
        "(0 if a > b). chunks(xs, n) must split a list into consecutive chunks of size n, keeping the final "
        "shorter chunk. Fix both in place; do not rename the functions.",
        '''
from sumrange import sum_range, chunks
assert sum_range(1, 5) == 15, "sum_range(1,5)"
assert sum_range(3, 3) == 3, "sum_range(3,3)"
assert sum_range(5, 1) == 0, "sum_range(5,1)"
assert sum_range(-2, 2) == 0, "sum_range(-2,2)"
assert chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]], "chunks odd"
assert chunks([1, 2, 3, 4], 2) == [[1, 2], [3, 4]], "chunks even"
assert chunks([], 3) == [], "chunks empty"
assert chunks([1, 2], 5) == [[1, 2]], "chunks n>len"
''',
        files={"sumrange.py": "def sum_range(a, b):\n    return sum(range(a, b))\n\n\ndef chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs) - n, n)]\n"},
        reference={"sumrange.py": "def sum_range(a, b):\n    return sum(range(a, b + 1))\n\n\ndef chunks(xs, n):\n    return [xs[i:i + n] for i in range(0, len(xs), n)]\n"},
    ),
    Task(
        "train-002-json-normalizer", "train",
        "Create normalize.py with normalize(obj) that recursively: sorts dict keys alphabetically, strips leading/"
        "trailing whitespace from string values (not keys), and removes dict entries whose value is None. "
        "List order is preserved; lists are normalized element-wise. Return a new object; do not mutate the input.",
        '''
import copy
from normalize import normalize
src = {"b": "  x ", "a": None, "c": [{"z": 1, "k": None, "m": " q"}, " s ", None], "d": {"y": None}}
orig = copy.deepcopy(src)
out = normalize(src)
assert src == orig, "input was mutated"
assert out == {"b": "x", "c": [{"m": "q", "z": 1}, "s", None], "d": {}}, out
assert list(out) == ["b", "c", "d"], list(out)
assert list(out["c"][0]) == ["m", "z"], list(out["c"][0])
assert normalize(5) == 5 and normalize(" a ") == "a"
''',
        reference={"normalize.py": '''def normalize(obj):
    if isinstance(obj, dict):
        return {k: normalize(obj[k]) for k in sorted(obj) if obj[k] is not None}
    if isinstance(obj, list):
        return [normalize(v) for v in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj
'''},
    ),
    Task(
        "train-003-csv-aggregator", "train",
        "Read sales.csv (header: region,amount) and write totals.json mapping each region to the sum of its "
        "amounts rounded to 2 decimals. Create aggregate.py so that `python aggregate.py` does this from the "
        "current directory. Handle quoted fields and skip rows with a non-numeric amount.",
        '''
write("sales.csv", 'region,amount\\n"North, East",10.10\\nSouth,5\\n"North, East",0.25\\nSouth,abc\\nWest,3.333\\nWest,3.333\\n')
if os.path.exists("totals.json"):
    os.remove("totals.json")
p = run("aggregate.py")
assert p.returncode == 0, p.stderr
got = json.loads(read("totals.json"))
assert got == {"North, East": 10.35, "South": 5.0, "West": 6.67}, got
''',
        files={"sales.csv": "region,amount\nNorth,10\nSouth,5\nNorth,2.5\n"},
        reference={"aggregate.py": '''import csv, json
tot = {}
with open("sales.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try:
            a = float(row["amount"])
        except (ValueError, TypeError):
            continue
        tot[row["region"]] = tot.get(row["region"], 0.0) + a
with open("totals.json", "w", encoding="utf-8") as f:
    json.dump({k: round(v, 2) for k, v in tot.items()}, f)
'''},
    ),
    Task(
        "train-004-cli-arg-parser", "train",
        "Create cli.py implementing `python cli.py greet NAME [--shout] [--times N]`. It prints `Hello, NAME!` "
        "N times (default 1), one per line; with --shout the line is upper-cased. Missing NAME or an unknown "
        "subcommand must exit with a non-zero status. Use argparse.",
        '''
p = run("cli.py", "greet", "Ann")
assert p.returncode == 0 and p.stdout.splitlines() == ["Hello, Ann!"], (p.stdout, p.stderr)
p = run("cli.py", "greet", "Bo", "--times", "3")
assert p.stdout.splitlines() == ["Hello, Bo!"] * 3, p.stdout
p = run("cli.py", "greet", "Cy", "--shout", "--times", "2")
assert p.stdout.splitlines() == ["HELLO, CY!"] * 2, p.stdout
assert run("cli.py", "greet").returncode != 0, "missing NAME must fail"
assert run("cli.py", "nope", "x").returncode != 0, "unknown subcommand must fail"
''',
        reference={"cli.py": '''import argparse
ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
g = sub.add_parser("greet")
g.add_argument("name")
g.add_argument("--shout", action="store_true")
g.add_argument("--times", type=int, default=1)
a = ap.parse_args()
line = f"Hello, {a.name}!"
if a.shout:
    line = line.upper()
for _ in range(a.times):
    print(line)
'''},
    ),
    Task(
        "train-005-log-error-extractor", "train",
        "app.log has lines `YYYY-MM-DD HH:MM:SS LEVEL component: message`. Create extract.py so that "
        "`python extract.py` writes errors.json: a list, in file order, of objects with keys time (the "
        "`YYYY-MM-DD HH:MM:SS` text), component and message, for ERROR lines only (not WARNING/INFO/ERRORS). "
        "Messages may contain colons. Ignore lines that do not match the format.",
        '''
write("app.log", "2024-01-01 10:00:00 INFO web: started\\n2024-01-01 10:00:05 ERROR db: timeout: after 30s\\nnot a log line\\n2024-01-01 10:01:00 WARNING db: slow\\n2024-01-01 10:02:00 ERROR web: 500 on /x\\n2024-01-01 10:03:00 ERRORS web: nope\\n")
if os.path.exists("errors.json"):
    os.remove("errors.json")
p = run("extract.py")
assert p.returncode == 0, p.stderr
got = json.loads(read("errors.json"))
assert got == [
    {"time": "2024-01-01 10:00:05", "component": "db", "message": "timeout: after 30s"},
    {"time": "2024-01-01 10:02:00", "component": "web", "message": "500 on /x"},
], got
''',
        files={"app.log": "2024-01-01 10:00:00 INFO web: ok\n2024-01-01 10:00:01 ERROR db: down\n"},
        reference={"extract.py": '''import json, re
pat = re.compile(r"^(\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}) (\\w+) ([^:\\s]+): (.*)$")
out = []
with open("app.log", encoding="utf-8") as f:
    for line in f:
        m = pat.match(line.rstrip("\\n"))
        if m and m.group(2) == "ERROR":
            out.append({"time": m.group(1), "component": m.group(3), "message": m.group(4)})
with open("errors.json", "w", encoding="utf-8") as f:
    json.dump(out, f)
'''},
    ),
    Task(
        "train-006-ini-to-json", "train",
        "Create ini2json.py so `python ini2json.py in.ini out.json` converts an INI file to JSON: each section "
        "becomes an object of its keys. Value typing: `true`/`false` (any case) -> boolean, an optional minus "
        "followed by digits -> integer, everything else stays a string. Keep key and section case as written.",
        '''
write("in.ini", "[server]\\nHost = example.org\\nport = 8080\\ndebug = True\\n\\n[db]\\nretries = -3\\nname = x1\\nflag = false\\nratio = 1.5\\n")
p = run("ini2json.py", "in.ini", "out.json")
assert p.returncode == 0, p.stderr
got = json.loads(read("out.json"))
assert got == {"server": {"Host": "example.org", "port": 8080, "debug": True},
               "db": {"retries": -3, "name": "x1", "flag": False, "ratio": "1.5"}}, got
assert type(got["server"]["port"]) is int and got["server"]["debug"] is True
''',
        reference={"ini2json.py": '''import configparser, json, re, sys
cp = configparser.ConfigParser()
cp.optionxform = str
cp.read(sys.argv[1], encoding="utf-8")
def conv(v):
    if v.lower() == "true":
        return True
    if v.lower() == "false":
        return False
    if re.fullmatch(r"-?\\d+", v):
        return int(v)
    return v
with open(sys.argv[2], "w", encoding="utf-8") as f:
    json.dump({s: {k: conv(v) for k, v in cp[s].items()} for s in cp.sections()}, f)
'''},
    ),
    Task(
        "train-007-markdown-toc", "train",
        "Create toc.py with make_toc(md: str) -> str. For every heading of level 1-3 (`#`, `##`, `###` followed "
        "by a space) outside fenced code blocks (``` fences) emit one line: two spaces of indent per level "
        "beyond 1, then `- [Title](#slug)`. slug = title lower-cased, spaces -> `-`, then every character that "
        "is not a letter, digit, `-` or `_` removed. Lines joined with \\n, no trailing newline. "
        "Levels 4+ are ignored.",
        '''
from toc import make_toc
md = "# Intro Section\\ntext\\n```\\n# not a heading\\n```\\n## What's New?\\n#### deep\\n### Sub_part 2\\n#NoSpace\\n"
want = "- [Intro Section](#intro-section)\\n  - [What's New?](#whats-new)\\n    - [Sub_part 2](#sub_part-2)"
assert make_toc(md) == want, repr(make_toc(md))
assert make_toc("no headings") == ""
''',
        reference={"toc.py": '''import re
def make_toc(md):
    out, fence = [], False
    for line in md.splitlines():
        if line.lstrip().startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r"^(#{1,3}) (.+)$", line)
        if not m:
            continue
        lvl, title = len(m.group(1)), m.group(2).strip()
        slug = re.sub(r"[^a-z0-9_\\-]", "", title.lower().replace(" ", "-"))
        out.append("  " * (lvl - 1) + f"- [{title}](#{slug})")
    return "\\n".join(out)
'''},
    ),
    Task(
        "train-008-semver-sorter", "train",
        "Create semver.py with sort_versions(versions) -> new list sorted ascending by Semantic Versioning 2.0 "
        "precedence. Versions look like MAJOR.MINOR.PATCH or MAJOR.MINOR.PATCH-PRERELEASE (dot-separated "
        "identifiers). Numeric fields compare numerically; a pre-release is lower than its release; numeric "
        "pre-release identifiers compare numerically and are lower than alphanumeric ones; alphanumeric ones "
        "compare lexically; with equal prefixes the longer identifier list is greater.",
        '''
from semver import sort_versions
src = ["2.0.0", "1.0.0-rc.1", "1.10.0", "1.0.0", "1.0.0-beta.11", "1.2.0", "1.0.0-alpha", "1.0.0-beta.2", "1.0.0-beta", "1.0.0-alpha.1", "1.0.0-alpha.beta"]
want = ["1.0.0-alpha", "1.0.0-alpha.1", "1.0.0-alpha.beta", "1.0.0-beta", "1.0.0-beta.2", "1.0.0-beta.11", "1.0.0-rc.1", "1.0.0", "1.2.0", "1.10.0", "2.0.0"]
cp = list(src)
assert sort_versions(src) == want, sort_versions(src)
assert src == cp, "input mutated"
''',
        reference={"semver.py": '''import functools
def _pre(a, b):
    for x, y in zip(a, b):
        if x == y:
            continue
        xn, yn = x.isdigit(), y.isdigit()
        if xn and yn:
            return -1 if int(x) < int(y) else 1
        if xn:
            return -1
        if yn:
            return 1
        return -1 if x < y else 1
    return (len(a) > len(b)) - (len(a) < len(b))
def _cmp(a, b):
    core = lambda v: tuple(int(p) for p in v.split("-", 1)[0].split("."))
    ca, cb = core(a), core(b)
    if ca != cb:
        return -1 if ca < cb else 1
    pa = a.split("-", 1)[1].split(".") if "-" in a else None
    pb = b.split("-", 1)[1].split(".") if "-" in b else None
    if pa is None and pb is None:
        return 0
    if pa is None:
        return 1
    if pb is None:
        return -1
    return _pre(pa, pb)
def sort_versions(versions):
    return sorted(versions, key=functools.cmp_to_key(_cmp))
'''},
    ),
    Task(
        "train-009-env-template-check", "train",
        "Create envcheck.py so `python envcheck.py TEMPLATE ACTUAL` compares two dotenv files. Template keys are "
        "the `KEY=...` lines that are not blank and do not start with `#`. Exit 0 if every template key is "
        "present in ACTUAL with a non-empty value (surrounding quotes count as empty if nothing is inside); "
        "otherwise print each missing-or-empty key on its own line, sorted alphabetically, to stdout and exit 1.",
        '''
write("t.env", "# comment\\nDB_URL=\\nAPI_KEY=changeme\\n\\nZED=1\\nNAME=x\\n")
write("ok.env", "DB_URL=postgres://h/db\\nAPI_KEY=abc\\nZED=1\\nNAME='y'\\nEXTRA=1\\n")
write("bad.env", "DB_URL=\\nAPI_KEY=abc\\nNAME=\\"\\"\\n# ZED=1\\n")
p = run("envcheck.py", "t.env", "ok.env")
assert p.returncode == 0 and p.stdout.strip() == "", (p.returncode, p.stdout, p.stderr)
p = run("envcheck.py", "t.env", "bad.env")
assert p.returncode == 1, p.returncode
assert p.stdout.split() == ["DB_URL", "NAME", "ZED"], p.stdout
''',
        reference={"envcheck.py": '''import sys
def parse(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip()] = v.strip().strip("'\\"").strip()
    return d
t, a = parse(sys.argv[1]), parse(sys.argv[2])
missing = sorted(k for k in t if not a.get(k))
for k in missing:
    print(k)
sys.exit(1 if missing else 0)
'''},
    ),
    Task(
        "train-010-sqlite-migration", "train",
        "app.db (SQLite) has table users(id, name, email). Create migrate.py so `python migrate.py` migrates it "
        "in place: add column `active INTEGER NOT NULL DEFAULT 1` and create index `idx_users_email` on "
        "users(email). It must be idempotent (running twice succeeds and changes nothing more) and must keep "
        "all existing rows.",
        '''
for _ in range(2):
    p = run("migrate.py")
    assert p.returncode == 0, p.stderr
con = sqlite3.connect("app.db")
cols = {r[1]: r for r in con.execute("PRAGMA table_info(users)")}
assert "active" in cols, "no active column"
assert cols["active"][2].upper() == "INTEGER" and cols["active"][3] == 1 and str(cols["active"][4]) == "1", cols["active"]
assert [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_users_email'")] == ["idx_users_email"]
idx = [r[2] for r in con.execute("PRAGMA index_info(idx_users_email)")]
assert idx == ["email"], idx
rows = list(con.execute("SELECT id, name, email, active FROM users ORDER BY id"))
assert rows == [(1, "ann", "ann@x.io", 1), (2, "bob", "bob@x.io", 1), (3, "cy", "cy@x.io", 1)], rows
''',
        setup=_seed_users_db,
        reference={"migrate.py": '''import sqlite3
con = sqlite3.connect("app.db")
cols = [r[1] for r in con.execute("PRAGMA table_info(users)")]
if "active" not in cols:
    con.execute("ALTER TABLE users ADD COLUMN active INTEGER NOT NULL DEFAULT 1")
con.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)")
con.commit()
con.close()
'''},
    ),
    Task(
        "train-011-dedup-preserve-order", "train",
        "Create dedup.py with dedup(items) -> new list containing the first occurrence of each item in original "
        "order. Items may be unhashable (lists, dicts); equality is `==`. Do not mutate the input. "
        "Case-sensitive for strings.",
        '''
from dedup import dedup
assert dedup([3, 1, 3, 2, 1]) == [3, 1, 2]
assert dedup([[1], [1], [2], [1]]) == [[1], [2]]
assert dedup([{"a": 1}, {"a": 1}, {"a": 2}]) == [{"a": 1}, {"a": 2}]
assert dedup(["a", "A", "a"]) == ["a", "A"]
assert dedup([]) == []
src = [1, 1, 2]
dedup(src)
assert src == [1, 1, 2], "input mutated"
''',
        reference={"dedup.py": '''def dedup(items):
    out = []
    for it in items:
        if it not in out:
            out.append(it)
    return out
'''},
    ),
    Task(
        "train-012-http-status-summary", "train",
        "access.log has lines `METHOD PATH STATUS DURATIONms` (e.g. `GET /a 200 12ms`). Create summary.py so "
        "`python summary.py` writes summary.json with integer counts for keys 2xx, 3xx, 4xx, 5xx and total "
        "(total = number of valid lines). Lines that are malformed (not exactly 4 fields, non-numeric status, "
        "status outside 200-599) are skipped and not counted.",
        '''
write("access.log", "GET /a 200 12ms\\nPOST /b 404 3ms\\ngarbage\\nGET /c 503 99ms\\nGET /d 301 1ms\\nGET /e abc 5ms\\nGET /f 200 7ms\\nGET /g 700 1ms\\nGET /h 200\\n")
if os.path.exists("summary.json"):
    os.remove("summary.json")
p = run("summary.py")
assert p.returncode == 0, p.stderr
got = json.loads(read("summary.json"))
assert got == {"2xx": 2, "3xx": 1, "4xx": 1, "5xx": 1, "total": 5}, got
''',
        files={"access.log": "GET /a 200 12ms\nGET /b 500 3ms\n"},
        reference={"summary.py": '''import json
c = {"2xx": 0, "3xx": 0, "4xx": 0, "5xx": 0, "total": 0}
with open("access.log", encoding="utf-8") as f:
    for line in f:
        p = line.split()
        if len(p) != 4 or not p[2].isdigit():
            continue
        s = int(p[2])
        if not 200 <= s <= 599:
            continue
        c[f"{s // 100}xx"] += 1
        c["total"] += 1
with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(c, f)
'''},
    ),
]


def heldout_allowed() -> bool:
    return os.environ.get(HELDOUT_ENV) == "1"


def _manifest_ids(split: str) -> List[str]:
    with open(_CONFIG_DIR / f"{split}_manifest.json", encoding="utf-8") as f:
        return list(json.load(f)["task_ids"])


def _all_for_split(split: str) -> Dict[str, Task]:
    if split == "train":
        return {t.task_id: t for t in TRAIN}
    if split == "heldout":
        if not heldout_allowed():
            raise PermissionError(f"held-out split is sealed; set {HELDOUT_ENV}=1 (final evaluation only)")
        from bench.tasks.heldout import HELDOUT  # lazy: never imported during train/evolution
        return {t.task_id: t for t in HELDOUT}
    raise ValueError(f"unknown split: {split!r}")


def list_tasks_for_split(split: str) -> List[Task]:
    """Tasks in manifest order. Raises PermissionError for heldout unless the env gate is set.
    Raises ValueError if the catalog and the manifest disagree."""
    by_id = _all_for_split(split)
    ids = _manifest_ids(split)
    if sorted(ids) != sorted(by_id):
        raise ValueError(f"{split} manifest and catalog differ: {sorted(set(ids) ^ set(by_id))}")
    return [by_id[i] for i in ids]


def get_task(task_id: str) -> Task:
    split = "heldout" if task_id.startswith("heldout-") else "train"
    by_id = _all_for_split(split)
    if task_id not in by_id:
        raise KeyError(task_id)
    return by_id[task_id]
