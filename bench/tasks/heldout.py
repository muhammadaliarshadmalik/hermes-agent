"""Sealed held-out tasks. Imported ONLY when HERMES_BENCH_ALLOW_HELDOUT=1."""
from bench.tasks.catalog import Task

HELDOUT = [
    Task(
        "heldout-001-interval-merger", "heldout",
        "Create intervals.py with merge(intervals) taking a list of [start, end] pairs (start <= end, unsorted, "
        "possibly empty) and returning a new sorted list of merged [start, end] lists. Overlapping AND touching "
        "intervals (end == next start) merge. Do not mutate the input.",
        '''
from intervals import merge
src = [[8, 10], [1, 3], [2, 6], [15, 18], [6, 7], [17, 20]]
cp = [list(x) for x in src]
assert merge(src) == [[1, 7], [8, 10], [15, 20]], merge(src)
assert src == cp, "input mutated"
assert merge([]) == []
assert merge([[5, 5]]) == [[5, 5]]
assert merge([[1, 10], [2, 3]]) == [[1, 10]]
assert merge([[1, 2], [3, 4]]) == [[1, 2], [3, 4]]
''',
        reference={"intervals.py": '''def merge(intervals):
    out = []
    for s, e in sorted([list(i) for i in intervals]):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
'''},
    ),
    Task(
        "heldout-002-nested-config-diff", "heldout",
        "Create confdiff.py with diff(a, b) comparing two nested dicts. Return {'added': {...}, 'removed': {...}, "
        "'changed': {...}} where keys are dotted paths (e.g. 'db.port'). added maps path -> new value, removed "
        "maps path -> old value, changed maps path -> [old, new]. Recurse into dicts only; lists and scalars are "
        "compared with == as leaves. If a path is a dict on one side and a scalar on the other, report it as changed.",
        '''
from confdiff import diff
a = {"db": {"host": "h", "port": 1, "opts": {"x": 1}}, "tags": [1, 2], "gone": True, "same": 0, "t": {"k": 1}}
b = {"db": {"host": "h", "port": 2, "opts": {"x": 1, "y": 2}}, "tags": [1, 3], "new": {"n": 1}, "same": 0, "t": 5}
want = {"added": {"db.opts.y": 2, "new": {"n": 1}}, "removed": {"gone": True},
        "changed": {"db.port": [1, 2], "tags": [[1, 2], [1, 3]], "t": [{"k": 1}, 5]}}
assert diff(a, b) == want, diff(a, b)
assert diff({}, {}) == {"added": {}, "removed": {}, "changed": {}}
''',
        reference={"confdiff.py": '''def diff(a, b):
    res = {"added": {}, "removed": {}, "changed": {}}
    def walk(x, y, pre):
        for k in set(x) | set(y):
            p = f"{pre}.{k}" if pre else k
            if k not in x:
                res["added"][p] = y[k]
            elif k not in y:
                res["removed"][p] = x[k]
            elif isinstance(x[k], dict) and isinstance(y[k], dict):
                walk(x[k], y[k], p)
            elif x[k] != y[k]:
                res["changed"][p] = [x[k], y[k]]
    walk(a, b, "")
    return res
'''},
    ),
    Task(
        "heldout-003-rate-limiter-bug", "heldout",
        "limiter.py has a sliding-window rate limiter with bugs. Spec: RateLimiter(limit, window).allow(key, now) "
        "returns True if fewer than `limit` allowed requests for that key happened within the last `window` "
        "seconds, and records the request; otherwise returns False and records nothing. A request made at time t "
        "stops counting when now - t >= window. Keys are independent. Fix the bugs without changing the API.",
        '''
from limiter import RateLimiter
r = RateLimiter(2, 10)
seq = [("a", 0, True), ("a", 1, True), ("a", 2, False), ("a", 9.99, False), ("a", 10, True),
       ("a", 10.5, False), ("a", 11, True), ("a", 11.5, False), ("b", 2, True)]
for key, now, want in seq:
    got = r.allow(key, now)
    assert got is want, (key, now, want, got)
''',
        files={"limiter.py": '''class RateLimiter:
    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.hits = {}

    def allow(self, key, now):
        q = self.hits.setdefault(key, [])
        while q and now - q[0] > self.window:
            q.pop(0)
        if len(q) > self.limit:
            return False
        q.append(now)
        return True
'''},
        reference={"limiter.py": '''class RateLimiter:
    def __init__(self, limit, window):
        self.limit = limit
        self.window = window
        self.hits = {}

    def allow(self, key, now):
        q = self.hits.setdefault(key, [])
        while q and now - q[0] >= self.window:
            q.pop(0)
        if len(q) >= self.limit:
            return False
        q.append(now)
        return True
'''},
    ),
    Task(
        "heldout-004-dependency-cycle", "heldout",
        "Create deps.py with find_cycle(graph) where graph maps node -> list of nodes it depends on (nodes that "
        "appear only as dependencies are allowed). Return None if acyclic. Otherwise return one cycle as a list "
        "of nodes whose first and last element are equal and where every consecutive pair is an edge in the "
        "graph (e.g. ['a','b','a']). A self-dependency is a cycle (['a','a']).",
        '''
from deps import find_cycle
def valid(g, c):
    return isinstance(c, list) and len(c) >= 2 and c[0] == c[-1] and all(b in g.get(a, []) for a, b in zip(c, c[1:]))
assert find_cycle({"a": ["b"], "b": ["c"], "c": []}) is None
assert find_cycle({}) is None
assert find_cycle({"a": ["b", "c"], "b": ["c"], "c": ["d"]}) is None
for g in ({"a": ["b"], "b": ["c"], "c": ["a"]}, {"a": ["a"]}, {"x": ["y"], "y": [], "a": ["b"], "b": ["c"], "c": ["b"]},
          {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": ["a"]}):
    c = find_cycle(g)
    assert valid(g, c), (g, c)
''',
        reference={"deps.py": '''def find_cycle(graph):
    state, stack = {}, []
    def dfs(n):
        state[n] = 1
        stack.append(n)
        for m in graph.get(n, []):
            if state.get(m) == 1:
                return stack[stack.index(m):] + [m]
            if state.get(m) is None:
                c = dfs(m)
                if c:
                    return c
        stack.pop()
        state[n] = 2
        return None
    for n in list(graph):
        if state.get(n) is None:
            c = dfs(n)
            if c:
                return c
    return None
'''},
    ),
]
