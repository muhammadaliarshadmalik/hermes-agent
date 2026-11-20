"""CLI: python -m bench run --candidate <dir> --split train|heldout [options]"""
from __future__ import annotations

import argparse
import os
import sys

from bench.tasks.catalog import HELDOUT_ENV


def main(argv=None) -> int:
    from bench.runner.executor import BundleError, load_baseline_config, run_split

    cfg = load_baseline_config()
    ap = argparse.ArgumentParser(prog="python -m bench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a candidate bundle on a task split")
    r.add_argument("--candidate", required=True, help="path to candidate bundle directory")
    r.add_argument("--split", required=True, choices=["train", "heldout"])
    r.add_argument("--repeat", type=int, default=cfg["default_repeat"])
    r.add_argument("--seed", type=int, default=cfg["default_seed"])
    r.add_argument("--out", default=None, help="output directory (default bench/runs/<run_id>)")
    r.add_argument("--timeout", type=float, default=cfg["default_timeout_seconds"])
    r.add_argument("--max-steps", type=int, default=cfg["default_max_steps"])
    r.add_argument("--model", default=None)
    r.add_argument("--provider", default=None)
    r.add_argument("--mock", action="store_true", help="offline deterministic run, no model")
    a = ap.parse_args(argv)

    if a.repeat < 1:
        print("error: --repeat must be >= 1", file=sys.stderr)
        return 2
    if a.split == "heldout" and os.environ.get(HELDOUT_ENV) != "1":
        print(f"error: held-out split is sealed; set {HELDOUT_ENV}=1 (final evaluation only)", file=sys.stderr)
        return 2
    if not a.mock and not a.model:
        print("error: --model is required for live runs (no silent default); use --mock for offline", file=sys.stderr)
        return 2
    try:
        s = run_split(a.candidate, a.split, repeat=a.repeat, seed=a.seed, out=a.out, timeout=a.timeout,
                      max_steps=a.max_steps, model=a.model, provider=a.provider, mock=a.mock)
    except PermissionError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except BundleError as e:
        print(f"error: invalid candidate bundle: {e}", file=sys.stderr)
        return 1
    suite = s["aggregate"]["suite"]["pass_rate"]
    print(f"run_id={s['run_id']} candidate={s['candidate_id']} split={s['split']} repeat={s['repeat']} mock={s['mock']}")
    print(f"pass_rate mean={suite['mean']:.3f} std={suite['std']:.3f}")
    flaky = [t for t, v in s["aggregate"]["per_task"].items() if v["flaky"]]
    print(f"flaky_tasks={flaky}")
    print(f"out={s['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
