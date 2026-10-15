#!/usr/bin/env python3
"""Summarize evaluation runs: mean +/- std across evaluation seeds.

Reads the validation rollouts of one run directory per evaluation seed and
prints mean@N and unbiased pass@k per benchmark and overall, each as
mean +/- population std across seeds.

Usage:
  summarize_eval.py RUN_DIR [RUN_DIR ...] [--n 16] [--pass-at-ks 1,2,4,8]

where each RUN_DIR contains rollout_details/val_step_0.jsonl(.gz).
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "third_party" / "verl"))

try:
    from verl.utils.canonical_eval import aggregate_rollouts, validate_eval_provenance
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "canonical_eval", REPO_ROOT / "third_party/verl/verl/utils/canonical_eval.py")
    _module = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_module)
    aggregate_rollouts = _module.aggregate_rollouts
    validate_eval_provenance = _module.validate_eval_provenance


def _load(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _find_dump(run_dir: Path) -> Path:
    for name in ("val_step_0.jsonl", "val_step_0.jsonl.gz"):
        candidate = run_dir / "rollout_details" / name
        if candidate.exists():
            return candidate
    raise SystemExit(f"no val_step_0 dump under {run_dir}")


def _fingerprint(rows: list[dict], summary: dict, n: int) -> dict:
    """Everything that must agree before two runs may be averaged together."""
    benchmarks = {name: int(values["questions"]) for name, values in summary["benchmarks"].items()}
    return {
        **validate_eval_provenance(rows, n=n),
        "benchmarks": tuple(sorted(benchmarks.items())),
        "rows": len(rows),
        "n": n,
    }


def _require_comparable(fingerprints: dict[str, dict]) -> None:
    if len(fingerprints) < 2:
        return
    reference_name, reference = next(iter(fingerprints.items()))
    problems = []
    seeds = [fingerprint["base_seed"] for fingerprint in fingerprints.values()]
    if len(seeds) != len(set(seeds)):
        raise SystemExit("duplicate evaluation seeds; each run must use a distinct base_seed")
    for name, fingerprint in fingerprints.items():
        for field, value in fingerprint.items():
            if field == "base_seed":
                continue
            if value != reference[field]:
                problems.append(f"  {field}: {reference_name}={reference[field]!r} vs {name}={value!r}")
    if not problems:
        return
    message = ("runs are not comparable; refusing to average them:\n" + "\n".join(problems))
    raise SystemExit(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--pass-at-ks", type=str, default="1,2,4,8")
    parser.add_argument("--allow-subset", action="store_true",
                        help="aggregate a subset of the benchmarks")
    args = parser.parse_args()
    ks = [int(k) for k in args.pass_at_ks.split(",") if k.strip()]

    summaries = []
    fingerprints = {}
    for run_dir in args.run_dirs:
        run_dir = run_dir.resolve()
        run_name = str(run_dir)
        if run_name in fingerprints:
            raise SystemExit(f"duplicate run directory: {run_dir}")
        rows = _load(_find_dump(run_dir))
        summary = aggregate_rollouts(
            rows, n=args.n, require_math_benchmarks=not args.allow_subset, pass_at_ks=ks)
        summaries.append((run_name, summary))
        fingerprints[run_name] = _fingerprint(rows, summary, args.n)
        print(f"[loaded] {run_dir} ({len(rows)} rows)")

    _require_comparable(fingerprints)

    metrics = [f"mean@{args.n}"] + [f"pass@{k}" for k in ks]
    scopes = ["overall"] + sorted(summaries[0][1]["benchmarks"].keys())
    print(f"\nseeds/runs: {[name for name, _ in summaries]}")
    header = "scope      " + "".join(f"{m:>22}" for m in metrics)
    print(header)
    for scope in scopes:
        cells = []
        for metric in metrics:
            values = []
            for _, summary in summaries:
                source = summary["overall"] if scope == "overall" else summary["benchmarks"][scope]
                values.append(float(source[metric]) * 100.0)
            mean = statistics.mean(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            cells.append(f"{mean:6.2f} +/- {std:4.2f}   ")
        print(f"{scope:10s} " + "".join(f"{c:>22}" for c in cells))


if __name__ == "__main__":
    main()
