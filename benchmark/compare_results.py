#!/usr/bin/env python3
# Copyright 2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Developer: Alişah Özcan

"""Compare two nvbench JSON results of benchmark_merge_ntt.

States are matched by benchmark name and axis values; the mean GPU time of
each state is compared. A state counts as slower (or faster) only when the
change exceeds both the threshold and the measured noise of the two runs.
Exits with status 1 if any state is slower.

Usage: compare_results.py before.json after.json [--threshold 3] [--all]
                          [--skip AXIS=VALUE ...]
"""

import argparse
import json
import math
import sys

TIME_TAG = "nv/cold/time/gpu/mean"
NOISE_TAG = "nv/cold/time/gpu/stdev/relative"


def summary_value(state, tag):
    for summary in state.get("summaries") or []:
        if summary.get("tag") == tag:
            for item in summary.get("data", []):
                if item.get("name") == "value":
                    return float(item["value"])
    return None


def load(path):
    with open(path) as f:
        data = json.load(f)
    results = {}
    for bench in data["benchmarks"]:
        for state in bench["states"]:
            if state.get("is_skipped"):
                continue
            time = summary_value(state, TIME_TAG)
            if time is None:
                continue
            axes = tuple(
                (axis["name"], str(axis["value"]))
                for axis in state.get("axis_values") or []
            )
            noise = summary_value(state, NOISE_TAG) or 0.0
            results[(bench["name"], axes)] = (time, noise)
    return results


def state_name(key):
    bench, axes = key
    return bench + " " + " ".join(f"{name}={value}" for name, value in axes)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="minimum change in percent (default: 3)")
    parser.add_argument("--all", action="store_true",
                        help="print every state, not only the changed ones")
    parser.add_argument("--skip", action="append", default=[],
                        metavar="AXIS=VALUE",
                        help="ignore states with this axis value "
                             "(e.g. Workload=single); can be repeated")
    args = parser.parse_args()

    before = load(args.before)
    after = load(args.after)
    skipped = {tuple(item.split("=", 1)) for item in args.skip}
    common = sorted(key for key in before.keys() & after.keys()
                    if not skipped & set(key[1]))

    rows = []
    log_ratios = {}
    for key in common:
        (t0, n0), (t1, n1) = before[key], after[key]
        change = (t1 / t0 - 1.0) * 100.0
        noise = (n0 + n1) * 100.0
        if change > max(args.threshold, noise):
            status = "SLOWER"
        elif -change > max(args.threshold, noise):
            status = "faster"
        else:
            status = "same"
        rows.append((change, status, key, t0, t1, noise))
        log_ratios.setdefault(key[0], []).append(math.log(t1 / t0))

    rows.sort(key=lambda row: -row[0])
    for change, status, key, t0, t1, noise in rows:
        if args.all or status != "same":
            print(f"{status:>6} {change:+7.2f}% (noise {noise:5.2f}%) "
                  f"{t0 * 1e6:10.3f} us -> {t1 * 1e6:10.3f} us  "
                  f"{state_name(key)}")

    print()
    for bench, values in sorted(log_ratios.items()):
        geomean = (math.exp(sum(values) / len(values)) - 1.0) * 100.0
        print(f"{bench}: {len(values)} states, geometric mean change "
              f"{geomean:+.2f}%")

    slower = sum(1 for row in rows if row[1] == "SLOWER")
    faster = sum(1 for row in rows if row[1] == "faster")
    missing = len(before.keys() ^ after.keys())
    print(f"{len(rows)} states compared: {slower} slower, {faster} faster"
          + (f", {missing} not present in both runs" if missing else ""))
    return 1 if slower else 0


if __name__ == "__main__":
    sys.exit(main())
