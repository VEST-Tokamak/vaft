#!/usr/bin/env python3
"""Compare two NUBEAM Plasma State files, profile by profile.

Usage: compare-plasma-state.py REFERENCE.cdf CANDIDATE.cdf [--changes state_changes.cdf]

Exact agreement is neither expected nor the goal. NUBEAM is a Monte Carlo code:
the distribution function is built from a finite particle sample, so every run
carries statistical noise that is largest where the target bins are smallest --
near the magnetic axis. The shipped reference states were also produced by an
older build of the code. What this script is for is establishing that the
profiles agree in shape and magnitude, which is what would break if the build
were computing the wrong physics.

Two metrics per profile:

  rel_integral  |sum(b) - sum(a)| / |sum(a)|
                Disagreement in the integrated quantity. Monte Carlo noise
                largely cancels in a sum, so this is the number to read for
                total beam power, torque and driven current.

  rel_l2        ||b - a||_2 / ||a||_2
                Point-by-point disagreement. Stays stubbornly large under low
                particle counts even when the integral matches, because it
                sees the per-bin scatter directly.
"""
import argparse
import sys

import numpy as np
from netCDF4 import Dataset

# The quantities a NUBEAM run exists to produce. Reported first and in this
# order; everything else follows in the full table.
HEADLINE = [
    ("pbe", "beam power to electrons"),
    ("pbi", "beam power to ions"),
    ("pbth", "beam power to thermalization"),
    ("nbeami", "fast ion density"),
    ("curbeam", "beam-driven current"),
    ("tqbe", "torque to electrons"),
    ("tqbi", "torque to ions"),
    ("tqbjxb", "JxB torque"),
    ("pfuse", "fusion power to electrons"),
    ("pfusi", "fusion power to ions"),
    ("eperp_beami", "fast ion perpendicular energy"),
    ("epll_beami", "fast ion parallel energy"),
    ("sbedep", "beam electron deposition"),
]


def metrics(a, b):
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        return None, None, "shape %s vs %s" % (a.shape, b.shape)
    finite = np.isfinite(a) & np.isfinite(b)
    if not finite.all():
        a, b = a[finite], b[finite]
    if a.size == 0:
        return None, None, "no finite values"
    sa, sb = a.sum(), b.sum()
    na = np.linalg.norm(a)
    if na == 0.0 and np.linalg.norm(b) == 0.0:
        return 0.0, 0.0, "both identically zero"
    rel_integral = abs(sb - sa) / abs(sa) if sa != 0.0 else float("nan")
    rel_l2 = np.linalg.norm(b - a) / na if na != 0.0 else float("nan")
    return rel_integral, rel_l2, ""


def fmt(x):
    if x is None:
        return "     -"
    if not np.isfinite(x):
        return "    --"
    return "%6.2f%%" % (100.0 * x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--changes", help="state_changes.cdf; restricts the full "
                                      "table to variables NUBEAM actually wrote")
    args = ap.parse_args()

    ref = Dataset(args.reference)
    cand = Dataset(args.candidate)

    written = None
    if args.changes:
        try:
            written = set(Dataset(args.changes).variables)
        except OSError as exc:
            print("warning: could not read %s (%s)" % (args.changes, exc),
                  file=sys.stderr)

    shared = [
        name for name in ref.variables
        if name in cand.variables
        and ref.variables[name].dtype.kind == "f"
        and ref.variables[name].size > 1
    ]
    if written is not None:
        shared = [n for n in shared if n in written]

    print("reference: %s" % args.reference)
    print("candidate: %s" % args.candidate)
    for label, tag in (("reference", ref), ("candidate", cand)):
        try:
            version = "".join(
                c.decode() if isinstance(c, bytes) else str(c)
                for c in tag.variables["version_id"][:]
            ).strip()
            print("  %s schema version: %s" % (label, version))
        except (KeyError, IndexError):
            pass
    print()

    rows = []
    for name in shared:
        rel_integral, rel_l2, note = metrics(
            ref.variables[name][:], cand.variables[name][:]
        )
        rows.append((name, rel_integral, rel_l2, note))
    by_name = {r[0]: r for r in rows}

    header = "%-18s %9s %9s   %s" % ("profile", "integral", "L2", "")
    print("NUBEAM output profiles")
    print(header)
    print("-" * 60)
    headline_seen = set()
    for name, description in HEADLINE:
        row = by_name.get(name)
        if row is None:
            continue
        headline_seen.add(name)
        print("%-18s %9s %9s   %s%s"
              % (name, fmt(row[1]), fmt(row[2]), description,
                 (" [%s]" % row[3]) if row[3] else ""))

    rest = [r for r in rows if r[0] not in headline_seen]
    # Worst disagreement first: that is where a real defect would show.
    rest.sort(key=lambda r: (-(r[1] if r[1] is not None and np.isfinite(r[1]) else -1)))
    if rest:
        print()
        print("other profiles (worst integral disagreement first)")
        print(header)
        print("-" * 60)
        for name, rel_integral, rel_l2, note in rest:
            print("%-18s %9s %9s   %s"
                  % (name, fmt(rel_integral), fmt(rel_l2), note))

    finite = [r[1] for r in rows if r[1] is not None and np.isfinite(r[1])]
    if finite:
        print()
        print("%d profiles compared; median integral disagreement %.2f%%, "
              "worst %.2f%%"
              % (len(finite), 100.0 * float(np.median(finite)),
                 100.0 * max(finite)))


if __name__ == "__main__":
    main()
