#!/usr/bin/env python3
"""Compare two NUBEAM Plasma State files, profile by profile.

Usage: compare-plasma-state.py REFERENCE.cdf CANDIDATE.cdf [--changes state_changes.cdf]
                               [--tolerance FRACTION]

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

Verdict (exit status 1 on any of these, 0 otherwise):

  * fewer than --min-profiles (default 10) of the 13 headline profiles could
    be compared -- a candidate that lacks most of them is not judged on the
    few it has;
  * any headline profile that could not be compared at all: shape mismatch,
    no finite values, or a reference of zero against a non-zero candidate;
  * the median rel_integral over the headline profiles exceeds --tolerance
    (default 0.25) -- the median, because single profiles are noise-dominated
    at these particle counts (VALIDATION.md records tqbjxb at 84% against a
    75% seed-to-seed floor) while a build computing the wrong physics shifts
    most of them at once; the recorded references sit at a 2-4% median;
  * any of the profiles VALIDATION.md found resolved above the noise (RESOLVED
    below) exceeds --ceiling (default 0.5) -- so half the headline cannot be
    100% wrong behind a good median.

Profiles that are identically zero in both files (pfuse/pfusi in a non-DT
case) carry no information and are left out of the count and the median.
Masked values -- netCDF fill, ~1e37 -- are excluded, never summed.
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

#: The headline profiles VALIDATION.md found resolved above their seed-to-seed
#: noise floor on both reference cases (pbe 0.3-1.9%, nbeami ~5%, sbedep ~4%,
#: curbeam 5-9%, pbi/eperp_beami ~10%). A large disagreement in one of these
#: is evidence, not scatter, so each is held to --ceiling on its own; the rest
#: (pbth, tqbe, tqbi, tqbjxb, pfuse, pfusi) only enter through the median.
RESOLVED = ("pbe", "pbi", "nbeami", "curbeam", "eperp_beami", "epll_beami", "sbedep")


#: Anything this large in a Plasma State is a fill value, not a quantity. netCDF
#: masks them on read; a plain float array of the same file would not, and the
#: default fill (9.97e36) then dominates every sum it enters.
FILL_MAGNITUDE = 1e30


def _values(x):
    """A float array with masked, non-finite and fill entries turned into NaN."""
    x = np.ma.masked_invalid(np.ma.asarray(x, dtype=float)).ravel()
    x = np.ma.masked_where(np.abs(np.ma.filled(x, 0.0)) > FILL_MAGNITUDE, x)
    return np.ma.filled(x, np.nan)


def metrics(a, b):
    a = _values(a)
    b = _values(b)
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
    # A reference that sums to zero against a candidate that does not is an
    # unbounded disagreement, not an undefined one: report it as infinite so
    # the verdict sees it rather than dropping the row.
    if sa != 0.0:
        rel_integral = abs(sb - sa) / abs(sa)
    else:
        rel_integral = 0.0 if sb == 0.0 else float("inf")
    rel_l2 = np.linalg.norm(b - a) / na if na != 0.0 else float("inf")
    return rel_integral, rel_l2, ""


def verdict(rows, tolerance, ceiling, min_profiles):
    """(ok, reason) for the headline rows; `reason` names the first failure."""
    names = {name for name, _ in HEADLINE}
    headline = [row for row in rows if row[0] in names]
    informative = [row for row in headline if row[3] != "both identically zero"]
    for name, rel_integral, _, note in informative:
        if rel_integral is None or not np.isfinite(rel_integral):
            return False, "%s could not be compared%s" % (name, (" (%s)" % note) if note else "")
    values = [rel_integral for _, rel_integral, _, _ in informative]
    if len(values) < min_profiles:
        return False, ("only %d of %d headline profiles could be compared; %d are required"
                       % (len(values), len(HEADLINE), min_profiles))
    median = float(np.median(values))
    if median > tolerance:
        return False, ("headline median integral disagreement %.2f%% exceeds the tolerance of %.2f%%"
                       % (100.0 * median, 100.0 * tolerance))
    for name, rel_integral, _, _ in informative:
        if name in RESOLVED and rel_integral > ceiling:
            return False, ("%s disagrees by %.2f%%, above the %.2f%% ceiling for a profile "
                           "resolved above the noise" % (name, 100.0 * rel_integral, 100.0 * ceiling))
    return True, ("headline median integral disagreement %.2f%% is within %.2f%% "
                  "over %d profiles" % (100.0 * median, 100.0 * tolerance, len(values)))


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
    ap.add_argument("--tolerance", type=float, default=0.25,
                    help="exit 1 when the median integral disagreement over the "
                         "headline profiles exceeds this fraction (default 0.25)")
    ap.add_argument("--ceiling", type=float, default=0.5,
                    help="exit 1 when a profile resolved above the noise (%s) "
                         "disagrees by more than this fraction (default 0.5)"
                         % ", ".join(RESOLVED))
    ap.add_argument("--min-profiles", type=int, default=10,
                    help="exit 1 when fewer headline profiles than this could be "
                         "compared (default 10 of %d)" % len(HEADLINE))
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

    ok, reason = verdict(rows, args.tolerance, args.ceiling, args.min_profiles)
    if not ok:
        print("FAIL: " + reason, file=sys.stderr)
        return 1
    print(reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
