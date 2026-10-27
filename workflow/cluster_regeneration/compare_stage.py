#!/usr/bin/env python3
"""Stage-level comparison of regenerated ODS products against reference products.

usage: compare_stage.py --new-root <filedb> --ref-root <filedb> --shots-file f --out out.json [--stages ...]
For each shot and stage ODS (diagnostics, eddy, efit, impa), groups leaves by their
index-free path ("magnetics.b_field_pol_probe.:.field.data") and records: leaves only in
one side, shape mismatches, exact-equal leaves, differing numeric leaves with the max
relative difference, string differences, and NaN element counts on both sides.
"""
import argparse
import collections
import gzip
import json
import sys
from pathlib import Path
import numpy as np

STAGES = {
    "diagnostics": "omas/diagnostics/{s}/output/diagnostics.json.gz",
    "eddy": "omas/eddy/{s}/output/eddy.json.gz",
    "efit": "omas/efit/magnetic/{s}/output/efit.json.gz",
    "impa": "omas/impa/{s}/output/impa.json.gz",
}


def load(p):
    with gzip.open(p, "rt") as h:
        return json.load(h)


def flatten(obj, path=(), out=None):
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, path + (k,), out)
    elif isinstance(obj, list) and obj and all(isinstance(x, dict) for x in obj):
        for i, v in enumerate(obj):
            flatten(v, path + (i,), out)
    else:
        out[path] = obj
    return out


def group_of(path):
    return ".".join(":" if isinstance(p, int) else str(p) for p in path)


def as_array(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return np.asarray(float(v))
    if isinstance(v, list):
        try:
            a = np.asarray(v, dtype=float)
            return a
        except (ValueError, TypeError):
            return None
    return None


def compare(new, ref):
    fn, fr = flatten(new), flatten(ref)
    G = collections.defaultdict(lambda: collections.Counter())
    maxrel = collections.defaultdict(float)
    maxabs = collections.defaultdict(float)
    for path in set(fn) | set(fr):
        g = group_of(path)
        c = G[g]
        if path not in fr:
            c["only_new"] += 1
            a = as_array(fn[path])
            if a is not None:
                c["nan_new"] += int(np.isnan(a).sum())
                c["n_new"] += a.size
            continue
        if path not in fn:
            c["only_ref"] += 1
            a = as_array(fr[path])
            if a is not None:
                c["nan_ref"] += int(np.isnan(a).sum())
                c["n_ref"] += a.size
            continue
        a, b = as_array(fn[path]), as_array(fr[path])
        if a is None or b is None:
            c["equal" if fn[path] == fr[path] else "str_diff"] += 1
            continue
        c["nan_new"] += int(np.isnan(a).sum())
        c["nan_ref"] += int(np.isnan(b).sum())
        c["n_new"] += a.size
        c["n_ref"] += b.size
        if a.shape != b.shape:
            c["shape_diff"] += 1
            continue
        if np.array_equal(a, b, equal_nan=True):
            c["equal"] += 1
            continue
        c["num_diff"] += 1
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            scale = max(np.nanmax(np.abs(b[both])), 1e-300)
            dabs = float(np.nanmax(np.abs(a[both] - b[both])))
            maxabs[g] = max(maxabs[g], dabs)
            maxrel[g] = max(maxrel[g], dabs / scale)
        c["nan_pattern_diff"] += int((np.isnan(a) != np.isnan(b)).sum())
    return {g: {**dict(c), **({"max_rel": maxrel[g], "max_abs": maxabs[g]} if g in maxrel else {})} for g, c in G.items()}


def match_slices(new, ref, tol=2e-5):
    """Pair equilibrium time slices by time (never by index); compare only the pairs."""
    def slices(o):
        eq = o.get("equilibrium", {})
        slices_, times = eq.get("time_slice", []), list(eq.get("time", []))
        times += [None] * (len(slices_) - len(times))
        return [(float(s.get("time", t)), s) for s, t in zip(slices_, times)]
    sn, sr = slices(new), slices(ref)
    pairs, used = [], set()
    for tn, s in sn:
        best = min(((abs(tn - tr), j) for j, (tr, _) in enumerate(sr) if j not in used), default=None)
        if best and best[0] <= tol:
            used.add(best[1])
            pairs.append((s, sr[best[1]][1]))
    extra = {"slices_new": len(sn), "slices_ref": len(sr), "slices_matched": len(pairs),
             "times_only_new": sorted(round(t * 1e3, 3) for t, _ in sn if all(abs(t - tr) > tol for tr, _ in sr)),
             "times_only_ref": sorted(round(t * 1e3, 3) for t, _ in sr if all(abs(t - tn) > tol for tn, _ in sn))}
    new = {"equilibrium": {"time_slice": [p[0] for p in pairs]}, "code": new.get("equilibrium", {}).get("code", {})}
    ref = {"equilibrium": {"time_slice": [p[1] for p in pairs]}, "code": ref.get("equilibrium", {}).get("code", {})}
    return new, ref, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-root", type=Path, required=True)
    ap.add_argument("--ref-root", type=Path, required=True)
    ap.add_argument("--shots-file", required=True)
    ap.add_argument("--stages", nargs="+", default=list(STAGES))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for s in map(int, open(a.shots_file).read().split()):
        for st in a.stages:
            pn, pr = a.new_root / STAGES[st].format(s=s), a.ref_root / STAGES[st].format(s=s)
            key = f"{s}/{st}"
            if not pn.exists() or not pr.exists():
                res[key] = {"new": pn.exists(), "ref": pr.exists()}
                continue
            try:
                new, ref = load(pn), load(pr)
                extra = {}
                if st == "efit":
                    new, ref, extra = match_slices(new, ref)
                res[key] = {"new": True, "ref": True, **extra, "groups": compare(new, ref)}
            except Exception as e:
                res[key] = {"new": True, "ref": True, "error": repr(e)[:300]}
            print(key, file=sys.stderr, flush=True)
    a.out.write_text(json.dumps(res))


if __name__ == "__main__":
    main()
