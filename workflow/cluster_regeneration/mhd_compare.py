#!/usr/bin/env python3
"""Compare two mhd_linear stage products (DCON/RDCON/STRIDE), slices paired by time.

usage: mhd_compare.py <new mhd_linear.json.gz> <reference mhd_linear.json.gz>

Each product stores one ``<solver>`` record per (time slice, n) in
``mhd_linear.code.parameters``. Every numeric scalar of a record (element texts and attributes,
keyed by their indexed path inside the record, e.g. ``surface[3]/delta_prime@real``) is compared
between the two products for the (slice, n) cases whose times agree within 2e-5 s (nearest
match). Cases and scalar names present on one side only are listed; a NaN on one side only
counts as a difference.
"""

from __future__ import annotations

import gzip
import json
import math
import sys
import xml.etree.ElementTree as ET

TOLERANCE_S = 2e-5


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _scalars(element, prefix, out):
    seen = {}
    for child in element:
        seen[child.tag] = seen.get(child.tag, -1) + 1
        path = f"{prefix}{child.tag}[{seen[child.tag]}]"
        value = _number((child.text or "").strip())
        if value is not None:
            out[path] = value
        for name, attribute in child.attrib.items():
            value = _number(attribute)
            if value is not None:
                out[f"{path}@{name}"] = value
        _scalars(child, path + "/", out)


def records(path):
    with gzip.open(path, "rt") as handle:
        product = json.load(handle)["mhd_linear"]
    times = [float(t) for t in product.get("time", [])]
    out = {}
    root = ET.fromstring(product.get("code", {}).get("parameters", "<parameters/>"))
    for solver in root.iter("solver"):
        index = int(solver.get("time_slice", -1))
        if not 0 <= index < len(times):
            continue
        values = {}
        _scalars(solver, "", values)
        out[(times[index], int(solver.get("n_tor", 0)))] = values
    return out


def pair(new, ref):
    pairs, used = [], set()
    for t, n in sorted(new):
        candidates = [(abs(t - tr), (tr, nr)) for tr, nr in ref if nr == n and (tr, nr) not in used]
        if candidates:
            gap, key = min(candidates)
            if gap <= TOLERANCE_S:
                used.add(key)
                pairs.append(((t, n), key))
    return pairs


def main(new_path, ref_path):
    new, ref = records(new_path), records(ref_path)
    pairs = pair(new, ref)
    paired_new = {a for a, _ in pairs}
    paired_ref = {b for _, b in pairs}
    print(f"cases new={len(new)} ref={len(ref)} paired={len(pairs)}")
    print("only new:", sorted((round(t, 6), n) for t, n in set(new) - paired_new))
    print("only ref:", sorted((round(t, 6), n) for t, n in set(ref) - paired_ref))
    worst, one_sided = {}, set()
    for key_new, key_ref in pairs:
        a, b = new[key_new], ref[key_ref]
        one_sided |= set(a) ^ set(b)
        for name in set(a) & set(b):
            x, y = a[name], b[name]
            if math.isnan(x) or math.isnan(y):
                rel = 0.0 if (math.isnan(x) and math.isnan(y)) else math.inf
            else:
                rel = abs(x - y) / max(abs(y), 1e-300) if x != y else 0.0
            if rel > worst.get(name, (0.0,))[0]:
                worst[name] = (rel, key_new, x, y)
    for name, (rel, key, x, y) in sorted(worst.items(), key=lambda item: -item[1][0])[:25]:
        print(f"{name:48s} rel={rel:.3e} at t={key[0]:.6f} n={key[1]}: {x:.6g} vs {y:.6g}")
    if one_sided:
        print(f"{len(one_sided)} scalar names on one side only, e.g. {sorted(one_sided)[:5]}")
    if not worst and not one_sided:
        print("all paired numeric scalars identical")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
