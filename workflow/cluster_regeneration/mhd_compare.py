#!/usr/bin/env python3
"""Compare two mhd_linear stage products (DCON/RDCON/STRIDE), slices paired by time.

usage: mhd_compare.py <new mhd_linear.json.gz> <reference mhd_linear.json.gz>

Each product stores one ``<solver>`` record per (time slice, n) in
``mhd_linear.code.parameters``. The numeric scalars of every record (child texts and
attributes, e.g. ``msing``, Delta', energies) are compared between the two products on the
slices whose times agree within 2e-5 s; slices present on one side only are listed.
"""

from __future__ import annotations

import gzip
import json
import sys
import xml.etree.ElementTree as ET


def _number(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


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
        for element in solver.iter():
            if element is solver:
                continue
            value = _number((element.text or "").strip())
            if value is not None:
                values[element.tag] = value
            for name, attribute in element.attrib.items():
                value = _number(attribute)
                if value is not None:
                    values[f"{element.tag}@{name}"] = value
        out[(round(times[index], 5), int(solver.get("n_tor", 0)))] = values
    return out


def main(new_path, ref_path):
    new, ref = records(new_path), records(ref_path)
    shared = sorted(set(new) & set(ref))
    print(f"cases new={len(new)} ref={len(ref)} paired={len(shared)}")
    print("only new:", sorted(set(new) - set(ref)))
    print("only ref:", sorted(set(ref) - set(new)))
    worst = {}
    for key in shared:
        a, b = new[key], ref[key]
        for name in set(a) & set(b):
            scale = max(abs(b[name]), 1e-300)
            rel = abs(a[name] - b[name]) / scale
            if rel > worst.get(name, (0.0, None))[0]:
                worst[name] = (rel, key, a[name], b[name])
    for name, (rel, key, a, b) in sorted(worst.items(), key=lambda item: -item[1][0])[:25]:
        print(f"{name:40s} rel={rel:.3e} at t={key[0]} n={key[1]}: {a:.6g} vs {b:.6g}")
    if not worst:
        print("all paired numeric scalars identical")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
