"""Structural comparison of two EFIT Green-table directories (issue #194).

    PYTHONPATH=$PWD python workflow/efit_tables/compare_tables.py vaft/data/efit /scratch/tables/legacy-39915 \\
        --json compare.json --markdown compare.md

Both directories are read through the same parser (``f90nml`` for
``mhdin.dat``, the record reader in ``vaft.code.efit.efund`` for the
tables), so what is reported is a property of the tables, not of two code
paths.  The comparison is deliberately layered:

1. ``mhdin.dat``: counts, grid, flags; per-element geometry of vessel,
   flux loops and probes (expected identical); PF groups (expected to agree
   on centroid and turns, not on discretization).
2. ``ep`` and the ``gridpc`` record of ``ec``: plasma-only Green functions
   that depend on the grid and the diagnostic positions alone -- if they
   match, the two EFUNDs computed the same thing on the same grid.
3. ``rfcoil`` and ``gridfc``: the F-coil responses, where the PF
   discretization shows.
4. ``rv``: the vessel responses, the discriminator for the A/B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import f90nml
import numpy as np

from vaft.code.efit.efund import MHDIN_NAME, read_fortran_arrays, read_table_manifest, table_identity

_GEOMETRY_KEYS = {
    "vessel": ("rvs", "zvs", "wvs", "hvs", "avs", "avs2"),
    "flux_loops": ("rsi", "zsi"),
    "probes": ("xmp2", "ymp2", "smp2"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _delta(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    if a.shape != b.shape:
        return {"shape_a": list(a.shape), "shape_b": list(b.shape), "comparable": False}
    diff = np.abs(a - b)
    scale = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(scale > 0, diff / scale, 0.0)
    return {
        "comparable": True,
        "n": int(a.size),
        "identical": bool(np.array_equal(a, b)),
        "max_abs": float(diff.max()) if a.size else 0.0,
        "max_rel": float(rel.max()) if a.size else 0.0,
        "rms_rel": float(np.sqrt(np.mean(rel**2))) if a.size else 0.0,
        "argmax_abs": int(diff.argmax()) if a.size else None,
    }


def read_mhdin(directory: Path) -> f90nml.Namelist:
    return f90nml.read(str(directory / MHDIN_NAME))


def compare_mhdin(a: f90nml.Namelist, b: f90nml.Namelist) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ma, mb = a["machinein"], b["machinein"]
    out["counts"] = {
        key: {"a": ma.get(key), "b": mb.get(key), "same": ma.get(key) == mb.get(key)}
        for key in ("nfcoil", "nfsum", "nsilop", "magpri", "necoil", "nesum", "nvesel", "nvsum", "nacoil")
    }
    ia, ib = a["in5"], b["in5"]
    out["in5"] = {
        key: {"a": ia.get(key), "b": ib.get(key), "same": ia.get(key) == ib.get(key)}
        for key in sorted(set(ia) | set(ib))
    }
    ta, tb = a["in3"], b["in3"]
    out["islpfc"] = {"a": ta.get("islpfc", ia.get("islpfc")), "b": tb.get("islpfc", ib.get("islpfc"))}
    out["geometry"] = {}
    for block, keys in _GEOMETRY_KEYS.items():
        out["geometry"][block] = {key: _delta(_array(ta.get(key, [])), _array(tb.get(key, []))) for key in keys}
    out["geometry"]["vessel"]["vsid"] = {
        "same": list(ta.get("vsid", [])) == list(tb.get("vsid", []))
    }
    amp_a = np.mod(_array(ta.get("amp2", [])), 360.0)
    amp_b = np.mod(_array(tb.get("amp2", [])), 360.0)
    out["geometry"]["probes"]["amp2_mod_360"] = _delta(amp_a, amp_b)
    out["geometry"]["probes"]["amp2_spelling"] = {
        "a": sorted(set(_array(ta.get("amp2", [])).tolist())),
        "b": sorted(set(_array(tb.get("amp2", [])).tolist())),
    }
    out["geometry"]["vessel"]["rsisvs"] = {
        **_delta(_array(ta.get("rsisvs", [])), _array(tb.get("rsisvs", []))),
        "a_range": [float(np.min(_array(ta["rsisvs"]))), float(np.max(_array(ta["rsisvs"])))] if "rsisvs" in ta else None,
        "b_range": [float(np.min(_array(tb["rsisvs"]))), float(np.max(_array(tb["rsisvs"])))] if "rsisvs" in tb else None,
    }
    out["pf_groups"] = _compare_pf_groups(ta, tb, int(ma["nfsum"]), int(mb["nfsum"]))
    return out


def _pf_group_table(in3: f90nml.Namelist, nfsum: int) -> list[dict[str, Any]]:
    rf, zf, wf, hf = (_array(in3[key]) for key in ("rf", "zf", "wf", "hf"))
    fcid = np.asarray(in3["fcid"], dtype=int).reshape(-1)
    fcturn = _array(in3["fcturn"])
    turnfc = _array(in3.get("turnfc", [1.0] * nfsum))
    rows = []
    for group in range(1, nfsum + 1):
        mask = fcid == group
        turns = fcturn[mask]
        weight = turns / turns.sum() if turns.sum() else turns
        rows.append(
            {
                "group": group,
                "elements": int(mask.sum()),
                "turns": float(turns.sum()),
                "turnfc": float(turnfc[group - 1]) if turnfc.size >= group else None,
                "r": float(np.dot(weight, rf[mask])),
                "z": float(np.dot(weight, zf[mask])),
                "z_min": float((zf[mask] - hf[mask] / 2).min()),
                "z_max": float((zf[mask] + hf[mask] / 2).max()),
                "r_min": float((rf[mask] - wf[mask] / 2).min()),
                "r_max": float((rf[mask] + wf[mask] / 2).max()),
            }
        )
    return rows


def _compare_pf_groups(ta, tb, nfsum_a: int, nfsum_b: int) -> dict[str, Any]:
    rows_a = _pf_group_table(ta, nfsum_a)
    rows_b = _pf_group_table(tb, nfsum_b)
    paired = []
    for row_a, row_b in zip(rows_a, rows_b):
        paired.append(
            {
                "group": row_a["group"],
                "elements": [row_a["elements"], row_b["elements"]],
                "turns": [row_a["turns"], row_b["turns"]],
                "turns_diff": row_b["turns"] - row_a["turns"],
                "r_centroid": [row_a["r"], row_b["r"]],
                "z_centroid": [row_a["z"], row_b["z"]],
                "centroid_shift_m": float(np.hypot(row_b["r"] - row_a["r"], row_b["z"] - row_a["z"])),
                "z_extent": [[row_a["z_min"], row_a["z_max"]], [row_b["z_min"], row_b["z_max"]]],
                "r_extent": [[row_a["r_min"], row_a["r_max"]], [row_b["r_min"], row_b["r_max"]]],
            }
        )
    return {"nfsum": [nfsum_a, nfsum_b], "groups": paired, "total_turns": [sum(r["turns"] for r in rows_a), sum(r["turns"] for r in rows_b)]}


def _table_records(directory: Path, name: str) -> list[np.ndarray] | None:
    path = directory / name
    if not path.is_file():
        return None
    return read_fortran_arrays(path, int32_records=(0,) if name.startswith("ec") else ())


def compare_records(a_dir: Path, b_dir: Path, name: str, labels: list[str], *, shapes: list[tuple[int, int]] | None = None) -> dict[str, Any]:
    ra = _table_records(a_dir, name)
    rb = _table_records(b_dir, name)
    if ra is None or rb is None:
        return {"present": [ra is not None, rb is not None]}
    out: dict[str, Any] = {"present": [True, True], "records": [len(ra), len(rb)], "by_record": {}}
    for index, label in enumerate(labels):
        if index >= len(ra) or index >= len(rb):
            out["by_record"][label] = {"present": [index < len(ra), index < len(rb)]}
            continue
        delta = _delta(ra[index].astype(float), rb[index].astype(float))
        if shapes and index < len(shapes) and delta.get("comparable") and shapes[index] is not None:
            rows, cols = shapes[index]
            if ra[index].size == rows * cols:
                ma = ra[index].reshape((rows, cols), order="F")
                mb = rb[index].reshape((rows, cols), order="F")
                per_column = []
                for column in range(cols):
                    per_column.append(_delta(ma[:, column], mb[:, column])["max_rel"])
                delta["max_rel_per_column"] = [float(value) for value in per_column]
        out["by_record"][label] = delta
    return out


def inventory(directory: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file():
            files[path.name] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    return {"dir": str(directory), "files": files, "identity": table_identity(directory), "manifest": read_table_manifest(directory)}


def compare(a_dir: Path, b_dir: Path, *, sample_columns: int = 64) -> dict[str, Any]:
    a_dir, b_dir = a_dir.expanduser(), b_dir.expanduser()
    nml_a, nml_b = read_mhdin(a_dir), read_mhdin(b_dir)
    ma, mb = nml_a["machinein"], nml_b["machinein"]
    report: dict[str, Any] = {
        "a": inventory(a_dir),
        "b": inventory(b_dir),
        "mhdin": compare_mhdin(nml_a, nml_b),
    }
    suffixes = sorted({p.name[2:-4] for d in (a_dir, b_dir) for p in d.glob("ec*.ddd")})
    report["tables"] = {}
    for suffix in suffixes:
        nsilop, magpri, nfsum, nvsum, nesum = (int(ma[k]) for k in ("nsilop", "magpri", "nfsum", "nvsum", "nesum"))
        ec = _table_records(a_dir, f"ec{suffix}.ddd")
        nw = nh = None
        if ec:
            nw, nh = (int(v) for v in ec[0][:2])
        nwnh = (nw or 0) * (nh or 0)
        block: dict[str, Any] = {"grid": [nw, nh]}
        block["ep"] = compare_records(a_dir, b_dir, f"ep{suffix}.ddd", ["rsilpc", "rmp2pc"])
        block["ec"] = compare_records(
            a_dir, b_dir, f"ec{suffix}.ddd", ["mw_mh", "rgrid_zgrid", "gridfc", "gridpc"],
            shapes=[None, None, (nwnh, nfsum), None],
        )
        block["rfcoil"] = compare_records(
            a_dir, b_dir, "rfcoil.ddd", ["gsilfc", "gmp2fc"], shapes=[(nsilop, nfsum), (magpri, nfsum)]
        )
        block["brzgfc"] = compare_records(a_dir, b_dir, "brzgfc.dat", ["brgrfc", "bzgrfc"])
        block["rv"] = compare_records(
            a_dir, b_dir, f"rv{suffix}.ddd", ["gsilvs", "gmp2vs", "ggridvs", "gfcvs", "gecvs", "gvsvs"],
            shapes=[None, None, None, (nfsum, nvsum), None, None],
        )
        block["re"] = compare_records(a_dir, b_dir, f"re{suffix}.ddd", ["rsilec", "rmp2ec", "gridec"])
        report["tables"][suffix] = block
    return report


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def markdown(report: dict[str, Any]) -> str:
    lines = ["# Green-table comparison", ""]
    lines.append(f"- A: `{report['a']['dir']}` ({report['a']['identity']['provenance']})")
    lines.append(f"- B: `{report['b']['dir']}` ({report['b']['identity']['provenance']})")
    lines.append("")
    lines.append("## mhdin.dat")
    lines.append("")
    lines.append("| count | A | B |")
    lines.append("| --- | --- | --- |")
    for key, row in report["mhdin"]["counts"].items():
        mark = "" if row["same"] else " **≠**"
        lines.append(f"| {key} | {row['a']} | {row['b']}{mark} |")
    lines.append("")
    lines.append("| &in5 | A | B |")
    lines.append("| --- | --- | --- |")
    for key, row in report["mhdin"]["in5"].items():
        mark = "" if row["same"] else " **≠**"
        lines.append(f"| {key} | {row['a']} | {row['b']}{mark} |")
    lines.append(f"| islpfc (in3) | {report['mhdin']['islpfc']['a']} | {report['mhdin']['islpfc']['b']} |")
    lines.append("")
    lines.append("| geometry | key | n | identical | max abs | max rel |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for block, keys in report["mhdin"]["geometry"].items():
        for key, delta in keys.items():
            if not isinstance(delta, dict) or "max_abs" not in delta:
                continue
            lines.append(
                f"| {block} | {key} | {delta.get('n')} | {delta.get('identical')} | {_fmt(delta.get('max_abs'))} | {_fmt(delta.get('max_rel'))} |"
            )
    probes = report["mhdin"]["geometry"]["probes"]
    lines.append("")
    lines.append(f"Probe angle spelling: A {probes['amp2_spelling']['a']}, B {probes['amp2_spelling']['b']} (mod 360 identical: {probes['amp2_mod_360'].get('identical')}).")
    rs = report["mhdin"]["geometry"]["vessel"]["rsisvs"]
    lines.append(f"rsisvs range: A {rs.get('a_range')}, B {rs.get('b_range')}.")
    lines.append("")
    pf = report["mhdin"]["pf_groups"]
    lines.append(f"## PF groups (nfsum A {pf['nfsum'][0]}, B {pf['nfsum'][1]}; total turns A {pf['total_turns'][0]:.0f}, B {pf['total_turns'][1]:.0f})")
    lines.append("")
    lines.append("| group | elements A/B | turns A/B | Δturns | centroid shift [m] | z extent A | z extent B |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in pf["groups"]:
        za, zb = row["z_extent"]
        lines.append(
            f"| {row['group']} | {row['elements'][0]}/{row['elements'][1]} | {row['turns'][0]:.0f}/{row['turns'][1]:.0f} | {row['turns_diff']:+.0f} | {row['centroid_shift_m']:.4f} | [{za[0]:.3f}, {za[1]:.3f}] | [{zb[0]:.3f}, {zb[1]:.3f}] |"
        )
    for suffix, block in report["tables"].items():
        lines.append("")
        lines.append(f"## Tables at {block['grid'][0]}x{block['grid'][1]}")
        lines.append("")
        lines.append("| file | record | present A/B | n | identical | max abs | max rel | rms rel |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for file_key in ("ep", "ec", "rfcoil", "brzgfc", "rv", "re"):
            entry = block.get(file_key) or {}
            present = entry.get("present", [False, False])
            if not all(present):
                lines.append(f"| {file_key} | – | {present[0]}/{present[1]} | | | | | |")
                continue
            for label, delta in entry.get("by_record", {}).items():
                if "max_abs" in delta:
                    lines.append(
                        f"| {file_key} | {label} | ✓/✓ | {delta['n']} | {delta['identical']} | {_fmt(delta['max_abs'])} | {_fmt(delta['max_rel'])} | {_fmt(delta['rms_rel'])} |"
                    )
                else:
                    lines.append(f"| {file_key} | {label} | {delta.get('present')} | | | | | |")
        for file_key, label in (("rfcoil", "gsilfc"), ("rfcoil", "gmp2fc"), ("ec", "gridfc"), ("rv", "gfcvs")):
            delta = ((block.get(file_key) or {}).get("by_record") or {}).get(label) or {}
            columns = delta.get("max_rel_per_column")
            if columns:
                lines.append("")
                lines.append(f"{file_key}/{label} max relative difference per F-coil group: " + ", ".join(f"{i + 1}: {v:.3g}" for i, v in enumerate(columns)))
    lines.append("")
    lines.append("## Inventory")
    lines.append("")
    lines.append("| file | size A | size B | sha256 A | sha256 B |")
    lines.append("| --- | --- | --- | --- | --- |")
    names = sorted(set(report["a"]["files"]) | set(report["b"]["files"]))
    for name in names:
        fa = report["a"]["files"].get(name)
        fb = report["b"]["files"].get(name)
        lines.append(
            f"| {name} | {fa['size'] if fa else '–'} | {fb['size'] if fb else '–'} | {fa['sha256'][:12] if fa else '–'} | {fb['sha256'][:12] if fb else '–'} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", type=Path, help="table directory A (e.g. the bundled vaft/data/efit)")
    parser.add_argument("b", type=Path, help="table directory B (e.g. a generated table)")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    report = compare(args.a, args.b)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    text = markdown(report)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
