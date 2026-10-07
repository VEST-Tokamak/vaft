"""Plasma-current waveforms of the atlas's record discharges (#1456, conference 2026-10-12).

The operational-space atlas stars a handful of record discharges (the largest current,
the longest pulse, the highest beta_N, ...). This script extracts ``magnetics.ip`` --
the shot-era calibrated inner Rogowski signal the diagnostics stage writes -- for those
shots from the FileDB diagnostics products, so the notebook can draw their I_p(t) in
the stars' colours without reading the 40-50 MB products itself.

Read-only on the FileDB: it opens ``omas/diagnostics/<shot>/output/diagnostics.json.gz``
and writes nothing outside ``--out``, where it puts ``record_ip.csv`` (shot, time_s,
ip_A) and ``record_ip_MANIFEST.json`` (each product's sha256 and the signal's
``method_name``). A shot without a diagnostics product is listed as missing, not
filled in.

    python build_record_ip.py --shots 44801 41664 44740 42963 40325 42986 39915 39917 42962 \\
        --filedb /srv/vest.filedb --out ~/runs/campaign/atlas/lane_v
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import gzip
import hashlib
import json
from pathlib import Path


def product_path(filedb: Path, shot: int) -> Path:
    return filedb / "omas" / "diagnostics" / str(shot) / "output" / "diagnostics.json.gz"


def read_ip(path: Path):
    """``(time [s], ip [A], method_name)`` of the first ``magnetics.ip`` channel."""
    with gzip.open(path, "rt") as handle:
        ods = json.load(handle)
    channel = ods["magnetics"]["ip"][0]
    time, data = channel["time"], channel["data"]
    if len(time) != len(data):
        raise ValueError(f"{path}: magnetics.ip time ({len(time)}) and data ({len(data)}) lengths differ")
    return time, data, channel.get("method_name", "")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--filedb", type=Path, default=Path("/srv/vest.filedb"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows, products, missing = [], {}, []
    for shot in args.shots:
        path = product_path(args.filedb, shot)
        if not path.is_file():
            missing.append(shot)
            continue
        time, data, method = read_ip(path)
        rows += [(shot, t, v) for t, v in zip(time, data)]
        products[str(shot)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "samples": len(time), "method_name": method}
    with open(out / "record_ip.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot", "time_s", "ip_A"])
        writer.writerows(rows)
    manifest = {"generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "signal": "magnetics.ip[0]",
                "filedb": str(args.filedb), "shots": args.shots, "missing": missing, "products": products}
    (out / "record_ip_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(products)} shot(s) written to {out / 'record_ip.csv'}; missing: {missing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
