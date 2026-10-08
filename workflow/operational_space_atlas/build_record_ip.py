"""Plasma-current waveforms of the atlas's record discharges (#1456, conference 2026-10-12).

The operational-space atlas stars a handful of record discharges (the largest current,
the longest pulse, the highest beta_N, ...). This script extracts ``magnetics.ip`` --
the shot-era calibrated inner Rogowski signal the diagnostics stage writes -- for those
shots from the FileDB diagnostics products, so the notebook can draw their I_p(t) in
the stars' colours (and the representative trajectory's discharge) without reading the 40-50 MB products itself.

Read-only on the FileDB: it opens ``omas/diagnostics/<shot>/output/diagnostics.json.gz``
and writes nothing outside ``--out``, where it puts ``record_ip.csv`` (shot, time_s,
ip_A) and ``record_ip_MANIFEST.json`` (each product's sha256 and the signal's
``method_name``). A shot without a diagnostics product is listed as missing, not
filled in.

With ``--onset`` it also writes ``record_onsets.csv``: the plasma window of each shot from
:func:`vaft.omas.plasma_timing.plasma_timing` on the same product (H-alpha by label is
authoritative, the plasma-current pulse the fallback), with the source that answered, the
light/current agreement and the flags, so the notebook can align the waveforms at onset.

    python build_record_ip.py --shots 44801 41664 44740 42963 40325 42986 39915 39917 42962 39916 \\
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


def plasma_window(path: Path) -> dict:
    """The shared plasma window of one diagnostics product, with its provenance; never a guess."""
    from vaft.omas import load
    from vaft.omas.plasma_timing import plasma_timing

    try:
        timing = plasma_timing(load(path))
    except Exception as exc:  # noqa: BLE001 -- recorded per shot, the rest still run
        return {"onset_s": "", "offset_s": "", "source": "", "agreement": "",
                "onset_delta_s": "", "flags": f"{type(exc).__name__}: {exc}"[:200]}
    return {"onset_s": timing.onset if timing.found else "", "offset_s": timing.offset if timing.found else "",
            "source": timing.source or "", "agreement": timing.agreement,
            "onset_delta_s": "" if timing.onset_delta_s is None else timing.onset_delta_s,
            "flags": ";".join(timing.flags) if timing.found else (timing.fallback_reason or "no plasma found")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shots", type=int, nargs="+", required=True)
    parser.add_argument("--filedb", type=Path, default=Path("/srv/vest.filedb"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--onset", action="store_true",
                        help="also write record_onsets.csv from vaft.omas.plasma_timing (needs vaft)")
    args = parser.parse_args(argv)

    out = args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows, products, missing, onsets = [], {}, [], []
    for shot in args.shots:
        path = product_path(args.filedb, shot)
        if not path.is_file():
            missing.append(shot)
            continue
        time, data, method = read_ip(path)
        rows += [(shot, t, v) for t, v in zip(time, data)]
        if args.onset:
            onsets.append({"shot": shot, **plasma_window(path)})
        products[str(shot)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "samples": len(time), "method_name": method}
    with open(out / "record_ip.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["shot", "time_s", "ip_A"])
        writer.writerows(rows)
    manifest = {"generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "signal": "magnetics.ip[0]",
                "filedb": str(args.filedb), "shots": args.shots, "missing": missing, "products": products}
    if args.onset:
        import vaft

        with open(out / "record_onsets.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["shot", "onset_s", "offset_s", "source", "agreement",
                                                       "onset_delta_s", "flags"])
            writer.writeheader()
            writer.writerows(onsets)
        manifest["onsets"] = {"function": "vaft.omas.plasma_timing.plasma_timing", "vaft": vaft.__version__,
                              "vaft_file": vaft.__file__}
    (out / "record_ip_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(products)} shot(s) written to {out / 'record_ip.csv'}; missing: {missing or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
