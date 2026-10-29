#!/usr/bin/env python3
"""Merge the sharded comparison tables into the notebook's checked-in snapshot.

``build_comparison_tables.py`` writes one CSV per check and shard under each
run directory. This script concatenates the shards, drops the duplicate rows a
resumed shard can leave, reduces the per-channel vacuum table to one row per
source, shot, wall model and channel family, and writes gzip CSVs plus
``snapshot_manifest.json`` (row counts, SHA-256, sources) to the notebook data
directory.

Usage::

    python workflow/pipeline_improvement/make_snapshot.py \\
        --tables ~/runs/pipeline_improvement/tables/efit ~/runs/pipeline_improvement/tables/vacuum \\
        --output notebooks/data/pipeline_improvement_public_vs_main
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

#: Columns that identify one row of each table; duplicates on them are resume leftovers.
KEYS = {
    "slices": ("side", "shot", "time_ms"),
    "fit": ("side", "shot", "time_ms"),
    "virial": ("side", "shot", "time_ms"),
    "thomson": ("side", "shot", "time_ms"),
    "thomson_channels": ("side", "shot", "time_ms", "channel"),
    "constraint_points": ("side", "shot", "time_ms", "family", "channel"),
    "vacuum": ("source", "shot", "wall_model", "name"),
    "onset": ("source", "shot", "family", "t_ms"),
    "onset_traces": ("source", "shot", "channel", "t_ms"),
    "stability_legacy": ("shot", "time_ms", "solver", "n"),
    "stability_legacy_surfaces": ("shot", "time_ms", "solver", "n", "m"),
}
ATLAS = ("stability_current_atlas_n.csv", "stability_current_atlas_surfaces.csv",
         "stability_current_schema.json", "stability_current_README.md")


def merged(tables: Sequence[Path], name: str) -> pd.DataFrame | None:
    frames = []
    for root in tables:
        for path in sorted(root.glob(f"{name}*.csv")):
            stem = path.stem
            # "fit_0of4" and "fit" belong to "fit"; "fit_points" would not.
            if stem != name and not (stem.startswith(name + "_") and stem[len(name) + 1:].replace("of", "").isdigit()):
                continue
            frames.append(pd.read_csv(path, low_memory=False))
    if not frames:
        return None
    table = pd.concat(frames, ignore_index=True)
    keys = [k for k in KEYS[name] if k in table.columns]
    return table.drop_duplicates(subset=keys, keep="last") if keys else table


def vacuum_by_family(vacuum: pd.DataFrame) -> pd.DataFrame:
    """One row per source, shot, wall model and family, over channels both sources evaluated.

    Restricting to channels present on both sides makes the medians compare the
    same sensors; a channel one product dropped would otherwise shift its side.
    """
    evaluated = vacuum[vacuum["status"] == "evaluated"].copy()
    evaluated["normalized_residual"] = pd.to_numeric(evaluated["normalized_residual"], errors="coerce")
    sources = evaluated.groupby(["shot", "wall_model", "name"])["source"].nunique()
    common = sources[sources == 2].reset_index()[["shot", "wall_model", "name"]]
    matched = evaluated.merge(common, on=["shot", "wall_model", "name"])
    grouped = matched.groupby(["source", "shot", "wall_model", "family"])
    out = grouped["normalized_residual"].agg(["median", "count"]).reset_index()
    out = out.rename(columns={"median": "normalized_residual_median", "count": "channels"})
    improvement = grouped["improvement"].median().reset_index(name="improvement_median")
    return out.merge(improvement, on=["source", "shot", "wall_model", "family"])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--tables", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tables = [p.expanduser() for p in args.tables]
    args.output.mkdir(parents=True, exist_ok=True)

    written: dict[str, dict[str, object]] = {}

    def write(frame: pd.DataFrame, name: str) -> None:
        path = args.output / f"{name}.csv.gz"
        frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})
        written[path.name] = {"rows": int(len(frame)), "sha256": sha256(path)}

    for name in KEYS:
        frame = merged(tables, name)
        if frame is None:
            continue
        if name == "vacuum":
            # A load error superseded by a rerun of the same source and shot is
            # not a result; only load errors no rerun replaced stay counted.
            rerun = frame[frame["status"] != "load_error"][["source", "shot"]].drop_duplicates()
            stale = frame["status"].eq("load_error") & frame.set_index(["source", "shot"]).index.isin(
                rerun.set_index(["source", "shot"]).index)
            frame = frame[~stale]
            write(vacuum_by_family(frame), "vacuum_by_family")
            status = frame.groupby(["source", "wall_model", "status"], dropna=False).size()
            written["vacuum_status"] = {"counts": {" / ".join(map(str, k)): int(v) for k, v in status.items()}}
            continue
        if name == "onset":
            # The snapshot keeps the three panels' families (the EFIT sub-families
            # stay in the run tables) at four significant digits.
            frame = frame[frame["family"].isin(("ip", "probe", "loop", "error"))].copy()
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce").map(lambda v: float(f"{v:.4g}"))
        if name == "onset_traces":
            frame = frame.copy()
            for column in ("measured", "vacuum"):
                frame[column] = pd.to_numeric(frame[column], errors="coerce").map(lambda v: float(f"{v:.5g}"))
        write(frame, name)
    for root in tables:
        for name in ATLAS:
            source = root / name
            if source.exists() and name not in written:
                target = args.output / (name + ".gz" if name.endswith(".csv") else name)
                if name.endswith(".csv"):
                    pd.read_csv(source).to_csv(target, index=False, compression={"method": "gzip", "mtime": 0})
                else:
                    target.write_bytes(source.read_bytes())
                written[target.name] = {"rows": None, "sha256": sha256(target)}
    manifests = {}
    for root in tables:
        for manifest in sorted(root.glob("manifest*.json")):
            manifests[f"{root}/{manifest.name}"] = json.loads(manifest.read_text())
    (args.output / "snapshot_manifest.json").write_text(json.dumps({
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "producer": "workflow/pipeline_improvement/make_snapshot.py",
        "tables": [str(t) for t in tables],
        "files": written,
        "runs": manifests,
    }, indent=1, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)) + "\n", encoding="utf-8")
    for name, entry in written.items():
        print(name, entry.get("rows"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
