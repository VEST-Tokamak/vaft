#!/usr/bin/env python3
"""Uncertainty band of the Thomson electron-pressure peaking, per slice (#579).

    python workflow/efit_uncertainty_calibration/thomson_peaking_band.py \\
        --filedb ~/runs/campaign/filedb --slices model_spread.csv --output peaking_band.csv

``sensitivity_pressure.py`` compares each member's EFIT peaking ``p(0)/<p>_V``
with the peaking of the fitted Thomson ``p_e``.  Telling the profile bases apart
by shape needs to know how well that Thomson peaking is itself determined.  This
script answers it with a parametric bootstrap over the channels:

1. map the Thomson channels through the slice's magnetic EFIT equilibrium (the
   one ``core_profiles`` was mapped with), matched by time;
2. perturb every channel's ``T_e`` and ``n_e`` by a Gaussian draw of its own
   ``data_error_upper``, unclipped, so the draws are centred on what the
   central fit sees (a draw the production fit refuses is dropped and counted);
3. refit with the production settings of ``build_core_profiles_ods``
   (``T_e`` polynomial order 2, ``n_e`` exponential order 2);
4. compute ``p_e(0) / <p_e>_V`` on the equilibrium's own grid, exactly as
   ``sensitivity_pressure`` computes ``peaking_e``.

Per slice: the unperturbed (central) peaking, the 16th/50th/84th percentiles
over the draws that fitted, and how many did.  Thomson remains a validation
quantity: nothing here selects a member.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

#: The fit ``build_core_profiles_ods`` uses for an electron-only core_profiles slice.
PRODUCTION_FIT: Mapping[str, Any] = {
    "Te_order": 2, "Ne_order": 2, "fitting_function_te": "polynomial", "fitting_function_ne": "exponential",
}
_AXIS = np.linspace(0.0, 1.0, 51)


def _load_ods(path: Path):
    from omas import load_omas_json

    if path.suffix != ".gz":
        return load_omas_json(str(path), consistency_check=False)
    with gzip.open(path, "rt", encoding="utf-8") as handle, \
            tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as staged:
        staged.write(handle.read())
    try:
        return load_omas_json(staged.name, consistency_check=False)
    finally:
        Path(staged.name).unlink(missing_ok=True)


def profile_peaking(equilibrium, index: int, pressure: Callable[[np.ndarray], np.ndarray]) -> float:
    """``p(0) / <p>_V`` of a profile given on ``rho_tor_norm``, integrated on the slice's own grid.

    The same definition as ``sensitivity_pressure.member_pressure``'s
    ``peaking_e``: ``<p>_V = W / 1.5 / V`` from
    :func:`vaft.validation.kinetic_state.integrated_pressure_ratio`, which needs
    no volume profile (the magnetics EFIT product carries none).
    """
    from vaft.validation.kinetic_state import integrated_pressure_ratio

    p_axis = np.asarray(pressure(_AXIS), float)
    if not np.all(np.isfinite(p_axis)):
        return math.nan
    full = integrated_pressure_ratio(equilibrium, index, _AXIS, p_axis)
    if not full.get("available") or not full.get("volume_m3"):
        return math.nan
    mean = full["w_e_j"] / 1.5 / full["volume_m3"]
    return float(p_axis[0] / mean) if mean > 0 else math.nan


def peaking_band(thomson, equilibrium, time_s: float, *, draws: int = 200, seed: int = 579,
                 fit: Mapping[str, Any] = PRODUCTION_FIT,
                 tolerance_s: float = 5.0e-4, equilibrium_index: int | None = None) -> dict[str, Any]:
    """Central value and bootstrap band of the Thomson ``p_e`` peaking at ``time_s``.

    ``thomson`` is the Thomson ODS (modified during the call and restored);
    ``equilibrium`` an equilibrium ODS whose slice nearest ``time_s`` maps the
    channels and gives the volume; ``equilibrium_index`` names the slice
    instead, for a one-slice ODS made from a g-file.
    """
    from vaft.process import profile
    from vaft.validation.kinetic_state import slice_at_time

    if equilibrium_index is not None:
        index = int(equilibrium_index)
    else:
        try:
            index, _offset = slice_at_time(equilibrium, time_s, ids="equilibrium", tolerance_s=tolerance_s)
        except LookupError as missing:  # EFIT kept no slice here: no geometry to map onto
            return {"central": math.nan, "n_draws": 0, "reason": f"no equilibrium slice: {missing}"[:160]}
    mapped = profile.equilibrium_mapping_thomson_scattering(
        thomson, equilibrium, time=None if equilibrium_index is not None else time_s)
    points = profile._thomson_points(thomson, time_s * 1e3, 1.0)
    k = points["time_index"]
    channels = thomson["thomson_scattering.channel"]
    original = [(np.array(channels[i]["t_e.data"], float), np.array(channels[i]["n_e.data"], float))
                for i in range(len(channels))]

    def one_peaking() -> float:
        n_e, t_e, *_ = profile.profile_fitting_thomson_scattering(thomson, time_s * 1e3, mapped, **fit)
        return profile_peaking(equilibrium, index, lambda x: n_e(x) * t_e(x))

    def write(t_e: np.ndarray, n_e: np.ndarray) -> None:
        for i, (te0, ne0) in enumerate(original):
            te, ne = te0.copy(), ne0.copy()
            te[k], ne[k] = t_e[i], n_e[i]
            channels[i]["t_e.data"], channels[i]["n_e.data"] = te, ne

    usable = (np.isfinite(points["t_e"]) & np.isfinite(points["n_e"]) & np.isfinite(points["t_e_std"])
              & np.isfinite(points["n_e_std"]) & (points["t_e_std"] > 0) & (points["n_e_std"] > 0))
    # Value and sigma criteria of the fit's own channel filter (positions are mapped above).
    result: dict[str, Any] = {"time_s": float(points["time"]), "channels": int(usable.sum())}
    try:
        try:
            result["central"] = one_peaking()
        except Exception as error:  # the slice cannot be fitted at all
            return {**result, "central": math.nan, "n_draws": 0, "reason": repr(error)[:120]}
        rng = np.random.default_rng(seed)
        values = []
        for _ in range(int(draws)):
            t_e = points["t_e"] + rng.normal(0.0, 1.0, points["t_e"].size) * np.nan_to_num(points["t_e_std"])
            n_e = points["n_e"] + rng.normal(0.0, 1.0, points["n_e"].size) * np.nan_to_num(points["n_e_std"])
            write(t_e, n_e)
            try:
                value = one_peaking()
            except Exception:  # a draw the production fit refuses is not a peaking
                continue
            if math.isfinite(value):
                values.append(value)
    finally:
        for i, (te0, ne0) in enumerate(original):
            channels[i]["t_e.data"], channels[i]["n_e.data"] = te0, ne0
    result["n_draws"] = len(values)
    if values:
        q16, q50, q84 = np.percentile(values, [16, 50, 84])
        result.update({"q16": float(q16), "q50": float(q50), "q84": float(q84)})
    return result


#: The #891 working setting, whose member geometry is preferred.
WORKING_SETTING = "p2f1_probe_x3.62_loop_x2.15_dia_x16_ip_x4_floor2pct_psiexit"


def member_geometry(scan_dirs: Sequence[Path], shot: int, time_s: float) -> tuple[Path | None, str | None]:
    """A member g-file for the slice: the working setting's, else the first (2,1), else any member's."""
    time_dir = f"t{int(round(time_s * 1e6)):07d}"
    candidates = []
    for scan in scan_dirs:
        root = Path(scan) / f"shot_{shot}" / time_dir
        if root.is_dir():
            candidates.extend(sorted(d for d in root.iterdir() if d.is_dir()))
    rank = lambda d: (d.name != WORKING_SETTING, not d.name.startswith("p2f1_"), d.name)  # noqa: E731
    for directory in sorted(candidates, key=rank):
        gfiles = sorted(directory.glob("g0*"))
        if gfiles:
            return gfiles[0], directory.name
    return None, None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filedb", type=Path, required=True,
                        help="FileDB with omas/thomson/<shot> and omas/efit/magnetic/<shot> products")
    parser.add_argument("--slices", type=Path, required=True, help="CSV with shot and time_efit_s columns")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scan-dirs", type=Path, nargs="*", default=(),
                        help="weight_scan --output directories: take each slice's geometry from a member "
                             "g-file (the working setting's when it has one) instead of the FileDB EFIT product")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=579)
    args = parser.parse_args(argv)

    with args.slices.open(newline="", encoding="utf-8") as handle:
        slices = sorted({(int(r["shot"]), float(r["time_efit_s"])) for r in csv.DictReader(handle)})
    root = args.filedb.expanduser() / "omas"
    cache: dict[int, tuple[Any, Any]] = {}
    rows = []
    for shot, time_s in slices:
        if shot not in cache:
            thomson = sorted((root / "thomson" / str(shot) / "output").glob("thomson.json*"))
            efit = [] if args.scan_dirs else sorted((root / "efit" / "magnetic" / str(shot) / "output").glob("efit.json*"))
            # Slices are sorted by shot, so one shot's products are held at a time.
            cache = {shot: (_load_ods(thomson[0]) if thomson else None, _load_ods(efit[0]) if efit else None)}
        thomson, equilibrium = cache[shot]
        row: dict[str, Any] = {"shot": shot, "time_efit_s": time_s}
        index = None
        if args.scan_dirs:
            gfile, setting = member_geometry(args.scan_dirs, shot, time_s)
            row["geometry"] = setting or "none"
            if gfile is not None:
                from vaft.data import read_geqdsk

                equilibrium, index = read_geqdsk(str(gfile)).to_omas(ods=None, time_index=0), 0
            else:
                equilibrium = None
        else:
            row["geometry"] = "filedb efit product"
        if thomson is None or equilibrium is None:
            rows.append({**row, "reason": "no thomson product or no geometry"})
            continue
        try:
            row.update(peaking_band(thomson, equilibrium, time_s, draws=args.draws, seed=args.seed,
                                    equilibrium_index=index))
        except Exception as error:  # one slice must not stop the table
            row["reason"] = repr(error)[:160]
        rows.append(row)
        print(row, flush=True)
    columns = ["shot", "time_efit_s", "geometry", "time_s", "channels", "central", "q16", "q50", "q84", "n_draws",
               "reason"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} slices -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
