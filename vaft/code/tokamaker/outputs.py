"""Collect and parse TokaMaker outputs from a working directory.

The runner leaves two artefacts: an EFIT g-file (``g<shot>.<time_ms>``) and a
``tokamaker_result.json`` sidecar with convergence status, global statistics,
and the solved coil currents. The g-file round-trips into an ODS equilibrium
subtree through ``vaft.data.eqdsk`` — exactly like the EFIT/TES adapters —
and the sidecar is flattened into ``TokaMakerResult.scalars``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .config import (
    TokaMakerConfig,
    TokaMakerEvolutionResult,
    TokaMakerResult,
    TokaMakerStabilityResult,
    TokaMakerStepRecord,
)

_log = logging.getLogger(__name__)

SIDECAR_NAME = "tokamaker_result.json"
EVOLUTION_SIDECAR_NAME = "tokamaker_evolution.json"
STABILITY_SIDECAR_NAME = "tokamaker_stability.json"

# get_stats() keys surfaced as flat scalars when present in the sidecar.
_STATS_KEYS = (
    "Ip", "Ip_centroid", "kappa", "kappaU", "kappaL", "delta", "deltaU", "deltaL",
    "R_geo", "a_geo", "vol", "q_0", "q_95", "P_ax", "P_max", "W_MHD", "beta_pol",
    "dflux", "tflux", "l_i", "beta_tor", "beta_n",
)


def _find_one(workdir: Path, patterns: list[str], last: bool = False) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(workdir.glob(pat))
        if hits:
            return hits[-1] if last else hits[0]
    return None


def _parse_gfile(gfile: Path):
    """Parse one g-file into ``(geqdsk_tuple, ods, error_message)`` best-effort."""
    try:
        from vaft.data.eqdsk import read_geqdsk

        parsed = read_geqdsk(gfile)
        return (parsed,), parsed.to_omas(ods=None, time_index=0), None
    except Exception as exc:
        _log.warning("Could not parse TokaMaker g-file %s: %s", gfile, exc)
        return (), None, f"{gfile.name}: {exc}"


def parse_stats_sidecar(path: Path) -> dict[str, Any]:
    """Flatten the ``tokamaker_result.json`` sidecar into a scalar dict.

    ``stats`` entries listed in ``_STATS_KEYS`` are promoted to top-level keys;
    the coil currents, targets, and run metadata are kept under their own keys.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    scalars: dict[str, Any] = {}
    stats = payload.get("stats") or {}
    for key in _STATS_KEYS:
        if key in stats:
            scalars[key] = stats[key]
    for key in ("converged", "coil_currents_A", "targets", "f0", "cocos",
                "o_point", "diverted", "error", "shot", "time_s"):
        if key in payload:
            scalars[key] = payload[key]
    return scalars


def collect_tokamaker_outputs(
    workdir: str | Path, config: Optional[TokaMakerConfig] = None
) -> TokaMakerResult:
    """Collect TokaMaker output files from a working directory and parse them.

    The g-file is converted to an ODS equilibrium subtree via
    ``vaft.data.eqdsk.read_geqdsk(...).to_omas()``; the JSON sidecar is
    flattened into ``TokaMakerResult.scalars``.

    Parsing is best-effort: a malformed g-file or sidecar leaves the other
    outputs intact and records the failure under the ``_geqdsk_error`` /
    ``_parse_error`` keys of ``TokaMakerResult.scalars`` instead of raising.
    """
    base = Path(workdir).expanduser()

    # Requiring a digit after the leading "g" keeps unrelated files (e.g.
    # ``gpec_*`` outputs or ``geometry.json`` in a reused workdir) from being
    # mistaken for a g-file. In a workdir holding several runs the NEWEST
    # (latest-time) g-file is picked; the runner overrides this with the exact
    # file it wrote, so the heuristic only governs post-hoc collection.
    gfile = _find_one(base, ["g[0-9]*.[0-9]*", "g[0-9]*"], last=True)
    stats_file = _find_one(base, [SIDECAR_NAME])
    mesh_file = _find_one(base, ["vest_gs_mesh_*.h5"])
    logs = tuple(sorted(p for p in base.glob("*.log") if p.is_file()))

    geqdsk: tuple[Any, ...] = ()
    ods = None
    geqdsk_error: Optional[str] = None
    if gfile is not None:
        geqdsk, ods, geqdsk_error = _parse_gfile(gfile)

    scalars: dict[str, Any] = {}
    if stats_file is not None:
        try:
            scalars = parse_stats_sidecar(stats_file)
        except Exception as exc:
            scalars = {"_parse_error": f"{stats_file.name}: {exc}"}
            _log.warning("Could not parse TokaMaker sidecar %s: %s", stats_file, exc)
    if geqdsk_error is not None:
        scalars["_geqdsk_error"] = geqdsk_error

    return TokaMakerResult(
        returncode=None,
        workdir=base,
        gfile=gfile,
        stats_file=stats_file,
        mesh_file=mesh_file,
        logs=logs,
        geqdsk=geqdsk,
        ods=ods,
        scalars=scalars,
    )


def _merge_equilibrium(gfiles, times):
    """Best-effort multi-slice equilibrium merge via the standard mapping.

    Returns ``(ods_or_None, error_message)``. Explicit ``times`` [s] are
    passed so the merge never depends on millisecond rounding of filenames.
    """
    if not gfiles:
        return None, ""
    try:
        from omas import ODS

        from vaft.machine_mapping.equilibrium import equilibrium as merge_equilibrium

        ods = ODS(consistency_check=False)
        merge_equilibrium(ods, [str(path) for path in gfiles], options={"times": list(times)})
        return ods, ""
    except Exception as exc:
        _log.warning("Could not merge TokaMaker g-files into an equilibrium IDS: %s", exc)
        return None, str(exc)


def collect_tokamaker_evolution_outputs(
    workdir: str | Path, config: Optional[TokaMakerConfig] = None
) -> TokaMakerEvolutionResult:
    """Rebuild an evolution result from a working directory (collect-only).

    Parses ``tokamaker_evolution.json`` back into step records, re-resolves
    the per-slice g-file paths, and re-merges the multi-slice equilibrium IDS
    best-effort (failures land in ``scalars['_merge_error']``/``'_parse_error'``).
    """
    base = Path(workdir).expanduser()
    sidecar_file = _find_one(base, [EVOLUTION_SIDECAR_NAME])
    mesh_file = _find_one(base, ["vest_gs_mesh_*.h5"])

    records: list[TokaMakerStepRecord] = []
    times: tuple[float, ...] = ()
    scalars: dict[str, Any] = {}
    vacuum = False
    if sidecar_file is not None:
        try:
            payload = json.loads(sidecar_file.read_text(encoding="utf-8"))
            times = tuple(float(t) for t in payload.get("times", ()))
            vacuum = bool(payload.get("vacuum", False))
            for entry in payload.get("steps", ()):
                gfile_name = entry.get("gfile")
                gfile = base / gfile_name if gfile_name else None
                records.append(TokaMakerStepRecord(
                    index=int(entry["index"]),
                    time=float(entry["time"]),
                    converged=bool(entry["converged"]),
                    error=str(entry.get("error", "")),
                    gfile=gfile if (gfile is not None and gfile.is_file()) else None,
                    stats=entry.get("stats", {}),
                    coil_currents_A=entry.get("coil_currents_A", {}),
                    vessel_currents_A=entry.get("vessel_currents_A", {}),
                    probe_fields=entry.get("probe_fields", {}),
                ))
        except Exception as exc:
            scalars["_parse_error"] = f"{sidecar_file.name}: {exc}"
            _log.warning("Could not parse evolution sidecar %s: %s", sidecar_file, exc)

    gfiles = tuple(rec.gfile for rec in records if rec.gfile is not None)
    gtimes = tuple(rec.time for rec in records if rec.gfile is not None)
    ods, merge_error = _merge_equilibrium(gfiles, gtimes)
    if merge_error:
        scalars["_merge_error"] = merge_error

    scalars.setdefault("n_steps", len(records))
    scalars.setdefault("n_failed", sum(1 for rec in records if not rec.converged))
    scalars.setdefault("vacuum", vacuum)

    return TokaMakerEvolutionResult(
        returncode=None,
        workdir=base,
        times=times,
        steps=tuple(records),
        gfiles=gfiles,
        sidecar_file=sidecar_file,
        mesh_file=mesh_file,
        ods=ods,
        scalars=scalars,
    )


def collect_tokamaker_stability_outputs(
    workdir: str | Path, config: Optional[TokaMakerConfig] = None
) -> TokaMakerStabilityResult:
    """Rebuild a stability result from the sidecar and .npz files (collect-only)."""
    base = Path(workdir).expanduser()
    stats_file = _find_one(base, [STABILITY_SIDECAR_NAME])
    eig_file = _find_one(base, ["tokamaker_eig_td.npz", "tokamaker_eig_wall.npz"])
    gfile = _find_one(base, ["g[0-9]*.[0-9]*"])

    tau_wall: tuple[float, ...] = ()
    gamma = None
    scalars: dict[str, Any] = {}
    if stats_file is not None:
        try:
            payload = json.loads(stats_file.read_text(encoding="utf-8"))
            wall = payload.get("wall", {})
            vertical = payload.get("vertical", {})
            tau_wall = tuple(float(t) for t in wall.get("tau_wall_s", ()))
            if tau_wall:
                scalars["tau_wall_s"] = list(tau_wall)
                scalars["tau_wall_max_s"] = tau_wall[0]
            if vertical.get("gamma_s") is not None:
                gamma = float(vertical["gamma_s"])
                scalars["gamma_s"] = gamma
                scalars["stable"] = gamma <= 0.0
            for section in (wall, vertical):
                if section.get("error"):
                    scalars.setdefault("_errors", []).append(section["error"])
        except Exception as exc:
            scalars["_parse_error"] = f"{stats_file.name}: {exc}"
            _log.warning("Could not parse stability sidecar %s: %s", stats_file, exc)

    return TokaMakerStabilityResult(
        returncode=None,
        workdir=base,
        tau_wall_s=tau_wall,
        gamma_s=gamma,
        eig_file=eig_file,
        stats_file=stats_file,
        gfile=gfile,
        scalars=scalars,
    )
