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

from .config import TokaMakerConfig, TokaMakerResult

_log = logging.getLogger(__name__)

SIDECAR_NAME = "tokamaker_result.json"

# get_stats() keys surfaced as flat scalars when present in the sidecar.
_STATS_KEYS = (
    "Ip", "Ip_centroid", "kappa", "kappaU", "kappaL", "delta", "deltaU", "deltaL",
    "R_geo", "a_geo", "vol", "q_0", "q_95", "P_ax", "P_max", "W_MHD", "beta_pol",
    "dflux", "tflux", "l_i", "beta_tor", "beta_n",
)


def _find_one(workdir: Path, patterns: list[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(workdir.glob(pat))
        if hits:
            return hits[0]
    return None


def parse_stats_sidecar(path: Path) -> dict[str, Any]:
    """Flatten the ``tokamaker_result.json`` sidecar into a scalar dict.

    ``stats`` entries listed in ``_STATS_KEYS`` are promoted to top-level keys;
    the coil currents, targets, and run metadata are kept under their own keys.
    """
    payload = json.loads(path.read_text())
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
    # mistaken for a g-file.
    gfile = _find_one(base, ["g[0-9]*.[0-9]*", "g[0-9]*"])
    stats_file = _find_one(base, [SIDECAR_NAME])
    mesh_file = _find_one(base, ["vest_gs_mesh_*.h5"])
    logs = tuple(sorted(p for p in base.glob("*.log") if p.is_file()))

    geqdsk: tuple[Any, ...] = ()
    ods = None
    geqdsk_error: Optional[str] = None
    if gfile is not None:
        try:
            from vaft.data.eqdsk import read_geqdsk
            parsed = read_geqdsk(gfile)
            geqdsk = (parsed,)
            ods = parsed.to_omas(ods=None, time_index=0)
        except Exception as exc:
            geqdsk = ()
            ods = None
            geqdsk_error = f"{gfile.name}: {exc}"
            _log.warning("Could not parse TokaMaker g-file %s: %s", gfile, exc)

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
