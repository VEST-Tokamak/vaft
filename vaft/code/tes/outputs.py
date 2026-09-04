"""Collect and parse TES outputs.

TES writes a standard EFIT g-file/a-file plus a ``.RESULT`` summary. The g-file
round-trips into an ODS equilibrium subtree through ``vaft.data.eqdsk`` — exactly
like the EFIT adapter — and the ``.RESULT`` scalars are parsed into a flat dict.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from .config import TESConfig, TESResult


_log = logging.getLogger(__name__)

_FLOAT = r"[-+]?\d+\.?\d*(?:[eE][+-]?\d+)?"

# Scalar tokens in a .RESULT header look like ``KEY[unit]  VALUE`` or ``KEY  VALUE``,
# several per line. This captures the label (incl. any [unit]) and the value.
_RESULT_TOKEN = re.compile(r"([A-Za-z][A-Za-z0-9_]*(?:\[[^\]]*\])?)\s+(" + _FLOAT + r")")

# Labels we surface with friendly keys; everything else is still captured raw.
_SCALAR_ALIASES = {
    "IP[kA]": "ip_kA",
    "Rgeo[m]": "rgeo", "Rcur[m]": "rcur", "Rmag[m]": "rmag",
    "Zgeo[m]": "zgeo", "Zcur[m]": "zcur", "Zmag[m]": "zmag",
    "MAJOR_R[m]": "major_r", "BT0[T]": "bt0", "MINOR_A[m]": "minor_a",
    "KAPPA": "kappa", "KAPPA_U": "kappa_u", "KAPPA_L": "kappa_l",
    "DELTA_U": "delta_u", "DELTA_L": "delta_l",
    "PSIA": "psia", "PSIB": "psib", "NXPOINT": "nxpoint",
    "VOL_P[m^3]": "vol_p", "LI": "li", "LI(3)": "li3", "WMHD[MJ]": "wmhd_MJ",
    "BETAP": "betap", "BETAP_TSC": "betap_tsc", "BETAT[%]": "betat_pct", "BETAN[%]": "betan_pct",
    "Q0": "q0", "Q95": "q95", "Q100": "q100",
}


def _find_one(workdir: Path, patterns: list[str]) -> Optional[Path]:
    for pat in patterns:
        hits = sorted(workdir.glob(pat))
        if hits:
            return hits[0]
    return None


def parse_result_scalars(result_file: Path) -> dict[str, Any]:
    """Parse the scalar header block of a TES ``.RESULT`` file into a flat dict.

    Stops at the first list block (iso-flux / coil-update sections) so only the
    equilibrium scalars are captured. Friendly aliases from ``_SCALAR_ALIASES``
    are added alongside the raw labels.
    """
    scalars: dict[str, Any] = {}
    for line in result_file.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("["):          # reached a list block
            break
        if "FLUX-DIFFERENCE" in line or "COIL CURRENT" in line:
            break
        for label, value in _RESULT_TOKEN.findall(line):
            val = float(value)
            scalars[label] = val
            alias = _SCALAR_ALIASES.get(label)
            if alias:
                scalars[alias] = val
    return scalars


def parse_result_coils(result_file: Path) -> list[dict[str, float | int]]:
    """Parse the 'EXT. COIL CURRENT UPDATED' block: base -> updated : delta [kA].

    ``index`` is the 1-based coil number as written by TES and stays an ``int``;
    the three current entries are floats in kA.
    """
    text = result_file.read_text(encoding="utf-8")
    block = re.search(r"EXT\. COIL CURRENT UPDATED.*?\n([\s\S]+?)(?=\n\n|\Z)", text)
    coils: list[dict[str, float | int]] = []
    if not block:
        return coils
    for line in block.group(1).splitlines():
        # ``strip`` so an indented block parses too; TES pads these lines to a
        # fixed width, and an anchored match would otherwise return no coils at all.
        m = re.match(
            r"\[\s*(\d+)\]\s*(" + _FLOAT + r")\s*-->\s*(" + _FLOAT + r")\s*:\s*(" + _FLOAT + r")",
            line.strip(),
        )
        if m:
            coils.append({
                "index": int(m.group(1)),
                "base_kA": float(m.group(2)),
                "updated_kA": float(m.group(3)),
                "delta_kA": float(m.group(4)),
            })
    return coils


def collect_tes_outputs(workdir: str | Path, config: Optional[TESConfig] = None) -> TESResult:
    """Collect TES output files from a working directory and parse them.

    The g-file is converted to an ODS equilibrium subtree via
    ``vaft.data.eqdsk.read_geqdsk(...).to_omas()``; ``.RESULT`` scalars and the
    updated coil currents are parsed into ``TESResult.scalars``.

    Parsing is best-effort: a malformed g-file or ``.RESULT`` leaves the other
    outputs intact and records the failure under the ``_geqdsk_error`` /
    ``_parse_error`` keys of ``TESResult.scalars`` instead of raising.
    """
    base = Path(workdir).expanduser()

    # EFIT names these ``g<shot>.<time>`` / ``a<shot>.<time>``. Requiring a digit
    # after the leading letter keeps unrelated files (e.g. ``gpec_*`` outputs in a
    # reused workdir) from being mistaken for a g-file or a-file.
    gfile = _find_one(base, ["g[0-9]*.[0-9]*", "g[0-9]*"])
    afile = _find_one(base, ["a[0-9]*.[0-9]*", "a[0-9]*"])
    result_file = _find_one(base, ["*.RESULT"])
    bndry = _find_one(base, ["*.BNDRY"])
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
            _log.warning("Could not parse TES g-file %s: %s", gfile, exc)

    scalars: dict[str, Any] = {}
    if result_file is not None:
        try:
            scalars = parse_result_scalars(result_file)
            scalars["coils"] = parse_result_coils(result_file)
        except Exception as exc:
            scalars = {"_parse_error": f"{result_file.name}: {exc}"}
            _log.warning("Could not parse TES .RESULT %s: %s", result_file, exc)
    if geqdsk_error is not None:
        scalars["_geqdsk_error"] = geqdsk_error

    return TESResult(
        returncode=None,
        workdir=base,
        gfile=gfile,
        afile=afile,
        result_file=result_file,
        bndry=bndry,
        logs=logs,
        geqdsk=geqdsk,
        ods=ods,
        scalars=scalars,
    )
