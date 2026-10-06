"""Compare MITIM's TGLF local inputs with VAFT's, key by key (#1588 stage A2).

MITIM builds TGLF's local input in its own Python (``MITIMstate.to_tglf``: its
own splines, derivatives and normalisations), as VAFT does in
``prepare_tglf_input``. Before any flux is compared, the two inputs for the same
``input.gacode`` and the same ``r/a`` are compared parameter by parameter, so a
flux difference can be traced to the parameter that caused it.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

__all__ = ["PHYSICS_KEYS", "compare_tglf_inputs", "effective_tglf_controls", "read_input_tglf"]

#: Keys that describe the plasma at the surface (geometry, gradients, species,
#: collisionality, beta, rotation); everything else is a solver control.
PHYSICS_KEYS = re.compile(
    r"^(RMIN_LOC|RMAJ_LOC|DRMAJDX_LOC|ZMAJ_LOC|DZMAJDX_LOC|Q_LOC|Q_PRIME_LOC|P_PRIME_LOC|"
    r"KAPPA_LOC|S_KAPPA_LOC|DELTA_LOC|S_DELTA_LOC|ZETA_LOC|S_ZETA_LOC|BETAE|XNUE|ZEFF|DEBYE|"
    r"VEXB|VEXB_SHEAR|BETA_LOC|KX0_LOC|SIGN_BT|SIGN_IT|NS|"
    r"(ZS|MASS|AS|TAUS|RLNS|RLTS|VPAR|VPAR_SHEAR|VNS_SHEAR|VTS_SHEAR)_\d+)$")


def _value(text: str) -> Any:
    token = text.strip().strip("'\"")
    lowered = token.lower()
    if lowered in (".true.", "true", "t", ".t."):
        return True
    if lowered in (".false.", "false", "f", ".f."):
        return False
    try:
        number = float(token.replace("d", "e").replace("D", "E"))
    except ValueError:
        return token
    return int(number) if re.fullmatch(r"[+-]?\d+", token) else number


def read_input_tglf(path: str | Path) -> dict[str, Any]:
    """``KEY=VALUE`` lines of an ``input.tglf``; comments (``#``) and blanks skipped."""
    parameters: dict[str, Any] = {}
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        parameters[key.strip().upper()] = _value(value)
    return parameters


#: The s-alpha geometry inputs, read only when GEOMETRY_FLAG = 0.
_SALPHA = re.compile(r"^[A-Z_]+_SA$")


def effective_tglf_controls(parameters: Mapping[str, Any]) -> dict[str, Any]:
    """The controls TGLF actually runs with, after its own start-up presets.

    ``tglf_startup.f90`` (GACODE b493397) has ``USE_PRESETS = .TRUE.`` hard-coded:
    it zeroes WDIA_TRAPPED, then for SAT_RULE 2/3 sets XNU_MODEL = 3, WDIA_TRAPPED = 1
    and UNITS GYRO -> CGYRO; for SAT_RULE 1 sets XNU_MODEL = 2; for SAT_RULE 0 sets
    UNITS = GYRO and XNU_MODEL = 2; with USE_BPER sets ALPHA_MACH = 0; and SAT_RULE 0
    rounds NMODES > 2 up to 4. The s-alpha ``*_SA`` inputs are dropped unless
    GEOMETRY_FLAG = 0. Two inputs that differ only in what these overwrite run
    identically, which is what a comparison should see.
    """
    def integer(key, default):
        try:
            return int(float(parameters.get(key, default)))
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be an integer, got {parameters.get(key)!r}") from None

    out = {key: value for key, value in parameters.items()
           if not (_SALPHA.match(key) and integer("GEOMETRY_FLAG", 1) != 0)}
    sat = integer("SAT_RULE", 0)
    out["WDIA_TRAPPED"] = 0.0
    if sat in (2, 3):
        out["XNU_MODEL"] = 3
        out["WDIA_TRAPPED"] = 1.0
        if str(out.get("UNITS", "GYRO")).upper() == "GYRO":
            out["UNITS"] = "CGYRO"
    elif sat == 1:
        out["XNU_MODEL"] = 2
    elif sat == 0:
        out["UNITS"] = "GYRO"
        out["XNU_MODEL"] = 2
        if integer("NMODES", 2) > 2:
            out["NMODES"] = 4
    if bool(out.get("USE_BPER", False)):
        out["ALPHA_MACH"] = 0.0
    return out


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def compare_tglf_inputs(
    vaft: Mapping[str, Any],
    mitim: Mapping[str, Any],
    *,
    rtol: float = 1e-3,
    atol: float = 1e-6,
    keys: Iterable[str] | None = None,
    effective: bool = True,
) -> list[dict[str, Any]]:
    """One row per key present in either input.

    Each row has ``key``, ``kind`` (``physics`` or ``control``), ``vaft``, ``mitim``,
    ``status`` (``agree``, ``differ``, ``vaft_only``, ``mitim_only``) and, for two
    numbers, ``abs_diff`` and ``rel_diff`` = |a - b| / max(|a|, |b|). A key that one
    side leaves to TGLF's default is ``*_only``, not a disagreement on the value.
    With ``effective`` (the default), both sides first go through
    :func:`effective_tglf_controls`, so a control TGLF overwrites at start-up does not
    count as a difference.
    """
    if effective:
        vaft, mitim = effective_tglf_controls(vaft), effective_tglf_controls(mitim)
    names = sorted(set(keys) if keys is not None else set(vaft) | set(mitim))
    rows = []
    for key in names:
        row: dict[str, Any] = {"key": key, "kind": "physics" if PHYSICS_KEYS.match(key) else "control",
                               "vaft": vaft.get(key), "mitim": mitim.get(key),
                               "abs_diff": None, "rel_diff": None}
        if key not in mitim:
            row["status"] = "vaft_only"
        elif key not in vaft:
            row["status"] = "mitim_only"
        else:
            a, b = _numeric(vaft[key]), _numeric(mitim[key])
            if a is None or b is None:
                row["status"] = "agree" if str(vaft[key]).strip().upper() == str(mitim[key]).strip().upper() else "differ"
            else:
                difference = abs(a - b)
                scale = max(abs(a), abs(b))
                row["abs_diff"] = difference
                row["rel_diff"] = difference / scale if scale > 0 else 0.0
                close = math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
                row["status"] = "agree" if close else "differ"
        rows.append(row)
    return rows
