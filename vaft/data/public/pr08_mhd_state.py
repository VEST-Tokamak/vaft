"""PR08 global MHD states as one canonical row-oriented table (#1736).

The ITPA PR08 profile database (:mod:`.itpa_profile`) also carries, in each
discharge's ``_0d.dat`` file, the global equilibrium quantities of a few time
slices.  :func:`pr08_mhd_state_table` turns those into one row per 0D record,
with columns named by the quantity identities of :mod:`vaft.formula.boundaries`
(``edge_safety_factor_95``, ``internal_inductance_li3``, ``normalized_beta``,
...) and their units in ``table.attrs["units"]``, so the population can be drawn
by :func:`vaft.plot.operational_space.operational_space_population` beside the
full-equilibrium states of :func:`vaft.omas.equilibrium_state_table` (#1620).
It does not touch :func:`~.itpa_profile.pr08_to_omas`: the ODS mapping and this
table read the same files for different purposes.

Provenance
----------
Every quantity column ``q`` has a sibling ``q_provenance`` on every row:

* ``source_direct`` -- the PR08 variable itself, converted only in unit or to a
  magnitude (PR08 signs follow the machine's current and field directions);
* ``deterministic_derived`` -- computed from ``source_direct`` columns of the same
  row by a registered VAFT function (:func:`vaft.formula.boundaries.kink_coordinates`,
  :func:`vaft.formula.stability.beta_N_from_beta_a_B0_Ip`, ...);
* ``source_corrected`` -- the source value of a file with a known whole-file unit slip
  (:data:`SOURCE_CORRECTIONS`: JT-60U ``IP`` in MA, ITER-scenario and JT-60U 39713
  ``BETMHD`` in percent), rescaled while the file is the pinned one; ``state_notes`` says so;
* ``source_invalid`` -- the source holds a value that cannot be the quantity
  (a non-positive radius, elongation, current, field, volume or ``l_i``; a
  ``BETMHD`` outside [0, 1]; an ``IP`` whose registered q95 estimate is more than
  :data:`UNIT_SLIP_FACTOR` off the source ``Q95``, i.e. not in amperes, as in
  JT-60U 16168); the cell is NaN and ``state_notes`` says why;
* ``missing`` -- the source holds nothing.

``equilibrium_derived`` (from a full equilibrium, :mod:`vaft.omas.equilibrium_state`)
and ``model_imputed`` (a parametric equilibrium fitted to fill a gap) are named in
:data:`PROVENANCE_KINDS` for tables that combine sources; this adapter produces
neither.  ``table.attrs["quantity_sources"]`` records, per column, the PR08
variable, its unit, its definition from the PR08 manual and the transformation.

Definitions that matter
-----------------------
* ``internal_inductance_li3`` is PR08 ``LI`` = 2 int B_p^2 dV / (mu0^2 I_p^2 R_geo):
  the l_i(3) form with the geometric radius, which the registry quantity admits
  ("carry the radius used"); ``li3_reference_radius`` holds that radius.
* ``toroidal_field`` is ``|BT|``, the vacuum field at ``RGEO``, as in the H-mode
  database; ``normalized_current``, ``normalized_beta`` and the shape coordinates
  use it. This is B at the geometric radius, whereas
  :func:`vaft.omas.equilibrium_state_table` normalizes with ``b0`` at the DD
  reference radius: ``table.attrs["conventions"]`` (:data:`PR08_CONVENTIONS`)
  states the field and radius on each table so the two are not merged as one
  population on a Troyon or ``*_in`` plane.
* ``edge_safety_factor_95`` is ``|Q95|`` from the equilibrium fit; the source has
  no ``q_psi`` at the boundary, so ``edge_safety_factor`` is not a column.
* ``poloidal_beta`` is ``BEPMHD`` (equilibrium); the diamagnetic ``BEPDIA`` is the
  separate ``poloidal_beta_diamagnetic``.

Nothing is fitted or filled.  Missing in the source stays missing.
"""

from __future__ import annotations

import math
import re
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from ._fetch import FetchError, cache_dir, fetch, sha256_of
from .itpa_profile import PR08_BASE_URL, PR08_REFERENCE, read_pr08_0d
from .schema import ColumnSpec

__all__ = [
    "MHD_STATE_COLUMNS",
    "PR08_MACHINES",
    "PROVENANCE_KINDS",
    "SOURCE_CORRECTIONS",
    "UNIT_SLIP_FACTOR",
    "fetch_pr08_population",
    "mhd_state_coverage",
    "pr08_inventory",
    "pr08_mhd_state_table",
    "pr08_release_inventory",
    "projection_coverage",
    "read_pr08_zero_d",
]

PROVENANCE_KINDS = ("source_direct", "source_corrected", "equilibrium_derived", "deterministic_derived",
                    "model_imputed", "source_invalid", "missing")

_IP_IN_MA = ("IP", 1e6, "IP is stored in MA in this file (q95 estimate 1.2e6 x the source Q95 on every record); x1e6")
_BETMHD_IN_PERCENT = ("BETMHD", 0.01, "BETMHD is stored in percent in this file (outside [0, 1] on every record); "
                                      "x0.01")

#: Unit slips of whole files of the pinned release, corrected only while the file is the pinned one
#: (its SHA-256 equals PR08_RELEASE's).  Each was found by the decisive checks of _unit_checks, holds on
#: every record of the file, and gives values consistent with the file's other quantities: JT-60U
#: 16107/16168 at 2.46 MA; the ITER scenarios at beta_N ~ 1.8-2.0; JT-60U 39713 at beta_t 1.49 %,
#: beta_N 2.6 beside its BEPMHD 1.30.  The checks still run after a correction, so a wrong one is rejected.
SOURCE_CORRECTIONS: dict = {
    ("jt60u", "16107"): (_IP_IN_MA,),
    ("jt60u", "16168"): (_IP_IN_MA,),
    ("jt60u", "39713"): (_BETMHD_IN_PERCENT,),
    **{("iter", shot): (_BETMHD_IN_PERCENT,) for shot in ("10020201", "10020202", "10020203", "10030201",
                                                          "10050201", "30040201", "30040202")},
}

PR08_SOURCE_DATABASE = "ITPA PR08 profile database"
PR08_SOURCE_RELEASE = "PR08 public release (tokamak-profiledb.ccfe.ac.uk, directories 2005-2008)"


@dataclass(frozen=True)
class Pr08Machine:
    """How a PR08 machine directory enters the table."""

    name: str
    machine_class: str
    dataset_type: str
    #: why this machine's Q95 is not an equilibrium q95 (empty when it is)
    q95_not_equilibrium: str = ""


#: PR08 machine directory -> canonical name, machine class and dataset type.  The ITPA ITER
#: records are predictive scenario simulations, not measurements.  T-10's Q95 equals the ITER
#: guideline estimate of its own 0D shape (iter_q95_coordinates) to 1e-4 on 40 of its 48
#: records: it is that estimate, not an equilibrium q95, and is not taken as one.
PR08_MACHINES: dict[str, Pr08Machine] = {
    "aug": Pr08Machine("AUG", "conventional_tokamak", "experimental"),
    "cmod": Pr08Machine("C-MOD", "conventional_tokamak", "experimental"),
    "d3d": Pr08Machine("DIII-D", "conventional_tokamak", "experimental"),
    "ftu": Pr08Machine("FTU", "conventional_tokamak", "experimental"),
    "iter": Pr08Machine("ITER", "conventional_tokamak", "design"),
    "jet": Pr08Machine("JET", "conventional_tokamak", "experimental"),
    "jt60u": Pr08Machine("JT-60U", "conventional_tokamak", "experimental"),
    "mast": Pr08Machine("MAST", "spherical_tokamak", "experimental"),
    "rtp": Pr08Machine("RTP", "conventional_tokamak", "experimental"),
    "t10": Pr08Machine("T-10", "conventional_tokamak", "experimental",
                       q95_not_equilibrium="T-10 Q95 is the ITER guideline estimate of the 0D shape, not an "
                                           "equilibrium q95"),
    "tftr": Pr08Machine("TFTR", "conventional_tokamak", "experimental"),
    "ts": Pr08Machine("TORE SUPRA", "conventional_tokamak", "experimental"),
    "txtr": Pr08Machine("TEXTOR", "conventional_tokamak", "experimental"),
}

_IDENTITY = {
    "record_id": ColumnSpec("str", "Unique '<machine>:<shot>:<time ms>' identifier, '#<n>' appended when a "
                                   "discharge repeats a time."),
    "machine": ColumnSpec("str", "Canonical machine name (PR08_MACHINES)."),
    "machine_class": ColumnSpec("str", "'spherical_tokamak' or 'conventional_tokamak', stated per machine."),
    "dataset_type": ColumnSpec("str", "'experimental' or 'design' (ITER scenario predictions)."),
    "shot": ColumnSpec("str", "Discharge as PR08 names it (PR08 shots may carry letter suffixes)."),
    "time_s": ColumnSpec("s", "PR08 TIME of the 0D record."),
    "source_database": ColumnSpec("str", "Source database."),
    "source_release": ColumnSpec("str", "Source release."),
    "source_record_id": ColumnSpec("str", "'<machine dir>/<shot dir>/<file>#<row>' in the release."),
    "source_kind": ColumnSpec("str", "Source dimensionality: '0D'."),
    "source_sha256": ColumnSpec("str", "SHA-256 of the 0D file read."),
    "phase": ColumnSpec("str", "PR08 PHASE (OHM, L, H, HGELM, ...), as given."),
    "state": ColumnSpec("str", "PR08 STATE ('STEADY' or 'TRANS'), as given."),
    "config": ColumnSpec("str", "PR08 CONFIG (SN, LSN, USN, DN, LIM, TOP, BOT, OUT, IN, IW), as given."),
    "state_notes": ColumnSpec("str", "Source values rejected, and why; empty when none."),
}


@dataclass(frozen=True)
class _Direct:
    column: str
    unit: str
    source: str
    source_unit: str
    definition: str
    scale: float = 1.0
    magnitude: bool = False
    positive: bool = True   # a value <= 0 cannot be the quantity


#: Quantities read directly from the 0D file.  Definitions abridged from the PR08 manual
#: (DOCS/PR08MAN/pdbman.html, "PR08 0D Variables").
_DIRECT = (
    _Direct("plasma_current", "MA", "IP", "A", "plasma current (sign: +ve anti-clockwise from above)",
            1e-6, magnitude=True),
    _Direct("toroidal_field", "T", "BT", "T", "vacuum toroidal field at the geometric axis RGEO", magnitude=True),
    _Direct("major_radius", "m", "RGEO", "m", "geometric major radius, (R_min + R_max)/2 at the axis elevation, "
                                                "from the equilibrium fit"),
    _Direct("minor_radius", "m", "AMIN", "m", "horizontal minor radius from the equilibrium fit"),
    _Direct("magnetic_axis_radius", "m", "RMAG", "m", "major radius of the magnetic axis"),
    _Direct("elongation", "-", "KAPPA", "-", "plasma elongation from the equilibrium fit"),
    _Direct("triangularity", "-", "DELTA", "-", "mean triangularity of the boundary", positive=False),
    _Direct("cross_section_area", "m^2", "AREA", "m^2", "poloidal cross-section area"),
    _Direct("plasma_volume", "m^3", "VOL", "m^3", "plasma volume"),
    _Direct("edge_safety_factor_95", "-", "Q95", "-", "safety factor at 95 % poloidal flux, equilibrium fit",
            magnitude=True),
    _Direct("safety_factor_axis", "-", "QAXIS", "-", "safety factor on the magnetic axis", magnitude=True),
    _Direct("internal_inductance_li3", "-", "LI", "-", "2 int B_p^2 dV / (mu0^2 I_p^2 R_geo)"),
    _Direct("poloidal_beta", "-", "BEPMHD", "-", "absolute poloidal beta from the equilibrium fit"),
    _Direct("poloidal_beta_diamagnetic", "-", "BEPDIA", "-", "absolute corrected poloidal beta, diamagnetic loop"),
    _Direct("toroidal_beta", "%", "BETMHD", "-", "absolute toroidal beta from the equilibrium fit (a fraction; "
                                                  "x100 to percent)", 100.0),
    _Direct("normalized_beta", "% m T/MA", "BETNMHD", "% m T/MA",
            "100 BETMHD AMIN BT / IP[MA] (signed BT and IP in the source formula)", magnitude=True),
)

#: Which toroidal field and which radius the field-normalized columns use, as
#: ``table.attrs["conventions"]`` declares them (the equilibrium-state table declares
#: :data:`vaft.omas.equilibrium_state.EQUILIBRIUM_STATE_CONVENTIONS`, b0 at R_ref).
PR08_CONVENTIONS = {
    **{column: {"b_field_definition": "|BT|, vacuum toroidal field at the geometric axis",
                "radius_reference": "major_radius (RGEO, geometric, R_geo)", "radius_symbol": "R_geo"}
       for column in ("normalized_current", "normalized_beta", "toroidal_field", "inverse_cylindrical_q",
                      "kink_safety_factor_elliptic", "kink_safety_factor_cylindrical",
                      "edge_safety_factor_95_estimate_iter", "edge_safety_factor_95_estimate_start")},
    "internal_inductance_li3": {"b_field_definition": "none (B_p only)",
                                "radius_reference": "major_radius (RGEO): PR08 LI is normalised by R_geo",
                                "radius_symbol": "R_geo"},
}

#: Deterministic coordinates: column -> (unit, how).
_DERIVED = {
    "aspect_ratio": ("-", "major_radius / minor_radius"),
    "inverse_aspect_ratio": ("-", "vaft.formula.equilibrium.inverse_aspect_ratio_from_a_R(minor_radius, major_radius)"),
    "area_elongation": ("-", "cross_section_area / (pi minor_radius^2)"),
    "normalized_current": ("MA m^-1 T^-1", "plasma_current / (minor_radius toroidal_field)"),
    "normalized_beta": ("% m T/MA", "vaft.formula.stability.beta_N_from_beta_a_B0_Ip(toroidal_beta, minor_radius, "
                                    "toroidal_field, plasma_current), where BETNMHD is absent"),
    "inverse_cylindrical_q": ("-", "1 / vaft.formula.equilibrium.q_cyl_from_B_R_epsilon_kappa_I(toroidal_field, "
                                   "major_radius, inverse_aspect_ratio, area_elongation, plasma_current [A])"),
    "kink_safety_factor_elliptic": ("-", "vaft.formula.boundaries.kink_coordinates(a, R, B, kappa, I_p)"),
    "kink_safety_factor_cylindrical": ("-", "vaft.formula.boundaries.cylindrical_kink_coordinates(a, R, B, kappa, I_p)"),
    "edge_safety_factor_95_estimate_iter": ("-", "vaft.formula.boundaries.iter_q95_coordinates(a, R, B, kappa, "
                                                 "delta, I_p) with the boundary KAPPA and DELTA in place of the "
                                                 "kappa_95 and delta_95 it is defined on (the release has almost no "
                                                 "KAPPA95); the function's docstring puts the resulting over-estimate "
                                                 "near 27 % at the ITER design point"),
    "edge_safety_factor_95_estimate_start": ("-", "vaft.formula.boundaries.start_q95_coordinates(a, R, B, kappa, "
                                                  "delta, I_p, configuration) on the boundary shape as above; "
                                                  "configuration from CONFIG: limiter for LIM/TOP/BOT/OUT/IN/IW, "
                                                  "double_null for DN, missing for single null or unknown"),
}

_QUANTITIES = {d.column: ColumnSpec(d.unit, f"PR08 {d.source}: {d.definition}") for d in _DIRECT}
for _name, (_unit, _how) in _DERIVED.items():
    _QUANTITIES.setdefault(_name, ColumnSpec(_unit, f"Deterministic: {_how}"))
_QUANTITIES["li3_reference_radius"] = ColumnSpec("m", "Radius R in the l_i(3) of internal_inductance_li3 (RGEO).")

#: The canonical columns: identity, then each quantity followed by its provenance.
MHD_STATE_COLUMNS: dict[str, ColumnSpec] = dict(_IDENTITY)
for _name, _spec in _QUANTITIES.items():
    MHD_STATE_COLUMNS[_name] = _spec
    if _name != "li3_reference_radius":
        MHD_STATE_COLUMNS[f"{_name}_provenance"] = ColumnSpec("str", f"How {_name} was obtained (PROVENANCE_KINDS).")


# ---------------------------------------------------------------------------
# Release inventory and download
# ---------------------------------------------------------------------------

def pr08_release_inventory() -> pd.DataFrame:
    """The PR08 release as pinned in this package (offline).

    Returns
    -------
    pandas.DataFrame
        One row per discharge: ``machine_dir``, ``shot``, ``directory`` (its own
        release directory), ``sha256_0d`` and ``kinds`` (file kinds present).
    """
    from ._pr08_release import PR08_RELEASE

    rows = [{"machine_dir": m, "shot": s, "directory": d, "sha256_0d": h, "kinds": kinds}
            for (m, s), (d, h, kinds) in PR08_RELEASE.items()]
    return pd.DataFrame(rows, columns=["machine_dir", "shot", "directory", "sha256_0d", "kinds"])


_HREF = re.compile(r'href="([^"?/][^"]*)"')


def _listing(url: str, timeout: float) -> list:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            text = response.read().decode("latin-1")
    except Exception as exc:  # network, HTTP or decoding failure: an upstream problem
        raise FetchError(f"{url}: {exc}") from exc
    return _HREF.findall(text)


def pr08_inventory(*, base_url: str = PR08_BASE_URL, timeout: float = 60.0) -> pd.DataFrame:
    """Crawl the live PR08 directory index (network).

    Parameters
    ----------
    base_url : str, optional
        The ``itpa_pr08`` directory URL [str].
    timeout : float, optional
        Per-request timeout [s].

    Returns
    -------
    pandas.DataFrame
        One row per ``pr08_<machine>_<shot>_0d.dat`` file found: ``machine_dir``,
        ``shot``, ``directory``, ``kinds`` (of that shot, in that directory) and
        ``other_files`` (directory entries that are not a discharge file).
        Compare it with :func:`pr08_release_inventory` to see whether the pinned
        snapshot is still the release.

    Raises
    ------
    FetchError
        The index could not be read.
    """
    base = base_url.rstrip("/") + "/"
    rows = []
    for machine in [m[:-1] for m in _listing(base, timeout) if m.endswith("/")]:
        for directory in [s[:-1] for s in _listing(f"{base}{machine}/", timeout) if s.endswith("/")]:
            entries = _listing(f"{base}{machine}/{directory}/", timeout)
            pattern = re.compile(rf"pr08_{re.escape(machine)}_([^_/]+)_(0d|1d|2d|com)\.dat")
            found: dict = {}
            other = []
            for entry in entries:
                match = pattern.fullmatch(entry)
                if match:
                    found.setdefault(match.group(1), set()).add(match.group(2))
                else:
                    other.append(entry)
            for shot, kinds in sorted(found.items()):
                if "0d" in kinds:
                    rows.append({"machine_dir": machine, "shot": shot, "directory": directory,
                                 "kinds": tuple(k for k in ("0d", "1d", "2d", "com") if k in kinds),
                                 "other_files": tuple(other)})
    return pd.DataFrame(rows, columns=["machine_dir", "shot", "directory", "kinds", "other_files"])


@dataclass(frozen=True)
class Pr08ZeroD:
    """One discharge's 0D file, read."""

    machine_dir: str
    shot: str
    directory: str
    path: Path
    sha256: str
    zero_d: pd.DataFrame


def fetch_pr08_population(*, machines: Optional[Sequence[str]] = None, cache=None,
                          timeout: float = 120.0) -> list:
    """Download (checksum-verified) and read every pinned PR08 0D file.

    Parameters
    ----------
    machines : sequence of str, optional
        PR08 machine directories to fetch (``"jet"``, ``"mast"``, ...); default all.
    cache : path-like, optional
        Cache directory, default :func:`vaft.data.public._fetch.cache_dir`.
    timeout : float, optional
        Per-file timeout [s].

    Returns
    -------
    list of Pr08ZeroD
        One per discharge whose file arrived.  A file that cannot be downloaded
        is skipped with a warning naming it; a checksum mismatch raises.
    """
    inventory = pr08_release_inventory()
    if machines is not None:
        inventory = inventory[inventory["machine_dir"].isin(list(machines))]
    out = []
    for row in inventory.itertuples(index=False):
        name = f"pr08_{row.machine_dir}_{row.shot}_0d.dat"
        url = f"{PR08_BASE_URL}/{row.machine_dir}/{row.directory}/{name}"
        target = cache_dir(cache) / "pr08" / row.machine_dir / row.directory
        try:
            path = fetch(url, sha256=row.sha256_0d, filename=name, cache=target, timeout=timeout)
        except FetchError as exc:
            warnings.warn(f"PR08 {row.machine_dir}/{row.directory}/{name} not downloaded ({exc})", stacklevel=2)
            continue
        out.append(read_pr08_zero_d(path, row.machine_dir, row.shot, directory=row.directory))
    return out


def read_pr08_zero_d(path, machine_dir: str, shot: str, *, directory: Optional[str] = None) -> Pr08ZeroD:
    """Read one local PR08 ``_0d.dat`` file for :func:`pr08_mhd_state_table`.

    Parameters
    ----------
    path : path-like
        The 0D file [path].
    machine_dir : str
        PR08 machine directory (a key of :data:`PR08_MACHINES`) [str].
    shot : str
        PR08 shot [str].
    directory : str, optional
        Release directory the file sits in; default ``shot``.

    Returns
    -------
    Pr08ZeroD
    """
    path = Path(path)
    return Pr08ZeroD(machine_dir, str(shot), str(directory or shot), path, sha256_of(path), read_pr08_0d(path))


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

def _number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number


def _direct_value(spec: _Direct, raw) -> tuple:
    value = _number(raw)
    if not np.isfinite(value):
        return math.nan, "missing"
    if spec.magnitude:
        value = abs(value)
    if spec.positive and value <= 0.0:
        return math.nan, "source_invalid"
    return value * spec.scale, "source_direct"


#: A unit slip in the source is a power of ten (A for MA is 1e6); a ratio this far from 1
#: between two source quantities that must agree to tens of percent is a slip, not physics.
UNIT_SLIP_FACTOR = 100.0


def _unit_checks(row: dict) -> list:
    """Source values in the wrong unit, found by decisive internal consistency; each becomes source_invalid."""
    from vaft.formula import boundaries as _b

    notes = []
    beta = row["toroidal_beta"]
    if row["toroidal_beta_provenance"] in ("source_direct", "source_corrected") and not 0.0 <= beta <= 100.0:
        # BETMHD is a fraction: outside [0, 1] it is a sign or unit error (percent given as a fraction)
        notes.append(f"BETMHD = {beta / 100.0:g} is not a beta fraction in [0, 1]")
        row["toroidal_beta"], row["toroidal_beta_provenance"] = math.nan, "source_invalid"
    values = [row[c] for c in ("minor_radius", "major_radius", "toroidal_field", "elongation", "triangularity",
                               "plasma_current", "edge_safety_factor_95")]
    if all(np.isfinite(v) for v in values):
        try:
            estimate = float(_b.iter_q95_coordinates(*values[:6]))
        except (ValueError, ZeroDivisionError, FloatingPointError):
            estimate = math.nan
        ratio = estimate / values[6] if np.isfinite(estimate) else math.nan
        if np.isfinite(ratio) and not 1.0 / UNIT_SLIP_FACTOR < ratio < UNIT_SLIP_FACTOR:
            # q95 ~ 1/I_p: an IP stored in MA instead of A puts the estimate 1e6 off the source Q95
            notes.append(f"IP = {row['plasma_current'] * 1e6:g} A gives a q95 estimate {ratio:.3g} x the source "
                         f"Q95: IP is not in A")
            row["plasma_current"], row["plasma_current_provenance"] = math.nan, "source_invalid"
    return notes


#: PR08 CONFIG (manual) -> START configuration of Akers et al. (2000); single null has no constant.
_START_CONFIGURATION = {**{code: "limiter" for code in ("LIM", "TOP", "BOT", "OUT", "IN", "IW")},
                        "DN": "double_null"}


def _derive(row: dict) -> None:
    """Fill the deterministic columns of one row from its direct columns, through registered functions."""
    from vaft.formula import boundaries as _b
    from vaft.formula.equilibrium import inverse_aspect_ratio_from_a_R, q_cyl_from_B_R_epsilon_kappa_I
    from vaft.formula.stability import beta_N_from_beta_a_B0_Ip

    a, r, b = row["minor_radius"], row["major_radius"], row["toroidal_field"]
    ip, kappa, delta = row["plasma_current"], row["elongation"], row["triangularity"]
    area = row["cross_section_area"]

    def put(column, inputs, compute):
        if all(np.isfinite(v) for v in inputs):
            try:
                value = float(compute())
            except (ValueError, ZeroDivisionError, FloatingPointError):
                value = math.nan
            if np.isfinite(value):
                row[column], row[f"{column}_provenance"] = value, "deterministic_derived"
                return
        row[column], row[f"{column}_provenance"] = math.nan, "missing"

    put("aspect_ratio", (a, r), lambda: r / a)
    put("inverse_aspect_ratio", (a, r), lambda: inverse_aspect_ratio_from_a_R(a, r))
    put("area_elongation", (area, a), lambda: area / (math.pi * a * a))
    put("normalized_current", (ip, a, b), lambda: ip / (a * b))
    if row["normalized_beta_provenance"] == "missing":   # a rejected source value is not replaced
        put("normalized_beta", (row["toroidal_beta"], a, b, ip),
            lambda: beta_N_from_beta_a_B0_Ip(row["toroidal_beta"], a, b, ip))
    kappa_a = row["area_elongation"]
    put("inverse_cylindrical_q", (b, r, a, kappa_a, ip),
        lambda: 1.0 / q_cyl_from_B_R_epsilon_kappa_I(b, r, a / r, kappa_a, ip * 1e6))
    put("kink_safety_factor_elliptic", (a, r, b, kappa, ip), lambda: _b.kink_coordinates(a, r, b, kappa, ip))
    put("kink_safety_factor_cylindrical", (a, r, b, kappa, ip),
        lambda: _b.cylindrical_kink_coordinates(a, r, b, kappa, ip))
    put("edge_safety_factor_95_estimate_iter", (a, r, b, kappa, delta, ip),
        lambda: _b.iter_q95_coordinates(a, r, b, kappa, delta, ip))
    configuration = _START_CONFIGURATION.get(str(row.get("config") or "").strip().upper())
    put("edge_safety_factor_95_estimate_start", (a, r, b, kappa, delta, ip, 1.0 if configuration else math.nan),
        lambda: _b.start_q95_coordinates(a, r, b, kappa, delta, ip, configuration=configuration))


def _corrections(discharge: Pr08ZeroD) -> tuple:
    """The pinned corrections of this file, or none when its content is not the pinned release's."""
    from ._pr08_release import PR08_RELEASE

    key = (discharge.machine_dir, discharge.shot)
    pinned = PR08_RELEASE.get(key)
    if key not in SOURCE_CORRECTIONS or pinned is None or pinned[1] != discharge.sha256:
        return ()
    return SOURCE_CORRECTIONS[key]


def _label(value) -> Optional[str]:
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else str(value).strip()


def pr08_mhd_state_table(discharges: Iterable[Pr08ZeroD]) -> pd.DataFrame:
    """The canonical global MHD-state table of PR08 discharges.

    Parameters
    ----------
    discharges : iterable of Pr08ZeroD
        From :func:`fetch_pr08_population` or :func:`read_pr08_zero_d`.

    Returns
    -------
    pandas.DataFrame
        One row per 0D record, columns :data:`MHD_STATE_COLUMNS`;
        ``attrs["units"]`` (registry units, ``"-"`` dimensionless),
        ``attrs["descriptions"]`` and ``attrs["quantity_sources"]`` (per column:
        PR08 variable, unit, definition, transformation, or the derivation);
        ``attrs["conventions"]`` (:data:`PR08_CONVENTIONS`): the field and radius
        behind ``normalized_current`` and ``normalized_beta``.
    """
    rows = []
    seen: dict = {}
    for discharge in discharges:
        machine = PR08_MACHINES.get(discharge.machine_dir,
                                    Pr08Machine(discharge.machine_dir.upper(), "", "experimental"))
        corrections = _corrections(discharge)
        for index, raw in enumerate(discharge.zero_d.to_dict("records")):
            # names are upper case in the manual; a few files write them in lower case
            record = {str(key).strip().upper(): value for key, value in raw.items()}
            row = {name: None for name in MHD_STATE_COLUMNS}
            time_s = _number(record.get("TIME"))
            key = f"{machine.name}:{discharge.shot}:{int(round(time_s * 1000.0)) if np.isfinite(time_s) else 'na'}"
            seen[key] = seen.get(key, 0) + 1
            row.update(
                record_id=key if seen[key] == 1 else f"{key}#{seen[key]}",
                machine=machine.name, machine_class=machine.machine_class, dataset_type=machine.dataset_type,
                shot=discharge.shot, time_s=time_s, source_database=PR08_SOURCE_DATABASE,
                source_release=PR08_SOURCE_RELEASE,
                source_record_id=f"{discharge.machine_dir}/{discharge.directory}/{discharge.path.name}#{index}",
                source_kind="0D", source_sha256=discharge.sha256,
                phase=_label(record.get("PHASE")), state=_label(record.get("STATE")),
                config=_label(record.get("CONFIG")),
            )
            notes = []
            for name, factor, reason in corrections:
                if name in record and np.isfinite(_number(record[name])):
                    record[name] = _number(record[name]) * factor
                    notes.append(reason)
            corrected = {name for name, _factor, _reason in corrections}
            for spec in _DIRECT:
                value, kind = _direct_value(spec, record.get(spec.source))
                if kind == "source_direct" and spec.source in corrected:
                    kind = "source_corrected"
                row[spec.column], row[f"{spec.column}_provenance"] = value, kind
                if kind == "source_invalid":
                    notes.append(f"{spec.source} = {_number(record.get(spec.source)):g} cannot be {spec.column}")
            if machine.q95_not_equilibrium and row["edge_safety_factor_95_provenance"] == "source_direct":
                row["edge_safety_factor_95"], row["edge_safety_factor_95_provenance"] = math.nan, "source_invalid"
                notes.append(machine.q95_not_equilibrium)
            if not np.isfinite(time_s):
                notes.append("no TIME in the record")
            notes += _unit_checks(row)
            row["state_notes"] = "; ".join(notes)
            row["li3_reference_radius"] = (row["major_radius"]
                                           if row["internal_inductance_li3_provenance"] == "source_direct"
                                           else math.nan)
            _derive(row)
            rows.append(row)
    table = pd.DataFrame(rows, columns=list(MHD_STATE_COLUMNS))
    for name, spec in MHD_STATE_COLUMNS.items():
        if spec.unit not in ("str", "bool", "int"):
            table[name] = pd.to_numeric(table[name], errors="coerce").astype(float)
    table.attrs["units"] = {k: v.unit for k, v in MHD_STATE_COLUMNS.items()}
    table.attrs["descriptions"] = {k: v.description for k, v in MHD_STATE_COLUMNS.items()}
    sources = {d.column: {"source_variable": d.source, "source_unit": d.source_unit, "dimensionality": "0D",
                          "definition": d.definition,
                          "transformation": " ".join(filter(None, ("magnitude" if d.magnitude else "",
                                                                   f"x{d.scale:g}" if d.scale != 1.0 else ""))) or "none",
                          "reference": PR08_REFERENCE}
               for d in _DIRECT}
    for name, (_unit, how) in _DERIVED.items():
        sources.setdefault(name, {}).update({"deterministic": how})
    table.attrs["quantity_sources"] = sources
    table.attrs["conventions"] = {k: dict(v) for k, v in PR08_CONVENTIONS.items()}
    return table


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def _quantity_columns(table: pd.DataFrame) -> list:
    units = table.attrs.get("units", {})
    return [c for c, unit in units.items() if c in table.columns and unit not in ("str", "bool", "int")
            and c not in ("time_s", "li3_reference_radius")]


def mhd_state_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """States and finite values per machine and quantity.

    Parameters
    ----------
    table : pandas.DataFrame
        A canonical table with ``machine`` and ``attrs["units"]``.

    Returns
    -------
    pandas.DataFrame
        Index machine, columns ``states`` then one count per quantity column.
    """
    columns = _quantity_columns(table)
    counts = table.groupby("machine", sort=True)[columns].count()
    counts.insert(0, "states", table.groupby("machine", sort=True).size())
    return counts


def projection_coverage(table: pd.DataFrame) -> pd.DataFrame:
    """Which registered operational-space projections each machine's states can be drawn on.

    A state is eligible for a projection when both axis quantities are columns
    of the table in the axis unit and finite on that row; the registry
    (:func:`vaft.diagram._op_space.list_projections`) is read, not a list.

    Parameters
    ----------
    table : pandas.DataFrame
        A canonical table with ``machine`` and ``attrs["units"]``.

    Returns
    -------
    pandas.DataFrame
        One row per (projection, machine): ``eligible``, ``states`` and
        ``excluded_by`` (the axis quantities missing on the excluded rows, or
        why the projection is not supported by the table at all).
    """
    from vaft.diagram._op_space import get_projection, list_projections

    units = table.attrs.get("units", {})
    rows = []
    for key in list_projections():
        projection = get_projection(key)
        axes = (projection.x, projection.y)
        unsupported = [f"{q.name} [{q.unit}] not a column" if q.name not in units else
                       f"{q.name} in {units[q.name]}, axis in {q.unit}"
                       for q in axes if units.get(q.name) != q.unit]
        for machine, group in table.groupby("machine", sort=True):
            if unsupported:
                rows.append({"projection": key, "machine": machine, "eligible": 0, "states": len(group),
                             "excluded_by": "; ".join(unsupported)})
                continue
            finite = group[[q.name for q in axes]].notna()
            eligible = finite.all(axis=1)
            missing = [q.name for q in axes if (~finite[q.name] & ~eligible).any()]
            rows.append({"projection": key, "machine": machine, "eligible": int(eligible.sum()),
                         "states": len(group), "excluded_by": ", ".join(missing)})
    return pd.DataFrame(rows, columns=["projection", "machine", "eligible", "states", "excluded_by"])
