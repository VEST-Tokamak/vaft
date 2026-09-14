"""TRANSP profiles as VAFT's kinetic container: the one conversion in this layer.

Every other module under :mod:`vaft.code` keeps its code's own names and
units and converts nothing -- :mod:`~vaft.code.transp.outputs` says so at
length.  This one converts, deliberately, because the alternative is worse:
turning a TRANSP run into a ``.kin`` or a pfile needs the two radial grids
reconciled and the electrostatic potential differentiated, and doing that at
each call site is how the grids came to be conflated in the first place.  So
it happens once, here, next to the transcript it reads, and everything it
decides is written into :attr:`KineticProfiles.provenance`.

**The result is on the zone-boundary grid.**  ``psi_norm`` is
``PLFLX / PLFLXA``, both of which live on ``XB``, and the E x B frequency is
a derivative of ``VRPOT`` -- also on ``XB`` -- with respect to that flux.  So
choosing ``XB`` leaves the two quantities that a consumer keys on untouched,
and interpolates only the smooth kinetic profiles that arrive on ``X``.
Measured on the MAST reference run at 750 ms, the choice is not symmetric:

- onto ``XB`` the edge ``psi_norm`` stays exactly 1.0 -- the last closed flux
  surface, which is what GPEC's ``read_kin`` re-splines its uniform [0, 1]
  grid against -- and the cost is extrapolating the outermost zone centre,
  which moves ``n_e`` by at most 4.2% of its peak;
- onto ``X`` the edge falls to 0.9857, losing that surface outright, and
  ``psi_norm`` moves by up to 0.036.

``target_grid="X"`` is available for a caller who would rather not touch the
measured profiles, and records that it was chosen.

The converter this replaces did the opposite by default -- it computed the
E x B frequency on ``XB`` and then interpolated it onto ``X`` along with a
``psi_norm`` it had already mis-normalised.

**Units come from the file.**  Each variable declares its own, and an
unrecognised one raises rather than being assumed; a wrong guess between
``N/CM**3`` and ``N/M**3`` is a silent factor of a million, which is exactly
what a ``--density-unit`` flag on the converter this replaces could produce.
The E x B frequency needs no conversion at all and it is worth saying why:
``VRPOT`` is in volts and ``PLFLX`` in webers per radian, and a volt per
weber is an inverse second, so ``-dPhi/dpsi`` is already rad/s.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from vaft.data.kinetic_profiles import KineticProfiles, PsiNormalization

from .outputs import TranspFormatError, TranspSlice, read_transp_output

__all__ = [
    "PROFILE_SOURCES",
    "TRANSP_UNIT_LADDER",
    "exb_frequency",
    "kinetic_profiles_from_slice",
    "read_transp_profiles",
]

#: Which TRANSP variable fills which container field, and the grid it is on.
#: Every one of these is a zone-centre quantity; the two that are not --
#: ``psi_norm`` and the E x B frequency -- are derived rather than copied.
PROFILE_SOURCES: Mapping[str, str] = {
    "n_e": "NE",
    "n_i": "NI",
    "T_e": "TE",
    "T_i": "TI",
    "omega_tor": "OMEGA",
}

#: The unit each variable may declare, and the factor into the container's.
#: Read from the file and looked up here; an unrecognised string raises.
#: Spelling is normalised for lookup -- TRANSP writes ``N/CM**3`` and
#: ``RAD/SEC`` -- but the string as written is what provenance records.
TRANSP_UNIT_LADDER: Mapping[str, tuple[str, float]] = {
    "n/cm**3": ("m^-3", 1.0e6),
    "n/m**3": ("m^-3", 1.0),
    "ev": ("eV", 1.0),
    "kev": ("eV", 1.0e3),
    "rad/sec": ("rad/s", 1.0),
    "rad/s": ("rad/s", 1.0),
}

#: The grid the conversion lands on unless a caller says otherwise.
DEFAULT_TARGET_GRID = "XB"


def exb_frequency(state: TranspSlice) -> np.ndarray:
    """The E x B frequency on the zone boundaries [rad/s].

    ``-dPhi/dpsi`` from ``VRPOT`` and ``PLFLX``, both of which TRANSP writes
    on ``XB``, so nothing is interpolated to compute it.  No unit conversion
    either: volts per weber-per-radian is radians per second.

    The sign is TRANSP's own.  ``VRPOT`` is the radial electrostatic
    potential and ``PLFLX`` the poloidal flux per radian measured from the
    axis, so the E x B rotation frequency is minus the derivative of the one
    with respect to the other.
    """
    potential = state.on_xb("VRPOT")
    flux = state.on_xb("PLFLX")
    for name, unit, expected in (
        ("VRPOT", state.units("VRPOT"), "VOLTS"),
        ("PLFLX", state.units("PLFLX"), "Wb/rad"),
    ):
        if unit.lower() != expected.lower():
            raise TranspFormatError(
                f"{name} declares {unit!r} rather than {expected!r}; the E x B "
                "frequency is a volt per weber-per-radian and comes out in rad/s "
                "only because of those two units, so a different pair needs a "
                "conversion this function does not make"
            )
    return -np.gradient(
        np.asarray(potential, dtype=float), np.asarray(flux, dtype=float), edge_order=2
    )


def _to_container_units(state: TranspSlice, name: str, field: str) -> tuple[np.ndarray, str]:
    """A zone-centre variable in the container's units, refusing an unknown one."""
    from vaft.data.kinetic_profiles import KINETIC_UNITS

    declared = state.units(name)
    unit, factor = TRANSP_UNIT_LADDER.get(declared.lower(), (None, None))
    if unit is None:
        raise TranspFormatError(
            f"{name} declares the unit {declared!r}, which this module cannot convert "
            f"to {KINETIC_UNITS[field]} for {field}. Guessing a factor here is how a "
            "density ends up wrong by a million"
        )
    if unit != KINETIC_UNITS[field]:
        raise TranspFormatError(
            f"{name} is in {declared!r}, which converts to {unit}, but {field} is "
            f"carried in {KINETIC_UNITS[field]}"
        )
    return np.asarray(state.on_x(name), dtype=float) * factor, declared


def _onto(
    values: np.ndarray, source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, int]:
    """``values`` interpolated from ``source`` onto ``target``, counting clamps.

    ``numpy.interp`` holds the end values beyond the source range rather than
    extrapolating, which is the conservative choice for a profile whose
    outermost point is already the least trustworthy -- but it is a choice,
    so the number of points it applies to comes back with the result and is
    recorded.
    """
    outside = int(np.count_nonzero((target < source[0]) | (target > source[-1])))
    return np.interp(target, source, values), outside


def kinetic_profiles_from_slice(
    state: TranspSlice,
    *,
    target_grid: str = DEFAULT_TARGET_GRID,
) -> KineticProfiles:
    """Convert one time of a TRANSP run into the kinetic container.

    ``psi_norm`` is ``PLFLX / PLFLXA`` -- the flux enclosed by each zone
    boundary over the flux enclosed by the plasma boundary -- and not
    ``(P - P[0]) / (P[-1] - P[0])``, which would declare the innermost zone
    boundary to be the magnetic axis and move the whole grid inward.

    ``target_grid`` is ``"XB"`` by default; see the module docstring for what
    the choice costs either way.  Whichever is chosen, which quantities were
    interpolated and how many points fell outside their source range is
    recorded in the result's provenance.
    """
    grid = str(target_grid).upper()
    if grid not in ("X", "XB"):
        raise TranspFormatError(
            f"target_grid must be 'X' or 'XB', not {target_grid!r}; TRANSP has those "
            "two radial grids and interpolating onto a third is a caller's business"
        )

    x, xb = np.asarray(state.x, dtype=float), np.asarray(state.xb, dtype=float)
    target = xb if grid == "XB" else x
    source_name = Path(state._output.path).name

    psi_xb = np.asarray(state.psi_norm_xb, dtype=float)
    omega_exb_xb = exb_frequency(state)

    provenance: dict[str, str] = {
        "psi_norm": (
            f"{source_name} PLFLX / PLFLXA at t = {state.time_s:.6f} s, on XB"
        ),
        "omega_exb": (
            f"{source_name} -d(VRPOT)/d(PLFLX) at t = {state.time_s:.6f} s, on XB "
            "[VOLTS per Wb/rad = rad/s]"
        ),
        "target_grid": (
            f"reported on the {grid} grid; TRANSP writes psi_norm and the E x B "
            f"frequency on XB and the kinetic profiles on X"
        ),
    }

    if grid == "XB":
        psi_norm, omega_exb = psi_xb, omega_exb_xb
    else:
        psi_norm, clamped_psi = _onto(psi_xb, xb, x)
        omega_exb, clamped_omega = _onto(omega_exb_xb, xb, x)
        provenance["psi_norm"] += f", interpolated onto X ({clamped_psi} points clamped)"
        provenance["omega_exb"] += f", interpolated onto X ({clamped_omega} points clamped)"

    fields: dict[str, np.ndarray] = {"omega_exb": omega_exb}
    for field, name in PROFILE_SOURCES.items():
        if name not in state._output.variables:
            continue
        values, declared = _to_container_units(state, name, field)
        note = f"{source_name} {name} at t = {state.time_s:.6f} s, on X [{declared}]"
        if grid == "XB":
            values, clamped = _onto(values, x, xb)
            note += f", interpolated onto XB ({clamped} points clamped)"
        fields[field] = values
        provenance[field] = note

    return KineticProfiles(
        psi_norm=psi_norm,
        **fields,
        normalization=PsiNormalization(
            method="transp_plflx_over_plflxa",
            source=source_name,
            axis_value=0.0,
            edge_value=1.0,
        ),
        extras={"x": x, "xb": xb},
        provenance=provenance,
        source=str(state._output.path),
    )


def read_transp_profiles(
    path: str | Path,
    *,
    time_s: float,
    target_grid: str = DEFAULT_TARGET_GRID,
) -> KineticProfiles:
    """Read one time of a TRANSP ``.CDF`` as kinetic profiles.

    A convenience over :func:`~vaft.code.transp.outputs.read_transp_output`
    and :func:`kinetic_profiles_from_slice` for the common case of wanting
    one time out of a run.  The sample actually taken is the nearest to
    ``time_s`` and is recorded in the result's provenance, because a run's
    samples are irregular and which one was used is part of the answer.
    """
    with read_transp_output(path) as output:
        return kinetic_profiles_from_slice(
            output.slice(float(time_s)), target_grid=target_grid
        )
