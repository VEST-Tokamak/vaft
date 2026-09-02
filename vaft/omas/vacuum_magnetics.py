"""Synthetic vacuum magnetics and plasma-residual QA for the eddy stage.

The eddy-current reconstruction is validated physically, by forward-modeling the
magnetic response of the reconstructed vacuum current system, rather than by
plotting the fitted eddy currents (issue #139).  For each magnetic observable
``i``::

    S_measured_i(t) = S_coil_i(t) + S_eddy_i(t) + S_plasma_i(t) + error_i(t)
    S_vacuum_i(t)   = S_coil_i(t) + S_eddy_i(t)
    S_residual_i(t) = S_measured_i(t) - S_vacuum_i(t)  ~  S_plasma_i(t)

so before plasma formation the residual should sit near the baseline noise, and
at plasma-current onset it should emerge coherently across channels at the
physically expected time.  A nonzero residual *after* breakdown is the plasma
signal and is not a failure; a large or incoherent residual *before* it is.

Conventions
-----------
Both sides of every comparison carry the same physical quantity and unit:

* flux loops compare ``magnetics.flux_loop.*.flux.data`` [Wb] against the
  Green's-function flux response, which :func:`vaft.formula.green.green_r`
  already returns as full poloidal flux in Wb -- no ``2*pi`` enters here;
* B probes compare ``magnetics.b_field_pol_probe.*.field.data`` [T] against the
  field response projected onto the probe's own sensitive axis,
  ``(cos(poloidal_angle), -sin(poloidal_angle))`` in ``(R, Z)`` -- the IMAS
  convention, read per channel from the ODS so a probe mounted differently from
  the rest projects differently (issue #169).  An ODS that stores no angle for a
  channel falls back to
  :data:`~vaft.machine_mapping.magnetics.POLOIDAL_ANGLE`, VEST's +Bz default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from vaft.ods_access import path_count as _count, path_value
from vaft.formula.statistics import (
    fractional_rms_improvement,
    rms,
    sigma_threshold_crossing,
)
from vaft.validation.imas import (
    VALIDITY_VALID,
    resolve_signal_time,
    validity_mask,
)
from vaft.machine_mapping.magnetics import (
    INBOARD_FLUX_LOOP_MAX_R,
    INBOARD_PROBE_MAX_R,
    OUTBOARD_FLUX_LOOP_MIN_R,
    OUTBOARD_PROBE_MIN_R,
    POLOIDAL_ANGLE,
    SIDE_PROBE_MIN_ABS_Z,
    vfit_plasma_mgods_startend,
)

__all__ = [
    "VacuumChannel",
    "VacuumMagneticsError",
    "eddy_improvement",
    "plasma_onset_time",
    "probe_family",
    "residual_onset",
    "residual_rms",
    "select_vacuum_channels",
    "synthetic_vacuum_magnetics",
    "vacuum_magnetics_metrics",
]

B_FIELD_POL_PROBE = "b_field_pol_probe"
FLUX_LOOP = "flux_loop"

#: How many channels of each family the default selection keeps.  Two per family
#: gives the inboard/outboard B-probe and flux-loop coverage issue #139 requires
#: -- enough that agreement cannot be a local response-matrix coincidence -- and
#: keeps the validation figures readable and the forward model fast.
DEFAULT_PER_FAMILY = 2

#: Residual onset is declared where the residual first leaves the pre-plasma
#: noise band by this many standard deviations.
ONSET_SIGMA = 5.0


class VacuumMagneticsError(ValueError):
    """Raised when an ODS cannot support the vacuum-magnetics forward model."""


@dataclass(frozen=True)
class VacuumChannel:
    """One magnetic observable, measured and forward-modeled on a shared grid."""

    name: str
    kind: str
    family: str
    index: int
    r: float
    z: float
    unit: str
    time: np.ndarray
    measured: np.ndarray
    coil: np.ndarray
    coil_eddy: np.ndarray

    @property
    def residual(self) -> np.ndarray:
        """Measured minus the coil+eddy synthetic -- the plasma signal estimate."""
        return self.measured - self.coil_eddy

    @property
    def coil_residual(self) -> np.ndarray:
        """Measured minus the coil-only synthetic, the eddy term's baseline."""
        return self.measured - self.coil


# ---------------------------------------------------------------------------
# Scalar QA helpers.  Pure array functions, so the physics tests can drive them
# directly with synthetic signals.
# ---------------------------------------------------------------------------

def residual_rms(residual: np.ndarray, window: np.ndarray) -> float:
    """RMS of ``residual`` over the boolean ``window``."""
    return rms(np.asarray(residual, dtype=float)[window])


def eddy_improvement(
    coil_residual: np.ndarray, residual: np.ndarray, window: np.ndarray
) -> float:
    """``1 - RMS(measured - coil+eddy) / RMS(measured - coil)`` over ``window``.

    1.0 is a perfect vacuum reconstruction, 0.0 means the eddy term added
    nothing, and a negative value means it made the agreement worse.
    """
    return fractional_rms_improvement(
        np.asarray(coil_residual, dtype=float)[window],
        np.asarray(residual, dtype=float)[window],
    )


def residual_onset(
    time: np.ndarray,
    residual: np.ndarray,
    window: np.ndarray,
    *,
    sigma: float = ONSET_SIGMA,
) -> float:
    """First time after ``window`` where ``residual`` leaves its noise band.

    The band is the mean and standard deviation of the residual *inside*
    ``window`` (the pre-plasma stretch), so the threshold is the channel's own
    measured noise rather than a global constant.  ``nan`` when the residual
    never emerges.
    """
    return sigma_threshold_crossing(time, residual, window, sigma=sigma)


def plasma_onset_time(ods: Any) -> float:
    """The stage's own plasma-current onset, from ``magnetics.ip``."""
    start, end = vfit_plasma_mgods_startend(ods)
    if start < 0 or end <= start:
        raise VacuumMagneticsError(
            "cannot locate the plasma-current onset from magnetics.ip.0; "
            "the eddy ODS carries no usable plasma current"
        )
    return float(start)


def probe_family(kind: str, r: float, z: float) -> str:
    """The EFIT-submitted family a channel at ``(r, z)`` belongs to."""
    if kind == FLUX_LOOP:
        if r < INBOARD_FLUX_LOOP_MAX_R:
            return "inboard_flux_loop"
        if r > OUTBOARD_FLUX_LOOP_MIN_R:
            return "outboard_flux_loop"
        return "flux_loop"
    if abs(z) > SIDE_PROBE_MIN_ABS_Z:
        return "side"
    if r < INBOARD_PROBE_MAX_R:
        return "inboard"
    if r > OUTBOARD_PROBE_MIN_R:
        return "outboard"
    return "other"


# ---------------------------------------------------------------------------
# Channel selection and the forward model
# ---------------------------------------------------------------------------

def _signal(ods: Any, path: str) -> np.ndarray | None:
    """A 1-D waveform at ``path``, or ``None`` when the ODS carries none.

    Through the shared non-mutating accessor (issue #118): a bare read would
    materialize a placeholder at every channel this probes and finds empty.
    """
    values = path_value(ods, path)
    if values is None:
        return None
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim != 1 or array.size < 2:
        return None
    return array


def _usable_in_window(
    ods: Any,
    base: str,
    time: np.ndarray,
    window: tuple[float, float] | None,
    min_validity: int,
) -> bool:
    """Whether the IDS says this channel has any usable sample in ``window``.

    Step one of the three #189 asks a downstream consumer to take: consult the
    validity the diagnostics stage established, *then* apply the consumer's own
    preconditions.  Signal health is not rediscovered here -- an integrator
    that railed at 0.31 s is the diagnostics stage's finding, and the eddy
    stage only needs to know whether anything usable is left in the interval it
    cares about.

    An ODS carrying no validity accepts everything, so this changes nothing for
    data produced before the quality layer existed.
    """
    accepted = validity_mask(ods, base, min_validity=min_validity)
    if accepted.size != time.size:
        return bool(accepted.all())
    if window is not None:
        inside = (time >= window[0]) & (time <= window[1])
        if not inside.any():
            return False
        accepted = accepted[inside]
    return bool(accepted.any())


def _position(ods: Any, base: str) -> tuple[float, float] | None:
    """A channel's ``(r, z)``, or ``None`` when the ODS does not carry one.

    A channel with a waveform but no stored geometry cannot be forward-modelled,
    so it is skipped rather than crashed on -- and probed through the shared
    accessor, because a bare read would leave a ``position`` branch behind on
    the way to that crash.
    """
    r, z = path_value(ods, f"{base}.r"), path_value(ods, f"{base}.z")
    if r is None or z is None:
        return None
    try:
        return float(r), float(z)
    except (TypeError, ValueError):
        return None


def _poloidal_angle(ods: Any, base: str) -> float:
    """The probe's sensitive-axis angle, defaulting to VEST's +Bz orientation.

    Read per channel rather than taken from the constant so an ODS carrying a
    measured misalignment -- the IMPA Bz sensors write one -- projects onto the
    axis it declares.
    """
    stored = path_value(ods, f"{base}.poloidal_angle")
    if stored is None:
        return POLOIDAL_ANGLE
    try:
        angle = float(stored)
    except (TypeError, ValueError):
        return POLOIDAL_ANGLE
    return angle if np.isfinite(angle) else POLOIDAL_ANGLE


def _candidates(
    ods: Any,
    *,
    window: tuple[float, float] | None = None,
    min_validity: int = VALIDITY_VALID,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(_count(ods, f"magnetics.{B_FIELD_POL_PROBE}")):
        base = f"magnetics.{B_FIELD_POL_PROBE}.{index}"
        data = _signal(ods, f"{base}.field.data")
        time = resolve_signal_time(ods, f"{base}.field")
        if data is None or time is None or data.size != time.size:
            continue
        if not _usable_in_window(ods, f"{base}.field", time, window, min_validity):
            continue
        # Then the eddy stage's own precondition, which is a different question
        # from health: a flatlined channel may be perfectly valid data and
        # still carry no information to correlate a forward model against.
        if not np.isfinite(data).any() or float(np.nanstd(data)) == 0.0:
            continue
        position = _position(ods, f"{base}.position")
        if position is None:
            continue
        r, z = position
        rows.append(
            {
                "kind": B_FIELD_POL_PROBE,
                "index": index,
                "r": r,
                "z": z,
                "poloidal_angle": _poloidal_angle(ods, base),
                "unit": "T",
                "family": probe_family(B_FIELD_POL_PROBE, r, z),
                "name": str(ods.get(f"{base}.name", f"Bp {index}") or f"Bp {index}"),
                "time": time,
                "data": data,
            }
        )
    for index in range(_count(ods, f"magnetics.{FLUX_LOOP}")):
        base = f"magnetics.{FLUX_LOOP}.{index}"
        data = _signal(ods, f"{base}.flux.data")
        time = resolve_signal_time(ods, f"{base}.flux")
        if data is None or time is None or data.size != time.size:
            continue
        if not _usable_in_window(ods, f"{base}.flux", time, window, min_validity):
            continue
        if not np.isfinite(data).any() or float(np.nanstd(data)) == 0.0:
            continue
        position = _position(ods, f"{base}.position.0")
        if position is None:
            continue
        r, z = position
        rows.append(
            {
                "kind": FLUX_LOOP,
                "index": index,
                "r": r,
                "z": z,
                "unit": "Wb",
                "family": probe_family(FLUX_LOOP, r, z),
                "name": str(ods.get(f"{base}.name", f"FL {index}") or f"FL {index}"),
                "time": time,
                "data": data,
            }
        )
    return rows


def select_vacuum_channels(
    ods: Any,
    *,
    per_family: int = DEFAULT_PER_FAMILY,
    channels: Sequence[tuple[str, int]] | None = None,
    window: tuple[float, float] | None = None,
    min_validity: int = VALIDITY_VALID,
) -> list[dict[str, Any]]:
    """Choose the magnetic channels to validate, grouped by EFIT family.

    Without ``channels`` this keeps up to ``per_family`` live channels from each
    family, which gives the inboard/outboard B-probe and flux-loop coverage the
    validation contract requires.  Pass ``channels`` as ``(kind, index)`` pairs
    to select explicitly.

    ``window`` restricts the validity question to the interval the caller
    actually validates over, so a channel that fails late in the discharge is
    still selected for a pre-plasma comparison.  ``min_validity`` is the floor
    on the Data Dictionary code; the default accepts flagged-but-valid channels,
    because a threshold that has not been justified on a VEST population must
    not silently remove data (#189, non-goals).
    """
    rows = _candidates(ods, window=window, min_validity=min_validity)
    if channels is not None:
        wanted = {(str(kind), int(index)) for kind, index in channels}
        chosen = [row for row in rows if (row["kind"], row["index"]) in wanted]
        missing = wanted - {(row["kind"], row["index"]) for row in chosen}
        if missing:
            listed = ", ".join(f"{kind}[{index}]" for kind, index in sorted(missing))
            raise VacuumMagneticsError(
                f"requested magnetic channels are absent or carry no usable data: {listed}"
            )
        return chosen

    selected: list[dict[str, Any]] = []
    for family in ("inboard", "outboard", "side", "other", "inboard_flux_loop",
                   "outboard_flux_loop", "flux_loop"):
        members = [row for row in rows if row["family"] == family]
        selected.extend(members[:per_family])
    return selected


def _currents(ods: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coil_count = _count(ods, "pf_active.coil")
    loop_count = _count(ods, "pf_passive.loop")
    if coil_count == 0:
        raise VacuumMagneticsError("ODS carries no pf_active coils")
    if loop_count == 0:
        raise VacuumMagneticsError(
            "ODS carries no pf_passive loops; the eddy stage has not run on it"
        )
    time = np.asarray(ods["pf_active.time"], dtype=float)
    coil = np.array(
        [np.asarray(ods[f"pf_active.coil.{i}.current.data"], dtype=float)
         for i in range(coil_count)]
    )
    loop = np.array(
        [np.asarray(ods[f"pf_passive.loop.{i}.current"], dtype=float)
         for i in range(loop_count)]
    )
    if coil.shape[1] != time.size or loop.shape[1] != time.size:
        raise VacuumMagneticsError(
            "pf_active/pf_passive currents do not share the pf_active time grid"
        )
    return time, coil, loop


def synthetic_vacuum_magnetics(
    ods: Any,
    *,
    per_family: int = DEFAULT_PER_FAMILY,
    channels: Sequence[tuple[str, int]] | None = None,
    window: tuple[float, float] | None = None,
    min_validity: int = VALIDITY_VALID,
) -> tuple[VacuumChannel, ...]:
    """Forward-model the coil and coil+eddy response at selected magnetics.

    The synthetic signals live on the ``pf_active`` time grid, which the eddy
    solve also writes ``pf_passive`` onto; measured signals are interpolated onto
    it and everything is clipped to the overlap of the two grids rather than
    extrapolated beyond where the diagnostics were mapped.
    """
    from vaft.omas.process_wrapper import compute_point_response_ods

    # Checked first: "the eddy solve has not run on this ODS" is the more
    # actionable diagnosis than "no usable channels" when both are true.
    time, coil_currents, loop_currents = _currents(ods)
    rows = select_vacuum_channels(
        ods,
        per_family=per_family,
        channels=channels,
        window=window,
        min_validity=min_validity,
    )
    if not rows:
        raise VacuumMagneticsError(
            "no magnetic channel carries usable measured data for vacuum validation"
        )
    n_coil, n_loop = coil_currents.shape[0], loop_currents.shape[0]

    measured_end = min(float(row["time"][-1]) for row in rows)
    measured_start = max(float(row["time"][0]) for row in rows)
    inside = (time >= measured_start) & (time <= measured_end)
    if inside.sum() < 2:
        raise VacuumMagneticsError(
            "the pf_active grid and the mapped magnetics do not overlap in time"
        )
    time = time[inside]
    coil_currents = coil_currents[:, inside]
    loop_currents = loop_currents[:, inside]

    psi, b_z, b_r = compute_point_response_ods(
        ods, [[row["r"], row["z"]] for row in rows]
    )

    built: list[VacuumChannel] = []
    for position, row in enumerate(rows):
        if row["kind"] == FLUX_LOOP:
            response = psi[position]
        else:
            angle = row["poloidal_angle"]
            # DD: poloidal_angle is clockwise from +R, so the sensitive axis is
            # (cos, -sin) in (R, Z).  Projecting with (cos, +sin) inverts every
            # probe; it did so here until issue #288, cancelling against a
            # stored angle that was wrong the same way.
            response = b_r[position] * np.cos(angle) - b_z[position] * np.sin(angle)
        coil = response[:n_coil] @ coil_currents
        eddy = response[n_coil : n_coil + n_loop] @ loop_currents
        built.append(
            VacuumChannel(
                name=row["name"],
                kind=row["kind"],
                family=row["family"],
                index=row["index"],
                r=row["r"],
                z=row["z"],
                unit=row["unit"],
                time=time,
                measured=np.interp(time, row["time"], row["data"]),
                coil=coil,
                coil_eddy=coil + eddy,
            )
        )
    return tuple(built)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def vacuum_magnetics_metrics(
    channels: Iterable[VacuumChannel],
    *,
    plasma_onset: float,
    plasma_current: tuple[np.ndarray, np.ndarray] | None = None,
    sigma: float = ONSET_SIGMA,
) -> dict[str, Any]:
    """Quantitative QA for one shot's vacuum-magnetics validation.

    Everything issue #139 asks the eddy stage to record: the pre-plasma residual
    RMS with and without the eddy response, the improvement from adding it, the
    residual and plasma-current onsets and their difference, and how coherently
    the residual onset appears across channels.

    ``plasma_onset`` bounds the pre-plasma validation window and comes from the
    pipeline's own Ip-based detector, which is deliberately conservative.  The
    onset *comparison* uses ``plasma_current`` -- the measured ``(time, Ip)`` --
    put through the same 5-sigma-above-window-noise rule as every residual, so
    ``onset_delta`` compares like with like instead of two different detectors.
    Without it the delta falls back to the window boundary.
    """
    channels = tuple(channels)
    if not channels:
        raise VacuumMagneticsError("no channels to compute vacuum metrics for")

    reference_time = channels[0].time
    reference_window = reference_time < plasma_onset
    current_onset = float("nan")
    if plasma_current is not None:
        ip_time, ip_values = plasma_current
        current_onset = residual_onset(
            reference_time,
            np.interp(reference_time, np.asarray(ip_time, float), np.asarray(ip_values, float)),
            reference_window,
            sigma=sigma,
        )
    if not np.isfinite(current_onset):
        current_onset = float(plasma_onset)

    rows: list[dict[str, Any]] = []
    for channel in channels:
        window = channel.time < plasma_onset
        if window.sum() < 2:
            raise VacuumMagneticsError(
                f"channel {channel.name!r} has no pre-plasma samples before "
                f"t={plasma_onset:.5f}s to validate against"
            )
        onset = residual_onset(channel.time, channel.residual, window, sigma=sigma)
        rows.append(
            {
                "name": channel.name,
                "kind": channel.kind,
                "family": channel.family,
                "index": channel.index,
                "unit": channel.unit,
                "pre_plasma_samples": int(window.sum()),
                "residual_rms_coil": residual_rms(channel.coil_residual, window),
                "residual_rms_coil_eddy": residual_rms(channel.residual, window),
                "improvement": eddy_improvement(
                    channel.coil_residual, channel.residual, window
                ),
                "residual_onset": onset,
                "onset_delta": (
                    float(onset - current_onset) if np.isfinite(onset) else float("nan")
                ),
            }
        )

    onsets = np.array(
        [row["residual_onset"] for row in rows if np.isfinite(row["residual_onset"])]
    )
    improvements = np.array(
        [row["improvement"] for row in rows if np.isfinite(row["improvement"])]
    )
    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in rows if row["family"] == family]
        family_improvements = [
            row["improvement"] for row in members if np.isfinite(row["improvement"])
        ]
        families[family] = {
            "channels": len(members),
            "median_improvement": (
                float(np.median(family_improvements)) if family_improvements else float("nan")
            ),
        }

    return {
        "schema_version": 1,
        "plasma_onset": float(plasma_onset),
        "plasma_current_onset": float(current_onset),
        "channels": rows,
        "families": families,
        "summary": {
            "channel_count": len(rows),
            "median_improvement": (
                float(np.median(improvements)) if improvements.size else float("nan")
            ),
            "min_improvement": (
                float(np.min(improvements)) if improvements.size else float("nan")
            ),
            # How tightly the residual appears across channels: a real plasma
            # emerges everywhere at once, a response-matrix artifact does not.
            "onset_coherence": (
                float(np.ptp(onsets)) if onsets.size > 1 else float("nan")
            ),
            "median_onset_delta": (
                float(np.median(onsets) - current_onset) if onsets.size else float("nan")
            ),
            "channels_without_onset": sum(
                1 for row in rows if not np.isfinite(row["residual_onset"])
            ),
        },
    }
