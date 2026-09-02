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
    "channel_residual_metrics",
    "eddy_improvement",
    "evaluation_mask",
    "plasma_free_residual",
    "plasma_onset_time",
    "probe_family",
    "residual_onset",
    "residual_rms",
    "select_vacuum_channels",
    "synthetic_vacuum_magnetics",
    "vacuum_magnetics_metrics",
    "vacuum_response",
    "vacuum_residual_metrics",
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
    #: Which samples the diagnostics stage says are usable (issue #189).
    #: ``None`` when the ODS carries no validity, which is not the same as
    #: "none are usable" -- see :attr:`usable`.
    valid: np.ndarray | None = None

    @property
    def usable(self) -> np.ndarray:
        """The per-sample validity mask, all-True when the ODS declares none."""
        if self.valid is None:
            return np.ones(self.time.size, dtype=bool)
        return np.asarray(self.valid, dtype=bool)

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

    Through the shared non-mutating accessor (issue #118): an ODS creates paths
    on access, so probing a channel that has no waveform would leave a malformed
    leaf behind -- invisible to ``flat()`` and fatal to the next consistency
    check.
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
    per_family: int | None = DEFAULT_PER_FAMILY,
    channels: Sequence[tuple[str, int]] | None = None,
    window: tuple[float, float] | None = None,
    min_validity: int = VALIDITY_VALID,
) -> list[dict[str, Any]]:
    """Choose the magnetic channels to validate, grouped by EFIT family.

    Without ``channels`` this keeps up to ``per_family`` live channels from each
    family, which gives the inboard/outboard B-probe and flux-loop coverage the
    routine validation contract requires.  ``per_family=None`` keeps every
    usable channel instead -- the compact subset is right for per-shot QA and
    too small to qualify a machine model (issue #190).  Pass ``channels`` as
    ``(kind, index)`` pairs to select explicitly.

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
    """Coil and passive-loop currents on the `pf_active` grid.

    Both are read from the ODS. A plasma-free benchmark that needs the wall
    driven by the PF coils alone does not inject currents here; it passes the
    ODS that :func:`vaft.validation.vacuum_benchmark.benchmark_wall_currents`
    returns, whose loops were solved without any plasma source (#190).
    """
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


def vacuum_response(
    ods: Any,
    *,
    per_family: int | None = DEFAULT_PER_FAMILY,
    channels: Sequence[tuple[str, int]] | None = None,
    window: tuple[float, float] | None = None,
    validity_window: tuple[float, float] | None = None,
    min_validity: int = VALIDITY_VALID,
) -> tuple[list[dict[str, Any]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Select channels and compute their geometry response, once.

    Returns the selected rows and the ``(psi, b_z, b_r)`` Green's-function
    triple for them. The triple is a function of geometry alone, so it can be
    handed to :func:`synthetic_vacuum_magnetics` as ``response=`` for every
    re-solve of the wall with different resistances (#308). Returning the rows
    alongside it is what lets the consumer verify the two still agree.
    """
    from vaft.omas.process_wrapper import compute_point_response_ods

    rows = select_vacuum_channels(
        ods,
        per_family=per_family,
        channels=channels,
        window=window if validity_window is None else validity_window,
        min_validity=min_validity,
    )
    if not rows:
        raise VacuumMagneticsError(
            "no magnetic channel carries usable measured data for vacuum validation"
        )
    positions = np.array([[row["r"], row["z"]] for row in rows], dtype=float)
    psi, b_z, b_r = compute_point_response_ods(ods, positions.tolist())
    return rows, (psi, b_z, b_r, positions)


def synthetic_vacuum_magnetics(
    ods: Any,
    *,
    per_family: int | None = DEFAULT_PER_FAMILY,
    channels: Sequence[tuple[str, int]] | None = None,
    window: tuple[float, float] | None = None,
    validity_window: tuple[float, float] | None = None,
    min_validity: int = VALIDITY_VALID,
    response: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[VacuumChannel, ...]:
    """Forward-model the coil and coil+eddy response at selected magnetics.

    The synthetic signals live on the ``pf_active`` time grid, which the eddy
    solve also writes ``pf_passive`` onto; measured signals are interpolated onto
    it and everything is clipped to the overlap of the two grids rather than
    extrapolated beyond where the diagnostics were mapped.

    ``window`` narrows the grid further, to the interval the caller actually
    evaluates over -- a benchmark's plasma-free stretch, say.  ``validity_window``
    is a separate question: which interval's validity decides whether a channel
    is *selected* at all.  They differ whenever a caller must model past the
    interval it judges by, which the eddy stage does: it selects on pre-plasma
    validity but needs post-onset samples for the residual to emerge into.
    Unset, ``validity_window`` follows ``window``.

    The passive-loop currents are whatever the ODS carries. For a plasma-free
    benchmark pass the ODS from
    :func:`vaft.validation.vacuum_benchmark.benchmark_wall_currents`, whose
    loops were solved from the PF coils alone (#190).

    ``response`` is the bundle from :func:`vacuum_response`. The geometry
    Green's functions it holds depend only on coil/loop/sensor positions, not
    on any current or resistance, so a calibration that re-solves the wall
    many times (#308) computes them once and passes them back in: on the real
    machine they are ~97% of this function's cost. Unset, they are computed
    here through the same function, so the two paths select identically by
    construction. The bundle carries the positions it was computed for, and
    they are compared to the selection -- not merely counted -- because two
    selections of equal size but different channels contract silently and
    wrongly otherwise (an 81% error was measured on the real machine).
    """
    # Checked first: "the eddy solve has not run on this ODS" is the more
    # actionable diagnosis than "no usable channels" when both are true.
    time, coil_currents, loop_currents = _currents(ods)
    if response is None:
        rows, response = vacuum_response(
            ods,
            per_family=per_family,
            channels=channels,
            window=window,
            validity_window=validity_window,
            min_validity=min_validity,
        )
    else:
        rows = select_vacuum_channels(
            ods,
            per_family=per_family,
            channels=channels,
            window=window if validity_window is None else validity_window,
            min_validity=min_validity,
        )
        if not rows:
            raise VacuumMagneticsError(
                "no magnetic channel carries usable measured data for vacuum validation"
            )
    psi, b_z, b_r, positions = response
    selected = np.array([[row["r"], row["z"]] for row in rows], dtype=float)
    if positions.shape != selected.shape or not np.array_equal(positions, selected):
        raise VacuumMagneticsError(
            f"precomputed response was built for {len(positions)} positions that do "
            f"not match the {len(rows)} channels selected here; it must come from "
            "vacuum_response() on the same selection"
        )
    n_coil, n_loop = coil_currents.shape[0], loop_currents.shape[0]

    measured_end = min(float(row["time"][-1]) for row in rows)
    measured_start = max(float(row["time"][0]) for row in rows)
    inside = (time >= measured_start) & (time <= measured_end)
    if window is not None:
        inside &= (time >= window[0]) & (time <= window[1])
    if inside.sum() < 2:
        raise VacuumMagneticsError(
            "the pf_active grid and the mapped magnetics do not overlap in time"
            + ("" if window is None else f" within the requested window {window}")
        )
    time = time[inside]
    coil_currents = coil_currents[:, inside]
    loop_currents = loop_currents[:, inside]

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
        quantity = "flux" if row["kind"] == FLUX_LOOP else "field"
        node = f"magnetics.{row['kind']}.{row['index']}.{quantity}"
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
                # Resampled onto the model grid by nearest sample: validity is a
                # per-sample state, not something to interpolate between.
                valid=validity_mask(
                    ods, node, times=time, min_validity=min_validity
                ),
            )
        )
    return tuple(built)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def plasma_free_residual(
    channels: Sequence[VacuumChannel],
    window: tuple[float, float] | None = None,
    *,
    normalize: bool = False,
) -> np.ndarray:
    """Stack ``measured - coil_eddy`` over the usable samples of every channel.

    This is the objective a wall-resistance calibration minimises (issue #308):
    with the model driven by the PF coils alone over a plasma-free interval,
    what is left is the passive-wall response the resistances control.

    Channels are concatenated in the order given, each restricted by
    :func:`evaluation_mask`, so the result is a single 1-D vector suitable for
    a least-squares fit. It is deliberately not reduced to a scalar: a fitter
    wants residuals, and the summary statistics already live in
    :func:`channel_residual_metrics`.

    ``normalize`` divides each channel's block by the RMS of its own measured
    signal over the same samples, which puts B-probes in tesla and flux loops
    in webers on a comparable footing. Without it a fit is dominated by
    whichever quantity happens to carry the larger numbers. Channels whose
    measured RMS is zero are left unscaled rather than divided by zero.
    """
    blocks: list[np.ndarray] = []
    for channel in channels:
        mask = evaluation_mask(channel, window)
        if not np.any(mask):
            continue
        residual = np.asarray(channel.measured, dtype=float)[mask] - np.asarray(
            channel.coil_eddy, dtype=float
        )[mask]
        if normalize:
            scale = float(np.sqrt(np.mean(np.asarray(channel.measured, dtype=float)[mask] ** 2)))
            if scale > 0.0:
                residual = residual / scale
        blocks.append(residual)
    if not blocks:
        return np.zeros(0, dtype=float)
    return np.concatenate(blocks)


def evaluation_mask(
    channel: VacuumChannel, window: tuple[float, float] | None
) -> np.ndarray:
    """Which samples of ``channel`` a comparison may actually use.

    The interval the caller asked for, intersected with what the diagnostics
    stage says is usable (issue #189).  Model agreement is only meaningful over
    samples that are measurements, and an unusable stretch left in would be
    charged to the model.

    The interval is half-open, ``[start, end)``.  Both callers derive its upper
    bound from a plasma-onset time, and that time is itself a grid sample --
    the detector returns ``float(time[start_index])`` -- so an inclusive bound
    would fold the plasma's own first sample into a nominally plasma-free
    window, and would silently widen the pre-plasma statistics the eddy stage
    has always reported over ``time < plasma_onset``.
    """
    inside = np.ones(channel.time.size, dtype=bool) if window is None else (
        (channel.time >= window[0]) & (channel.time < window[1])
    )
    return inside & channel.usable


def channel_residual_metrics(
    channel: VacuumChannel,
    *,
    window: tuple[float, float] | None = None,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Measured-versus-model agreement for one channel over one interval.

    This is a *comparison*, and says nothing about whether the measurement is
    sound: a large residual is evidence about the model (issue #190), and must
    never be written back into the channel's source validity (#253 §10).

    Every quantity #190 asks for per channel, in the channel's own units except
    where explicitly normalized:

    ``residual_rms_coil`` / ``residual_rms_coil_eddy``
        Agreement without and with the passive-wall response.  The pair is the
        point: their ratio is what the eddy model is worth here.
    ``improvement``
        ``1 - RMS(coil+eddy) / RMS(coil)``.  1.0 is perfect, 0.0 means the wall
        term added nothing, negative means it made agreement worse.
    ``residual_bias`` / ``residual_trend``
        A residual that is offset, or walking, is structured rather than noisy
        and points at something the model does not represent.
    ``normalized_residual``
        Residual RMS as a fraction of what the channel actually swung, so a
        quiet channel and a strongly driven one are comparable.
    ``correlation``
        Measured against coil+eddy.  Near 1 with a large residual means the
        dynamics are right and the gain is wrong -- a calibration question, not
        a wall-model one.

    A channel with too few usable samples is reported ``excluded`` with a
    reason rather than being counted as a model failure.
    """
    from vaft.formula.statistics import (
        dynamic_range,
        linear_trend,
        pearson_correlation,
        residual_bias,
    )

    mask = evaluation_mask(channel, window)
    row: dict[str, Any] = {
        "name": channel.name,
        "kind": channel.kind,
        "family": channel.family,
        "index": channel.index,
        "unit": channel.unit,
        "r": channel.r,
        "z": channel.z,
        "samples": int(mask.sum()),
        "window_start": float(channel.time[mask][0]) if mask.any() else float("nan"),
        "window_end": float(channel.time[mask][-1]) if mask.any() else float("nan"),
    }
    if int(mask.sum()) < int(min_samples):
        usable = int(channel.usable.sum())
        row["status"] = "excluded"
        row["reason"] = (
            f"only {int(mask.sum())} usable sample(s) in the evaluation window; "
            f"the channel declares {usable} usable of {channel.time.size}"
        )
        return row

    measured = channel.measured[mask]
    residual = channel.residual[mask]
    span = dynamic_range(measured)
    row.update(
        {
            "status": "evaluated",
            "reason": "",
            "measured_rms": rms(measured),
            "measured_dynamic_range": span,
            "residual_rms_coil": rms(channel.coil_residual[mask]),
            "residual_rms_coil_eddy": rms(residual),
            "improvement": eddy_improvement(channel.coil_residual, channel.residual, mask),
            "residual_bias": residual_bias(residual),
            "residual_trend": linear_trend(channel.time[mask], residual),
            "normalized_residual": (
                rms(residual) / span if span > 0 else float("nan")
            ),
            "correlation": pearson_correlation(measured, channel.coil_eddy[mask]),
        }
    )
    return row


def vacuum_residual_metrics(
    channels: Iterable[VacuumChannel],
    *,
    window: tuple[float, float] | None = None,
    min_samples: int = 2,
) -> dict[str, Any]:
    """Per-channel and per-family measured-versus-model agreement (issue #190).

    The one residual kernel in VAFT: the routine eddy-stage QA
    (:func:`vacuum_magnetics_metrics`) reports the same numbers over its
    pre-plasma window, so a per-shot verdict and a machine-model benchmark
    cannot drift apart in what they mean by "residual RMS".

    Metrics only -- no thresholds, no verdict.  What counts as acceptable
    depends on the study, and #190 is explicit that broad acceptance thresholds
    must wait until the VEST benchmark distribution has been inspected.
    """
    rows = [
        channel_residual_metrics(channel, window=window, min_samples=min_samples)
        for channel in channels
    ]
    evaluated = [row for row in rows if row["status"] == "evaluated"]

    def spread(key: str) -> dict[str, float]:
        values = np.array(
            [row[key] for row in evaluated if np.isfinite(row[key])], dtype=float
        )
        if values.size == 0:
            return {"median": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in rows}):
        members = [row for row in evaluated if row["family"] == family]
        improvements = [
            row["improvement"] for row in members if np.isfinite(row["improvement"])
        ]
        families[family] = {
            "channels": sum(1 for row in rows if row["family"] == family),
            "evaluated": len(members),
            "median_improvement": (
                float(np.median(improvements)) if improvements else float("nan")
            ),
        }

    return {
        "schema_version": 1,
        "window": None if window is None else [float(window[0]), float(window[1])],
        "channels": rows,
        "families": families,
        "summary": {
            "channel_count": len(rows),
            "evaluated": len(evaluated),
            "excluded": len(rows) - len(evaluated),
            "improvement": spread("improvement"),
            "normalized_residual": spread("normalized_residual"),
            "correlation": spread("correlation"),
            "improved_fraction": (
                float(
                    np.mean([row["improvement"] > 0.0 for row in evaluated])
                )
                if evaluated
                else float("nan")
            ),
        },
    }


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

    # The residual statistics come from the shared kernel, so this per-shot
    # verdict and the machine-model benchmark (#190) cannot disagree about what
    # "residual RMS" means. What stays here is the onset analysis, which is
    # specific to asking where the plasma signal emerges.
    pre_plasma = (float("-inf"), float(plasma_onset))
    rows = vacuum_residual_metrics(channels, window=pre_plasma)["channels"]
    for channel, row in zip(channels, rows):
        window = evaluation_mask(channel, pre_plasma)
        if row["status"] != "evaluated":
            # The eddy stage's own policy: for routine per-shot QA a channel it
            # cannot validate is an actionable gap, not a row to skip. The
            # benchmark's policy differs, which is why it lives with the caller
            # rather than in the kernel.
            raise VacuumMagneticsError(
                f"channel {channel.name!r} has no usable pre-plasma samples before "
                f"t={plasma_onset:.5f}s to validate against"
            )
        onset = residual_onset(channel.time, channel.residual, window, sigma=sigma)
        row["pre_plasma_samples"] = int(window.sum())
        row["residual_onset"] = onset
        row["onset_delta"] = (
            float(onset - current_onset) if np.isfinite(onset) else float("nan")
        )

    onsets = np.array(
        [row["residual_onset"] for row in rows if np.isfinite(row["residual_onset"])]
    )
    improvements = np.array(
        [row["improvement"] for row in rows if np.isfinite(row["improvement"])]
    )
    families = {
        family: {
            "channels": sum(1 for row in rows if row["family"] == family),
            "median_improvement": (
                float(
                    np.median(
                        [
                            row["improvement"]
                            for row in rows
                            if row["family"] == family and np.isfinite(row["improvement"])
                        ]
                    )
                )
                if any(
                    row["family"] == family and np.isfinite(row["improvement"])
                    for row in rows
                )
                else float("nan")
            ),
        }
        for family in sorted({row["family"] for row in rows})
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
