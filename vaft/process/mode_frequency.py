"""Predicted lab-frame mode-frequency tracks from an equilibrium and a rotation profile.

A magnetic perturbation with toroidal mode number ``n`` that is carried by the
plasma at its resonant surface ``|q| = |m/n|`` is seen in the laboratory frame
at ``f = n f_phi``, where ``f_phi`` is the toroidal rotation frequency of that
surface (issue #460).  The functions here turn an equilibrium (which locates
the surface) and a toroidal-rotation profile (which says how fast it turns)
into that prediction, as a time track that any fluctuation view -- a Mirnov,
soft X-ray, ECE or reflectometry spectrogram -- can draw over its own map.

The prediction is a hypothesis to compare with, never an identification: a
spectrogram ridge that follows the ``(2, 1)`` track is *consistent* with a
2/1 mode convected by the toroidal flow, nothing more.  Poloidal flow, the
``E x B`` and diamagnetic drifts, a mode's own frame frequency, and the
difference between the impurity rotation a CES measures and the bulk flow all
move a real mode off the track; none of them is modelled.  Mode-number
identification from the phase across a probe array is a separate analysis
(:func:`vaft.process.magnetics.toroidal_mode_analysis`).

The model is named in every result (``model="toroidal_rotation"``), so a
figure or a record built from a track says which physical approximation drew
it.

Notation
--------
q            : safety factor; resonance is |q| = |m/n|                     [-]
m, n         : poloidal and toroidal mode numbers of the hypothesis          [-]
rho_s        : flux coordinate of the rational surface                       [-]
v_phi        : toroidal ion velocity on the outboard midplane              [m/s]
R_out        : outboard-midplane major radius of the rational surface        [m]
omega_phi    : toroidal angular rotation frequency, v_phi / R_out        [rad/s]
f_phi        : toroidal rotation frequency, omega_phi / (2 pi)              [Hz]
f_pred       : predicted lab-frame frequency, n f_phi                       [Hz]

Conventions
-----------
**Rigid rotation of a flux surface.**  ``omega_phi`` is constant on a flux
surface to lowest order, so it is the quantity interpolated; a velocity is
turned into it at the major radius where it was taken.  VAFT's profile
reconstruction maps CES channels on the outboard midplane, so a stored
``velocity.toroidal`` is an outboard-midplane velocity and is divided by
``R_out`` of the surface (``equilibrium.profiles_1d.r_outboard``), not by the
axis radius: on VEST, with ``R_axis ~ 0.4 m`` and ``R_out`` up to ``0.7 m``,
the axis radius would overstate the edge rotation frequency by up to ~75 %.
A stored ``rotation_frequency_tor`` already is ``omega_phi`` and is used as is.

**Signs.**  ``f_phi`` keeps the sign of the stored rotation (positive along
the toroidal direction of the source's COCOS) and ``f_pred = n f_phi`` keeps
the sign of ``n`` as requested; resonance itself is ``|q| = |m/n|``, so the
COCOS sign of ``q`` never decides whether a surface exists.  A single probe's
spectrogram cannot tell the propagation direction, so a view draws
``|f_pred|``.

**Gaps, never extrapolation.**  A sample is valid only where the surface
exists, the rotation grid covers it radially and the rotation profiles cover
its time; elsewhere the track holds NaN and states why.

Provenance
----------
.. [1] The Doppler-shift relation ``omega_lab = n omega_phi`` for a mode
   convected by toroidal rotation, e.g. R. J. La Haye, Phys. Plasmas 13,
   055501 (2006), Sec. II; issue #460 for the request and result contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

__all__ = [
    "MODE_FREQUENCY_MODELS",
    "MODE_FREQUENCY_STATUSES",
    "ModeFrequencyTrack",
    "ModeFrequencyTracks",
    "mode_frequency_tracks",
]

#: The prediction models :func:`mode_frequency_tracks` implements.
MODE_FREQUENCY_MODELS = ("toroidal_rotation",)

#: Why a track sample is (or is not) valid, in the order the checks run.
MODE_FREQUENCY_STATUSES = (
    "valid",
    "no_equilibrium_profile",   # the slice has no usable q(psi)
    "no_surface",               # |q| never reaches m/n (or this branch) in the slice
    "outside_rotation_time",    # no rotation profile brackets the slice time
    "no_surface_coordinate",    # the root lacks the coordinate the rotation grid uses
    "outside_rotation_radius",  # the rotation grid does not reach the surface
    "no_major_radius",          # a velocity needs R_out and the slice has none
)

#: Rotation leaves read, in order of preference: an angular frequency first
#: (no radius needed), then the DD velocity, then its obsolescent spelling.
_ROTATION_LEAVES = (
    ("rotation_frequency_tor", "angular"),
    ("velocity.toroidal", "velocity"),
    ("velocity_tor", "velocity"),
)

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class ModeFrequencyTrack:
    """One predicted frequency track: one ``(m, n)`` on one root of ``|q| = m/n``.

    Every array runs over the equilibrium slices, in time order. ``branch``
    counts roots outward from the axis (0 innermost), so a reversed-shear
    profile gives a ``branch=1`` track wherever it has a second root; roots are
    associated across time by that order, not followed continuously.
    ``predicted_frequency`` is ``n * toroidal_rotation_frequency``, signed, NaN
    wherever ``valid`` is false, and ``status`` says why per sample.
    """

    m: int
    n: int
    q: float
    branch: int
    time: np.ndarray
    psi_norm: np.ndarray
    rho_pol_norm: np.ndarray
    rho_tor_norm: np.ndarray
    r_outboard: np.ndarray
    toroidal_rotation_frequency: np.ndarray
    predicted_frequency: np.ndarray
    valid: np.ndarray
    status: tuple[str, ...]
    model: str = "toroidal_rotation"

    @property
    def label(self) -> str:
        """``2/1 (q = 2)``: the mode and the surface it is predicted on."""
        return f"{self.m}/{self.n} (q = {self.q:g})"


@dataclass(frozen=True)
class ModeFrequencyTracks:
    """The tracks of one request, the model that drew them and how.

    ``tracks`` follow the request order of the modes, then the branch.
    ``provenance`` names the equilibrium and rotation sources, the coordinate
    the rotation was evaluated on, the radius it was divided by and the
    interpolation policy, as plain JSON-serialisable values.
    """

    model: str
    modes: tuple[tuple[int, int], ...]
    tracks: tuple[ModeFrequencyTrack, ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[ModeFrequencyTrack]:
        return iter(self.tracks)

    def __len__(self) -> int:
        return len(self.tracks)


# --- reading ---------------------------------------------------------------------


def _value(ods: Any, path: str) -> Any:
    from vaft.ods_access import path_value

    try:
        return path_value(ods, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _count(ods: Any, path: str) -> int:
    from vaft.ods_access import path_count

    try:
        return path_count(ods, path)
    except (KeyError, IndexError, TypeError, ValueError):
        return 0


def _array(ods: Any, path: str) -> np.ndarray | None:
    value = _value(ods, path)
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return array if array.size else None


def _scalar(ods: Any, path: str) -> float | None:
    array = _array(ods, path)
    if array is None or array.size != 1 or not np.isfinite(array[0]):
        return None
    return float(array[0])


def _slice_time(ods: Any, container: str, index: int) -> float | None:
    own = _scalar(ods, f"{container}.{index}.time")
    if own is not None:
        return own
    ids = container.split(".", 1)[0]
    shared = _array(ods, f"{ids}.time")
    if shared is not None and index < shared.size and np.isfinite(shared[index]):
        return float(shared[index])
    return None


@dataclass(frozen=True)
class _EquilibriumSlice:
    index: int
    time: float
    psi_norm: np.ndarray | None
    q: np.ndarray | None
    rho_tor_norm: np.ndarray | None
    r_outboard: np.ndarray | None
    psi_norm_reference: str
    r_outboard_source: str = ""   # "stored", "derived" or "" (unavailable)


def _equilibrium_slice(ods: Any, index: int, time: float) -> _EquilibriumSlice:
    from vaft.data._derived import is_rho_pol_proxy, rho_tor_profile

    base = f"equilibrium.time_slice.{index}"
    q = _array(ods, f"{base}.profiles_1d.q")
    psi = _array(ods, f"{base}.profiles_1d.psi")
    if q is None or psi is None or q.size != psi.size or q.size < 2:
        return _EquilibriumSlice(index, time, None, None, None, None, "")
    axis = _scalar(ods, f"{base}.global_quantities.psi_axis")
    edge = _scalar(ods, f"{base}.global_quantities.psi_boundary")
    reference = "global"
    if axis is None or edge is None or axis == edge:
        axis, edge, reference = float(psi[0]), float(psi[-1]), "profile_ends"
    if not np.isfinite(axis) or not np.isfinite(edge) or axis == edge:
        return _EquilibriumSlice(index, time, None, None, None, None, "")
    psi_norm = (psi - axis) / (edge - axis)
    finite = np.isfinite(psi_norm)
    if np.count_nonzero(finite) < 2 or not np.all(np.diff(psi_norm[finite]) > 0):
        return _EquilibriumSlice(index, time, None, None, None, None, "")

    rho_tor = None
    stored = _array(ods, f"{base}.profiles_1d.rho_tor_norm")
    ends = (psi - psi[0]) / (psi[-1] - psi[0]) if psi[-1] != psi[0] else psi_norm
    if (
        stored is not None and stored.size == psi.size and np.all(np.isfinite(stored))
        and abs(stored[0]) < 1e-6 and abs(stored[-1] - 1.0) < 1e-6
        and np.all(np.diff(stored) >= 0.0) and not is_rho_pol_proxy(stored, ends)
    ):
        rho_tor = stored
    else:
        derived = rho_tor_profile(q, psi)
        if derived is not None and derived.rho_tor_norm.size == psi.size:
            rho_tor = np.asarray(derived.rho_tor_norm, dtype=float)
    r_outboard = _array(ods, f"{base}.profiles_1d.r_outboard")
    source = "stored"
    if r_outboard is None or r_outboard.size != psi.size or not np.any(np.isfinite(r_outboard)):
        r_outboard = _derived_r_outboard(ods, index, psi)
        source = "derived" if r_outboard is not None else ""
    return _EquilibriumSlice(index, time, psi_norm, q, rho_tor, r_outboard, reference, source)


def _derived_r_outboard(ods: Any, index: int, psi: np.ndarray) -> np.ndarray | None:
    """``r_outboard`` of a slice that stores none, from its 2-D flux map.

    The map, axis and boundary outline are only *read* (through
    :mod:`vaft.ods_access`, which never materialises a path) and handed to
    :func:`vaft.process.equilibrium.psi_to_radial`, the midplane split
    ``update_equilibrium_profiles_1d_radial_coordinates`` uses; nothing is
    written to the caller's object. ``None`` when the slice cannot serve it.
    """
    from vaft.process.equilibrium import psi_to_radial

    base = f"equilibrium.time_slice.{index}"
    grid_r = _array(ods, f"{base}.profiles_2d.0.grid.dim1")
    grid_z = _array(ods, f"{base}.profiles_2d.0.grid.dim2")
    psi_2d = _value(ods, f"{base}.profiles_2d.0.psi")
    boundary_r = _array(ods, f"{base}.boundary.outline.r")
    axis_r = _scalar(ods, f"{base}.global_quantities.magnetic_axis.r")
    axis_z = _scalar(ods, f"{base}.global_quantities.magnetic_axis.z")
    if any(v is None for v in (grid_r, grid_z, psi_2d, boundary_r, axis_r, axis_z)):
        return None
    try:
        psi_2d = np.asarray(psi_2d, dtype=float)
        if psi_2d.shape != (grid_r.size, grid_z.size):
            if psi_2d.shape != (grid_z.size, grid_r.size):
                return None
            psi_2d = psi_2d.T  # stored (Z, R); OMAS orders profiles_2d (dim1=R, dim2=Z)
        row = psi_2d[:, int(np.argmin(np.abs(grid_z - axis_z)))]
        _, outboard = psi_to_radial(psi, row, grid_r, boundary_r, axis_r)
    except Exception:  # noqa: BLE001 - an underivable slice is a gap, not a failure
        return None
    outboard = np.asarray(outboard, dtype=float).reshape(-1)
    if outboard.size != psi.size or not np.any(np.isfinite(outboard)):
        return None
    return outboard


@dataclass(frozen=True)
class _RotationSlice:
    index: int
    time: float
    coordinate: str          # "rho_tor_norm" or "rho_pol_norm"
    grid: np.ndarray         # ascending, finite
    values: np.ndarray       # on grid
    kind: str                # "angular" [rad/s] or "velocity" [m/s]
    leaf: str
    proxy: bool = False      # grid.rho_tor_norm was the sqrt(psi_N) proxy, read as rho_pol_norm


def _rotation_slices(ods: Any, ion_index: int) -> list[_RotationSlice]:
    slices: list[_RotationSlice] = []
    for index in range(_count(ods, "core_profiles.profiles_1d")):
        time = _slice_time(ods, "core_profiles.profiles_1d", index)
        if time is None:
            continue
        base = f"core_profiles.profiles_1d.{index}"
        for leaf, kind in _ROTATION_LEAVES:
            values = _array(ods, f"{base}.ion.{ion_index}.{leaf}")
            if values is None:
                continue
            for coordinate in ("rho_tor_norm", "rho_pol_norm"):
                grid = _array(ods, f"{base}.grid.{coordinate}")
                if grid is None or grid.size != values.size:
                    continue
                if coordinate == "rho_tor_norm" and _is_proxy_grid(ods, base, grid):
                    # The sqrt(psi_N) proxy older files wrote under this name
                    # (#276) is a poloidal radius: it is read at the root's
                    # rho_pol_norm, never at its rho_tor_norm.
                    coordinate, proxy = "rho_pol_norm", True
                else:
                    proxy = False
                keep = np.isfinite(grid) & np.isfinite(values)
                if np.count_nonzero(keep) < 2:
                    continue
                order = np.argsort(grid[keep])
                slices.append(_RotationSlice(
                    index, time, coordinate, grid[keep][order], values[keep][order], kind,
                    f"core_profiles.profiles_1d.{{i}}.ion.{ion_index}.{leaf}", proxy,
                ))
                break
            else:
                continue
            break
    slices.sort(key=lambda s: s.time)
    return slices


def _is_proxy_grid(ods: Any, base: str, grid: np.ndarray) -> bool:
    """Whether a core profile's ``grid.rho_tor_norm`` is the ``sqrt(psi_N)`` proxy.

    The equilibrium side's test (:func:`vaft.data._derived.is_rho_pol_proxy`),
    against the profile's own ``grid.psi`` when it stores one, else a uniform
    ``psi_N``; a stored ``grid.rho_pol_norm`` equal to it settles it too.
    """
    from vaft.data._derived import is_rho_pol_proxy

    psi = _array(ods, f"{base}.grid.psi")
    psi_norm = None
    if psi is not None and psi.size == grid.size and np.all(np.isfinite(psi)) and psi[-1] != psi[0]:
        psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
    if is_rho_pol_proxy(grid, psi_norm):
        return True
    rho_pol = _array(ods, f"{base}.grid.rho_pol_norm")
    return bool(
        rho_pol is not None and rho_pol.size == grid.size
        and np.allclose(rho_pol, grid, rtol=0.0, atol=1e-9, equal_nan=False)
    )


def _modes(modes: Any) -> tuple[tuple[int, int], ...]:
    if modes is None or isinstance(modes, (str, bytes)):
        raise ValueError(f"modes takes (m, n) pairs, e.g. [(2, 1), (4, 2)]; got {modes!r}")
    pairs = list(modes)
    if len(pairs) == 2 and all(isinstance(v, (int, np.integer)) for v in pairs):
        pairs = [tuple(pairs)]  # one (m, n) passed bare
    if not pairs:
        raise ValueError("modes is empty: name at least one (m, n)")
    result: list[tuple[int, int]] = []
    for pair in pairs:
        try:
            m, n = pair
        except (TypeError, ValueError):
            raise ValueError(f"modes holds {pair!r}; each entry is an (m, n) pair") from None
        try:
            whole = float(m) == int(m) and float(n) == int(n)
        except (TypeError, ValueError):
            whole = False
        if not whole:
            raise ValueError(f"mode {pair!r} is not a pair of whole mode numbers")
        m, n = int(m), int(n)
        if m == 0 or n == 0:
            raise ValueError(f"mode ({m}, {n}) has q = m/n zero or undefined")
        if (m, n) not in result:
            result.append((m, n))
    return tuple(result)


def _omega_at(rotation: _RotationSlice, root: Any, r_out: float | None) -> tuple[float | None, str]:
    """``omega_phi`` of one rotation slice at one root, or ``(None, status)``."""
    x = root.rho_tor_norm if rotation.coordinate == "rho_tor_norm" else root.rho_pol_norm
    if x is None or not np.isfinite(x):
        return None, "no_surface_coordinate"
    if not rotation.grid[0] <= x <= rotation.grid[-1]:
        return None, "outside_rotation_radius"
    value = float(np.interp(x, rotation.grid, rotation.values))
    if rotation.kind == "angular":
        return value, "valid"
    if r_out is None or not np.isfinite(r_out) or r_out <= 0.0:
        return None, "no_major_radius"
    return value / r_out, "valid"


def mode_frequency_tracks(ods, modes, *, model="toroidal_rotation", ion_index=0, time_tolerance=0.0):
    """Predicted lab-frame frequency of each ``(m, n)`` from the rotation at ``|q| = m/n``.

    For every equilibrium slice the rational surfaces are located by
    :func:`vaft.process.equilibrium.rational_surfaces` (the one resolver; modes
    that share ``m/n`` share it), the toroidal rotation is evaluated there and
    ``f_pred = n f_phi`` follows. Each root of each mode is its own track, so a
    reversed-shear ``q = 2`` gives two tracks and is never reduced to one.
    The result is renderer-independent: any fluctuation view can draw it.

    Parameters
    ----------
    ods : ODS or IMAS entry
        Holds ``equilibrium.time_slice`` (``profiles_1d.q``, ``psi``,
        ``r_outboard``, ``rho_tor_norm``; ``global_quantities.psi_axis``,
        ``psi_boundary``) and ``core_profiles.profiles_1d`` with the ion's
        ``rotation_frequency_tor`` or ``velocity.toroidal`` on
        ``grid.rho_tor_norm`` (else ``grid.rho_pol_norm``) [-].
    modes : sequence of (int, int)
        The ``(m, n)`` hypotheses; ``(2, 1)`` and ``(4, 2)`` share ``q = 2``
        and give ``f_phi`` and ``2 f_phi`` [-].
    model : str, optional
        The prediction model; only ``"toroidal_rotation"`` exists [-].
    ion_index : int, optional
        Position of the ion in ``core_profiles.profiles_1d.{i}.ion`` whose
        rotation is used; 0 is VAFT's main ion [-].
    time_tolerance : float, optional
        How far outside the rotation profiles' time span an equilibrium slice
        may lie and still take the nearest profile; 0 pairs only slices the
        profiles bracket [s].

    Returns
    -------
    ModeFrequencyTracks
        ``model``, the normalised ``modes``, one :class:`ModeFrequencyTrack`
        per mode and root (time [s], surface coordinates [-], ``r_outboard``
        [m], ``toroidal_rotation_frequency`` and ``predicted_frequency`` [Hz],
        ``valid``, ``status``) and ``provenance`` [-].

    Raises
    ------
    ValueError
        ``model`` is not one of :data:`MODE_FREQUENCY_MODELS`, a mode is not a
        whole non-zero ``(m, n)``, ``time_tolerance`` is negative, the input
        has no equilibrium slice with a time, or no core profile carries a
        toroidal rotation of ion ``ion_index`` on a flux grid.

    Processing steps
    ----------------
    1. Read every timed equilibrium slice: ``psi_norm`` from the global
       ``psi_axis``/``psi_boundary`` (else the profile's ends), the authentic
       ``rho_tor_norm`` (stored, else integrated from ``q``) and ``r_outboard``
       (stored, else derived from the 2-D flux map at the axis height without
       writing the input).
    2. Read every timed rotation profile, preferring ``rotation_frequency_tor``
       over ``velocity.toroidal``, on ``rho_tor_norm`` over ``rho_pol_norm``; a
       ``grid.rho_tor_norm`` that is the ``sqrt(psi_N)`` proxy is read as the
       ``rho_pol_norm`` it is.
    3. Per slice, locate ``|q| = |m/n|`` once per distinct ratio with
       :func:`~vaft.process.equilibrium.rational_surfaces`.
    4. Per root, evaluate ``omega_phi`` on the two rotation profiles that
       bracket the slice time (linear in the profile's coordinate, a velocity
       divided by ``R_out`` of the root), then linearly in time.
    5. ``f_phi = omega_phi / 2 pi`` and ``f_pred = n f_phi`` per mode; any
       failed step leaves NaN and its status.

    Convention
    ----------
    Resonance is ``|q| = |m/n|``. ``f_phi`` carries the sign of the stored
    rotation (positive along the toroidal direction of the source COCOS) and
    ``f_pred`` that of ``n`` as given; a single-probe spectrogram shows
    ``|f_pred|``. A velocity is converted at the outboard-midplane radius of
    the surface, ``R_out``, because the stored velocity is an outboard-midplane
    (CES) value; ``R_axis`` is never substituted.

    Limitations
    -----------
    Toroidal rotation only: poloidal flow, ``E x B`` and diamagnetic drifts and
    the mode's own frame frequency are not modelled, and the measured impurity
    rotation stands in for the bulk flow, so agreement with a track is
    consistency with the ``(m, n)`` hypothesis, not identification. The
    rotation profile's flux coordinate is taken as the equilibrium's: a
    profile mapped through a different reconstruction is not re-mapped.
    Branches are associated across time by root order outward, not by
    continuity. Nothing is extrapolated, radially or in time.

    Applicability
    -------------
    Machine-independent. Any equilibrium with a q profile and any toroidal
    rotation profile on a normalised flux grid; on VEST the rotation is CES
    data reconstructed into ``core_profiles``.

    Provenance
    ----------
    .. [1] R. J. La Haye, "Neoclassical tearing modes and their control",
       Phys. Plasmas 13, 055501 (2006): the lab-frame frequency of a mode
       convected by toroidal rotation, ``omega = n Omega_phi``; issue #460 for
       the interpolation policy and the result contract.
    """
    from vaft.process.equilibrium import rational_surfaces

    if model not in MODE_FREQUENCY_MODELS:
        raise ValueError(f"model= takes one of {', '.join(MODE_FREQUENCY_MODELS)}; got {model!r}")
    requested = _modes(modes)
    tolerance = float(time_tolerance)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"time_tolerance must be a non-negative time in s; got {time_tolerance!r}")

    timed = [
        (index, _slice_time(ods, "equilibrium.time_slice", index))
        for index in range(_count(ods, "equilibrium.time_slice"))
    ]
    timed = sorted(((i, t) for i, t in timed if t is not None), key=lambda item: item[1])
    if not timed:
        raise ValueError("no equilibrium.time_slice with a time: the rational surfaces cannot be placed")
    rotation = _rotation_slices(ods, int(ion_index))
    if not rotation:
        raise ValueError(
            f"no core_profiles.profiles_1d carries a toroidal rotation of ion {ion_index} "
            "(rotation_frequency_tor or velocity.toroidal on grid.rho_tor_norm or grid.rho_pol_norm)"
        )
    rotation_times = np.array([s.time for s in rotation])
    slices = [_equilibrium_slice(ods, index, time) for index, time in timed]

    # 3. one resolution per slice and per distinct ratio
    resolved: list[dict[tuple[int, int], tuple[Any, ...]] | None] = []
    for eq in slices:
        if eq.q is None:
            resolved.append(None)
            continue
        try:
            surfaces = rational_surfaces(eq.psi_norm, eq.q, resonances=requested, rho_tor_norm=eq.rho_tor_norm)
        except ValueError:
            resolved.append(None)
            continue
        roots: dict[tuple[int, int], tuple[Any, ...]] = {}
        for surface in surfaces:
            for pair in surface.harmonics:
                roots[pair] = surface.roots
        resolved.append(roots)

    def bracket(time: float) -> tuple[list[tuple[_RotationSlice, float]], str]:
        """The rotation slices (with weights) that serve ``time``."""
        if rotation_times[0] <= time <= rotation_times[-1]:
            right = int(np.searchsorted(rotation_times, time, side="left"))
            if rotation_times[right] == time:
                return [(rotation[right], 1.0)], "valid"
            left = right - 1
            span = rotation_times[right] - rotation_times[left]
            weight = (time - rotation_times[left]) / span
            return [(rotation[left], 1.0 - weight), (rotation[right], weight)], "valid"
        nearest = int(np.argmin(np.abs(rotation_times - time)))
        if abs(rotation_times[nearest] - time) <= tolerance:
            return [(rotation[nearest], 1.0)], "valid"
        return [], "outside_rotation_time"

    # 4.-5. per distinct ratio and branch: omega_phi once, shared by its modes
    omega_cache: dict[tuple[float, int, int], tuple[float, str, Any, float]] = {}

    def omega(slice_position: int, pair: tuple[int, int], branch: int) -> tuple[float, str, Any, float]:
        eq = slices[slice_position]
        roots = resolved[slice_position]
        if roots is None:
            return np.nan, "no_equilibrium_profile", None, np.nan
        key = (abs(pair[0]) / abs(pair[1]), slice_position, branch)
        if key in omega_cache:
            return omega_cache[key]
        found = roots.get(pair, ())
        if branch >= len(found):
            result = (np.nan, "no_surface", None, np.nan)
        else:
            root = found[branch]
            r_out = None
            if eq.r_outboard is not None:
                usable = np.isfinite(eq.psi_norm) & np.isfinite(eq.r_outboard)
                if np.count_nonzero(usable) >= 2:
                    r_out = float(np.interp(root.psi_norm, eq.psi_norm[usable], eq.r_outboard[usable]))
            served, status = bracket(eq.time)
            value = 0.0
            for rotation_slice, weight in served:
                part, status = _omega_at(rotation_slice, root, r_out)
                if part is None:
                    break
                value += weight * part
            result = (value if status == "valid" else np.nan, status, root,
                      np.nan if r_out is None else r_out)
        omega_cache[key] = result
        return result

    tracks: list[ModeFrequencyTrack] = []
    times = np.array([eq.time for eq in slices])
    for pair in requested:
        branches = max([len(r.get(pair, ())) for r in resolved if r is not None] or [0])
        for branch in range(max(branches, 1)):
            columns = [omega(position, pair, branch) for position in range(len(slices))]
            angular = np.array([c[0] for c in columns], dtype=float)
            statuses = tuple(c[1] for c in columns)
            roots = [c[2] for c in columns]
            f_phi = angular / _TWO_PI
            valid = np.array([s == "valid" for s in statuses])
            tracks.append(ModeFrequencyTrack(
                m=pair[0], n=pair[1], q=abs(pair[0]) / abs(pair[1]), branch=branch,
                time=times.copy(),
                psi_norm=np.array([np.nan if r is None else r.psi_norm for r in roots], dtype=float),
                rho_pol_norm=np.array([
                    np.nan if r is None or r.rho_pol_norm is None else r.rho_pol_norm for r in roots
                ], dtype=float),
                rho_tor_norm=np.array([
                    np.nan if r is None or r.rho_tor_norm is None else r.rho_tor_norm for r in roots
                ], dtype=float),
                r_outboard=np.array([c[3] for c in columns], dtype=float),
                toroidal_rotation_frequency=np.where(valid, f_phi, np.nan),
                predicted_frequency=np.where(valid, pair[1] * f_phi, np.nan),
                valid=valid,
                status=statuses,
                model=model,
            ))

    provenance = {
        "model": model,
        "formula": "f_pred = n * f_phi(|q| = |m/n|), f_phi = omega_phi / (2 pi)",
        "rational_surface_resolver": "vaft.process.equilibrium.rational_surfaces",
        "equilibrium": {
            "source": "equilibrium.time_slice.{i}.profiles_1d (q, psi, rho_tor_norm, r_outboard)",
            "slices": [int(eq.index) for eq in slices],
            "psi_norm_reference": sorted({eq.psi_norm_reference for eq in slices if eq.psi_norm_reference}),
            "r_outboard": {
                "stored": [int(eq.index) for eq in slices if eq.r_outboard_source == "stored"],
                "derived": [int(eq.index) for eq in slices if eq.r_outboard_source == "derived"],
                "unavailable": [int(eq.index) for eq in slices if not eq.r_outboard_source],
                "derivation": "vaft.process.equilibrium.psi_to_radial on the profiles_2d psi row at the "
                              "magnetic-axis height, read without writing the input",
            },
        },
        "toroidal_rotation": {
            "source": sorted({s.leaf for s in rotation}),
            "ion_index": int(ion_index),
            "coordinate": sorted({s.coordinate for s in rotation}),
            "rho_tor_norm_proxy_profiles": [int(s.index) for s in rotation if s.proxy],
            "profiles": [int(s.index) for s in rotation],
            "time_span": [float(rotation_times[0]), float(rotation_times[-1])],
            "velocity_to_frequency": "omega_phi = v_phi / R_out(surface), "
                                     "R_out = equilibrium profiles_1d.r_outboard at the root",
        },
        "radial_interpolation": "linear in the rotation profile's coordinate; none outside its grid",
        "time_interpolation": (
            "linear between the rotation profiles bracketing each equilibrium time; none outside "
            f"their span (nearest profile within time_tolerance = {tolerance:g} s)"
        ),
        "interpretation": "consistency with the (m, n) hypothesis under toroidal convection, "
                          "not mode identification",
    }
    return ModeFrequencyTracks(model=model, modes=requested, tracks=tuple(tracks), provenance=provenance)
