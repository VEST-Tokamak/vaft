"""Resistive effective charge from the transformer balance and a conductivity model.

Infers one scalar ``Zeff_resistive`` over a time window: the effective charge
a stated parallel-conductivity model needs to reproduce the plasma resistance
observed through Romero's transformer balance (issue #1214).  It is a
*model-inferred* quantity, not the composition value ``sum n_i Z_i^2 / n_e``;
it absorbs every error in T_e, the equilibrium, the boundary voltage, the
conductivity model and any non-inductive current, and is named and stored so
that nobody mistakes it for a measurement.  Nothing here writes
``core_profiles.zeff``.

The chain, and where each step's state changes::

    boundary flux psi_B, I_p, li_3 per slice          [RomeroBoundaryFlux]
    -> smoothed in time, differentiated               [Smoothing]
    -> V_B, V_I, V_R = V_B - V_I, R_p = V_R/(I_p - I_ni)  [ObservedResistance]
    flux-surface state: T_e, n_e, <J.B>, <B^2>, V(psi) [FluxSurfaceState]
    -> sigma_par(Z) -> P_Ohm(Z) -> R_p^model(Z)        [ModelResistance]
    -> bounded scalar fit of V_R^model(Z) to V_R^obs   [ResistiveZeffInference]

Notation
--------
psi_B      : boundary poloidal flux, full flux in Romero's sign               [Wb]
V_B        : boundary loop voltage, -d psi_B / dt                             [V]
V_I        : inductive voltage, L_i dI_p/dt + I_p/2 dL_i/dt                   [V]
V_R        : resistive voltage, V_B - V_I = R_p (I_p - I_ni)                  [V]
L_i        : dimensional internal inductance, mu0 R0 li_3 / 2                 [H]
<J.B>      : flux-surface average of J.B                                      [A T m^-2]
<B^2>      : flux-surface average of B^2 (IMAS gm5)                           [T^2]
sigma_par  : parallel electrical conductivity                                 [S m^-1]

Conventions
-----------
**Romero's voltage convention, full webers.**  Every voltage is
``V = -d psi/dt`` on the full flux, so a positive ``V_B`` sustains a positive
``I_p`` (issue #781).  This is *not* the Ejima convention of
:func:`vaft.formula.equilibrium.loop_voltage_from_total_flux`
(``+2 pi d psi/dt`` on a per-radian flux, #354): the observed path accepts a
:class:`RomeroBoundaryFlux` only, which carries its convention tag, and
refuses anything else rather than guess which one an array is in.

**Ohmic power, not j_tor.**  The model resistance is the flux-surface
Ohmic dissipation over the current squared,
``R_p = int <E.B><J.B>/<B^2> dV / I_p^2`` with
``<E.B> = (<J.B> - <J_bs.B>) / sigma_par``.  ``V_R I_p`` is the same
dissipation in Romero's balance, so the two sides are one definition.  The
parallel current is never replaced by ``j_tor`` -- they differ at VEST
aspect ratio (#1214 Sec. 7.2).

**One scalar, not a profile.**  A single global resistance constrains a
single amplitude; ``Z_eff`` is held constant across radius and across the
inference window (#1214 Sec. 10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np

__all__ = [
    "ROMERO_VOLTAGE_CONVENTION",
    "RomeroBoundaryFlux",
    "Smoothing",
    "ObservedResistance",
    "FluxSurfaceState",
    "ModelResistance",
    "ResistiveZeffInference",
    "TabulatedConductivity",
    "smooth_local_polynomial",
    "observed_resistance",
    "parallel_conductivity",
    "model_resistance",
    "infer_resistive_zeff",
    "resistive_zeff_sensitivity",
]

#: The only voltage convention :func:`observed_resistance` accepts.
ROMERO_VOLTAGE_CONVENTION = "romero:V=-dpsi/dt,full_Wb"

#: Conductivity models :func:`parallel_conductivity` knows by name.
CONDUCTIVITY_MODELS = ("spitzer_nrl", "sauter_spitzer", "sauter", "redl")


# ---------------------------------------------------------------------------
# containers
# ---------------------------------------------------------------------------


def _as_1d(name: str, values, size: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1) if np.ndim(values) else np.asarray(
        [float(values)]
    )
    if size is not None and array.size != size:
        raise ValueError(f"{name} has {array.size} samples; expected {size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} holds non-finite values")
    return array


@dataclass(frozen=True)
class RomeroBoundaryFlux:
    """Boundary flux, plasma current and internal inductance in Romero's convention.

    Built by :func:`vaft.omas.resistive_zeff.romero_boundary_flux_ods` from an
    equilibrium, or directly from arrays a caller has already brought to the
    convention.  ``voltage_convention`` must be :data:`ROMERO_VOLTAGE_CONVENTION`;
    a flux in any other convention is refused at construction.
    """

    time: np.ndarray
    I_p: np.ndarray
    psi_boundary: np.ndarray
    li_3: np.ndarray
    R0: float
    flux_normalization: str
    flux_sign: float
    voltage_convention: str = ROMERO_VOLTAGE_CONVENTION
    li_definition: str = "li_3 = 2 int B_p^2 dV / (mu0^2 I_p^2 R0); L_i = mu0 R0 li_3 / 2"
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.voltage_convention != ROMERO_VOLTAGE_CONVENTION:
            raise ValueError(
                f"voltage convention {self.voltage_convention!r} is not Romero's "
                f"({ROMERO_VOLTAGE_CONVENTION!r}); bring the flux to full webers and "
                "V = -dpsi/dt before building a RomeroBoundaryFlux (#354, #1214)"
            )
        t = _as_1d("time", self.time)
        object.__setattr__(self, "time", t)
        for name in ("I_p", "psi_boundary", "li_3"):
            object.__setattr__(self, name, _as_1d(name, getattr(self, name), t.size))
        if t.size < 3 or np.any(np.diff(t) <= 0.0):
            raise ValueError("time must hold at least three strictly increasing samples")
        if not (np.isfinite(self.R0) and self.R0 > 0.0):
            raise ValueError(f"R0 must be positive, got {self.R0!r}")
        if np.any(self.li_3 <= 0.0):
            raise ValueError("li_3 must be positive at every sample")
        if self.flux_sign not in (-1.0, 1.0):
            raise ValueError(f"flux_sign must be +1 or -1, got {self.flux_sign!r}")

    @property
    def L_i(self) -> np.ndarray:
        """Dimensional internal inductance ``mu0 R0 li_3 / 2`` [H]."""
        from vaft.formula.constants import MU0

        # internal_inductance_from_li_3_R0 is scalar-only; the same identity.
        return 0.5 * MU0 * float(self.R0) * self.li_3

    def scaled_li(self, factor: float) -> "RomeroBoundaryFlux":
        """The same flux with ``li_3`` multiplied by ``factor`` (sensitivity)."""
        return replace(self, li_3=self.li_3 * float(factor))


@dataclass(frozen=True)
class Smoothing:
    """How the time series are smoothed before they are differentiated.

    ``method`` is ``"none"`` (raw samples) or ``"local_polynomial"`` (a
    least-squares polynomial of ``order`` fitted over ``window_s`` around each
    sample, :func:`smooth_local_polynomial`).  There is no default: the choice
    moves ``dL_i/dt`` and therefore ``V_R``, so it is stated and recorded.
    """

    method: str
    window_s: Optional[float] = None
    order: Optional[int] = None

    def __post_init__(self) -> None:
        if self.method == "none":
            if self.window_s is not None or self.order is not None:
                raise ValueError("method 'none' takes no window_s or order")
        elif self.method == "local_polynomial":
            if self.window_s is None or not self.window_s > 0.0:
                raise ValueError("local_polynomial needs a positive window_s")
            if self.order is None or int(self.order) < 0:
                raise ValueError("local_polynomial needs a non-negative order")
        else:
            raise ValueError(f"unknown smoothing method {self.method!r}")

    def apply(self, time, values) -> np.ndarray:
        """Smoothed ``values`` on ``time`` [any]."""
        if self.method == "none":
            return np.asarray(values, dtype=float)
        return smooth_local_polynomial(time, values, window_s=self.window_s, order=self.order)

    def scaled(self, factor: float) -> "Smoothing":
        """The same smoothing with the window multiplied by ``factor``."""
        if self.method == "none":
            return self
        return replace(self, window_s=self.window_s * float(factor))

    def describe(self) -> str:
        if self.method == "none":
            return "none"
        return f"local_polynomial(window_s={self.window_s:g}, order={int(self.order)})"


@dataclass(frozen=True)
class ObservedResistance:
    """Romero's balance evaluated on smoothed data, with the resistance it implies."""

    time: np.ndarray
    I_p: np.ndarray
    I_ni: np.ndarray
    L_i: np.ndarray
    li_3: np.ndarray
    dI_p_dt: np.ndarray
    dL_i_dt: np.ndarray
    V_B: np.ndarray
    V_I: np.ndarray
    V_R: np.ndarray
    R_p: np.ndarray
    inductive_fraction: np.ndarray
    flags: tuple
    provenance: Mapping[str, Any]

    def as_dict(self) -> dict:
        out = {
            name: getattr(self, name)
            for name in (
                "time", "I_p", "I_ni", "L_i", "li_3", "dI_p_dt", "dL_i_dt",
                "V_B", "V_I", "V_R", "R_p", "inductive_fraction",
            )
        }
        out["flags"] = self.flags
        out["provenance"] = dict(self.provenance)
        return out


@dataclass(frozen=True)
class FluxSurfaceState:
    """One equilibrium slice and its electron profiles on a common flux grid.

    Every profile is on ``psi_norm`` from axis to boundary.  ``j_dot_b`` and
    ``b2_average`` are flux-surface averages; ``volume`` is the volume
    enclosed by each surface.  ``j_bootstrap_dot_b`` is optional and labelled
    by ``bootstrap_model``; leaving it ``None`` means no bootstrap current.
    """

    time: float
    psi_norm: np.ndarray
    volume: np.ndarray
    T_e: np.ndarray
    n_e: np.ndarray
    q: np.ndarray
    r_inboard: np.ndarray
    r_outboard: np.ndarray
    j_dot_b: np.ndarray
    b2_average: np.ndarray
    I_p: float
    trapped_fraction: Optional[np.ndarray] = None
    j_bootstrap_dot_b: Optional[np.ndarray] = None
    bootstrap_model: str = "none"
    source: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        psi = _as_1d("psi_norm", self.psi_norm)
        if psi.size < 3 or np.any(np.diff(psi) <= 0.0):
            raise ValueError("psi_norm must be strictly increasing with at least three points")
        object.__setattr__(self, "psi_norm", psi)
        for name in ("volume", "T_e", "n_e", "q", "r_inboard", "r_outboard",
                     "j_dot_b", "b2_average"):
            object.__setattr__(self, name, _as_1d(name, getattr(self, name), psi.size))
        for name in ("trapped_fraction", "j_bootstrap_dot_b"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _as_1d(name, value, psi.size))
        if np.any(self.T_e <= 0.0) or np.any(self.n_e <= 0.0):
            raise ValueError("T_e and n_e must be positive on every surface; cut the grid "
                             "inside the last surface the profiles support")
        if np.any(self.b2_average <= 0.0):
            raise ValueError("<B^2> must be positive")
        if np.any(np.diff(self.volume) < 0.0):
            raise ValueError("volume must not decrease outward")
        if not np.isfinite(self.I_p) or self.I_p == 0.0:
            raise ValueError("I_p must be finite and non-zero")
        if (self.j_bootstrap_dot_b is None) != (self.bootstrap_model == "none"):
            raise ValueError("j_bootstrap_dot_b and bootstrap_model must be given together")

    def scaled(self, *, T_e: float = 1.0, n_e: float = 1.0) -> "FluxSurfaceState":
        """The same state with T_e and n_e multiplied (sensitivity)."""
        return replace(self, T_e=self.T_e * float(T_e), n_e=self.n_e * float(n_e))

    def without_bootstrap(self) -> "FluxSurfaceState":
        return replace(self, j_bootstrap_dot_b=None, bootstrap_model="none")

    def restricted(self, psi_min: float, psi_max: float) -> "FluxSurfaceState":
        """The surfaces with ``psi_min <= psi_norm <= psi_max`` only.

        For comparing models on the band a tabulated conductivity covers; the
        resistance of a restricted state is a partial one and says so in its
        source.
        """
        keep = (self.psi_norm >= psi_min) & (self.psi_norm <= psi_max)
        if keep.sum() < 3:
            raise ValueError(f"fewer than three surfaces in psi_norm [{psi_min}, {psi_max}]")
        cut = {name: getattr(self, name)[keep] for name in (
            "psi_norm", "volume", "T_e", "n_e", "q", "r_inboard", "r_outboard",
            "j_dot_b", "b2_average")}
        for name in ("trapped_fraction", "j_bootstrap_dot_b"):
            value = getattr(self, name)
            cut[name] = None if value is None else value[keep]
        return replace(self, **cut, source={**dict(self.source),
                                            "restricted_psi_norm": f"{psi_min:g}-{psi_max:g}"})


@dataclass(frozen=True)
class TabulatedConductivity:
    """A conductivity computed elsewhere at one fixed Z_eff, as a model callable.

    The seam through which a NEO conductivity case (#744) enters
    :func:`parallel_conductivity` and :func:`model_resistance` without an API
    change.  NEO builds its collision operator from its species list, so its
    conductivity belongs to the charge that list implies and to no other:
    asking for a different ``z_eff`` raises rather than rescaling.
    """

    name: str
    psi_norm: np.ndarray
    sigma: np.ndarray
    z_eff: float
    rtol: float = 1e-3

    def __call__(self, state: "FluxSurfaceState", z_eff: float, ln_lambda_profile) -> np.ndarray:
        if abs(float(z_eff) - self.z_eff) > self.rtol * self.z_eff:
            raise ValueError(
                f"{self.name} was computed at Z_eff = {self.z_eff:g} and cannot be "
                f"evaluated at {float(z_eff):g}; rerun it with that species list"
            )
        grid = np.asarray(self.psi_norm, dtype=float)
        if state.psi_norm.min() < grid.min() - 1e-9 or state.psi_norm.max() > grid.max() + 1e-9:
            raise ValueError(
                f"{self.name} covers psi_norm {grid.min():.3f}-{grid.max():.3f}; restrict the "
                "state to that band first (FluxSurfaceState.restricted)"
            )
        return np.interp(state.psi_norm, grid, np.asarray(self.sigma, dtype=float))


@dataclass(frozen=True)
class ModelResistance:
    """The plasma resistance a conductivity model predicts for one state and Z_eff."""

    time: float
    z_eff: float
    conductivity_model: str
    ln_lambda: str
    bootstrap_model: str
    R_p: float
    P_ohm_per_I2: float
    sigma_parallel: np.ndarray
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class ResistiveZeffInference:
    """The typed result of #1214 Sec. 11: an estimate, never an IMAS ``zeff``.

    ``status`` is ``"ok"``, ``"bound_hit"`` (the minimum sits on a bound and
    the value is poorly constrained), ``"non_monotonic"`` (R_p(Z) is not
    monotonic over the bounds) or ``"not_identifiable"`` (nothing to fit;
    ``reason`` says why).  ``zeff`` is ``None`` unless the status is ``ok``
    or ``bound_hit``.
    """

    quantity: str
    time_window: tuple
    resolved_times: np.ndarray
    estimate: Mapping[str, Any]
    observed: Mapping[str, Any]
    model: Mapping[str, Any]
    quality: Mapping[str, Any]
    provenance: Mapping[str, Any]
    sensitivity: Mapping[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None

    @property
    def zeff(self) -> Optional[float]:
        return self.estimate.get("zeff")

    @property
    def status(self) -> str:
        return self.estimate["status"]

    def as_row(self) -> dict:
        """Flat scalar mapping: survives ``code.parameters`` and a CSV row."""
        row = {
            "quantity": self.quantity,
            "t_start_s": float(self.time_window[0]),
            "t_end_s": float(self.time_window[1]),
            "n_times": int(np.size(self.resolved_times)),
            "reason": self.reason or "",
        }
        for prefix, section in (
            ("", self.estimate),
            ("model_", self.model),
            ("quality_", self.quality),
            ("provenance_", self.provenance),
        ):
            for key, value in section.items():
                if np.ndim(value) == 0 and not isinstance(value, Mapping):
                    row[f"{prefix}{key}"] = value
        for key, value in self.sensitivity.items():
            row[f"sensitivity_{key}"] = value
        return row


# ---------------------------------------------------------------------------
# observed path (#1214 Phases A-B)
# ---------------------------------------------------------------------------


def smooth_local_polynomial(time, values, *, window_s: float, order: int) -> np.ndarray:
    """Local least-squares polynomial smoothing on a non-uniform time axis.

    A Savitzky-Golay filter written for unevenly spaced samples: at each
    sample a polynomial of ``order`` is fitted by least squares to every
    sample within ``window_s / 2`` of it, and its value there is returned.
    EFIT slices arrive at about 1 ms with gaps, which the uniform-grid filter
    cannot take.

    Parameters
    ----------
    time : array_like
        Sample times, strictly increasing [s].
    values : array_like
        The sampled quantity, same length as ``time`` [any].
    window_s : float
        Full width of the fitting window [s].
    order : int
        Polynomial order; at least ``order + 2`` samples must fall in every
        window [-].

    Returns
    -------
    np.ndarray
        Smoothed values on ``time`` [any].

    Raises
    ------
    ValueError
        Lengths differ, time does not strictly increase, or a window holds
        fewer than ``order + 2`` samples -- a fit with no redundancy is an
        interpolation, not a smoothing, and is refused [-].

    Processing steps
    ----------------
    1. For each sample, select the samples with ``|t - t_i| <= window_s/2``.
    2. Fit a polynomial of ``order`` in ``t - t_i`` by least squares.
    3. Return its constant term.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    The window shrinks to one side at the ends of the record, where the fit
    extrapolates and is least reliable; cut a margin of ``window_s / 2`` off
    the inference window if the end samples matter.

    Provenance
    ----------
    .. [1] A. Savitzky and M. J. E. Golay, Anal. Chem. 36 (1964) 1627 --
           the uniform-grid special case.
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(values, dtype=float)
    if t.shape != x.shape or t.ndim != 1:
        raise ValueError("time and values must be one-dimensional and the same length")
    if np.any(np.diff(t) <= 0.0):
        raise ValueError("time must strictly increase")
    order = int(order)
    half = 0.5 * float(window_s)
    out = np.empty_like(x)
    for i, centre in enumerate(t):
        mask = np.abs(t - centre) <= half + 1e-12
        if mask.sum() < order + 2:
            raise ValueError(
                f"{mask.sum()} samples within {window_s:g} s of t = {centre:.4f} s; "
                f"order {order} needs at least {order + 2} -- widen the window"
            )
        coeffs = np.polynomial.polynomial.polyfit(t[mask] - centre, x[mask], order)
        out[i] = coeffs[0]
    return out


def observed_resistance(
    flux: RomeroBoundaryFlux,
    *,
    I_ni,
    smoothing: Smoothing,
) -> ObservedResistance:
    """Resistive voltage and plasma resistance from Romero's transformer balance.

    Smooths the plasma current, the boundary flux and the internal inductance,
    evaluates :func:`vaft.process.equilibrium.romero_flux_balance` on them, and
    reports ``V_R = V_B - V_I`` and ``R_p = V_R / (I_p - I_ni)`` sample by
    sample with the assumptions that produced them.

    Parameters
    ----------
    flux : RomeroBoundaryFlux
        Boundary flux, current and ``li_3`` in Romero's convention; nothing
        else is accepted [any].
    I_ni : float or array_like
        Non-inductively driven current. Required: ``0`` asserts a purely
        Ohmic discharge and is recorded as such, never assumed [A].
    smoothing : Smoothing
        How the series are smoothed before differentiation; required [any].

    Returns
    -------
    ObservedResistance
        ``V_B``, ``V_I``, ``V_R`` [V], ``R_p`` [Ohm], the derivatives,
        ``inductive_fraction = |V_I| / |V_B|`` and per-sample ``flags``
        (``R_p_nonpositive``, ``V_B_nonpositive``), plus the provenance of the
        flux convention, ``li`` definition, smoothing and current-source
        assumption [any].

    Raises
    ------
    TypeError
        ``flux`` is not a :class:`RomeroBoundaryFlux`: a bare loop-voltage
        array carries no convention and is refused [-].
    ValueError
        ``I_ni`` does not broadcast to the time axis, or
        :func:`~vaft.process.equilibrium.romero_flux_balance` rejects the
        series [-].

    Processing steps
    ----------------
    1. Smooth ``I_p``, ``psi_B`` and ``L_i`` with ``smoothing``.
    2. ``psi_C = psi_B + L_i I_p`` (Romero eqs. 35-36) and run the balance
       with ``R_p = 0`` so that its ``V_R`` is unused.
    3. ``V_R = V_B - V_I``; ``R_p = V_R / (I_p - I_ni)``.
    4. Flag samples whose ``V_B`` or ``R_p`` is not positive.

    Convention
    ----------
    Romero's: full-weber flux, ``V = -dpsi/dt``; a positive ``V_B`` drives a
    positive ``I_p`` and an Ohmic ``R_p`` is positive.  The Ejima loop voltage
    of :func:`vaft.formula.equilibrium.loop_voltage_from_total_flux` differs
    by sign and, on a per-radian flux, by 2 pi (#354), which is why only a
    tagged :class:`RomeroBoundaryFlux` gets in.

    Defaults
    --------
    None.  ``I_ni`` and ``smoothing`` have no default: an assumed value for
    either moves ``R_p`` and must be the caller's stated choice.

    Applicability
    -------------
    Machine-independent.  Closed-flux phase only, as the balance it calls.

    Limitations
    -----------
    ``R_p`` is a closure: it absorbs every error in ``V_B``, ``L_i`` and
    ``I_ni``.  ``dL_i/dt`` from reconstructed ``li_3`` is the noisiest term;
    where ``inductive_fraction`` is large the resistance is a small difference
    of large numbers and should not be fitted.

    Provenance
    ----------
    .. [1] J. A. Romero and JET-EFDA contributors, Nucl. Fusion 50 (2010)
           115002, eqs. (23)-(27), (35)-(40).
    .. [issue] #1214 Sec. 5 (observed path) and Sec. 4 (convention).
    """
    from vaft.process.equilibrium import romero_flux_balance

    if not isinstance(flux, RomeroBoundaryFlux):
        raise TypeError(
            "observed_resistance takes a RomeroBoundaryFlux, which carries its voltage "
            f"convention; got {type(flux).__name__}. Build one with "
            "vaft.omas.resistive_zeff.romero_boundary_flux_ods (#1214 Sec. 4)."
        )
    if not isinstance(smoothing, Smoothing):
        raise TypeError("smoothing must be a Smoothing")
    t = flux.time
    try:
        i_ni = np.array(np.broadcast_to(np.asarray(I_ni, dtype=float), t.shape))
    except ValueError:
        raise ValueError("I_ni must be a scalar or match the time axis") from None

    ip = smoothing.apply(t, flux.I_p)
    psi_b = smoothing.apply(t, flux.psi_boundary)
    li_3 = smoothing.apply(t, flux.li_3)
    l_i = np.asarray(replace(flux, li_3=li_3).L_i) if np.all(li_3 > 0.0) else None
    if l_i is None:
        raise ValueError("smoothing drove li_3 non-positive; narrow the window")
    psi_c = psi_b + l_i * ip

    balance = romero_flux_balance(t, ip, psi_b, psi_c, 0.0, i_ni)
    v_b, v_i = balance["V_B"], balance["V_I"]
    v_r = v_b - v_i
    with np.errstate(divide="ignore", invalid="ignore"):
        r_p = v_r / (ip - i_ni)
        inductive = np.abs(v_i) / np.abs(v_b)

    flags = tuple(
        tuple(
            name
            for name, bad in (("V_B_nonpositive", v_b[k] <= 0.0),
                              ("R_p_nonpositive", not r_p[k] > 0.0))
            if bad
        )
        for k in range(t.size)
    )
    if np.all(i_ni == 0.0):
        assumption = "ohmic: I_ni = 0 (stated by caller)"
    else:
        assumption = "I_ni supplied by caller"
    provenance = {
        "voltage_convention": flux.voltage_convention,
        "flux_normalization": flux.flux_normalization,
        "flux_sign": flux.flux_sign,
        "li_definition": flux.li_definition,
        "R0": flux.R0,
        "smoothing": smoothing.describe(),
        "derivative": "vaft.process.numerical.time_derivative (interval-weighted central)",
        "current_source_assumption": assumption,
        **{f"source_{k}": v for k, v in dict(flux.source).items()},
    }
    return ObservedResistance(
        time=t, I_p=ip, I_ni=i_ni, L_i=l_i, li_3=li_3,
        dI_p_dt=balance["dI_p_dt"], dL_i_dt=balance["dL_i_dt"],
        V_B=v_b, V_I=v_i, V_R=v_r, R_p=r_p, inductive_fraction=inductive,
        flags=flags, provenance=provenance,
    )


# ---------------------------------------------------------------------------
# forward model (#1214 Phases C-D)
# ---------------------------------------------------------------------------


def _ln_lambda_profile(state: FluxSurfaceState, ln_lambda) -> tuple[np.ndarray, str]:
    from vaft.formula.neoclassical import coulomb_logarithm_electron_sauter

    if isinstance(ln_lambda, str):
        if ln_lambda != "sauter":
            raise ValueError(f"ln_lambda must be a number or 'sauter', got {ln_lambda!r}")
        return np.asarray(coulomb_logarithm_electron_sauter(state.n_e, state.T_e), float), "sauter"
    value = float(ln_lambda)
    if not value > 0.0:
        raise ValueError(f"ln_lambda must be positive, got {ln_lambda!r}")
    return np.full(state.psi_norm.size, value), f"{value:g}"


def parallel_conductivity(
    state: FluxSurfaceState,
    z_eff: float,
    *,
    model: Union[str, Callable],
    ln_lambda: Union[float, str],
) -> np.ndarray:
    """Parallel electrical conductivity on every surface of a flux-surface state.

    Parameters
    ----------
    state : FluxSurfaceState
        Electron profiles and geometry on a flux grid [any].
    z_eff : float
        Effective charge, constant across radius; at least 1 [-].
    model : str or callable
        ``"spitzer_nrl"`` (``1/eta_par`` of the NRL Formulary, linear in Z),
        ``"sauter_spitzer"`` (Sauter's Spitzer with N(Z)), ``"sauter"`` or
        ``"redl"`` (the neoclassical corrections), or a callable
        ``model(state, z_eff, ln_lambda_profile) -> sigma`` -- the seam a NEO
        conductivity table plugs into without an API change [any].
    ln_lambda : float or str
        Coulomb logarithm: a number, or ``"sauter"`` for Sauter's electron
        ``ln Lambda_e(n_e, T_e)`` on every surface [-].

    Returns
    -------
    np.ndarray
        ``sigma_par`` on ``state.psi_norm`` [S/m].

    Raises
    ------
    ValueError
        ``z_eff`` below 1, an unknown model name, or a non-positive
        ``ln_lambda`` [-].

    Convention
    ----------
    Parallel, not perpendicular: ``spitzer_nrl`` is ``1/eta_par`` with the
    NRL coefficient 5.2e-5 Ohm m eV^1.5 (#1188).  The neoclassical models use
    the circular trapped fraction from ``epsilon = (R_out - R_in)/(R_out + R_in)``
    unless the state carries its own, and ``R = (R_out + R_in)/2`` in
    ``nu_e*`` -- as :func:`vaft.omas.neoclassical.compute_conductivity` does.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``spitzer_nrl`` and ``sauter_spitzer`` differ by tens of percent at
    ``Z > 1`` (linear Z against ``Z N(Z)``); the model name is part of the
    result.  The axis surface has ``epsilon = 0`` and no trapped particles.

    Provenance
    ----------
    .. [1] O. Sauter, C. Angioni and Y. R. Lin-Liu, Phys. Plasmas 6 (1999)
           2834; 9 (2002) 5140 (erratum).
    .. [2] A. Redl et al., Phys. Plasmas 28 (2021) 022502.
    .. [3] NRL Plasma Formulary (2019), p. 29.
    """
    from vaft.formula.equilibrium import spitzer_resistivity_from_T_e_Z_eff_ln_Lambda
    from vaft.formula.neoclassical import (
        electron_collisionality_sauter,
        redl_neoclassical_conductivity,
        sauter_neoclassical_conductivity,
        sauter_spitzer_conductivity,
        trapped_particle_fraction,
    )

    z = float(z_eff)
    if not z >= 1.0:
        raise ValueError(f"z_eff must be at least 1, got {z_eff!r}")
    lnl, _ = _ln_lambda_profile(state, ln_lambda)
    if callable(model):
        return np.asarray(model(state, z, lnl), dtype=float)
    if model == "spitzer_nrl":
        return 1.0 / np.asarray(spitzer_resistivity_from_T_e_Z_eff_ln_Lambda(state.T_e, z, lnl))
    sigma_sp = np.asarray(sauter_spitzer_conductivity(state.T_e, z, lnl), dtype=float)
    if model == "sauter_spitzer":
        return sigma_sp
    if model not in ("sauter", "redl"):
        raise ValueError(f"unknown conductivity model {model!r}; choose {CONDUCTIVITY_MODELS}")
    r_in, r_out = state.r_inboard, state.r_outboard
    major = 0.5 * (r_in + r_out)
    eps = np.clip((r_out - r_in) / (r_out + r_in), 0.0, 0.999)
    f_t = (state.trapped_fraction if state.trapped_fraction is not None
           else np.asarray(trapped_particle_fraction(eps), dtype=float))
    # nu_e* ~ eps^-3/2 diverges on the axis, where f_t = 0 makes it irrelevant:
    # evaluate off-axis only and give the axis the collisional limit.
    nu = np.full(eps.shape, 1e30)
    off = eps > 0.0
    if np.any(off):
        nu[off] = np.asarray(
            electron_collisionality_sauter(
                state.n_e[off], state.T_e[off], state.q[off], major[off], eps[off], z, lnl[off]
            ),
            dtype=float,
        )
    kernel = redl_neoclassical_conductivity if model == "redl" else sauter_neoclassical_conductivity
    return np.asarray(kernel(sigma_sp, f_t, nu, z), dtype=float)


def model_resistance(
    state: FluxSurfaceState,
    z_eff: float,
    *,
    model: Union[str, Callable],
    ln_lambda: Union[float, str],
) -> ModelResistance:
    """Plasma resistance a conductivity model predicts from the flux-surface Ohmic power.

    Parameters
    ----------
    state : FluxSurfaceState
        One slice: profiles, ``<J.B>``, ``<B^2>``, enclosed volume and the
        optional bootstrap ``<J_bs.B>`` [any].
    z_eff : float
        Effective charge, constant across radius [-].
    model : str or callable
        Conductivity model, as in :func:`parallel_conductivity` [any].
    ln_lambda : float or str
        Coulomb logarithm, as in :func:`parallel_conductivity` [-].

    Returns
    -------
    ModelResistance
        ``R_p = P_Ohm / I_p^2`` [Ohm], with the conductivity profile and the
        model, ``ln Lambda`` and bootstrap labels it came from [any].

    Raises
    ------
    ValueError
        Propagated from :func:`parallel_conductivity` [-].

    Processing steps
    ----------------
    1. ``sigma_par(psi)`` from :func:`parallel_conductivity`.
    2. ``<E.B> = (<J.B> - <J_bs.B>) / sigma_par`` -- ``<J_bs.B>`` is zero
       when the state carries none.
    3. ``P_Ohm = int <E.B><J.B>/<B^2> dV`` by the trapezoid rule on the
       enclosed volume.
    4. ``R_p = P_Ohm / I_p^2``.

    Convention
    ----------
    ``V_R I_p = P_Ohm`` is the resistive term of Romero's Poynting balance,
    so this ``R_p`` is the same quantity :func:`observed_resistance` closes
    for.  ``<E.J> ~ <E.B><J.B>/<B^2>`` treats ``J_par/B`` as a flux function;
    the parallel current is used, never ``j_tor`` (#1214 Sec. 7.2).

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    ``<E.J> ~ <E.B><J.B>/<B^2>`` is exact only when ``J_par/B`` is a flux
    function; the poloidal variation of the Pfirsch-Schlueter current is
    dropped, which at VEST aspect ratio is part of the model-form uncertainty
    rather than a resolved term.  Steady state is not assumed: the current
    profile is the equilibrium's own, and only its parallel resistivity is
    modelled.

    Provenance
    ----------
    .. [1] S. P. Hirshman and D. J. Sigmar, Nucl. Fusion 21 (1981) 1079 --
           the flux-surface parallel Ohm's law.
    .. [issue] #1214 Sec. 7.
    """
    from scipy.integrate import trapezoid

    sigma = parallel_conductivity(state, z_eff, model=model, ln_lambda=ln_lambda)
    _, lnl_label = _ln_lambda_profile(state, ln_lambda)
    jb = state.j_dot_b
    jbs = state.j_bootstrap_dot_b if state.j_bootstrap_dot_b is not None else 0.0
    e_dot_b = (jb - jbs) / sigma
    integrand = e_dot_b * jb / state.b2_average
    power = float(trapezoid(integrand, state.volume))
    per_i2 = power / state.I_p**2
    name = model if isinstance(model, str) else getattr(model, "name", getattr(model, "__name__", "custom"))
    return ModelResistance(
        time=float(state.time), z_eff=float(z_eff), conductivity_model=str(name),
        ln_lambda=lnl_label, bootstrap_model=state.bootstrap_model,
        R_p=per_i2, P_ohm_per_I2=per_i2, sigma_parallel=sigma,
        provenance={"power": "int <E.B><J.B>/<B^2> dV", **dict(state.source)},
    )


# ---------------------------------------------------------------------------
# inference (#1214 Phases E-F)
# ---------------------------------------------------------------------------


def _match_states(observed: ObservedResistance, states, tolerance_s: float):
    pairs = []
    for state in states:
        k = int(np.argmin(np.abs(observed.time - state.time)))
        if abs(observed.time[k] - state.time) > tolerance_s:
            raise ValueError(
                f"state at t = {state.time:.5f} s has no observed sample within "
                f"{tolerance_s:g} s; states are matched by time, never by index"
            )
        pairs.append((k, state))
    return pairs


def infer_resistive_zeff(
    observed: ObservedResistance,
    states: Sequence[FluxSurfaceState],
    *,
    model: Union[str, Callable],
    ln_lambda: Union[float, str],
    bounds: tuple,
    weights: Union[str, Sequence[float]],
    time_tolerance_s: float = 5e-5,
    scan_points: int = 41,
) -> ResistiveZeffInference:
    """Bounded scalar Z_eff that makes the model resistive voltage match the observed one.

    Parameters
    ----------
    observed : ObservedResistance
        Observed ``V_R`` history from :func:`observed_resistance` [any].
    states : sequence of FluxSurfaceState
        Flux-surface states at times inside ``observed.time`` [any].
    model : str or callable
        Conductivity model, as in :func:`parallel_conductivity` [any].
    ln_lambda : float or str
        Coulomb logarithm, as in :func:`parallel_conductivity` [-].
    bounds : tuple of float
        ``(Z_min, Z_max)``, explicit; the solution is never clipped [-].
    weights : str or sequence of float
        ``"uniform"`` or one weight per state; required [-].
    time_tolerance_s : float, optional
        Largest gap between a state and its observed sample [s].
    scan_points : int, optional
        Log-spaced Z samples used to test monotonicity first [-].

    Returns
    -------
    ResistiveZeffInference
        The #1214 Sec. 11 result: estimate (``zeff``, ``uncertainty``,
        ``bounds``, ``objective``, ``status``), observed and predicted
        voltages, quality and provenance [any].

    Raises
    ------
    ValueError
        No state, bounds that are not ``1 <= Z_min < Z_max``, weights of the
        wrong length, or a state with no observed sample at its time [-].

    Processing steps
    ----------------
    1. Match each state to the observed sample at its time.
    2. Drop samples flagged ``R_p_nonpositive``; none left gives
       ``not_identifiable``.
    3. Scan ``R_p^model(Z)`` over the bounds; a non-monotonic response gives
       ``non_monotonic`` without a fit.
    4. Minimise ``J(Z) = sum w (V_R^obs - R_p^model(Z) (I_p - I_ni))^2``
       with a bounded scalar search.
    5. Uncertainty from the curvature of ``J`` and the residual scatter
       (``None`` with one sample); a minimum within 0.1 % of the range from
       a bound gives ``bound_hit``, as does a scan whose smallest ``J`` is
       at an end point.

    Defaults
    --------
    ``time_tolerance_s`` 50 us is a numerical convenience below the 1e-4 s
    rounding of the Lane K state key (#1454); ``scan_points`` 41 is a
    numerical convenience.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    One amplitude only: Z_eff is constant across radius and time
    (#1214 Sec. 10).  The uncertainty is statistical; the systematic part
    comes from :func:`resistive_zeff_sensitivity`.

    Provenance
    ----------
    .. [issue] #1214 Secs. 9-11.
    """
    from scipy.optimize import minimize_scalar

    states = list(states)
    if not states:
        raise ValueError("no flux-surface state to fit")
    z_lo, z_hi = (float(b) for b in bounds)
    if not (1.0 <= z_lo < z_hi):
        raise ValueError(f"bounds must satisfy 1 <= Z_min < Z_max, got {bounds!r}")
    if isinstance(weights, str):
        if weights != "uniform":
            raise ValueError(f"weights must be 'uniform' or an array, got {weights!r}")
        w_all = np.ones(len(states))
    else:
        w_all = np.asarray(weights, dtype=float)
        if w_all.shape != (len(states),) or np.any(w_all < 0.0):
            raise ValueError("weights must be non-negative, one per state")

    pairs = _match_states(observed, states, time_tolerance_s)
    name = model if isinstance(model, str) else getattr(model, "name", getattr(model, "__name__", "custom"))
    model_info = {
        "conductivity_model": str(name),
        "ln_lambda": _ln_lambda_profile(states[0], ln_lambda)[1],
        "bootstrap_model": states[0].bootstrap_model,
        "current_source_assumption": observed.provenance.get("current_source_assumption"),
        "zeff_profile": "constant over radius and window",
    }
    window = (float(min(s.time for s in states)), float(max(s.time for s in states)))
    provenance = {
        **dict(observed.provenance),
        "estimator": "bounded least squares on V_R",
        "time_tolerance_s": time_tolerance_s,
    }
    try:
        import vaft

        provenance["vaft_version"] = getattr(vaft, "__version__", "unknown")
    except Exception:  # pragma: no cover - import of the package itself
        provenance["vaft_version"] = "unknown"

    keep = [(k, s, w) for (k, s), w in zip(pairs, w_all)
            if "R_p_nonpositive" not in observed.flags[k] and w > 0.0]

    def result(status, *, zeff=None, uncertainty=None, objective=None, reason=None,
               times=(), v_obs=(), v_model=(), r_obs=(), r_model=(), quality=None):
        return ResistiveZeffInference(
            quantity="Zeff_resistive (model-inferred; not a composition measurement)",
            time_window=window,
            resolved_times=np.asarray(times, dtype=float),
            estimate={"zeff": zeff, "uncertainty": uncertainty, "z_min": z_lo,
                      "z_max": z_hi, "objective": objective, "status": status},
            observed={"resistive_voltage": np.asarray(v_obs), "plasma_resistance": np.asarray(r_obs)},
            model={**model_info, "predicted_resistive_voltage": np.asarray(v_model),
                   "predicted_plasma_resistance": np.asarray(r_model)},
            quality=quality or {},
            provenance=provenance,
            reason=reason,
        )

    if not keep:
        return result("not_identifiable",
                      reason="no matched sample with a positive observed resistance")

    idx = np.array([k for k, _, _ in keep])
    w = np.array([wt for _, _, wt in keep])
    st = [s for _, s, _ in keep]
    drive = observed.I_p[idx] - observed.I_ni[idx]
    v_obs = observed.V_R[idx]

    def r_model(z):
        return np.array([model_resistance(s, z, model=model, ln_lambda=ln_lambda).R_p for s in st])

    def objective(z):
        return float(np.sum(w * (v_obs - r_model(z) * drive) ** 2))

    grid = np.geomspace(z_lo, z_hi, int(scan_points))
    scan = np.array([r_model(z) for z in grid])  # (n_z, n_samples)
    steps = np.diff(scan, axis=0)
    monotonic = bool(np.all(steps > 0.0) or np.all(steps < 0.0))
    quality = {"monotonic": monotonic, "n_samples": int(idx.size),
               "max_inductive_fraction": float(np.nanmax(observed.inductive_fraction[idx]))}
    if not monotonic:
        return result("non_monotonic", times=observed.time[idx], v_obs=v_obs,
                      r_obs=observed.R_p[idx], quality=quality,
                      reason="R_p^model(Z) is not monotonic over the bounds")

    j_scan = np.array([float(np.sum(w * (v_obs - row * drive) ** 2)) for row in scan])
    k0 = int(np.argmin(j_scan))
    lo = grid[max(k0 - 1, 0)]
    hi = grid[min(k0 + 1, grid.size - 1)]
    found = minimize_scalar(objective, bounds=(lo, hi), method="bounded",
                            options={"xatol": 1e-8 * max(1.0, lo)})
    z_star = float(found.x)
    j_star = float(found.fun)
    r_star = r_model(z_star)
    v_star = r_star * drive

    # Curvature of J: Gauss-Newton, J'' ~ 2 sum w (dV/dZ)^2; the difference is
    # kept inside the bounds, where the models are defined.
    dz = 1e-4 * z_star
    z_minus, z_plus = max(z_star - dz, z_lo), min(z_star + dz, z_hi)
    dv_dz = (r_model(z_plus) - r_model(z_minus)) * drive / (z_plus - z_minus)
    fisher = float(np.sum(w * dv_dz**2))
    if idx.size >= 2 and fisher > 0.0:
        # Residual variance per unit weight, J/(n-1), over the Fisher information.
        sigma_z = math.sqrt(j_star / (idx.size - 1) / fisher)
    else:
        sigma_z = None
    # On a bound when the scan's minimum is an end point (the unconstrained
    # minimum lies beyond it) or the solution sits within 0.1 % of the range.
    tol = 1e-3 * (z_hi - z_lo)
    on_bound = (k0 in (0, grid.size - 1) or z_star - z_lo <= tol or z_hi - z_star <= tol)
    rms = math.sqrt(j_star / np.sum(w))
    quality.update(
        residual_rms_V=rms,
        normalized_residual=rms / float(np.sqrt(np.average(v_obs**2, weights=w))),
        curvature=2.0 * fisher,
        bound_hit=bool(on_bound),
    )
    return result(
        "bound_hit" if on_bound else "ok",
        zeff=z_star, uncertainty=sigma_z, objective=j_star,
        times=observed.time[idx], v_obs=v_obs, v_model=v_star,
        r_obs=observed.R_p[idx], r_model=r_star, quality=quality,
        reason="minimum on a bound: poorly constrained" if on_bound else None,
    )


def resistive_zeff_sensitivity(
    flux: RomeroBoundaryFlux,
    states: Sequence[FluxSurfaceState],
    *,
    I_ni,
    smoothing: Smoothing,
    model: Union[str, Callable],
    ln_lambda: Union[float, str],
    bounds: tuple,
    weights: Union[str, Sequence[float]],
    relative_perturbation: float = 0.1,
    smoothing_factors: tuple = (0.5, 2.0),
    models: tuple = ("spitzer_nrl", "sauter", "redl"),
    time_tolerance_s: float = 5e-5,
) -> tuple:
    """Nominal resistive Z_eff and how far each input moves it.

    Parameters
    ----------
    flux : RomeroBoundaryFlux
        Boundary flux history [any].
    states : sequence of FluxSurfaceState
        Flux-surface states in the window [any].
    I_ni : float or array_like
        Non-inductive current, as in :func:`observed_resistance` [A].
    smoothing : Smoothing
        Nominal smoothing [any].
    model : str or callable
        Nominal conductivity model [any].
    ln_lambda : float or str
        Coulomb logarithm [-].
    bounds : tuple of float
        ``(Z_min, Z_max)`` [-].
    weights : str or sequence of float
        As in :func:`infer_resistive_zeff` [-].
    relative_perturbation : float, optional
        Fractional change applied to T_e, n_e and li_3, each way [-].
    smoothing_factors : tuple of float, optional
        Multipliers of the smoothing window [-].
    models : tuple of str, optional
        Alternative conductivity models compared with the nominal one [-].
    time_tolerance_s : float, optional
        As in :func:`infer_resistive_zeff` [s].

    Returns
    -------
    tuple
        ``(nominal, table)``: the nominal :class:`ResistiveZeffInference`,
        with its ``sensitivity`` filled with the largest shift per input, and
        a list of rows ``{"input", "setting", "zeff", "delta", "status"}``
        [any].

    Raises
    ------
    ValueError
        Propagated from :func:`infer_resistive_zeff` [-].

    Processing steps
    ----------------
    1. Nominal inference.
    2. Re-infer with T_e, n_e and li_3 scaled by ``1 +/- relative_perturbation``.
    3. Re-infer with the smoothing window scaled by each factor.
    4. Re-infer with each alternative conductivity model, and without the
       bootstrap current if the states carry one.
    5. ``delta = Z - Z_nominal``; the largest absolute delta per input goes
       into ``nominal.sensitivity``.

    Defaults
    --------
    ``relative_perturbation`` 0.1 and ``smoothing_factors`` (0.5, 2) are
    numerical-convenience probes of local sensitivity, not uncertainty
    estimates of VEST diagnostics; ``models`` is the conventional
    Spitzer/Sauter/Redl ladder of #1214 Sec. 3.

    Applicability
    -------------
    Machine-independent.

    Limitations
    -----------
    One-at-a-time perturbations; no correlation between inputs.  NEO enters
    only as a callable model the caller supplies.

    Provenance
    ----------
    .. [issue] #1214 Sec. 12 and Phase F.
    """
    kwargs = dict(model=model, ln_lambda=ln_lambda, bounds=bounds, weights=weights,
                  time_tolerance_s=time_tolerance_s)
    observed = observed_resistance(flux, I_ni=I_ni, smoothing=smoothing)
    nominal = infer_resistive_zeff(observed, states, **kwargs)
    z0 = nominal.zeff
    rows = []

    def run(label, setting, fn):
        try:
            res = fn()
            z, status = res.zeff, res.status
        except ValueError as error:
            z, status = None, f"error: {error}"
        delta = (z - z0) if (z is not None and z0 is not None) else None
        rows.append({"input": label, "setting": setting, "zeff": z, "delta": delta,
                     "status": status})

    d = float(relative_perturbation)
    for sign in (-1.0, 1.0):
        f = 1.0 + sign * d
        run("T_e", f"x{f:g}", lambda f=f: infer_resistive_zeff(
            observed, [s.scaled(T_e=f) for s in states], **kwargs))
        run("n_e", f"x{f:g}", lambda f=f: infer_resistive_zeff(
            observed, [s.scaled(n_e=f) for s in states], **kwargs))
        run("li_3", f"x{f:g}", lambda f=f: infer_resistive_zeff(
            observed_resistance(flux.scaled_li(f), I_ni=I_ni, smoothing=smoothing),
            states, **kwargs))
    for factor in smoothing_factors:
        alt = smoothing.scaled(factor)
        run("smoothing", alt.describe(), lambda alt=alt: infer_resistive_zeff(
            observed_resistance(flux, I_ni=I_ni, smoothing=alt), states, **kwargs))
    nominal_name = model if isinstance(model, str) else None
    for alt_model in models:
        if alt_model == nominal_name:
            continue
        run("conductivity_model", alt_model, lambda m=alt_model: infer_resistive_zeff(
            observed, states, **{**kwargs, "model": m}))
    if any(s.j_bootstrap_dot_b is not None for s in states):
        run("bootstrap", "none", lambda: infer_resistive_zeff(
            observed, [s.without_bootstrap() for s in states], **kwargs))

    worst = {}
    for row in rows:
        if row["delta"] is not None:
            worst[row["input"]] = max(worst.get(row["input"], 0.0), abs(row["delta"]))
    nominal = replace(nominal, sensitivity={f"max_abs_delta_{k}": v for k, v in worst.items()})
    return nominal, rows
