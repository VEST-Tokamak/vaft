"""Quasilinear flux decomposition of one TGLF run, validated against TGLF's own output (#1900).

TGLF's saturated flux for species ``s``, field ``f``, ky point ``k``, mode ``m`` and
quantity ``q`` is, in ``get_multiscale_spectrum`` (``tglf_multiscale_spectrum.f90``,
GACODE b493397)::

    flux[s, f, m, k, q] = W[s, f, m, k, q] * phi[k, m] * QLA[q, m]

* ``W`` is the quasilinear weight, ``out.tglf.QL_flux_spectrum``
  (:attr:`~vaft.code.gacode.tglf.outputs.TglfOutputs.ql_flux_spectrum`).
* ``phi`` is the saturated potential intensity, column ``phi`` of
  ``out.tglf.field_spectrum``.
* ``QLA`` is 1 for SAT_RULE 0, 1 and 2. SAT_RULE 3 applies a per-mode quasilinear
  amplitude factor:
  - particle flux: ``QLA_P = 2 f(x; 1.1, 0.6)``;
  - energy flux: ``QLA_E = 2 f(x; 0.75, 0.6)``;
  - toroidal stress, parallel stress and exchange: ``QLA_O = 2 * 0.8``.

  ``f(x; y1, y2)`` equals ``y1`` below ``x_ITG = 0.8`` and ``y2`` above ``x_TEM = 1.0``,
  and is linear between them. For each mode, ``x`` is
  ``|W_e / sum over ions of W_i|`` for the electrostatic energy weight, linearly
  interpolated to ``ky = kymax_out``. When ``kymax_out`` is at or below the first ky
  point, or an ion-weight sum bracketing it is zero, ``x = 0.5``.
  TGLF writes no file holding ``QLA``. It is rebuilt here from ``W`` and ``kymax_out``,
  the same way TGLF computes it.

``out.tglf.sum_flux_spectrum`` sums this over modes and integrates over ky with the
weights ``write_tglf_sum_flux_spectrum`` uses: ``dky0 * F[k-1] + dky1 * F[k]``.
For ``KYGRID_MODEL != 0`` these are log-trapezoid weights; for ``KYGRID_MODEL = 0``
the rule is ``dky0 = 0, dky1 = ky_1``.

:func:`flux_decomposition` rebuilds every entry of ``sum_flux_spectrum`` from these
factors. It returns the factors only when the rebuild matches TGLF within
:data:`RECONSTRUCTION_RTOL`. A decomposition that does not reproduce the solver's own
flux is not used, and the status reports why. Before ``QLA`` was included, SAT3
rebuilds came out at 0.67-0.83 of the native flux. With it, 240 runs (SAT0-3, ES and
EM) reproduce to <= 1.5e-15 (#1900).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "LINEAR_RTOL",
    "QLA_OTHER",
    "RECONSTRUCTION_RTOL",
    "ClosureSplit",
    "FluxDecomposition",
    "closure_split",
    "flux_decomposition",
    "ky_integration_weights",
    "sat3_qla_factors",
]

#: Largest accepted ``max|rebuilt - native| / max|native|`` per flux quantity. The
#: rebuild is exact up to the digits TGLF prints. The tolerance leaves room for a
#: platform that prints fewer digits, but it is 10^5 tighter than the 17-50 % gap a
#: missing factor produces.
RECONSTRUCTION_RTOL = 1e-6

#: Two runs have the same linear physics when ky grid, eigenvalues and QL weights agree
#: to this relative tolerance (TGLF is deterministic: equal presets give equal bits).
LINEAR_RTOL = 1e-9

#: SAT3 factor for toroidal stress, parallel stress and exchange (2 for real symmetry x 0.8).
QLA_OTHER = 2 * 0.8
_X_ITG, _X_TEM = 0.8, 1.0


def _transition(x: float, y1: float, y2: float) -> float:
    """``mode_transition_function`` of tglf_multiscale_spectrum.f90."""
    if x < _X_ITG:
        return y1
    if x > _X_TEM:
        return y2
    return y1 * (_X_TEM - x) / (_X_TEM - _X_ITG) + y2 * (x - _X_ITG) / (_X_TEM - _X_ITG)


def sat3_qla_factors(native: Any) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """``(QLA, x)`` for a SAT_RULE 3 run: ``QLA`` shaped ``(5, mode)``, ``x`` shaped ``(mode,)``.

    Returns ``None`` when the factors cannot be formed as TGLF forms them:
    adiabatic electrons, no ``kymax_out``, ``kymax_out`` beyond the ky grid, or
    missing weights.
    """
    weights = getattr(native, "ql_flux_spectrum", None)
    ky = getattr(native, "ky_spectrum", None)
    parameters = getattr(native, "saturation_parameters", None) or {}
    species = getattr(native, "spectral_species", None)
    if weights is None or ky is None or "kymax_out" not in parameters or not species or species[0] != 1:
        return None
    ky = np.asarray(ky, dtype=float)
    kmax = float(parameters["kymax_out"])
    nmodes = weights.shape[2]
    x = np.empty(nmodes)
    for mode in range(nmodes):
        if kmax <= ky[0]:
            x[mode] = 0.5
            continue
        if kmax > ky[-1]:
            return None
        i = int(np.argmax(ky >= kmax))
        ions = weights[1:, 0, mode, :, 1].sum(axis=0)        # sum over ions, phi, energy
        if ions[i] == 0.0 or ions[i - 1] == 0.0:
            x[mode] = 0.5
            continue
        ratio = np.abs(weights[0, 0, mode, i - 1:i + 1, 1] / ions[i - 1:i + 1])
        x[mode] = ((ratio[1] - ratio[0]) * kmax + (ky[i] * ratio[0] - ky[i - 1] * ratio[1])) / (ky[i] - ky[i - 1])
    qla = np.empty((5, nmodes))
    for mode in range(nmodes):
        qla[0, mode] = 2 * _transition(x[mode], 1.1, 0.6)
        qla[1, mode] = 2 * _transition(x[mode], 0.75, 0.6)
        qla[2:, mode] = QLA_OTHER
    return qla, x


def ky_integration_weights(ky: np.ndarray, kygrid_model: int = 1) -> np.ndarray:
    """``(2, ky)`` weights ``(dky0, dky1)`` of ``write_tglf_sum_flux_spectrum``.

    Entry ``k`` of ``sum_flux_spectrum`` is ``dky0[k] * F[k-1] + dky1[k] * F[k]``.
    """
    ky = np.asarray(ky, dtype=float)
    out = np.zeros((2, ky.size))
    out[1, 0] = ky[0]
    for k in range(1, ky.size):
        if kygrid_model != 0:
            d = np.log(ky[k] / ky[k - 1]) / (ky[k] - ky[k - 1])
            out[0, k] = ky[k - 1] * (ky[k] * d - 1.0)
            out[1, k] = ky[k] * (1.0 - ky[k - 1] * d)
        else:
            out[1, k] = ky[0]
    return out


def _integrate(per_ky: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Entries ``dky0 * F[k-1] + dky1 * F[k]`` along the ky axis (axis -2 of ``(..., ky, q)``)."""
    shifted = np.concatenate([np.zeros_like(per_ky[..., :1, :]), per_ky[..., :-1, :]], axis=-2)
    return weights[0][:, None] * shifted + weights[1][:, None] * per_ky


@dataclass(frozen=True)
class FluxDecomposition:
    """The factors of one TGLF run's saturated flux, or why they are withheld.

    ``status``:
    * ``validated``: the factors rebuild ``sum_flux_spectrum`` within ``tolerance``.
    * ``reconstruction_mismatch``: the rebuild disagrees, so no factor is returned.
    * ``unavailable``: an input the rebuild needs is missing.

    Array fields are ``None`` unless the decomposition is validated. Axes follow
    :class:`~vaft.code.gacode.tglf.outputs.TglfOutputs`:
    * ``weight``: ``(species, field, mode, ky, 5)``, the ``W`` above.
    * ``intensity``: ``(ky, mode)``, the ``phi`` above.
    * ``qla``: ``(5, mode)``.
    * ``x``: ``(mode,)``; SAT3 only, otherwise ``None``.
    * ``flux``: ``(species, field, mode, ky, 5)``, the product before ky integration.
    * ``ky_weights``: ``(2, ky)``.
    """

    status: str
    reason: str = ""
    relative_error: Optional[float] = None
    tolerance: float = RECONSTRUCTION_RTOL
    sat_rule: Optional[int] = None
    kygrid_model: Optional[int] = None
    ky: Optional[np.ndarray] = None
    weight: Optional[np.ndarray] = None
    intensity: Optional[np.ndarray] = None
    qla: Optional[np.ndarray] = None
    x: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    ky_weights: Optional[np.ndarray] = None

    @property
    def validated(self) -> bool:
        return self.status == "validated"

    def integrated(self, flux: Optional[np.ndarray] = None) -> np.ndarray:
        """``(species, field, 5)`` ky-integrated, mode-summed totals of ``flux`` (default: own)."""
        flux = self.flux if flux is None else flux
        return _integrate(flux.sum(axis=2), self.ky_weights).sum(axis=-2)


def flux_decomposition(native: Any, *, kygrid_model: int = 1,
                       tolerance: float = RECONSTRUCTION_RTOL) -> FluxDecomposition:
    """Factor a TGLF run's flux into ``W * phi * QLA`` and check the rebuild (module docstring).

    Parameters
    ----------
    native : TglfOutputs
        One run's native container.
    kygrid_model : int
        The run's ``KYGRID_MODEL`` (TGLF default 1). A wrong value is caught by the
        rebuild check, not silently accepted.
    tolerance : float
        Largest accepted relative rebuild error, per flux quantity.

    Returns
    -------
    FluxDecomposition
        Factors only when ``status == "validated"``.
    """
    parameters = getattr(native, "saturation_parameters", None) or {}
    sat = parameters.get("SAT_RULE")
    weights = getattr(native, "ql_flux_spectrum", None)
    field = getattr(native, "field_spectrum", None)
    native_sum = getattr(native, "sum_flux_spectrum", None)
    ky = getattr(native, "ky_spectrum", None)
    common = dict(tolerance=tolerance, sat_rule=sat, kygrid_model=kygrid_model)
    missing = [name for name, value in (("ql_flux_spectrum", weights), ("field_spectrum", field),
                                        ("sum_flux_spectrum", native_sum), ("ky_spectrum", ky),
                                        ("SAT_RULE", sat)) if value is None]
    if missing:
        return FluxDecomposition("unavailable", f"missing {', '.join(missing)}", **common)
    ky = np.asarray(ky, dtype=float)
    intensity = np.asarray(field[..., 1], dtype=float)
    nmodes = intensity.shape[1]
    if weights.shape[2] != nmodes or weights.shape[3] != ky.size or native_sum.shape[2] != ky.size:
        return FluxDecomposition("unavailable", "weight, intensity and flux grids disagree", **common)
    x = None
    if int(sat) == 3:
        factors = sat3_qla_factors(native)
        if factors is None:
            return FluxDecomposition("unavailable", "SAT3 amplitude factor cannot be formed "
                                     "(adiabatic electrons, no kymax_out, or kymax_out off the grid)",
                                     **common)
        qla, x = factors
    else:
        qla = np.ones((5, nmodes))
    weight = np.nan_to_num(np.asarray(weights, dtype=float))      # NaN = field not run
    flux = weight * intensity.T[None, None, :, :, None] * qla.T[None, None, :, None, :]
    ky_weights = ky_integration_weights(ky, kygrid_model)
    nfield = native_sum.shape[1]
    rebuilt = _integrate(flux.sum(axis=2), ky_weights)[:, :nfield]  # (species, field, ky, 5)
    target = np.asarray(native_sum, dtype=float)
    if rebuilt.shape != target.shape:
        return FluxDecomposition("unavailable", f"species/field axes disagree: {rebuilt.shape} vs "
                                 f"{target.shape}", **common)
    # A NaN field slot is one the run did not include (outputs.py); any other
    # non-finite native value is a flux that cannot be checked, so nothing is validated.
    present = ~np.all(np.isnan(target), axis=(0, 2, 3))
    if not present.any() or not np.all(np.isfinite(target[:, present])):
        return FluxDecomposition("unavailable", "sum_flux_spectrum is missing or not finite", **common)
    target, rebuilt = target[:, present], rebuilt[:, present]
    errors = []
    for q in range(target.shape[-1]):
        scale = float(np.max(np.abs(target[..., q])))
        difference = float(np.max(np.abs(rebuilt[..., q] - target[..., q])))
        # A quantity TGLF reports as exactly zero must be rebuilt as zero, not skipped.
        errors.append(difference / scale if scale > 0 else (0.0 if difference == 0.0 else np.inf))
    error = max(errors)
    if not error <= tolerance:
        return FluxDecomposition("reconstruction_mismatch",
                                 f"rebuild differs from sum_flux_spectrum by {error:.3g} (> {tolerance:g})",
                                 relative_error=error, **common)
    return FluxDecomposition("validated", relative_error=error, ky=ky, weight=weight,
                             intensity=intensity, qla=qla, x=x, flux=flux,
                             ky_weights=ky_weights, **common)


@dataclass(frozen=True)
class ClosureSplit:
    """Exact two-term split of the flux change ``b - a`` between two runs with equal linear physics.

    ``total = intensity_term + qla_term``, each ``(species, field, 5)`` in gyro-Bohm units:
    * ``intensity_term = sum W * (phi_b - phi_a) * QLA_a``
    * ``qla_term = sum W * phi_b * (QLA_b - QLA_a)``

    The split is exact but depends on the order: the intensity change is evaluated at
    ``a``'s amplitude factor. It describes the closure difference between the two
    saturation rules on a shared linear spectrum. It is not a decomposition into
    "pure saturation intensity" and anything else.
    ``status`` is ``split``, ``linear_differs`` or ``unavailable`` (a side not validated).
    """

    status: str
    reason: str = ""
    linear_max_rel_diff: Optional[float] = None
    total: Optional[np.ndarray] = None
    intensity_term: Optional[np.ndarray] = None
    qla_term: Optional[np.ndarray] = None


def _linear_difference(a: Any, b: Any) -> Optional[float]:
    """Largest relative difference of ky grid, eigenvalues and QL weights (``None``: shapes differ)."""
    worst = 0.0
    for name in ("ky_spectrum", "eigenvalue_spectrum", "ql_flux_spectrum"):
        left, right = getattr(a, name, None), getattr(b, name, None)
        if left is None or right is None or np.shape(left) != np.shape(right):
            return None
        left, right = np.nan_to_num(np.asarray(left, float)), np.nan_to_num(np.asarray(right, float))
        scale = float(np.max(np.abs(left))) or 1.0
        worst = max(worst, float(np.max(np.abs(left - right))) / scale)
    return worst


def closure_split(a: Any, b: Any, *, kygrid_model: int = 1,
                  linear_rtol: float = LINEAR_RTOL) -> ClosureSplit:
    """Split ``flux(b) - flux(a)`` into intensity and amplitude-factor terms (:class:`ClosureSplit`).

    Only for two runs whose ky grid, eigenvalues and QL weights agree within
    ``linear_rtol``. In TGLF b493397 that means SAT0/SAT1 at ``UNITS=GYRO``, or
    SAT2/SAT3. The agreement is measured here, not assumed from the SAT rules.
    """
    linear = _linear_difference(a, b)
    if linear is None or linear > linear_rtol:
        return ClosureSplit("linear_differs", "ky grid, eigenvalues or QL weights differ",
                            linear_max_rel_diff=linear)
    da, db = flux_decomposition(a, kygrid_model=kygrid_model), flux_decomposition(b, kygrid_model=kygrid_model)
    if not (da.validated and db.validated):
        return ClosureSplit("unavailable", f"decomposition: a {da.status}, b {db.status}",
                            linear_max_rel_diff=linear)
    w = da.weight
    intensity = w * (db.intensity - da.intensity).T[None, None, :, :, None] * da.qla.T[None, None, :, None, :]
    amplitude = w * db.intensity.T[None, None, :, :, None] * (db.qla - da.qla).T[None, None, :, None, :]
    return ClosureSplit("split", linear_max_rel_diff=linear,
                        total=db.integrated() - da.integrated(),
                        intensity_term=da.integrated(intensity), qla_term=da.integrated(amplitude))
