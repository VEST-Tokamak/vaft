"""EFIT goodness-of-fit and numerical-convergence metrics (issue #139).

Everything here is computed from quantities that were either **submitted to
EFIT** or **produced by the EFIT run** -- no independent experimental
comparison, no cross-code check, no synthetic truth, no uncertainty propagation
or sensitivity study. The question answered is narrow and internal: *is this fit
statistically acceptable against the uncertainties EFIT itself was given, and is
the solution numerically converged?*

The normalization, derived rather than assumed
----------------------------------------------
EFIT stores a per-channel chi-square. That value fixes the uncertainty it
actually fitted against, through the identity

.. code-block:: text

    chi2_i = ((measured_i - reconstructed_i) * weight_i / k)**2

where ``k`` reconciles the units the ODS stores with the units EFIT fitted in.
:func:`sigma_unit_factor` recovers ``k`` per family from EFIT's own chi-square
instead of hard-coding it, so the metric is self-validating and a future
convention change surfaces as a spread warning rather than as silently wrong
numbers.

On VEST ``k`` is 1 for B probes and 2*pi for flux loops -- the Wb versus Wb/rad
convention, since the ODS stores flux in Wb while the k-file writer divides by
2*pi on submission. The effective uncertainty is then ``sigma_i = k / weight_i``,
and the normalized residual ``z_i = (m_i - r_i) * weight_i / k`` is the residual
in units of that uncertainty, directly comparable across diagnostic families.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

__all__ = [
    "CONSTRAINT_STATES",
    "FIT_ROLES",
    "ConstraintTable",
    "FAMILIES",
    "classify_fit_role",
    "constraint_state",
    "constraint_table",
    "convergence_metrics",
    "efit_quality_metrics",
    "fit_quality_metrics",
    "normalized_residuals",
    "run_test_z",
    "sigma_unit_factor",
    "slice_times",
]

#: How a submitted constraint channel was classified.  ``generate_constraints_ods``
#: zeroes both ``measured`` and ``weight`` for a channel whose raw signal is
#: absent, and the k-file writer zeroes ``weight`` alone for a channel outside
#: the families EFIT fits, so the three states are decidable without consulting
#: the diagnostics ODS.
CONSTRAINT_STATES = ("enabled", "disabled", "missing")

#: Whether a family was fitted or merely handed back unchanged.  A prescribed
#: family's zero residual confirms an input was honoured; it says nothing about
#: fit quality, so it never enters a goodness-of-fit aggregate.
FIT_ROLES = ("fitted", "prescribed")

#: The constraint families, their display titles, units and scale factors.
FAMILIES = (
    ("bpol_probe", "Poloidal probes", "mT", 1e3, True),
    ("flux_loop", "Flux loops", "mWb", 1e3, True),
    ("pf_current", "PF currents", "kA", 1e-3, True),
)

#: Normalized-residual thresholds used for the outlier census.
OUTLIER_LEVELS = (2.0, 3.0)

#: EFIT's own defaults, from ``set_defaults.f90``.  A k-file that does not set
#: one of these still gets it, so a metric that ignored them would compare
#: against a threshold that was never in force.
EFIT_DEFAULTS = {
    "error": 1.0e-2,     # iteration exit tolerance, consumed only via `idone`
    "errmin": 1.0e-2,    # iconvr == 2 stopping precondition, and chkerr's threshold
    "saimin": 80.0,      # chi-square acceptance threshold, iconvr != 2
    "saicon": 80.0,      # chi-square stopping precondition, iconvr == 2
    "ierchk": 1,         # >0 means chkerr runs at all
    "mxiter": -25,
    "nxiter": 1,         # inner equilibrium-loop length
    "iconvr": 2,
}

#: Hard-coded in ``response_matrix.F90`` (``integer*4, parameter :: minite=8``),
#: with a TODO in the source questioning it.  No namelist can change it.
EFIT_MINITE = 8


# ---------------------------------------------------------------------------
# ODS access primitives
# ---------------------------------------------------------------------------

def _get(ods: Any, path: str, default: Any = None) -> Any:
    """Read one leaf, treating an absent or container-valued path as absent.

    OMAS auto-vivifies parts of the ``code.parameters`` subtree on read, so a
    missing leaf can come back as an empty container rather than raising.
    """
    try:
        value = ods[path]
    except (KeyError, ValueError, IndexError, TypeError):
        return default
    if value is None or hasattr(value, "keys"):
        return default
    return value


def _count(ods: Any, path: str) -> int:
    try:
        return len(ods[path])
    except (KeyError, ValueError, IndexError, TypeError):
        return 0


def _scalar(value: Any, scale: float = 1.0) -> float:
    try:
        return float(np.asarray(value)) * scale
    except (TypeError, ValueError):
        return float("nan")


def _array(ods: Any, path: str) -> np.ndarray | None:
    value = _get(ods, path)
    if value is None:
        return None
    array = np.asarray(value, dtype=float).reshape(-1)
    return array if array.size else None


def slice_times(ods: Any) -> np.ndarray:
    """Reconstruction time per equilibrium slice, falling back to slice index."""
    count = _count(ods, "equilibrium.time_slice")
    times = _array(ods, "equilibrium.time")
    if times is not None and times.size >= count:
        return np.asarray(times[:count], dtype=float)
    return np.asarray(
        [
            _scalar(_get(ods, f"equilibrium.time_slice.{index}.time", index))
            for index in range(count)
        ],
        dtype=float,
    )


def constraint_state(measured: float, weight: float) -> str:
    """Classify one channel from what the constraint builder wrote."""
    if np.isfinite(weight) and weight == 0.0:
        return "missing" if measured == 0.0 else "disabled"
    return "enabled"


@dataclass(frozen=True)
class ConstraintTable:
    """One EFIT constraint family at one time slice, channel by channel.

    Every channel is present, including the ones with no data: which channels
    went missing is exactly what the submitted-constraint validation exists to
    show.  Consumers that only care about fitted channels filter on ``state``.
    """

    family: str
    index: np.ndarray
    measured: np.ndarray
    reconstructed: np.ndarray
    uncertainty: np.ndarray
    weight: np.ndarray
    chi_squared: np.ndarray
    state: tuple[str, ...]
    source: tuple[str, ...]

    @property
    def residual(self) -> np.ndarray:
        return self.measured - self.reconstructed

    def mask(self, *states: str) -> np.ndarray:
        return np.array([item in states for item in self.state], dtype=bool)

    def count(self, state: str) -> int:
        return sum(1 for item in self.state if item == state)


def constraint_table(
    ods: Any, *, time_slice: int, family: str, is_array: bool = True, scale: float = 1.0
) -> ConstraintTable:
    """Read one constraint family at one slice into parallel arrays."""
    root = f"equilibrium.time_slice.{time_slice}.constraints.{family}"
    count = _count(ods, root) if is_array else 1
    index, measured, reconstructed, uncertainty, weight, chi = [], [], [], [], [], []
    state: list[str] = []
    source: list[str] = []
    for position in range(count):
        base = f"{root}.{position}" if is_array else root
        measured_value = _scalar(_get(ods, f"{base}.measured"), scale)
        weight_value = _scalar(_get(ods, f"{base}.weight"))
        index.append(position)
        measured.append(measured_value)
        reconstructed.append(_scalar(_get(ods, f"{base}.reconstructed"), scale))
        uncertainty.append(_scalar(_get(ods, f"{base}.measured_error_upper"), abs(scale)))
        weight.append(weight_value)
        chi.append(_scalar(_get(ods, f"{base}.chi_squared")))
        state.append(constraint_state(measured_value, weight_value))
        identifier = _get(ods, f"{base}.source")
        source.append(
            str(identifier) if identifier not in (None, "") else f"{family}[{position}]"
        )
    return ConstraintTable(
        family=family,
        index=np.asarray(index, dtype=float),
        measured=np.asarray(measured, dtype=float),
        reconstructed=np.asarray(reconstructed, dtype=float),
        uncertainty=np.asarray(uncertainty, dtype=float),
        weight=np.asarray(weight, dtype=float),
        chi_squared=np.asarray(chi, dtype=float),
        state=tuple(state),
        source=tuple(source),
    )


# ---------------------------------------------------------------------------
# A0-A2: role, normalization, normalized residual
# ---------------------------------------------------------------------------

def classify_fit_role(ods: Any, table: ConstraintTable, *, time_slice: int) -> str:
    """Whether ``table``'s family was fitted or handed back unchanged.

    A prescribed family -- ``exact`` set, or a residual and chi-square that are
    identically zero on every channel -- confirms that an input was honoured. It
    is not evidence about fit quality and must not dilute an aggregate.
    """
    root = f"equilibrium.time_slice.{time_slice}.constraints.{table.family}"
    for position in range(len(table.state)):
        if _scalar(_get(ods, f"{root}.{position}.exact", 0.0)) not in (0.0, float("nan")):
            return "prescribed"
    fitted = table.mask("enabled", "disabled")
    if not fitted.any():
        return "prescribed"
    residual = table.residual[fitted]
    chi = table.chi_squared[fitted]
    residual_zero = np.all(np.nan_to_num(residual) == 0.0)
    chi_zero = np.all(np.nan_to_num(chi) == 0.0)
    return "prescribed" if residual_zero and chi_zero else "fitted"


def sigma_unit_factor(table: ConstraintTable) -> tuple[float, float]:
    """Recover EFIT's units-of-fit factor ``k`` from its own chi-square.

    Returns ``(k, relative_spread)``.  ``k`` is the median of
    ``|(m - r) * w| / sqrt(chi2)`` over channels where all three are usable; the
    spread is the channel-to-channel scatter, which must be small -- a large one
    means the stored chi-square and the stored residual no longer describe the
    same fit, and every normalized residual built on it is suspect.
    """
    usable = (
        (table.weight > 0)
        & np.isfinite(table.chi_squared)
        & (table.chi_squared > 0)
        & np.isfinite(table.residual)
    )
    if not usable.any():
        return float("nan"), float("nan")
    ratios = np.abs(table.residual[usable] * table.weight[usable]) / np.sqrt(
        table.chi_squared[usable]
    )
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    if not ratios.size:
        return float("nan"), float("nan")
    k = float(np.median(ratios))
    spread = float(np.ptp(ratios) / k) if k else float("nan")
    return k, spread


def normalized_residuals(table: ConstraintTable, k: float) -> np.ndarray:
    """Residuals in units of the uncertainty EFIT assigned: ``z = (m-r)*w/k``."""
    if not np.isfinite(k) or k == 0.0:
        return np.full(table.residual.shape, np.nan)
    return table.residual * table.weight / k


# ---------------------------------------------------------------------------
# A6, A9: bias and residual structure
# ---------------------------------------------------------------------------

def run_test_z(values: np.ndarray) -> float:
    """Wald-Wolfowitz runs-test z-score for the sign sequence of ``values``.

    Coherent sign patterns across a spatially ordered channel array indicate an
    unmodelled field component rather than random scatter.  ``|z| > 2`` means
    the sign sequence is unlikely under independence: too few runs (negative z)
    is clustering, too many (positive z) is alternation.

    Channels are taken in array order, which is VEST's own channel ordering --
    not poloidal angle, since the EFIT ODS carries no ``magnetics`` IDS.
    """
    signs = np.sign(values[np.isfinite(values) & (values != 0.0)])
    n = signs.size
    if n < 3:
        return float("nan")
    positive = int(np.sum(signs > 0))
    negative = n - positive
    if positive == 0 or negative == 0:
        return float("nan")
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    expected = 2.0 * positive * negative / n + 1.0
    variance = (
        2.0 * positive * negative * (2.0 * positive * negative - n)
        / (n * n * (n - 1.0))
    )
    if variance <= 0.0:
        return float("nan")
    return float((runs - expected) / math.sqrt(variance))


def _lag1_autocorrelation(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size < 3:
        return float("nan")
    centered = finite - finite.mean()
    denominator = float(np.sum(centered * centered))
    if denominator == 0.0:
        return float("nan")
    return float(np.sum(centered[1:] * centered[:-1]) / denominator)


# ---------------------------------------------------------------------------
# Part A: goodness of fit
# ---------------------------------------------------------------------------

def fit_quality_metrics(ods: Any, *, time_slice: int) -> dict[str, Any]:
    """Goodness-of-fit metrics for one reconstructed time slice."""
    families: dict[str, Any] = {}
    total_chi = 0.0
    fitted_channels = 0
    for family, title, unit, scale, is_array in FAMILIES:
        table = constraint_table(
            ods, time_slice=time_slice, family=family, is_array=is_array
        )
        if not len(table.state):
            continue
        role = classify_fit_role(ods, table, time_slice=time_slice)
        k, spread = sigma_unit_factor(table)
        z = normalized_residuals(table, k)
        fitted = table.mask("enabled") & np.isfinite(z)
        chi_sum = float(np.nansum(table.chi_squared[table.mask("enabled")]))

        entry: dict[str, Any] = {
            "title": title,
            "display_unit": unit,
            "fit_role": role,
            "tier": {"fit_role": "metadata"},
            "channels": {state: table.count(state) for state in CONSTRAINT_STATES},
            "sigma_unit_factor": k,
            "sigma_unit_factor_spread": spread,
            "chi_squared_sum": chi_sum,
            # Display units, kept from the earlier slice for continuity.
            "residual_rms_display": (
                float(np.sqrt(np.mean((table.residual[fitted] * scale) ** 2)))
                if fitted.any()
                else float("nan")
            ),
        }
        if role == "fitted" and fitted.any():
            selected = z[fitted]
            count = selected.size
            bias = float(np.mean(selected))
            standard_error = 1.0 / math.sqrt(count)
            worst = int(np.argmax(np.abs(selected)))
            worst_index = int(np.flatnonzero(fitted)[worst])
            entry.update(
                {
                    "z_rms": float(np.sqrt(np.mean(selected**2))),
                    "z_bias": bias,
                    "z_bias_standard_error": standard_error,
                    "z_bias_significant": bool(abs(bias) > 2.0 * standard_error),
                    "z_abs_max": float(np.abs(selected).max()),
                    "z_abs_max_channel": table.source[worst_index],
                    "outlier_fraction": {
                        f"gt_{level:g}sigma": float(
                            np.mean(np.abs(selected) > level)
                        )
                        for level in OUTLIER_LEVELS
                    },
                    "residual_structure": {
                        "lag1_autocorrelation": _lag1_autocorrelation(selected),
                        "run_test_z": run_test_z(selected),
                        "ordering": "channel index (VEST array order, not poloidal angle)",
                    },
                }
            )
            total_chi += chi_sum
            fitted_channels += count
        families[family] = entry

    # Scalar constraints participate in the total but have no channel array.
    scalars: dict[str, Any] = {}
    for family in ("ip", "diamagnetic_flux"):
        root = f"equilibrium.time_slice.{time_slice}.constraints.{family}"
        chi = _scalar(_get(ods, f"{root}.chi_squared"))
        if not np.isfinite(chi):
            continue
        measured = _scalar(_get(ods, f"{root}.measured"))
        reconstructed = _scalar(_get(ods, f"{root}.reconstructed"))
        weight = _scalar(_get(ods, f"{root}.weight"))
        scalars[family] = {
            "measured": measured,
            "reconstructed": reconstructed,
            "chi_squared": chi,
            "z": float(math.copysign(math.sqrt(chi), measured - reconstructed))
            if chi >= 0
            else float("nan"),
            "sigma_from_weight": float(1.0 / weight) if weight else float("nan"),
        }
        total_chi += chi
        fitted_channels += 1

    aux = f"equilibrium.code.parameters.time_slice.{time_slice}.auxquantities"
    dof = _scalar(_get(ods, f"{aux}.degrees_of_freedom"))
    dof_inputs = {
        "num_input_data": _scalar(_get(ods, f"{aux}.num_input_data")),
        "num_fit_variables": _scalar(_get(ods, f"{aux}.num_fit_variables")),
        "num_hard_constraints": _scalar(_get(ods, f"{aux}.num_hard_constraints")),
    }
    reduced = float(total_chi / dof) if np.isfinite(dof) and dof > 0 else float("nan")

    shares = {
        name: (entry["chi_squared_sum"] / total_chi if total_chi else float("nan"))
        for name, entry in families.items()
        if entry["fit_role"] == "fitted"
    }
    shares.update(
        {
            name: (entry["chi_squared"] / total_chi if total_chi else float("nan"))
            for name, entry in scalars.items()
        }
    )

    return {
        "families": families,
        "scalars": scalars,
        "chi_squared_total": float(total_chi),
        "degrees_of_freedom": dof,
        "degrees_of_freedom_inputs": dof_inputs,
        # ~1 is consistent with the assigned uncertainties; >>1 indicates model
        # or calibration error; <<1 over-assigned uncertainties or over-fitting.
        "chi_squared_reduced": reduced,
        "chi_squared_share": shares,
        "fitted_channel_count": fitted_channels,
        "tier": {
            "chi_squared_reduced": "primary",
            "chi_squared_share": "primary",
            "z_rms": "primary",
            "z_bias": "primary",
            "z_abs_max": "primary",
            "outlier_fraction": "diagnostic",
            "residual_structure": "diagnostic",
            "sigma_unit_factor": "diagnostic",
            "residual_rms_display": "diagnostic",
        },
    }


# ---------------------------------------------------------------------------
# Part B: numerical convergence
# ---------------------------------------------------------------------------

def _bilinear(
    values: np.ndarray, x: np.ndarray, y: np.ndarray, at_x: float, at_y: float
) -> float:
    """Bilinear sample of ``values`` on the ``(x, y)`` grid, NaN outside it."""
    if at_x < x[0] or at_x > x[-1] or at_y < y[0] or at_y > y[-1]:
        return float("nan")
    i = int(np.clip(np.searchsorted(x, at_x) - 1, 0, x.size - 2))
    j = int(np.clip(np.searchsorted(y, at_y) - 1, 0, y.size - 2))
    tx = (at_x - x[i]) / (x[i + 1] - x[i]) if x[i + 1] != x[i] else 0.0
    ty = (at_y - y[j]) / (y[j + 1] - y[j]) if y[j + 1] != y[j] else 0.0
    return float(
        values[i, j] * (1 - tx) * (1 - ty)
        + values[i + 1, j] * tx * (1 - ty)
        + values[i, j + 1] * (1 - tx) * ty
        + values[i + 1, j + 1] * tx * ty
    )


def _relative_difference(values: Iterable[float]) -> float:
    finite = [value for value in values if np.isfinite(value)]
    if len(finite) < 2:
        return float("nan")
    scale = max(abs(value) for value in finite)
    if scale == 0.0:
        return 0.0
    return float((max(finite) - min(finite)) / scale)


def convergence_metrics(ods: Any, *, time_slice: int) -> dict[str, Any]:
    """Numerical-convergence metrics for one slice, from EFIT-native output."""
    root = f"equilibrium.time_slice.{time_slice}"
    parameters = f"equilibrium.code.parameters.time_slice.{time_slice}"

    # B1: EFIT's own verdict, the only place it is stated.
    aeqdsk = f"{parameters}.aeqdsk"
    jflag = _get(ods, f"{aeqdsk}.jflag")
    lflag = _get(ods, f"{aeqdsk}.lflag")
    verdict: dict[str, Any] = {
        "source": "aeqdsk" if jflag is not None else "unavailable",
        "jflag": None if jflag is None else int(jflag),
        "lflag": None if lflag is None else int(lflag),
        # Deliberately "accepted", not "converged": jflag starts at 1 and only
        # drops when chkerr objects, and chkerr judges terror against `errmin`
        # (iconvr == 2) rather than against the iteration exit tolerance. See
        # the error block's exit_ratio for the tolerance question.
        "accepted": None if jflag is None else bool(int(jflag) == 1 and int(lflag) == 0),
        "meaning": "chkerr raised nothing, or was disabled (ierchk <= 0)",
        "limiter_location": _get(ods, f"{aeqdsk}.limloc"),
        "q_method_flag": _get(ods, f"{aeqdsk}.qmflag"),
        "fit_type": _get(ods, f"{aeqdsk}.fit_type"),
    }
    if jflag is None:
        verdict["reason"] = "no a-file was parsed for this slice"

    # B2: two different tolerances, and they are not interchangeable.
    #
    # `terror` (a-file) is the final value of `errorm` from residu():
    #     errorm = max|psi - psi_previous| over the grid / |sidif| / relax
    # `cerror` records the same quantity per iteration, and the iteration exit
    # test is `errorm <= error`. So terror, cerror and in1.error are the same
    # normalized quantity and are directly comparable.
    #
    # EFIT's *acceptance* test is a different one: chkerr compares terror with
    # `errmin` when iconvr == 2 and with `error` otherwise. Those two can differ
    # by orders of magnitude, so both are reported along with which one was in
    # force -- and where a value came from EFIT's default rather than from the
    # k-file, that is said.
    def _setting(name: str) -> tuple[float, str]:
        for block in ("in1", "out1"):
            value = _scalar(_get(ods, f"{parameters}.{block}.{name}"))
            if np.isfinite(value):
                return value, block
        default = EFIT_DEFAULTS.get(name, float("nan"))
        return float(default), "efit_default"

    iconvr = _scalar(_get(ods, f"{parameters}.out1.iconvr"))
    tolerance_name = "errmin" if iconvr == 2 else "error"
    exit_tolerance, exit_source = _setting("error")
    acceptance_tolerance, acceptance_source = _setting(tolerance_name)

    terror = _scalar(_get(ods, f"{aeqdsk}.terror"))
    final_error = terror
    final_error_source = "aeqdsk.terror"
    if not np.isfinite(final_error):
        final_error = _scalar(
            _get(ods, f"{root}.convergence.grad_shafranov_deviation_value")
        )
        final_error_source = "convergence.grad_shafranov_deviation_value"

    def _ratio(tolerance: float) -> float:
        return (
            float(final_error / tolerance)
            if np.isfinite(final_error) and np.isfinite(tolerance) and tolerance
            else float("nan")
        )

    error_block = {
        "final_error": final_error,
        "final_error_source": final_error_source,
        "definition": "max|psi - psi_prev| over the grid / |sidif| / relax",
        # Did the Grad-Shafranov iteration reach the tolerance it was asked for?
        "exit_tolerance": exit_tolerance,
        "exit_tolerance_source": exit_source,
        "exit_ratio": _ratio(exit_tolerance),
        "reached_exit_tolerance": bool(
            np.isfinite(final_error)
            and np.isfinite(exit_tolerance)
            and final_error <= exit_tolerance
        ),
        # Which threshold EFIT's own acceptance check actually applied.
        "iconvr": iconvr,
        "acceptance_tolerance_name": tolerance_name,
        "acceptance_tolerance": acceptance_tolerance,
        "acceptance_tolerance_source": acceptance_source,
        "acceptance_ratio": _ratio(acceptance_tolerance),
        "within_acceptance_tolerance": bool(
            np.isfinite(final_error)
            and np.isfinite(acceptance_tolerance)
            and final_error < acceptance_tolerance
        ),
        "secondary_tolerance": _scalar(_get(ods, f"{parameters}.in1.serror")),
    }

    # `error` reaches the solver only through `idone` in residu(), which breaks
    # the inner `equilibrium: do ii=1,nxiter` loop. With nxiter == 1 that loop
    # runs a single pass regardless, and for iconvr == 2 the outer loop is left
    # through `ichisq`, which never consults `error`. The requested tolerance is
    # then inert, and `exit_ratio` says nothing about the solve -- recorded so a
    # large ratio is not mistaken for a convergence failure.
    nxiter, nxiter_source = _setting("nxiter")
    error_block["exit_tolerance_effective"] = not (
        iconvr == 2 and np.isfinite(nxiter) and abs(nxiter) <= 1
    )
    error_block["nxiter"] = nxiter
    error_block["nxiter_source"] = nxiter_source
    if not error_block["exit_tolerance_effective"]:
        error_block["exit_tolerance_inert_reason"] = (
            "iconvr=2 with nxiter=1: `error` is consumed only by `idone`, which "
            "breaks a single-pass inner loop, and the outer loop exits on "
            "`ichisq`. The requested tolerance never gates this run."
        )

    # chkerr's other acceptance test, on EFIT's own total chi-square.
    chisq_name = "saicon" if iconvr == 2 else "saimin"
    chisq_limit, chisq_source = _setting(chisq_name)
    chisq_value = _scalar(_get(ods, f"{aeqdsk}.chisq"))
    error_block["chi_squared_total"] = chisq_value
    error_block["chi_squared_limit_name"] = chisq_name
    error_block["chi_squared_limit"] = chisq_limit
    error_block["chi_squared_limit_source"] = chisq_source
    error_block["chi_squared_margin"] = (
        float(chisq_value / chisq_limit)
        if np.isfinite(chisq_value) and np.isfinite(chisq_limit) and chisq_limit
        else float("nan")
    )
    # The a-file total is `saisq`, which at an iconvr==2 stop has been reset to
    # `saiold` -- the previous iterate's value -- while the per-channel `chiout`
    # arrays hold the current iterate's. It also includes `saisref` (reference
    # flux loop) and `chiecc` (E-coils), neither of which has an OMAS constraint
    # family here. The two totals are therefore close but not equal by design.
    error_block["chi_squared_comparable_to_family_sum"] = False

    # B3: how the solve actually terminated.
    #
    # For iconvr == 2 -- EFIT's default and VEST's mode -- the outer loop is
    # left through `ichisq` (response_matrix.F90), which fires only when all of
    #     nniter >= minite (8, hard-coded)
    #     errorm <= errmin
    #     saisq  <= saicon
    #     |saisq - saiold| <= 0.10  or  saisq >= saiold   (chi-square stalled)
    # hold at once, and then restores the previous iterate (brsp = brsold,
    # saisq = saiold).
    #
    # So `terror <= errmin` and `chisq <= saicon` are *preconditions of
    # stopping*, not achievements: any run that stopped this way satisfies them
    # by construction.  The discriminator that carries information is whether
    # the run stopped that way at all, or instead exhausted its iterations.
    iterations = _scalar(_get(ods, f"{root}.convergence.iterations_n"))
    cap, cap_source = _setting("mxiter")
    cap = abs(cap)
    hit_cap = bool(
        np.isfinite(iterations) and np.isfinite(cap) and cap and iterations >= cap
    )
    iteration_block = {
        "iterations": iterations,
        "iteration_cap": cap,
        "iteration_cap_source": cap_source,
        "minimum_iterations": EFIT_MINITE,
        "hit_cap": hit_cap,
        # True when the solve left through the iconvr==2 criterion rather than
        # running out of iterations. This is the convergence question that has
        # content for this configuration.
        "stopped_on_criterion": bool(
            np.isfinite(iterations)
            and iterations >= EFIT_MINITE
            and not hit_cap
            and error_block["within_acceptance_tolerance"]
            and (
                not np.isfinite(error_block["chi_squared_margin"])
                or error_block["chi_squared_margin"] < 1.0
            )
        )
        if iconvr == 2
        else None,
    }

    # B4-B6: the approach, not just the endpoint.  Needs an m-file.
    history = _array(ods, f"{parameters}.meqdsk.variables.cerror.data")
    if history is None or history.size < 2:
        history_block: dict[str, Any] = {
            "available": False,
            "reason": "no m-file cerror history was mapped for this slice",
        }
    else:
        positive = history[np.isfinite(history) & (history > 0)]
        decreases = int(np.sum(history[1:] < history[:-1]))
        tail = positive[-min(positive.size, 5) :]
        if tail.size >= 3:
            rate = float(
                np.polyfit(np.arange(tail.size, dtype=float), np.log10(tail), 1)[0]
            )
        else:
            rate = float("nan")
        history_block = {
            "available": True,
            "iterations": int(history.size),
            "first_error": float(history[0]),
            "final_error": float(history[-1]),
            "monotonic_fraction": float(decreases / max(history.size - 1, 1)),
            "final_decade_rate": rate,
            "stagnated": bool(np.isfinite(rate) and abs(rate) < 0.05),
        }
        if not np.isfinite(terror):
            error_block["final_error"] = float(history[-1])
            error_block["final_error_source"] = "meqdsk.cerror"
        history_block["agrees_with_aeqdsk_terror"] = (
            bool(np.isclose(history[-1], terror, rtol=1e-6))
            if np.isfinite(terror)
            else None
        )

    # B7-B9: EFIT's own outputs must agree with each other.
    ip_values = [
        _scalar(_get(ods, f"{root}.global_quantities.ip")),
        _scalar(_get(ods, f"{root}.constraints.ip.reconstructed")),
        _scalar(_get(ods, f"{parameters}.aeqdsk.cpasma")),
    ]
    self_consistency: dict[str, Any] = {
        "ip_relative_spread": _relative_difference(ip_values),
        "ip_sources": {
            "global_quantities": ip_values[0],
            "constraint_reconstructed": ip_values[1],
            "aeqdsk_cpasma": ip_values[2],
        },
    }
    psi = _get(ods, f"{root}.profiles_2d.0.psi")
    r_grid = _array(ods, f"{root}.profiles_2d.0.grid.dim1")
    z_grid = _array(ods, f"{root}.profiles_2d.0.grid.dim2")
    if psi is not None and r_grid is not None and z_grid is not None:
        grid_psi = np.asarray(psi, dtype=float)
        psi_axis = _scalar(_get(ods, f"{root}.global_quantities.psi_axis"))
        psi_boundary = _scalar(_get(ods, f"{root}.global_quantities.psi_boundary"))
        axis_r = _scalar(_get(ods, f"{root}.global_quantities.magnetic_axis.r"))
        axis_z = _scalar(_get(ods, f"{root}.global_quantities.magnetic_axis.z"))
        span = abs(psi_boundary - psi_axis)
        if (
            grid_psi.shape == (r_grid.size, z_grid.size)
            and np.isfinite([psi_axis, axis_r, axis_z, span]).all()
            and span > 0
        ):
            # Evaluate the reconstructed flux map at the reported axis and
            # compare with the reported scalar.  The magnetic axis is a *local*
            # extremum inside the plasma, not the grid's global one -- on a
            # decaying, inboard-shifted slice the global minimum sits in a grid
            # corner -- so a global-extremum search is the wrong comparison.
            interpolated = _bilinear(grid_psi, r_grid, z_grid, axis_r, axis_z)
            self_consistency["psi_axis_grid_offset"] = (
                float(abs(interpolated - psi_axis) / span)
                if np.isfinite(interpolated)
                else float("nan")
            )
            self_consistency["psi_axis_grid_value"] = interpolated
            # The defining property of the axis: stationary flux.  Checked on
            # the 3x3 neighbourhood of the nearest grid node.
            row = int(np.argmin(np.abs(r_grid - axis_r)))
            column = int(np.argmin(np.abs(z_grid - axis_z)))
            rows = slice(max(row - 1, 0), row + 2)
            columns = slice(max(column - 1, 0), column + 2)
            window = grid_psi[rows, columns]
            centre = grid_psi[row, column]
            towards_boundary = np.sign(psi_boundary - psi_axis)
            self_consistency["magnetic_axis_is_local_extremum"] = bool(
                np.all(towards_boundary * (window - centre) >= 0.0)
            )

    settings = {
        name: _get(ods, f"{parameters}.{path}")
        for name, path in (
            ("error", "in1.error"),
            ("serror", "in1.serror"),
            ("mxiter", "in1.mxiter"),
            ("relax", "in1.relax"),
            ("iconvr", "out1.iconvr"),
            ("icurrt", "out1.icurrt"),
            ("kppcur", "in1.kppcur"),
            ("kffcur", "in1.kffcur"),
            ("nxiter", "out1.nxiter"),
        )
    }

    return {
        "verdict": verdict,
        "error": error_block,
        "iterations": iteration_block,
        "history": history_block,
        "self_consistency": self_consistency,
        "solver_settings": {
            key: (None if value is None else _json_safe(value))
            for key, value in settings.items()
        },
        "tier": {
            "verdict": "primary",
            "error.reached_exit_tolerance": "primary",
            "error.within_acceptance_tolerance": "primary",
            "error.exit_ratio": "primary",
            "error.acceptance_ratio": "primary",
            "error.chi_squared_margin": "diagnostic",
            "iterations.stopped_on_criterion": "primary",
            "iterations.hit_cap": "primary",
            "error.exit_tolerance_effective": "metadata",
            "iterations": "primary",
            "self_consistency.ip_relative_spread": "primary",
            "history": "diagnostic",
            "self_consistency.psi_axis_grid_offset": "diagnostic",
            "self_consistency.magnetic_axis_grid_distance_m": "diagnostic",
            "solver_settings": "metadata",
        },
    }


def _json_safe(value: Any) -> Any:
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.reshape(-1).tolist()


def efit_quality_metrics(ods: Any) -> dict[str, Any]:
    """Goodness-of-fit and convergence metrics for every reconstructed slice."""
    times = slice_times(ods)
    slices = []
    for index in range(times.size):
        slices.append(
            {
                "time": float(times[index]),
                "fit": fit_quality_metrics(ods, time_slice=index),
                "convergence": convergence_metrics(ods, time_slice=index),
            }
        )
    reduced = np.array(
        [entry["fit"]["chi_squared_reduced"] for entry in slices], dtype=float
    )
    finite = reduced[np.isfinite(reduced)]
    accepted = [
        entry["convergence"]["verdict"]["accepted"]
        for entry in slices
        if entry["convergence"]["verdict"]["accepted"] is not None
    ]
    reached = [
        entry["convergence"]["error"]["reached_exit_tolerance"]
        for entry in slices
        if np.isfinite(entry["convergence"]["error"]["final_error"])
    ]
    return {
        "schema_version": 1,
        "slice_count": len(slices),
        "slices": slices,
        "summary": {
            "chi_squared_reduced_median": (
                float(np.median(finite)) if finite.size else float("nan")
            ),
            "chi_squared_reduced_max": (
                float(np.max(finite)) if finite.size else float("nan")
            ),
            "slices_accepted_by_efit": sum(1 for value in accepted if value),
            "slices_with_verdict": len(accepted),
            # The separate question: did the iteration reach in1.error?
            "slices_reaching_exit_tolerance": sum(1 for value in reached if value),
            "slices_with_final_error": len(reached),
            "slices_hitting_iteration_cap": sum(
                1 for entry in slices if entry["convergence"]["iterations"]["hit_cap"]
            ),
            "slices_stopped_on_criterion": sum(
                1
                for entry in slices
                if entry["convergence"]["iterations"]["stopped_on_criterion"]
            ),
        },
    }
