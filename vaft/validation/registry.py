"""Stable, machine-readable metadata for named validation checks (#253 §5, #72).

Every check a report can contain is described **once** here -- its category,
the unit of its headline value, the provider it composes, and the tolerance
that turns its normalized residual into a status.  Results then stay compact
(#253 §4): a check returns its numbers and a status, and a consumer that needs
the unit, the method or the tolerance asks the registry by the check's key.

That is how #72's acceptance criterion *"every reported quantity names its
definition, units, tolerance, and source"* is met without repeating those on
every result instance.  What is deliberately **not** here (#253 §5): formula
derivations, assumptions, limitations, references.  Those belong in the
provider's docstring, which ``provider`` points at.

Tolerances are the one policy this layer owns: the boundary between "agrees",
"disagrees but is usable" and "not credible" for the *check itself*.  Whether a
``warn`` result may still be used downstream is the consumer's policy, not
this module's (#253 §7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "CHECKS",
    "CheckSpec",
    "MEASURES",
    "checks_in",
    "describe",
]


#: How a check's normalized residual is measured.  The tolerance pair in a
#: :class:`CheckSpec` is read against this scale.
MEASURES: Mapping[str, str] = {
    "z": "residual over its standard deviation; |z| is graded",
    "relative": "|value - reference| / |reference|",
    "log_ratio": "|ln(value / reference)|, for positive quantities; symmetric in the two",
    "reduced_chi_squared": "chi-square total over degrees of freedom",
    "rule": "no threshold: the status follows the rule in `method`",
}


@dataclass(frozen=True)
class CheckSpec:
    """What is stable about one check, independent of any particular result."""

    key: str
    """``<category>.<check>``, the key under which a report carries the result."""
    category: str
    unit: str
    """Unit of the headline value (``"1"`` dimensionless, ``""`` when none)."""
    provider: str
    """Dotted path of the canonical provider whose docstring defines the quantity."""
    method: str
    """One line: how the status is reached."""
    measure: str = "rule"
    """A key of :data:`MEASURES`."""
    tolerance: tuple[float, float] | None = None
    """``(warn, fail)`` on the measure: ``pass`` at or below ``warn``, ``warn`` at
    or below ``fail``, ``fail`` beyond.  ``None`` when ``measure`` is ``"rule"``."""

    def __post_init__(self) -> None:
        if self.measure not in MEASURES:
            raise ValueError(f"{self.key}: unknown measure {self.measure!r}")
        if (self.tolerance is None) != (self.measure == "rule"):
            raise ValueError(f"{self.key}: a tolerance is required exactly when the measure is not a rule")
        if self.tolerance is not None and not (0 <= self.tolerance[0] <= self.tolerance[1]):
            raise ValueError(f"{self.key}: tolerance must satisfy 0 <= warn <= fail")


def _spec(key: str, unit: str, provider: str, method: str, measure: str = "rule",
          tolerance: tuple[float, float] | None = None) -> CheckSpec:
    return CheckSpec(key, key.split(".", 1)[0], unit, provider, method, measure, tolerance)


_SPECS = (
    # -- verification: was the calculation performed as intended? ---------------
    _spec("verification.structure", "",
          "vaft.process.equilibrium.check_equilibrium_requirements",
          "fail on a structural error (grid/psi shape, degenerate flux range, non-finite psi, "
          "non-positive area or volume); warn on a missing LCFS or a non-monotonic profiles_1d.psi"),
    _spec("verification.continuity", "1",
          "vaft.validation.equilibrium.verify_continuity",
          "fail unless equilibrium.time is finite and strictly increasing; warn when Ip changes sign "
          "or steps by more than half its median between consecutive slices; not_available below two slices"),
    _spec("verification.convention", "",
          "vaft.process.cocos.validate_cocos",
          "fail on a violated COCOS sign relation; warn on a q-sign violation only; "
          "indeterminate when no index is declared or its inputs are unavailable"),
    _spec("verification.convergence", "1",
          "vaft.omas.efit_quality.convergence_metrics",
          "fail when code.output_flag < 0, EFIT's own verdict rejects the slice, or the final "
          "iteration error exceeds the acceptance tolerance; warn when it stopped at the iteration "
          "cap or short of its exit tolerance; not_available without any solver evidence"),
    # -- diagnostic fit: does the reconstruction reproduce what was measured? -----
    _spec("diagnostic_fit.bpol_probe", "1", "vaft.omas.efit_quality.fit_quality_metrics",
          "RMS of the normalized residual over fitted poloidal-probe channels", "z", (2.0, 4.0)),
    _spec("diagnostic_fit.flux_loop", "1", "vaft.omas.efit_quality.fit_quality_metrics",
          "RMS of the normalized residual over fitted flux-loop channels", "z", (2.0, 4.0)),
    _spec("diagnostic_fit.pf_current", "1", "vaft.omas.efit_quality.fit_quality_metrics",
          "RMS of the normalized residual over fitted PF-current channels", "z", (2.0, 4.0)),
    _spec("diagnostic_fit.ip", "1", "vaft.omas.efit_quality.fit_quality_metrics",
          "normalized residual of the plasma-current constraint", "z", (2.0, 4.0)),
    _spec("diagnostic_fit.diamagnetic_flux", "1", "vaft.omas.efit_quality.fit_quality_metrics",
          "normalized residual of the diamagnetic-flux constraint", "z", (2.0, 4.0)),
    _spec("diagnostic_fit.global", "1", "vaft.omas.efit_quality.fit_quality_metrics",
          "reduced chi-square over every fitted channel", "reduced_chi_squared", (2.0, 5.0)),
    # -- physical validity: does the result satisfy force balance and be plausible? --
    _spec("physical_validity.virial", "1",
          "vaft.omas.process_wrapper.compute_virial_equilibrium_quantities_ods",
          "Lao virial closure; indeterminate when any Shafranov integral, alpha, B_pa, beta_p or li "
          "is non-finite (near-singular denominator or missing boundary); fail outside the "
          "plausibility bounds 0 < beta_p <= 10 and 0 < li <= 3"),
    _spec("physical_validity.pressure_consistency", "1",
          "vaft.process.equilibrium.derive_global_descriptors",
          "volume-integral beta_p (2 mu0 <p> / <Bp>^2, from profiles_1d.pressure) against the "
          "surface-integral virial beta_p", "log_ratio", (0.223, 0.693)),
    _spec("physical_validity.q_profile", "1", "vaft.validation.equilibrium.validate_physical",
          "fail when profiles_1d.q is non-finite or changes sign; warn when it is not monotonic"),
    _spec("physical_validity.pressure_profile", "Pa", "vaft.validation.equilibrium.validate_physical",
          "fail when profiles_1d.pressure is non-finite; warn when it is negative anywhere or "
          "not non-increasing from axis to edge"),
    _spec("physical_validity.diamagnetic_flux", "Wb",
          "vaft.omas.process_wrapper.compute_diamagnetic_flux_measured_vs_computed",
          "reconstructed diamagnetic flux against magnetics.diamagnetic_flux at the slice time; "
          "indeterminate when the measurement is zero", "relative", (0.25, 1.0)),
    # -- independent validation: does it agree with measurements it never used? --
    _spec("independent_validation.kinetic_pressure", "Pa",
          "vaft.validation.equilibrium.validate_independent",
          "core_profiles thermal pressure on the equilibrium's flux coordinate; electron-only "
          "coverage can exceed the reconstruction (fail) but falls short of it at most as warn",
          "log_ratio", (0.262, 0.693)),
    _spec("independent_validation.thomson_pressure", "Pa",
          "vaft.validation.equilibrium.validate_independent",
          "n_e k T_e from thomson_scattering channels mapped to psi_n through profiles_2d psi; "
          "same electron-only asymmetry as kinetic_pressure", "log_ratio", (0.262, 0.693)),
    _spec("independent_validation.diamagnetic_energy", "J",
          "vaft.formula.equilibrium.virial_beta_pd_from_S_mu_rt",
          "stored energy from the measured diamagnetic flux through the virial diamagnetic beta_p, "
          "against the virial kinetic energy", "log_ratio", (0.262, 0.693)),
)

#: Every check, by key.  Insertion order is report order.
CHECKS: Mapping[str, CheckSpec] = {spec.key: spec for spec in _SPECS}


def describe(key: str) -> CheckSpec:
    """The registry entry for ``key`` (``"<category>.<check>"``)."""
    try:
        return CHECKS[key]
    except KeyError:
        raise KeyError(f"no registered validation check {key!r}; known: {sorted(CHECKS)}") from None


def checks_in(category: str) -> tuple[CheckSpec, ...]:
    """Every registered check of one category, in report order."""
    return tuple(spec for spec in CHECKS.values() if spec.category == category)
