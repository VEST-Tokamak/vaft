"""Typed scientific configuration for VEST EFIT inputs.

The defaults in this module preserve the routine VEST k-file semantics.  The
objects are independent of an EFIT installation, so configurations can be
validated, serialized, hashed, and scanned without running the binary.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
import hashlib
from itertools import product
import json
import math
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, Sequence


DIAGNOSTIC_GROUPS = frozenset(
    {
        "pf_current",
        "plasma_current",
        "diamagnetic_flux",
        "bpol_probe",
        "flux_loop",
    }
)


def _require_finite(name: str, value: float, *, positive: bool = False) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    if positive and float(value) <= 0:
        raise ValueError(f"{name} must be greater than zero")


def _legacy_coil_constraint_matrix() -> tuple[tuple[float, ...], ...]:
    """Return the routine 16-coil by 12-constraint matrix."""
    matrix = [[0.0 for _ in range(12)] for _ in range(16)]
    for column in range(7):
        matrix[0][column] = 1.0
        matrix[column + 1][column] = -1.0
    for column, upper in enumerate((8, 10, 12, 14), start=7):
        matrix[upper][column] = 1.0
        matrix[upper + 1][column] = -1.0
    matrix[13][11] = 1.0
    matrix[14][11] = -1.0
    return tuple(tuple(row) for row in matrix)


@dataclass(frozen=True)
class EFITProfileConfig:
    """Pressure and FF-prime representation written to the EFIT namelist."""

    kppcur: int = 2
    kffcur: int = 2
    kppfnc: int = 0
    kfffnc: int = 0
    pcurbd: int = 1
    fcurbd: int = 1

    def __post_init__(self) -> None:
        for name in ("kppcur", "kffcur"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            object.__setattr__(self, name, int(value))
        for name in ("kppfnc", "kfffnc"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            object.__setattr__(self, name, int(value))
        for name in ("pcurbd", "fcurbd"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value not in (0, 1)
            ):
                raise ValueError(f"{name} must be 0 or 1")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class EFITInitializationConfig:
    """Ellipse and low-current initialization controls."""

    rzero: float = 0.4
    zzero: float = 0.0
    minor_radius: float = 0.3
    elongation: float = 1.6
    current_threshold: float = 5_000.0

    def __post_init__(self) -> None:
        _require_finite("rzero", self.rzero, positive=True)
        _require_finite("zzero", self.zzero)
        _require_finite("minor_radius", self.minor_radius, positive=True)
        _require_finite("elongation", self.elongation, positive=True)
        _require_finite("current_threshold", self.current_threshold)
        if self.current_threshold < 0:
            raise ValueError("current_threshold must be non-negative")


@dataclass(frozen=True)
class EFITNumericsConfig:
    """Convergence, relaxation, and iteration controls.

    The four settings VEST actually stops on were, until issue #171, not
    expressible here at all: ``ERRMIN``, ``SAICON``, ``ICONVR`` and ``NXITER``
    were left to EFIT's own defaults, one of them behind a commented-out line
    in the writer.  That is why a VEST fit terminates on the chi-square
    criterion after eleven iterations while the ``ERROR`` written beside it
    asks for 1e-5 and is never reached.

    They are ``None`` by default and a ``None`` writes no key, so the emitted
    k-file is byte-identical to before and EFIT's default still applies.  A
    study sets them explicitly; nothing infers them.
    """

    relaxation: float = 1.0
    error_tolerance: float = 1.0e-5
    measurement_error_floor: float = 5.0e-4
    max_iterations: int = 100
    #: ``ERRMIN``: the relative-error floor the fit must reach. EFIT's default
    #: is 1e-2, two orders looser than the ``ERROR`` VEST writes.
    error_minimum: float | None = None
    #: ``SAICON``: the chi-square the fit is allowed to stop at. EFIT's
    #: default is 80, which VEST's flat-top passes on the way down.
    chi_squared_target: float | None = None
    #: ``ICONVR``: which criterion terminates the outer loop. VEST runs 2
    #: (chi-square) by not writing the key at all.
    convergence_mode: int | None = None
    #: ``NXITER``: the inner equilibrium iteration count per outer step.
    inner_iterations: int | None = None

    def __post_init__(self) -> None:
        _require_finite("relaxation", self.relaxation, positive=True)
        _require_finite("error_tolerance", self.error_tolerance, positive=True)
        _require_finite(
            "measurement_error_floor", self.measurement_error_floor, positive=True
        )
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, Integral)
            or self.max_iterations <= 0
        ):
            raise ValueError("max_iterations must be a positive integer")
        object.__setattr__(self, "max_iterations", int(self.max_iterations))
        for name in ("error_minimum", "chi_squared_target"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value, positive=True)
                object.__setattr__(self, name, float(value))
        for name in ("convergence_mode", "inner_iterations"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer or None")
            object.__setattr__(self, name, int(value))

    def termination_keys(self) -> dict[str, Any]:
        """The ``&IN1`` keys this configuration adds, in namelist spelling.

        Empty when every optional setting is ``None``, which is the routine
        case and the reason the k-file is unchanged by this addition.
        """
        keys: dict[str, Any] = {}
        if self.error_minimum is not None:
            keys["ERRMIN"] = self.error_minimum
        if self.chi_squared_target is not None:
            keys["SAICON"] = self.chi_squared_target
        if self.convergence_mode is not None:
            keys["ICONVR"] = self.convergence_mode
        if self.inner_iterations is not None:
            keys["NXITER"] = self.inner_iterations
        return keys


#: A tolerance so large the criterion it guards can never fire.  EFIT tests
#: ``value >= tolerance``, so this disables the check without disabling the
#: computation: the quantity is still evaluated and still written to the
#: a-file, and only the automatic rejection stops.
IGNORE_CRITERION = 1.0e6


@dataclass(frozen=True)
class EFITAcceptanceEnvelope:
    """The ``&incheck`` bounds EFIT accepts a reconstruction within.

    EFIT reads these from ``mhdin.dat`` in the table directory
    (``read_namelist.F90``) and ``chkerr`` tests every solution against them,
    raising the numbered failures that decide ``jflag``.  Its built-in
    defaults are DIII-D's -- ``aminor_min = 30 cm``, ``rcntr_min = 90 cm`` --
    and a machine that does not reach them has its equilibria rejected for
    being small rather than for being wrong.

    Lengths are **centimetres**, which is EFIT's convention here and not the
    metres used everywhere else in this package.  The geometric bounds are
    meant to be derived from the machine
    (:func:`vaft.machine_mapping.efund_geometry.vest_acceptance_envelope`),
    never fitted to a shot: a bound tuned until one discharge passes tests
    nothing.
    """

    #: Minor radius [cm]: at most half the limiter's radial extent, at least
    #: a few grid cells so an unresolved boundary is not accepted.
    aminor_min: float = 25.0
    aminor_max: float = 75.0
    #: Boundary centre [cm], inside the limiter by at least ``aminor_min``.
    rcntr_min: float = 30.0
    rcntr_max: float = 160.0
    zcntr_min: float = -165.0
    zcntr_max: float = 165.0
    #: Current centroid [cm], same reasoning as the boundary centre.
    rcurrt_min: float = 30.0
    rcurrt_max: float = 160.0
    zcurrt_min: float = -165.0
    zcurrt_max: float = 165.0
    #: Physical admissibility, not geometry: left at EFIT's values unless
    #: there is a physics argument to move them.
    elong_min: float = 0.8
    elong_max: float = 4.0
    li_min: float = 0.05
    li_max: float = 1.5
    betap_max: float = 6.0
    betat_max: float = 25.0
    qstar_min: float = 1.0
    qstar_max: float = 200.0
    qout_min: float = 1.0
    qout_max: float = 200.0
    sepin_check: float = -90.0
    gapin_min: float = -0.2
    gapout_min: float = -0.2
    gaptop_min: float = -0.2
    #: Measured-vs-reconstructed plasma current, left at EFIT's value.
    plasma_diff: float = 0.08
    #: The virial (Lao) consistency tolerances, failures #20 and #21.  At
    #: VEST's aspect ratio these are ill-conditioned and cannot decide the
    #: question they are asked (issue #649): ``sbpp`` is a difference of two
    #: nearly equal terms, negative on some slices, and ``sbli`` divides by
    #: ``alpha - 1``.  :data:`IGNORE_CRITERION` disables the rejection while
    #: leaving every quantity computed and written to the a-file.
    dbpli_diff: float = 0.05
    delbp_diff: float = 0.08

    def __post_init__(self) -> None:
        for name in fields(self):
            _require_finite(name.name, getattr(self, name.name))
            object.__setattr__(self, name.name, float(getattr(self, name.name)))
        for low, high in (
            ("aminor_min", "aminor_max"),
            ("rcntr_min", "rcntr_max"),
            ("zcntr_min", "zcntr_max"),
            ("rcurrt_min", "rcurrt_max"),
            ("zcurrt_min", "zcurrt_max"),
            ("elong_min", "elong_max"),
            ("li_min", "li_max"),
            ("qstar_min", "qstar_max"),
            ("qout_min", "qout_max"),
        ):
            if getattr(self, low) >= getattr(self, high):
                raise ValueError(f"{low} must be below {high}")
        if self.aminor_min <= 0:
            raise ValueError("aminor_min must be positive: a plasma has a size")

    def to_namelist(self) -> dict[str, float]:
        """The ``&incheck`` keys, in the spelling EFIT reads."""
        return {name.name: float(getattr(self, name.name)) for name in fields(self)}

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": 1, **self.to_namelist()}

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EFITConstraintConfig:
    """Diagnostic, diamagnetic-flux, coil, and passive-structure controls.

    ``group_weights`` overrides non-zero ODS channel weights for the named
    group.  A zero channel weight remains zero so a rejected channel cannot be
    accidentally re-enabled by a scan.
    """

    group_weights: Mapping[str, float] = field(default_factory=dict)
    uncertainty_mode: str = "legacy_weight"
    legacy_vbit: float = 10.0
    legacy_weight_scale: float = 10_000.0
    use_diamagnetic_flux: bool = True
    #: How the measured diamagnetic flux is written to DFLUX.  ``"imas"`` (the
    #: default since issue #385) writes the stored, signed value, which is what
    #: EFIT expects: it fits DFLUX against ``cdflux = integral (B_t - B_tv) dA``
    #: signed with B_t, so a diamagnetic plasma in a positive field is negative.
    #: ``"absolute"`` and ``"negative"`` force a sign and exist for controlled
    #: comparisons only; ``"absolute"`` was the historical default, inherited
    #: from a donor fitter that compared magnitudes.
    diamagnetic_flux_sign: str = "imas"
    diamagnetic_flux_input_units: str = "Wb"
    wall_current_mode: str = "measured"
    passive_structure_mode: str = "fixed_currents"
    coil_constraint_matrix: tuple[tuple[float, ...], ...] = field(
        default_factory=_legacy_coil_constraint_matrix
    )
    coil_constraint_targets: tuple[float, ...] = (0.0,) * 12
    nccoil: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.use_diamagnetic_flux, bool):
            raise ValueError("use_diamagnetic_flux must be boolean")
        unknown = set(self.group_weights) - DIAGNOSTIC_GROUPS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"unknown EFIT diagnostic group(s): {names}")
        for name, value in self.group_weights.items():
            _require_finite(f"group_weights[{name!r}]", value)
            if value < 0:
                raise ValueError(f"group_weights[{name!r}] must be non-negative")
        if self.uncertainty_mode not in {"legacy_weight", "standard_deviation"}:
            raise ValueError(
                "uncertainty_mode must be 'legacy_weight' or 'standard_deviation'"
            )
        _require_finite("legacy_vbit", self.legacy_vbit, positive=True)
        _require_finite("legacy_weight_scale", self.legacy_weight_scale, positive=True)
        if self.diamagnetic_flux_sign not in {"absolute", "imas", "negative"}:
            raise ValueError(
                "diamagnetic_flux_sign must be 'absolute', 'imas', or 'negative'"
            )
        if self.diamagnetic_flux_input_units not in {"Wb", "mWb"}:
            raise ValueError("diamagnetic_flux_input_units must be 'Wb' or 'mWb'")
        if self.wall_current_mode not in {"measured", "disabled"}:
            raise ValueError("wall_current_mode must be 'measured' or 'disabled'")
        if self.passive_structure_mode not in {
            "fixed_currents",
            "fit_currents",
            "disabled",
        }:
            raise ValueError(
                "passive_structure_mode must be 'fixed_currents', "
                "'fit_currents', or 'disabled'"
            )
        rows = tuple(
            tuple(float(value) for value in row) for row in self.coil_constraint_matrix
        )
        if not rows or not rows[0]:
            raise ValueError("coil_constraint_matrix must not be empty")
        columns = len(rows[0])
        if any(len(row) != columns for row in rows):
            raise ValueError("coil_constraint_matrix must be rectangular")
        if any(not math.isfinite(value) for row in rows for value in row):
            raise ValueError("coil_constraint_matrix values must be finite")
        if len(self.coil_constraint_targets) != columns:
            raise ValueError(
                "coil_constraint_targets length must equal the matrix column count"
            )
        if any(
            not math.isfinite(float(value)) for value in self.coil_constraint_targets
        ):
            raise ValueError("coil_constraint_targets values must be finite")
        if (
            isinstance(self.nccoil, bool)
            or not isinstance(self.nccoil, Integral)
            or self.nccoil < 0
        ):
            raise ValueError("nccoil must be a non-negative integer")
        object.__setattr__(self, "nccoil", int(self.nccoil))
        object.__setattr__(
            self, "group_weights", MappingProxyType(dict(self.group_weights))
        )
        object.__setattr__(self, "coil_constraint_matrix", rows)
        object.__setattr__(
            self,
            "coil_constraint_targets",
            tuple(float(value) for value in self.coil_constraint_targets),
        )


@dataclass(frozen=True)
class EFITScientificConfig:
    """Complete scientific configuration used to generate routine k-files."""

    profile: EFITProfileConfig = field(default_factory=EFITProfileConfig)
    initialization: EFITInitializationConfig = field(
        default_factory=EFITInitializationConfig
    )
    numerics: EFITNumericsConfig = field(default_factory=EFITNumericsConfig)
    constraints: EFITConstraintConfig = field(default_factory=EFITConstraintConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a sorted, JSON-compatible resolved representation."""
        return _canonical(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EFITScientificConfig":
        """Rebuild a validated configuration from :meth:`to_dict` output."""
        constraints = dict(payload.get("constraints", {}))
        if "coil_constraint_matrix" in constraints:
            constraints["coil_constraint_matrix"] = tuple(
                tuple(row) for row in constraints["coil_constraint_matrix"]
            )
        if "coil_constraint_targets" in constraints:
            constraints["coil_constraint_targets"] = tuple(
                constraints["coil_constraint_targets"]
            )
        return cls(
            profile=EFITProfileConfig(**dict(payload.get("profile", {}))),
            initialization=EFITInitializationConfig(
                **dict(payload.get("initialization", {}))
            ),
            numerics=EFITNumericsConfig(**dict(payload.get("numerics", {}))),
            constraints=EFITConstraintConfig(**constraints),
        )

    @property
    def sha256(self) -> str:
        """Stable hash of the resolved scientific configuration."""
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {
            item.name: _canonical(getattr(value, item.name)) for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _replace_path(value: Any, parts: Sequence[str], replacement: Any) -> Any:
    if not parts:
        return replacement
    head, *tail = parts
    if is_dataclass(value):
        if not hasattr(value, head):
            raise ValueError(f"unknown EFIT configuration path component: {head}")
        return replace(
            value,
            **{head: _replace_path(getattr(value, head), tail, replacement)},
        )
    if isinstance(value, Mapping):
        updated = dict(value)
        if not tail:
            updated[head] = replacement
        else:
            updated[head] = _replace_path(updated.get(head, {}), tail, replacement)
        return updated
    raise ValueError(f"cannot descend into EFIT configuration path component: {head}")


def efit_parameter_grid(
    base: EFITScientificConfig | None,
    axes: Mapping[str, Sequence[Any]],
) -> tuple[EFITScientificConfig, ...]:
    """Expand dotted scientific-config paths into a deterministic scan grid.

    Example paths include ``profile.kppcur``, ``numerics.relaxation``, and
    ``constraints.group_weights.bpol_probe``.
    """
    resolved_base = base or EFITScientificConfig()
    names = sorted(axes)
    if any(not values for values in axes.values()):
        raise ValueError("parameter-grid axes must not be empty")
    configurations = []
    for values in product(*(axes[name] for name in names)):
        candidate: Any = resolved_base
        for name, replacement_value in zip(names, values):
            candidate = _replace_path(candidate, name.split("."), replacement_value)
        if not isinstance(candidate, EFITScientificConfig):
            raise TypeError(
                "parameter-grid replacement did not produce EFITScientificConfig"
            )
        configurations.append(candidate)
    return tuple(configurations)


__all__ = [
    "DIAGNOSTIC_GROUPS",
    "EFITAcceptanceEnvelope",
    "IGNORE_CRITERION",
    "EFITConstraintConfig",
    "EFITInitializationConfig",
    "EFITNumericsConfig",
    "EFITProfileConfig",
    "EFITScientificConfig",
    "efit_parameter_grid",
]
