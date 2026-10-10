"""The common result of a non-axisymmetric coil operating-space study (#1178).

One container, :class:`CoilOperatingSpace`, carries what a GPEC resonant-field
study (#1884), a PENTRC NTV-torque study (#1887) and their plots (#1886) share:
the operating points of ``K`` coil groups and the metrics evaluated at each.
It is a table, not an ODS -- ``mhd_linear`` has no operating-point, phasor or
status axis -- and it imports no solver.

    samples : one row per operating point   sample_id, role, amplitude_<g>, phase_<g>
    values  : one row per (sample, metric)   sample_id, metric, value, unit, status [, m, psi_n_rational, psi_n]
    to_frame() ─▶ the two joined, plus amplitude_ratio_<g> and phase_rel_<g> against group 0

The schema was settled on #1886 with the plotting owner; field names change
only through that thread.

Notation
--------
K         : number of independently driven coil groups                   [-]
n         : toroidal mode number of the excitation                       [-]
A_g       : cosine amplitude of group g's sector currents           [A per turn]
delta_g   : phase of group g                                            [rad]
c_g       : excitation phasor, A_g exp(+i delta_g)                  [A per turn]
phi_k     : toroidal angle of sector k in the coil geometry's frame     [rad]

Conventions
-----------
**Excitation.**  Group ``g`` drives its sectors with
``I_k = A_g cos(n phi_k + delta_g)``
(:meth:`vaft.machine_mapping.coils_non_axisymmetric_geometry.CoilExcitation.from_mode`),
so ``I_k = Re(c_g exp(+i n phi_k))`` with ``c_g = A_g exp(+i delta_g)``.
Written as ``I(phi) = C_n exp(+i n phi) + C_{-n} exp(-i n phi)``, the
coefficient of ``exp(+i n phi)`` is ``C_n = c_g / 2``; in GPEC's and DCON's
``exp(i(m theta - n phi))`` basis the coefficient of ``exp(-i n phi)`` is
``conj(c_g) / 2`` -- the same excitation, the conjugate number.  ``A_g`` is
the cosine amplitude, not in general the largest sector current.  The
pattern's peak sits at ``n phi = -delta_g``: ``delta_g = 0`` puts it at
``phi = 0`` and, for ``n > 0``, increasing ``delta_g`` rotates it toward
``-phi`` of the frame the result names in ``convention["frame"]``.  With ``N``
equally spaced sectors the pattern is resolved only when ``2 n mod N != 0``
(six sectors alias ``n = 3``: the sine part vanishes and ``delta`` cannot be
recovered); the container cannot see the sector count and does not check it.

**Status per (sample, metric).**  A GPEC metric can be valid where PENTRC
failed at the same point, so status lives on the value row.  Only a
``valid`` row carries a number; every other status carries NaN, so a failed
or uncomputed point is never drawn as zero.

**Relative coordinates.**  Against group 0: ``amplitude_ratio_<g> = A_g / A_0``
and ``phase_rel_<g> = (delta_g - delta_0) mod 2 pi``; both NaN where an
amplitude involved is zero (a phase there is undefined; the metric is not).

Provenance
----------
.. [1178] Issue #1178, the coil amplitude/phase operating-space umbrella.
.. [1886] Issue #1886, the plot-side contract this schema answers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "COIL_EXCITATION_CONVENTION",
    "OPERATING_SPACE_ROLES",
    "OPERATING_SPACE_STATUSES",
    "CoilOperatingSpace",
    "relative_coil_coordinates",
]

#: Status of one (sample, metric) value.
OPERATING_SPACE_STATUSES = ("valid", "infeasible", "failed", "not_computed", "undefined")

#: Why a sample is in the result.  ``probe``, ``held_out`` and ``invariance``
#: are #1887's identification runs; ``held_out`` points are direct runs.
OPERATING_SPACE_ROLES = (
    "scan", "reference", "analytic_optimum", "numerical_optimum", "gpec_confirmation",
    "probe", "held_out", "invariance",
)

#: The excitation convention every result states (the ``frame`` is per result).
COIL_EXCITATION_CONVENTION = {
    "sector_current": "I_k = A cos(n phi_k + delta)  [A per turn]",
    "phasor": "c = A exp(+i delta)",
    "fourier_basis": "I(phi) = C_n exp(+i n phi) + C_-n exp(-i n phi)",
    "fourier_coefficient": "C_n = c / 2 (coefficient of exp(+i n phi); of exp(-i n phi), as in GPEC, conj(c) / 2)",
    "amplitude": "cosine amplitude A of the sector currents, not the largest sector current",
    "phase_zero": "delta = 0: pattern peak at phi = 0",
    "phase_direction": "pattern peak at n phi = -delta: for n > 0, increasing delta rotates it toward -phi",
}

_SAMPLE_COLUMNS = ("sample_id", "role")
_VALUE_COLUMNS = ("sample_id", "metric", "value", "unit", "status")
#: Optional value columns that tell two values of one metric at one sample apart.
_VALUE_DISTINGUISHERS = ("m", "psi_n_rational", "psi_n")


@dataclass(frozen=True)
class CoilOperatingSpace:
    """Operating points of ``K`` coil groups and the metrics evaluated at each (#1178).

    The tables are validated and copied at construction; treat them as read-only.
    """

    #: Toroidal mode number [-].
    n: int
    #: Coil-group names, in phasor order; group 0 is the relative-coordinate reference.
    groups: tuple[str, ...]
    #: One row per operating point: ``sample_id``, ``role``, ``amplitude_<g>`` [A per turn],
    #: ``phase_<g>`` [rad].
    samples: Any
    #: One row per (sample, metric): ``sample_id``, ``metric``, ``value``, ``unit``,
    #: ``status``; optionally ``m``, ``psi_n_rational``, ``psi_n``.
    values: Any
    #: :data:`COIL_EXCITATION_CONVENTION` plus ``frame``, a sentence naming the toroidal frame.
    convention: Mapping[str, str]
    #: ``{"kind": "regular", "axes": {...}}`` or ``{"kind": "irregular"}``.
    grid: Mapping[str, Any]
    #: Per metric (or metric family): what produced it -- field kind, windows,
    #: PENTRC method / grid / enclosed radius / kinetic identity.
    provenance: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: Optional radial profiles per sample id (e.g. PENTRC's accumulated torque).
    profiles: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        import pandas as pd

        if isinstance(self.n, bool) or int(self.n) != self.n or int(self.n) < 1:
            raise ValueError(f"n must be a positive integer, got {self.n!r}")
        object.__setattr__(self, "n", int(self.n))
        groups = tuple(str(g) for g in self.groups)
        object.__setattr__(self, "groups", groups)
        if not groups or len(set(groups)) != len(groups):
            raise ValueError("groups must be non-empty and unique")
        # Copies, so neither the caller's frames nor a later edit of these can
        # break what was validated (the dataclass is frozen; its tables too).
        samples, values = self.samples.copy(), self.values.copy()
        missing = [c for c in (*_SAMPLE_COLUMNS, *self._coordinate_columns()) if c not in samples.columns]
        if missing:
            raise ValueError(f"samples lack {missing}")
        missing = [c for c in _VALUE_COLUMNS if c not in values.columns]
        if missing:
            raise ValueError(f"values lack {missing}")
        shared = (set(samples.columns) & set(values.columns)) - {"sample_id"}
        if shared:
            raise ValueError(f"samples and values both carry {sorted(shared)}; to_frame would rename them")
        if samples.empty or values.empty:
            raise ValueError("a result needs at least one sample and one value")
        if samples["sample_id"].isna().any() or samples["sample_id"].duplicated().any():
            raise ValueError("sample_id must be present and unique")
        unknown = {str(r) for r in samples["role"]} - set(OPERATING_SPACE_ROLES)
        if unknown:
            raise ValueError(f"unknown roles {sorted(unknown)}; use {OPERATING_SPACE_ROLES}")
        coordinates = [f"{kind}_{g}" for g in groups for kind in ("amplitude", "phase")]
        for column in coordinates:
            if not pd.api.types.is_numeric_dtype(samples[column]) or pd.api.types.is_bool_dtype(samples[column]):
                raise ValueError(f"samples column {column!r} must be numeric")
            samples[column] = samples[column].astype(float)
        amplitudes = samples[[f"amplitude_{g}" for g in groups]].to_numpy()
        phases = samples[[f"phase_{g}" for g in groups]].to_numpy()
        if not (np.all(np.isfinite(amplitudes)) and np.all(amplitudes >= 0)):
            raise ValueError("amplitudes must be finite and not negative")
        if not np.all(np.isfinite(phases)):
            raise ValueError("phases must be finite")
        orphans = {str(v) for v in set(values["sample_id"]) - set(samples["sample_id"])}
        if orphans:
            raise ValueError(f"values name unknown samples {sorted(orphans)[:5]}")
        if values["metric"].isna().any() or (values["metric"].astype(str).str.strip() == "").any():
            raise ValueError("every value names its metric")
        unknown = {str(v) for v in values["status"]} - set(OPERATING_SPACE_STATUSES)
        if unknown:
            raise ValueError(f"unknown statuses {sorted(unknown)}; use {OPERATING_SPACE_STATUSES}")
        if not pd.api.types.is_numeric_dtype(values["value"]) or pd.api.types.is_bool_dtype(values["value"]):
            raise ValueError("the value column must be numeric (NaN for a non-valid row)")
        values["value"] = values["value"].astype(float)
        number = values["value"].to_numpy()
        valid = (values["status"] == "valid").to_numpy()
        if not np.all(np.isfinite(number[valid])):
            raise ValueError("a valid value must be finite")
        if np.any(np.isfinite(number[~valid])):
            raise ValueError("only a valid row carries a number; the others carry NaN, never zero")
        key = ["sample_id", "metric", *[c for c in _VALUE_DISTINGUISHERS if c in values.columns]]
        if values.duplicated(subset=key).any():
            raise ValueError(f"one value per {tuple(key)}")
        if values["unit"].isna().any() or (values["unit"].astype(str).str.strip() == "").any():
            raise ValueError("every value names its unit ('-' for dimensionless)")
        missing = [key for key in (*COIL_EXCITATION_CONVENTION, "frame") if key not in self.convention]
        if missing:
            raise ValueError(f"convention lacks {missing}")
        kind = self.grid.get("kind")
        if kind not in ("regular", "irregular"):
            raise ValueError("grid kind must be 'regular' or 'irregular'")
        if kind == "regular" and not self.grid.get("axes"):
            raise ValueError("a regular grid names its axes")
        object.__setattr__(self, "samples", samples)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "convention", dict(self.convention))
        object.__setattr__(self, "grid", dict(self.grid))
        object.__setattr__(self, "provenance", {k: dict(v) for k, v in self.provenance.items()})
        object.__setattr__(self, "profiles", dict(self.profiles))

    def _coordinate_columns(self) -> list[str]:
        return [f"{kind}_{g}" for g in self.groups for kind in ("amplitude", "phase")]

    def phasors(self) -> np.ndarray:
        """``(N, K)`` excitation phasors ``A_g exp(+i delta_g)`` in sample order [A per turn]."""
        a = self.samples[[f"amplitude_{g}" for g in self.groups]].to_numpy(dtype=float)
        delta = self.samples[[f"phase_{g}" for g in self.groups]].to_numpy(dtype=float)
        return a * np.exp(1j * delta)

    def metrics(self) -> tuple[str, ...]:
        """The metric names present, in first-appearance order [-]."""
        return tuple(dict.fromkeys(self.values["metric"]))

    def to_frame(self):
        """Long form: one row per (sample, metric), with the coordinates and relative coordinates.

        Returns
        -------
        pandas.DataFrame
            ``values`` joined with ``samples`` on ``sample_id``, plus
            ``amplitude_ratio_<g>`` and ``phase_rel_<g>`` for every group after
            the first (see :func:`relative_coil_coordinates`) [-].
        """
        coordinates = self.samples.copy()
        relative = relative_coil_coordinates(
            coordinates[[f"amplitude_{g}" for g in self.groups]].to_numpy(dtype=float),
            coordinates[[f"phase_{g}" for g in self.groups]].to_numpy(dtype=float),
        )
        for index, g in enumerate(self.groups[1:], start=1):
            coordinates[f"amplitude_ratio_{g}"] = relative["amplitude_ratio"][:, index]
            coordinates[f"phase_rel_{g}"] = relative["phase_rel"][:, index]
        return self.values.merge(coordinates, on="sample_id", how="left", validate="many_to_one")


def relative_coil_coordinates(amplitudes: Sequence, phases: Sequence) -> dict[str, np.ndarray]:
    """Amplitude ratios and relative phases of coil groups against group 0.

    Parameters
    ----------
    amplitudes : array_like
        ``(N, K)`` cosine amplitudes, not negative [A per turn].
    phases : array_like
        ``(N, K)`` phases [rad].

    Returns
    -------
    coordinates : dict
        ``amplitude_ratio`` (``A_g / A_0``) and ``phase_rel``
        (``(delta_g - delta_0) mod 2 pi``, in rad), each ``(N, K)``; column 0
        is 1 and 0 where defined [-, rad].

    Raises
    ------
    ValueError
        Shapes differ, or an amplitude is negative or not finite.

    Processing steps
    ----------------
    1. Ratio against group 0; NaN where ``A_0 = 0``.
    2. Phase difference wrapped to ``[0, 2 pi)``; NaN where ``A_0`` or ``A_g``
       is zero, since a phase of a zero excitation is undefined.

    Convention
    ----------
    Phases follow ``I_k = A cos(n phi_k + delta)``, phasor ``A exp(+i delta)``;
    a relative phase is that of group ``g`` minus group 0.

    Applicability
    -------------
    Machine-independent.

    Provenance
    ----------
    .. [1886] Issue #1886, ``amplitude_ratio`` and ``phase_31`` of the plot contract.
    """
    a = np.atleast_2d(np.asarray(amplitudes, dtype=float))
    delta = np.atleast_2d(np.asarray(phases, dtype=float))
    if a.shape != delta.shape:
        raise ValueError(f"amplitudes {a.shape} and phases {delta.shape} differ")
    if not np.all(np.isfinite(a)) or np.any(a < 0):
        raise ValueError("amplitudes must be finite and not negative")
    reference = a[:, :1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(reference > 0, a / np.where(reference > 0, reference, 1.0), np.nan)
    defined = (reference > 0) & (a > 0)
    phase_rel = np.where(defined, np.mod(delta - delta[:, :1], 2 * np.pi), np.nan)
    return {"amplitude_ratio": ratio, "phase_rel": phase_rel}
