"""Canonical static source profiles for TokaMaker's piecewise-linear input.

Public normalized coordinates run from axis (0) to boundary (1). OFT reverses
these internally. Native derivatives have opposite sign to COCOS 11 and use
flux per radian. The solver adjusts source amplitudes using global targets;
the tables preserve shape, while Ip and relative axis pressure set the scales.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from vaft.data.equilibrium import EquilibriumData
from vaft.process._equilibrium_parametric import convert_cocos


@dataclass(frozen=True)
class TokaMakerProfiles:
    """Native per-radian derivatives sampled axis to edge; pressure in Pa."""

    psi_n: np.ndarray
    pprime: np.ndarray
    ffprime: np.ndarray
    axis_pressure_Pa: float
    edge_pressure_Pa: float = 0.0
    ip_A: float | None = None

    def solver_tables(self) -> dict[str, dict[str, Any]]:
        """Return OFT linterp definitions; its normalization sets amplitudes."""
        return {"pp_prof": {"type": "linterp", "x": self.psi_n, "y": self.pprime},
                "ffp_prof": {"type": "linterp", "x": self.psi_n, "y": self.ffprime}}


def _validate(x, pp, ffp, pax):
    if (x.ndim != 1 or len(x) < 2 or pp.shape != x.shape or ffp.shape != x.shape
            or not np.isfinite(np.r_[x, pp, ffp, pax]).all()
            or np.any(np.diff(x) <= 0) or not np.isclose(x[0], 0, atol=1e-10, rtol=0)
            or not np.isclose(x[-1], 1, atol=1e-10, rtol=0)):
        raise ValueError("profiles need finite distinct axis-to-edge psi_n spanning [0,1]")
    if not np.any(pp) and not np.any(ffp):
        raise ValueError("both Grad-Shafranov sources are zero")
    if np.any(pp):
        integral = float(np.sum((pp[1:] + pp[:-1]) * np.diff(x) / 2))
        scale = float(np.sum((np.abs(pp[1:]) + np.abs(pp[:-1])) * np.diff(x) / 2))
        if abs(integral) <= 1e-12 * scale:
            raise ValueError("pressure source needs nonzero integrated shape for OFT normalization")
    if np.any(pp) and pax <= 0:
        raise ValueError("pressure source needs positive relative axis pressure for OFT normalization")
    if not np.any(pp) and pax != 0:
        raise ValueError("zero pressure source requires zero relative axis pressure")


def equilibrium_to_tokamaker_profiles(equilibrium: EquilibriumData) -> TokaMakerProfiles:
    """Convert canonical static source profiles to native OFT table shapes.

    The equilibrium must declare COCOS, a nonzero axis-to-boundary flux span,
    and complete pprime/ffprime tables. It is converted to COCOS 11; both
    derivatives are multiplied by -2*pi to obtain native per-radian units.
    Public table coordinates remain axis=0 and edge=1 because OFT performs
    the internal reversal. Ip must be finite and positive after conversion,
    matching the installed OFT target API. Bt sign does not change FFprime.

    Relative axis pressure is the integral of pprime from edge to axis; a
    stored pressure profile supplies only its edge offset. This keeps the
    derivative table authoritative. OFT uses zero edge pressure internally.
    Surface-current sheets and flow cannot be represented by these tables.
    """
    if equilibrium.convention.contradicted:
        raise ValueError("equilibrium COCOS declaration contradicts observed signs")
    model = equilibrium.metadata.get("model") if equilibrium.metadata.get("source_type") == "guazzotto_freidberg" else None
    if model is not None and (model.pressure_pedestal or model.bootstrap_fraction or model.mach_number):
        raise ValueError("surface-current or flow terms are unsupported by static OFT profiles")
    eq = convert_cocos(equilibrium, 11)
    if eq.ip is None or not np.isfinite(eq.ip) or eq.ip <= 0:
        raise ValueError("OFT profile normalization requires finite positive canonical Ip")
    if any(getattr(eq, name) is None for name in ("psi_1d", "pprime", "ffprime", "psi_axis", "psi_boundary")):
        raise ValueError("flux coordinates and pprime/ffprime profiles are required")
    span = float(eq.psi_boundary - eq.psi_axis)
    if not np.isfinite(span) or span == 0:
        raise ValueError("nonzero axis-to-boundary flux span is required")
    x = (np.asarray(eq.psi_1d, dtype=float) - eq.psi_axis) / span
    if x.ndim != 1 or np.shape(eq.pprime) != x.shape or np.shape(eq.ffprime) != x.shape:
        raise ValueError("source profile lengths must match psi_1d")
    if eq.pressure is not None and (np.shape(eq.pressure) != x.shape or not np.isfinite(eq.pressure).all()):
        raise ValueError("stored pressure must be finite and match psi_1d")
    order = np.argsort(x)
    x = x[order]
    pp = np.asarray(eq.pprime, dtype=float)[order]
    ffp = np.asarray(eq.ffprime, dtype=float)[order]
    # Trapezoidal integration of the declared piecewise-linear derivative.
    pax = -float(np.sum((pp[1:] + pp[:-1]) * np.diff(x) / 2) * span)
    edge_pressure = 0.0 if eq.pressure is None else float(np.asarray(eq.pressure)[order[-1]])
    native_pp, native_ffp = -2 * np.pi * pp, -2 * np.pi * ffp
    _validate(x, native_pp, native_ffp, pax)
    x = x.copy()
    x[0], x[-1] = 0.0, 1.0
    return TokaMakerProfiles(x, native_pp, native_ffp, pax, edge_pressure, float(eq.ip))


def profiles_for_config(config, equilibrium=None) -> TokaMakerProfiles | None:
    """Resolve power-law, canonical-equilibrium, or explicit native tables."""
    if config.profile_mode == "power_law":
        return None
    if config.profile_mode == "equilibrium":
        eq = equilibrium if equilibrium is not None else config.profile_equilibrium
        if eq is None:
            raise ValueError("profile_mode='equilibrium' requires an EquilibriumData")
        return equilibrium_to_tokamaker_profiles(eq)
    if config.profile_mode != "explicit":
        raise ValueError("profile_mode must be 'power_law', 'equilibrium', or 'explicit'")
    table: Mapping = config.profile_tables or {}
    if not {"psi_n", "pprime", "ffprime", "axis_pressure_Pa"} <= set(table):
        raise ValueError("explicit profiles require psi_n/pprime/ffprime and axis_pressure_Pa")
    x, pp, ffp = (np.asarray(table[name], dtype=float) for name in ("psi_n", "pprime", "ffprime"))
    pax = float(table["axis_pressure_Pa"])
    _validate(x, pp, ffp, pax)
    # Explicit tables use native derivatives per radian, with declared
    # axis-to-edge coordinates; no silent units or convention inference.
    x = x.copy()
    x[0], x[-1] = 0.0, 1.0
    return TokaMakerProfiles(x, pp.copy(), ffp.copy(), pax)


def profile_targets(profiles: TokaMakerProfiles | None, targets: Mapping[str, float]) -> dict[str, float]:
    """Set source amplitudes without a singular pure-pressure FF column."""
    result = dict(targets)
    if profiles is None:
        return result
    if profiles.ip_A is not None:
        result.setdefault("Ip", profiles.ip_A)
    if "R0" in result or "Ip_ratio" in result:
        raise ValueError("tabulated pressure normalization conflicts with R0/Ip_ratio targets")
    if np.any(profiles.pprime):
        result.setdefault("pax", profiles.axis_pressure_Pa)
    else:
        if result.get("pax", 0) != 0:
            raise ValueError("nonzero axis pressure target conflicts with zero pressure source")
        result.pop("pax", None)
    if not np.any(profiles.ffprime):
        # With FF'=0, Ip+pax gives two equations for one source amplitude.
        # Pressure fixes the unique source; Ip is an independently checked output.
        result.pop("Ip", None)
    return result


def source_profile_diagnostics(solver, profiles: TokaMakerProfiles) -> dict[str, Any]:
    """Record requested shape and actual native derivatives after the solve."""
    x, f, fp, pressure, pp = solver.get_profiles(profiles.psi_n)
    return {"psi_n": x, "requested_native_pprime": profiles.pprime,
            "requested_native_ffprime": profiles.ffprime,
            "realized_native_pprime": pp, "realized_native_ffprime": f * fp,
            "realized_pressure_Pa": pressure,
            "target_relative_axis_pressure_Pa": profiles.axis_pressure_Pa,
            "target_edge_pressure_Pa": profiles.edge_pressure_Pa,
            "target_Ip_A": profiles.ip_A}


__all__ = ["TokaMakerProfiles", "equilibrium_to_tokamaker_profiles"]
