"""Versioned effective-resistance calibration of the VEST passive wall (#308).

``pf_passive.loop[:].resistance`` in the shipped static geometry is not a
material property. Nine of the eleven wall regions carry the nominal hoop
resistance ``rho * 2*pi*R / A`` exactly; the outboard mid-chamber wall (``W1``)
and the inboard tungsten limiter (``W11``) carry that nominal value multiplied
by a **fitted band factor** -- twelve factors over twelve 20-loop z-bands for
``W1``, nineteen over eighteen 12-loop bands plus one 14-loop band for ``W11``.
Those factors come from the legacy ``Run_Effective_wall_Resistance_Fitting.m``
procedure, which writes ``Input_Geometry/Wall_coeff_<key>.mat`` holding
``Wall_Factor_{Inboard,Outboard,Side}``; the shipped asset reproduces vintage
``2303`` bitwise (all 950 loops, see :data:`LEGACY_CALIBRATIONS`).

This module separates the two: :func:`nominal_resistance` is the geometry,
:class:`WallResistanceCalibration` is the fit, and
:func:`calibrated_resistance` puts them back together. A calibration is a
first-class, versioned object so a benchmark or a fit records *which* wall it
was evaluated against rather than a frozen inheritance nobody can name.

Band order follows the donor's loop order: outboard bands run from the lowest
z upward, inboard bands from the highest z downward. The donor applies the
side-wall factors nowhere (that block is commented out), so ``side`` is
carried for provenance only and never applied.

The per-loop ``resistivity`` field is a separate, inconsistent vector -- it is
not the effective resistivity ``resistance * A / (2*pi*R)`` and matches no
legacy vintage. Only ``resistance`` feeds the eddy solve; ``resistivity`` is
read by the TokaMaker vessel export alone. That discrepancy is reported, not
repaired, here.
"""
from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

__all__ = [
    "LEGACY_CALIBRATIONS",
    "NOMINAL_RESISTIVITY",
    "WallResistanceCalibration",
    "band_factors",
    "band_layout",
    "calibrated_resistance",
    "identify_calibration",
    "load_legacy_wall_coeff",
    "nominal_resistance",
]

#: Material resistivities [Ohm m] the nominal hoop resistance is built from.
#: Stainless steel for every region but the inboard tungsten limiter.
NOMINAL_RESISTIVITY: Mapping[str, float] = {"stainless": 7.8e-7, "tungsten": 5.6e-8}

#: Region name -> material.  Everything not listed is stainless.
_REGION_MATERIAL: Mapping[str, str] = {"W11": "tungsten"}

#: (region, loops per band, band count); the last inboard band takes the
#: remaining 14 loops, exactly as the donor solver indexes them.
_OUTBOARD = ("W1", 20, 12)
_INBOARD = ("W11", 12, 19)


@dataclass(frozen=True)
class WallResistanceCalibration:
    """One vintage of fitted band factors.

    ``outboard`` has twelve entries, ``inboard`` nineteen. ``side`` is kept
    when the vintage carries it but is never applied (the donor does not).
    """

    key: str
    outboard: tuple[float, ...]
    inboard: tuple[float, ...]
    side: tuple[float, ...] | None = None
    source: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if len(self.outboard) != _OUTBOARD[2]:
            raise ValueError(f"outboard needs {_OUTBOARD[2]} factors, got {len(self.outboard)}")
        if len(self.inboard) != _INBOARD[2]:
            raise ValueError(f"inboard needs {_INBOARD[2]} factors, got {len(self.inboard)}")
        for name, values in (("outboard", self.outboard), ("inboard", self.inboard)):
            arr = np.asarray(values, dtype=float)
            if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
                raise ValueError(f"{name} factors must be finite and positive")

    def digest(self) -> str:
        """Stable fingerprint of the applied factors (side excluded)."""
        payload = np.concatenate(
            [np.asarray(self.outboard, float), np.asarray(self.inboard, float)]
        ).tobytes()
        return hashlib.sha1(payload).hexdigest()[:12]

    def replace(self, **changes: Any) -> "WallResistanceCalibration":
        return dataclasses.replace(self, **changes)


#: The vintage the shipped static geometry was built from.  Values are the
#: donor's ``Wall_coeff_2303.mat`` at full precision; applying them to the
#: nominal hoop resistance reproduces every shipped ``resistance`` bitwise.
#: Note the donor's own solver hard-codes vintage 38510 at run time; the
#: asset VAFT inherited predates that.
LEGACY_CALIBRATIONS: Mapping[str, WallResistanceCalibration] = {
    "2303": WallResistanceCalibration(
        key="2303",
        outboard=(
            1.9318654933282589, 1.0483214691687186, 0.8842783594497277,
            1.936962484731398, 2.194988295023534, 1.1262867668109,
            1.901442098879354, 1.4506659991244721, 1.4750544023799699,
            1.219001624648076, 0.8805563363790923, 0.8364739903194159,
        ),
        inboard=(
            1.616823116719205, 0.851810791694679, 0.40013657076422154,
            0.8236877331438898, 4.55803309971832, 5.309876643881609,
            9.139545108684423, 8.869419156063513, 6.944719297879274,
            8.144809080248708, 8.246496612186334, 8.937749755436615,
            9.211509269275012, 5.1109739462224795, 4.519444420358444,
            2.583651673088478, 0.6853019000872758, 0.6078158034215352,
            2.2361746936694495,
        ),
        source="VFIT_VEST-Equilibrium-Code/Input_Geometry/Wall_coeff_2303.mat",
        note="reproduces the shipped VEST_static_geometry pf_passive resistances bitwise",
    ),
}


def _loops(ods: Any) -> int:
    return int(len(ods["pf_passive.loop"]))


def band_layout(ods: Any) -> dict[str, list[np.ndarray]]:
    """Loop indices of every fitted band, in donor order.

    Derived from the loop names rather than hard-coded ranges, and checked
    against the donor's counts so a regenerated asset with a different loop
    order fails loudly instead of receiving factors on the wrong loops.
    """
    n = _loops(ods)
    for i in range(n):
        if f"pf_passive.loop.{i}.name" not in ods:
            raise ValueError(f"pf_passive.loop.{i} has no name; cannot band an unnamed wall")
    names = [str(ods[f"pf_passive.loop.{i}.name"]) for i in range(n)]
    layout: dict[str, list[np.ndarray]] = {}
    for region, per_band, count in (_OUTBOARD, _INBOARD):
        idx = np.flatnonzero(np.array(names) == region)
        if idx.size == 0:
            raise ValueError(f"no pf_passive loop is named {region!r}")
        if not np.array_equal(idx, np.arange(idx[0], idx[0] + idx.size)):
            raise ValueError(f"{region} loops are not contiguous; cannot band them")
        expected = per_band * count if region == _OUTBOARD[0] else per_band * (count - 1) + 14
        if idx.size != expected:
            raise ValueError(
                f"{region} has {idx.size} loops; the calibration expects {expected}"
            )
        bands = [idx[k * per_band:(k + 1) * per_band] for k in range(count - 1)]
        bands.append(idx[(count - 1) * per_band:])
        layout[region] = bands
    return layout


def nominal_resistance(ods: Any) -> np.ndarray:
    """``rho * 2*pi*R / A`` per loop, with ``R`` the mean outline radius.

    This is the value nine of eleven regions already carry; for the two fitted
    regions it is what the band factors multiply. The operation order is the
    one that reproduces the shipped asset bitwise.
    """
    n = _loops(ods)
    out = np.empty(n)
    for i in range(n):
        loop = ods[f"pf_passive.loop.{i}"]
        # Membership first: reading a missing OMAS path materializes it.
        for leaf in ("name", "element.0.area", "element.0.geometry.outline.r"):
            if leaf not in loop:
                raise ValueError(
                    f"pf_passive.loop.{i} lacks {leaf}; the nominal hoop resistance needs it"
                )
        material = _REGION_MATERIAL.get(str(loop["name"]), "stainless")
        rho = NOMINAL_RESISTIVITY[material]
        r_mean = float(np.mean(np.asarray(loop["element.0.geometry.outline.r"], dtype=float)))
        area = float(loop["element.0.area"])
        out[i] = rho * (2.0 * np.pi * r_mean) / area
    return out


def _factors_from_layout(
    layout: Mapping[str, list[np.ndarray]], n: int, calibration: WallResistanceCalibration
) -> np.ndarray:
    factors = np.ones(n)
    for band, value in zip(layout[_OUTBOARD[0]], calibration.outboard):
        factors[band] = value
    for band, value in zip(layout[_INBOARD[0]], calibration.inboard):
        factors[band] = value
    return factors


def band_factors(ods: Any, calibration: WallResistanceCalibration) -> np.ndarray:
    """Per-loop multiplier: the band's factor on fitted loops, 1 elsewhere."""
    return _factors_from_layout(band_layout(ods), _loops(ods), calibration)


def calibrated_resistance(ods: Any, calibration: WallResistanceCalibration) -> np.ndarray:
    """Nominal hoop resistance times the calibration's band factors."""
    return nominal_resistance(ods) * band_factors(ods, calibration)


def identify_calibration(ods: Any, *, rtol: float = 1e-12) -> dict[str, Any]:
    """Which known vintage the ODS's resistances were built from, if any.

    Returns the matching key (or ``None``), the worst relative deviation from
    that vintage, and a digest of the *measured* band factors so an unknown
    vintage is still fingerprinted reproducibly.
    """
    n = _loops(ods)
    layout = band_layout(ods)  # cheapest check first: is this even a banded VEST wall?
    for i in range(n):
        if f"pf_passive.loop.{i}.resistance" not in ods:
            raise ValueError(f"pf_passive.loop.{i} has no resistance")
    shipped = np.array([float(ods[f"pf_passive.loop.{i}.resistance"]) for i in range(n)])
    nominal = nominal_resistance(ods)
    ratio = shipped / nominal
    measured = np.concatenate(
        [[float(np.mean(ratio[b])) for b in layout[_OUTBOARD[0]]],
         [float(np.mean(ratio[b])) for b in layout[_INBOARD[0]]]]
    )
    digest = hashlib.sha1(np.round(measured, 12).tobytes()).hexdigest()[:12]
    fitted = np.concatenate(layout[_OUTBOARD[0]] + layout[_INBOARD[0]])
    free = np.setdiff1d(np.arange(n), fitted)
    best_key, best_err = None, float("inf")
    for key, cal in LEGACY_CALIBRATIONS.items():
        rebuilt = nominal * _factors_from_layout(layout, n, cal)
        err = float(np.max(np.abs(rebuilt / shipped - 1.0)))
        if err < best_err:
            best_key, best_err = key, err
    matched = best_key if best_err <= rtol else None
    return {
        "key": matched,
        "max_relative_deviation": best_err if matched else None,
        "nearest_key": best_key,
        "measured_factor_digest": digest,
        "unfitted_loops_nominal": bool(np.allclose(ratio[free], 1.0, rtol=1e-12, atol=0.0)),
    }


def load_legacy_wall_coeff(path: str | Path, *, key: str | None = None) -> WallResistanceCalibration:
    """Read a donor ``Wall_coeff_<key>.mat`` into a calibration."""
    from scipy.io import loadmat

    source = Path(path)
    data = loadmat(str(source))
    if "Wall_Factor_Inboard" not in data or "Wall_Factor_Outboard" not in data:
        raise ValueError(f"{source} does not carry Wall_Factor_Inboard/Outboard")
    side = data.get("Wall_Factor_Side")
    if key is None:
        stem = source.stem
        key = stem.split("_")[-1] if "_" in stem else stem
    return WallResistanceCalibration(
        key=str(key),
        outboard=tuple(float(v) for v in np.asarray(data["Wall_Factor_Outboard"]).ravel()),
        inboard=tuple(float(v) for v in np.asarray(data["Wall_Factor_Inboard"]).ravel()),
        side=None if side is None else tuple(float(v) for v in np.asarray(side).ravel()),
        source=str(source),
    )
