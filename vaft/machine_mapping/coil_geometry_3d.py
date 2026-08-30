"""Canonical VEST non-axisymmetric 3D coil geometry.

The primary geometry source is the three packaged GPEC-format coil files
``vaft/data/gpec/vest_{UP,MID,LOW}.dat``.  Each file header is four fields
``ncoil nsec npts nw``: number of coils (toroidal sectors) in the set, number
of sections, points per coil, and the winding-turn multiplier.  The body is
``ncoil * npts`` rows of Cartesian ``x y z`` coordinates in metres, one
*single* closed geometric filament per coil; the magnetic effect of the coil
is ``nw`` times the per-turn current (all three VEST sets carry ``nw = 20``).
This turn interpretation and the geometry were reviewed with 3D coil
developer Gwang-geun Seo.

This module is deliberately free of omas/matplotlib/``vaft.code`` imports so
both the machine-mapping layer and the GPEC input adapter can consume it
without creating dependency cycles.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from vaft.data.resources import data_path

from .utils import VestConfigurationError

__all__ = [
    "VEST_3D_COIL_SETS",
    "CoilSetSpec",
    "CoilFilament",
    "CoilSet3D",
    "Vest3DCoilConfig",
    "CoilExcitation",
    "parse_gpec_coil_dat",
    "load_vest_3d_coil_config",
]

_SECTOR_ANGLES_DEG = (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
# UP/LOW saddle filaments span 0-30 deg per sector, so their centroids sit at
# +15 deg relative to the sector origin; MID loops are centred on the origin.
_SADDLE_SECTOR_ANGLES_DEG = (15.0, 75.0, 135.0, 195.0, 255.0, 315.0)


@dataclass(frozen=True)
class CoilSetSpec:
    """Static metadata for one canonical VEST 3D coil set."""

    identifier: str
    dat_file: str
    description: str
    z_range: tuple[float, float]
    sector_angles_deg: tuple[float, ...] = _SECTOR_ANGLES_DEG
    provenance: str = (
        "GPEC-format coil description; geometry and 20-turn interpretation "
        "reviewed with 3D coil developer Gwang-geun Seo."
    )


VEST_3D_COIL_SETS: Mapping[str, CoilSetSpec] = {
    "UP": CoilSetSpec(
        identifier="VEST_3D_UP",
        dat_file="gpec/vest_UP.dat",
        description="Upper saddle coil array, 6 sectors at 60 deg spacing",
        z_range=(0.60, 1.14),
        sector_angles_deg=_SADDLE_SECTOR_ANGLES_DEG,
    ),
    "MID": CoilSetSpec(
        identifier="VEST_3D_MID",
        dat_file="gpec/vest_MID.dat",
        description=(
            "Mid-plane 12-inch circular coil array (20 turns), 6 sectors at "
            "60 deg spacing; shot-48226 reference geometry"
        ),
        z_range=(-0.17, 0.17),
    ),
    "LOW": CoilSetSpec(
        identifier="VEST_3D_LOW",
        dat_file="gpec/vest_LOW.dat",
        description="Lower saddle coil array, 6 sectors at 60 deg spacing",
        z_range=(-1.14, -0.60),
        sector_angles_deg=_SADDLE_SECTOR_ANGLES_DEG,
    ),
}


@dataclass(frozen=True)
class CoilFilament:
    """One physical winding path (one toroidal sector)."""

    points_xyz: np.ndarray  # (npts, 3) metres

    def __post_init__(self) -> None:
        self.points_xyz.setflags(write=False)

    @property
    def centroid_angle_deg(self) -> float:
        centroid = self.points_xyz.mean(axis=0)
        return math.degrees(math.atan2(centroid[1], centroid[0])) % 360.0

    @property
    def is_closed(self) -> bool:
        return bool(np.allclose(self.points_xyz[0], self.points_xyz[-1]))


@dataclass(frozen=True)
class CoilSet3D:
    """A canonical coil set: parsed geometry plus its static metadata."""

    name: str
    identifier: str
    turns: float
    filaments: tuple[CoilFilament, ...]
    sector_angles_deg: tuple[float, ...]
    dat_path: Path
    description: str
    provenance: str


@dataclass(frozen=True)
class Vest3DCoilConfig:
    """The full canonical VEST 3D coil configuration."""

    coil_sets: Mapping[str, CoilSet3D]

    def __getitem__(self, name: str) -> CoilSet3D:
        return self.coil_sets[name]


def parse_gpec_coil_dat(path: str | Path) -> tuple[int, int, int, float, np.ndarray]:
    """Parse a GPEC coil ``.dat`` file.

    Returns ``(ncoil, nsec, npts, nw, points)`` with ``points`` shaped
    ``(ncoil, npts, 3)`` in Cartesian metres.
    """
    path = Path(path)
    raw = path.read_text().split()
    if len(raw) < 4:
        raise VestConfigurationError(f"GPEC coil file too short: {path}")
    try:
        ncoil, nsec, npts = (int(float(token)) for token in raw[:3])
        nw = float(raw[3])
    except ValueError as exc:
        raise VestConfigurationError(
            f"Unparseable GPEC coil header {raw[:4]!r} in {path}"
        ) from exc
    values = np.asarray(raw[4:], dtype=float)
    if values.size != ncoil * npts * 3:
        raise VestConfigurationError(
            f"GPEC coil file {path} declares {ncoil} coils x {npts} points "
            f"but contains {values.size // 3} coordinate rows"
        )
    points = values.reshape(ncoil, npts, 3)
    return ncoil, nsec, npts, nw, points


def _validate_coil_set(name: str, spec: CoilSetSpec, coil_set: CoilSet3D) -> None:
    n_sectors = len(spec.sector_angles_deg)
    if len(coil_set.filaments) != n_sectors:
        raise VestConfigurationError(
            f"Coil set {name}: expected {n_sectors} sectors, "
            f"parsed {len(coil_set.filaments)} from {coil_set.dat_path}"
        )
    z_low, z_high = spec.z_range
    for filament, expected_angle in zip(coil_set.filaments, spec.sector_angles_deg):
        if not filament.is_closed:
            raise VestConfigurationError(
                f"Coil set {name}: filament near {expected_angle:.0f} deg is "
                f"not a closed loop in {coil_set.dat_path}"
            )
        angle = filament.centroid_angle_deg
        delta = (angle - expected_angle + 180.0) % 360.0 - 180.0
        if abs(delta) > 2.0:
            raise VestConfigurationError(
                f"Coil set {name}: filament centroid at {angle:.2f} deg does "
                f"not match declared sector angle {expected_angle:.0f} deg"
            )
        z_values = filament.points_xyz[:, 2]
        if z_values.min() < z_low - 1e-6 or z_values.max() > z_high + 1e-6:
            raise VestConfigurationError(
                f"Coil set {name}: filament z-range "
                f"[{z_values.min():.3f}, {z_values.max():.3f}] m outside the "
                f"expected [{z_low}, {z_high}] m"
            )


def load_vest_3d_coil_config(
    data_root: str | Path | None = None,
    *,
    coil_sets: Sequence[str] | None = None,
) -> Vest3DCoilConfig:
    """Load and validate the canonical VEST 3D coil configuration.

    ``data_root`` overrides the packaged ``vaft/data`` directory (tests);
    ``coil_sets`` selects a subset by canonical name (default: all three).
    """
    selected = tuple(coil_sets) if coil_sets is not None else tuple(VEST_3D_COIL_SETS)
    unknown = [name for name in selected if name not in VEST_3D_COIL_SETS]
    if unknown:
        raise VestConfigurationError(
            f"Unknown VEST 3D coil set(s) {unknown}; "
            f"available: {sorted(VEST_3D_COIL_SETS)}"
        )
    root = Path(data_root) if data_root is not None else data_path()
    loaded: dict[str, CoilSet3D] = {}
    for name in selected:
        spec = VEST_3D_COIL_SETS[name]
        dat_path = root / spec.dat_file
        if not dat_path.exists():
            raise VestConfigurationError(
                f"Coil geometry file for set {name} not found: {dat_path}"
            )
        ncoil, _nsec, _npts, nw, points = parse_gpec_coil_dat(dat_path)
        coil_set = CoilSet3D(
            name=name,
            identifier=spec.identifier,
            turns=nw,
            filaments=tuple(CoilFilament(points[index]) for index in range(ncoil)),
            sector_angles_deg=spec.sector_angles_deg,
            dat_path=dat_path,
            description=spec.description,
            provenance=spec.provenance,
        )
        _validate_coil_set(name, spec, coil_set)
        loaded[name] = coil_set
    return Vest3DCoilConfig(coil_sets=loaded)


@dataclass(frozen=True)
class CoilExcitation:
    """Run-specific excitation of one coil set: amperes per turn per sector."""

    coil_set: str
    currents_a: tuple[float, ...]

    @classmethod
    def from_mode(
        cls,
        coil_set: str,
        amplitude_a: float,
        n_tor: int,
        phase_deg: float = 0.0,
        sector_angles_deg: Sequence[float] = _SECTOR_ANGLES_DEG,
    ) -> "CoilExcitation":
        """Cosine sector pattern ``I_k = A cos(n * phi_k + phase)``."""
        currents = tuple(
            amplitude_a
            * math.cos(math.radians(n_tor * angle + phase_deg))
            for angle in sector_angles_deg
        )
        return cls(coil_set=coil_set, currents_a=currents)
