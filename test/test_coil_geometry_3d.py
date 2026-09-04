"""Canonical VEST 3D coil geometry: parser, loader, and excitation model.

These tests run against the packaged ``vaft/data/gpec/vest_{UP,MID,LOW}.dat``
files, pinning the header semantics (``ncoil nsec npts nw``, with ``nw`` the
winding-turn multiplier) and the physical placement of each coil set.
"""

from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from vaft.data.resources import data_path
from vaft.machine_mapping.coil_geometry_3d import (
    VEST_3D_COIL_SETS,
    CoilExcitation,
    load_vest_3d_coil_config,
    parse_gpec_coil_dat,
)
from vaft.machine_mapping.utils import VestConfigurationError


EXPECTED_HEADERS = {
    "UP": (6, 1, 420, 20.0),
    "MID": (6, 1, 100, 20.0),
    "LOW": (6, 1, 420, 20.0),
}


@pytest.mark.parametrize("set_name", sorted(EXPECTED_HEADERS))
def test_parse_gpec_coil_dat_headers_and_shapes(set_name):
    spec = VEST_3D_COIL_SETS[set_name]
    ncoil, nsec, npts, nw, points = parse_gpec_coil_dat(data_path(spec.dat_file))
    assert (ncoil, nsec, npts, nw) == EXPECTED_HEADERS[set_name]
    assert points.shape == (ncoil, npts, 3)


def test_loader_returns_all_canonical_sets():
    config = load_vest_3d_coil_config()
    assert sorted(config.coil_sets) == ["LOW", "MID", "UP"]
    for coil_set in config.coil_sets.values():
        assert coil_set.turns == 20.0
        assert len(coil_set.filaments) == 6
        assert all(filament.is_closed for filament in coil_set.filaments)


def test_coil_set_placement():
    config = load_vest_3d_coil_config()

    up = np.concatenate([f.points_xyz for f in config["UP"].filaments])
    assert 0.61 <= up[:, 2].min() <= 0.63
    assert 1.10 <= up[:, 2].max() <= 1.13

    low = np.concatenate([f.points_xyz for f in config["LOW"].filaments])
    assert -1.13 <= low[:, 2].min() <= -1.10
    assert -0.63 <= low[:, 2].max() <= -0.61

    mid = np.concatenate([f.points_xyz for f in config["MID"].filaments])
    radius = np.hypot(mid[:, 0], mid[:, 1])
    assert 0.79 <= radius.min() and radius.max() <= 0.82
    assert -0.16 <= mid[:, 2].min() and mid[:, 2].max() <= 0.16


def test_mid_filament_is_single_circular_loop():
    """MID draws one geometric loop; turns live only in the nw header."""
    config = load_vest_3d_coil_config(coil_sets=["MID"])
    filament = config["MID"].filaments[0]
    segments = np.diff(filament.points_xyz, axis=0)
    length = float(np.linalg.norm(segments, axis=1).sum())
    assert length == pytest.approx(2.0 * math.pi * 0.15, rel=0.01)


def test_sector_angles_match_geometry():
    config = load_vest_3d_coil_config()
    for coil_set in config.coil_sets.values():
        for filament, declared in zip(
            coil_set.filaments, coil_set.sector_angles_deg
        ):
            delta = (filament.centroid_angle_deg - declared + 180.0) % 360.0 - 180.0
            assert abs(delta) <= 2.0


def test_loader_rejects_unknown_set():
    with pytest.raises(VestConfigurationError, match="Unknown VEST 3D coil set"):
        load_vest_3d_coil_config(coil_sets=["SIDE"])


def test_loader_rejects_truncated_file(tmp_path):
    root = tmp_path / "data"
    (root / "gpec").mkdir(parents=True)
    for spec in VEST_3D_COIL_SETS.values():
        shutil.copy2(data_path(spec.dat_file), root / spec.dat_file)
    truncated = (root / "gpec/vest_MID.dat").read_text(encoding="utf-8").splitlines()[:-10]
    (root / "gpec/vest_MID.dat").write_text("\n".join(truncated) + "\n", encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="coordinate rows"):
        load_vest_3d_coil_config(root)


def test_loader_rejects_missing_file(tmp_path):
    with pytest.raises(VestConfigurationError, match="not found"):
        load_vest_3d_coil_config(tmp_path)


def test_excitation_from_mode_n1_pattern():
    excitation = CoilExcitation.from_mode("MID", 200.0, 1)
    assert excitation.coil_set == "MID"
    assert excitation.currents_a == pytest.approx(
        (200.0, 100.0, -100.0, -200.0, -100.0, 100.0)
    )


def test_excitation_from_mode_phase_shift():
    shifted = CoilExcitation.from_mode("UP", 100.0, 2, phase_deg=90.0)
    expected = tuple(
        100.0 * math.cos(math.radians(2 * angle + 90.0))
        for angle in (0.0, 60.0, 120.0, 180.0, 240.0, 300.0)
    )
    assert shifted.currents_a == pytest.approx(expected)
