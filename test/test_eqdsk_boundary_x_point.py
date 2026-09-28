"""The native EQDSK conversion writes ``boundary.x_point`` (#238).

``OMFITgeqdsk.to_omas`` used to supply the leaf and the native conversion left
it empty. It now carries the boundary X-points only -- the saddles that bound
the plasma -- because a reconstructed map has many other saddles in the vacuum.
The analytic Solov'ev family is the reference: its X-points are prescribed, so
the round trip through a g-file has a known answer.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vaft.data.eqdsk import from_equilibrium, read_geqdsk, to_omas
from vaft.process.equilibrium import calculate_q_profile_from_psi, solovev_example

PACKAGED_48224 = Path(__file__).resolve().parents[1] / "vaft" / "data" / "kineticEfit" / "g048224.00300"


def _geqdsk(topology: str):
    eq = solovev_example(topology)
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    q = calculate_q_profile_from_psi(
        eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis, eq.psi_boundary,
        np.clip(psi_n, 0.01, 0.99), axis_rz=eq.magnetic_axis, cocos=eq.convention.cocos,
    )
    return eq, from_equilibrium(dataclasses.replace(eq, q=np.asarray(q, dtype=float)))


def _written(ods) -> list[tuple[float, float]]:
    slice_ = ods["equilibrium.time_slice.0"]
    if "boundary.x_point" not in slice_:
        return []
    return [(float(slice_[f"boundary.x_point.{i}.r"]), float(slice_[f"boundary.x_point.{i}.z"]))
            for i in range(len(slice_["boundary.x_point"]))]


@pytest.mark.parametrize("topology", ["double_null", "single_null"])
def test_diverted_x_points_survive_the_gfile_round_trip(topology):
    eq, geqdsk = _geqdsk(topology)
    written = sorted(_written(to_omas(geqdsk)), key=lambda point: point[1])
    requested = sorted(eq.metadata["x_points_requested"], key=lambda point: point[1])
    assert len(written) == len(requested)
    cell = float(eq.r[1] - eq.r[0])
    for (r_w, z_w), (r_q, z_q) in zip(written, requested):
        assert np.hypot(r_w - r_q, z_w - z_q) < 0.05 * cell


def test_limited_plasma_has_no_boundary_x_point():
    _, geqdsk = _geqdsk("limited")
    assert _written(to_omas(geqdsk)) == []


def test_vacuum_saddles_of_a_reconstruction_are_not_boundary_x_points():
    """48224 at 300 ms is limited, and its map has seventeen saddles in the vacuum."""
    assert _written(to_omas(read_geqdsk(PACKAGED_48224))) == []


def test_x_point_search_is_derived_data():
    _, geqdsk = _geqdsk("double_null")
    assert _written(to_omas(geqdsk, allow_derived_data=False)) == []


def test_a_failed_search_warns_and_keeps_the_conversion(monkeypatch):
    import vaft.process.equilibrium as equilibrium

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(equilibrium, "derive_boundary_representation", fail)
    _, geqdsk = _geqdsk("double_null")
    with pytest.warns(UserWarning, match="boundary.x_point not written"):
        ods = to_omas(geqdsk)
    assert _written(ods) == []
    assert "equilibrium.time_slice.0.profiles_2d.0.psi" in ods
