"""dRsep through GEQDSK -> ODS conversion, the IMAS boundary leaves and native EFIT (#2325 phase D)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from test_separatrix_balance import _equilibrium, _exact  # noqa: E402

from vaft.data.equilibrium import SeparatrixBalanceSeries  # noqa: E402
from vaft.process.equilibrium import (  # noqa: E402
    derive_boundary_representation,
    native_efit_d_r_sep,
    separatrix_balance_series,
)


def _with_profiles(eq):
    """Give an analytic flux map the 1-D profiles a GEQDSK export needs.

    Only the flux map and boundary matter here; the profiles are simple and
    mutually consistent, not a force-balance solution.
    """
    from dataclasses import replace

    n = eq.r.size
    return replace(
        eq, psi_1d=np.linspace(eq.psi_axis, eq.psi_boundary, n), pressure=np.linspace(1.0e3, 0.0, n),
        f=np.full(n, 1.0), q=np.linspace(1.5, 4.0, n), ip=1.0e5, bt0=1.0, r0=1.0,
    )


def _ods_from(*equilibria):
    from vaft.data.eqdsk import from_equilibrium

    ods = None
    for index, eq in enumerate(equilibria):
        geqdsk = from_equilibrium(_with_profiles(eq))
        ods = geqdsk.to_omas(ods, time_index=index) if ods is not None else geqdsk.to_omas(time_index=index)
    return ods


@pytest.fixture(scope="module")
def lsn_ods():
    eq, psi_l, psi_u = _equilibrium(0.08)
    return _ods_from(eq), eq, psi_l, psi_u


def test_geqdsk_conversion_writes_the_dd_boundary_leaves(lsn_ods):
    ods, eq, psi_l, psi_u = lsn_ods
    ts = ods["equilibrium.time_slice.0"]
    assert int(ts["boundary.type"]) == 1
    assert int(ts["boundary_separatrix.type"]) == 1
    # boundary.x_point keeps its active-only meaning: the lower X-point.
    assert len(ts["boundary.x_point"]) == 1
    assert float(ts["boundary.x_point.0.z"]) < 0
    assert float(ts["boundary_secondary_separatrix.x_point.0.z"]) > 0
    # The secondary flux is in the ODS's own (weber) convention: compare with
    # the 2-D map at the X-point, which carries the same scale.
    r = np.asarray(ts["profiles_2d.0.grid.dim1"]); z = np.asarray(ts["profiles_2d.0.grid.dim2"])
    from scipy.interpolate import RectBivariateSpline

    spline = RectBivariateSpline(r, z, np.asarray(ts["profiles_2d.0.psi"]))
    expected = float(spline.ev(float(ts["boundary_secondary_separatrix.x_point.0.r"]),
                               float(ts["boundary_secondary_separatrix.x_point.0.z"])))
    assert float(ts["boundary_secondary_separatrix.psi"]) == pytest.approx(expected, rel=1e-6)
    # distance_inner_outer is unsigned and positive, at its own reference plane.
    distance = float(ts["boundary_secondary_separatrix.distance_inner_outer"])
    assert distance > 0
    assert distance == pytest.approx(abs(_exact(psi_l, psi_u)), rel=0.2)


def test_the_series_recovers_the_signed_value_from_the_ods(lsn_ods):
    ods, eq, psi_l, psi_u = lsn_ods
    before = set(ods.flat().keys())
    series = separatrix_balance_series(ods)
    assert set(ods.flat().keys()) == before, "the series must not modify its input"
    assert series.state == ("valid",)
    assert series.d_r_sep[0] == pytest.approx(_exact(psi_l, psi_u), abs=2e-4)
    assert series.d_r_sep[0] < 0
    assert np.isnan(series.native_d_r_sep[0])


def test_the_mirror_round_trip_flips_only_the_sign():
    lsn, psi_l, psi_u = _equilibrium(0.08)
    usn, _, _ = _equilibrium(0.08, flip=True)
    series = separatrix_balance_series(_ods_from(lsn, usn))
    assert series.state == ("valid", "valid")
    assert series.d_r_sep[1] == pytest.approx(-series.d_r_sep[0], abs=1e-6)


def test_states_distinguish_limited_unavailable_and_no_plasma():
    from vaft.data.equilibrium import Contour

    lsn, _, _ = _equilibrium(0.08)
    ods = _ods_from(lsn, lsn)
    # Slice 1 becomes a vacuum slice: no plasma is not limited.
    ods["equilibrium.time_slice.1.global_quantities.ip"] = 0.0
    series = separatrix_balance_series(ods)
    assert series.state == ("valid", "no_plasma")
    assert np.isnan(series.d_r_sep[1])
    assert series.topology[1] is None

    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    eq, _, _ = _equilibrium(0.01, 201, inset=3e-3)
    wall = Contour(float(np.max(eq.lcfs.r)) + 0.3 * (np.cos(theta) - 1.0), 0.75 * np.sin(theta))
    limited, _, _ = _equilibrium(0.01, 201, inset=3e-3, wall=wall)
    rep = derive_boundary_representation(limited)
    series = separatrix_balance_series({"equilibrium": {"time_slice": [{"time": 0.1}]}}, representations=[rep])
    assert series.state == ("limited",)
    assert np.isnan(series.d_r_sep[0])
    assert np.isfinite(series.geometric_d_r_sep[0])


def test_reprocessing_into_a_limited_state_leaves_no_stale_secondary(lsn_ods):
    from vaft.data.equilibrium import Contour
    from vaft.omas.update import write_boundary_topology

    ods, eq, _, _ = lsn_ods
    theta = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    inset, _, _ = _equilibrium(0.01, 201, inset=3e-3)
    wall = Contour(float(np.max(inset.lcfs.r)) + 0.3 * (np.cos(theta) - 1.0), 0.75 * np.sin(theta))
    limited, _, _ = _equilibrium(0.01, 201, inset=3e-3, wall=wall)
    copy = ods.copy()
    target = copy["equilibrium.time_slice.0"]
    write_boundary_topology(target, derive_boundary_representation(limited))
    assert int(target["boundary.type"]) == 0
    for leaf in ("boundary.x_point", "boundary_secondary_separatrix.psi",
                 "boundary_secondary_separatrix.x_point", "boundary_secondary_separatrix.distance_inner_outer"):
        assert leaf not in target, leaf
    # And writing the diverted state twice is idempotent.
    rep = derive_boundary_representation(eq)
    write_boundary_topology(target, rep, psi_scale=2 * np.pi)
    first = {key: value for key, value in target.flat().items()}
    write_boundary_topology(target, rep, psi_scale=2 * np.pi)
    assert set(target.flat().keys()) == set(first)


def test_update_equilibrium_boundary_topology_on_packaged_reconstructions():
    import vaft
    from vaft.data.eqdsk import read_geqdsk
    from vaft.omas.update import update_equilibrium_boundary_topology

    root = Path(vaft.__file__).parent / "data" / "efit"
    ods = read_geqdsk(root / "g039915.00319").to_omas(allow_derived_data=False)
    assert "boundary.type" not in ods["equilibrium.time_slice.0"]
    representations = update_equilibrium_boundary_topology(ods)
    ts = ods["equilibrium.time_slice.0"]
    assert int(ts["boundary.type"]) == 0  # the packaged VEST slice is limited
    assert "boundary_secondary_separatrix.psi" not in ts
    series = separatrix_balance_series(ods, time_indices=[0], representations=[representations[0]])
    assert series.state == ("limited",)


@pytest.mark.parametrize("drsep, limloc, expected", [
    (-1.2, "SNB", -0.012),
    (0.8, "SNT", 0.008),
    (0.1, "DN ", 0.001),
    (40.0, "IN ", np.nan),
    (40.0, "SNB", np.nan),
    (2.0, "MAR", np.nan),
    (1.0, "OUT", np.nan),
])
def test_native_efit_drsep_is_aligned_only_where_it_is_signed(drsep, limloc, expected):
    value, reason = native_efit_d_r_sep(drsep, limloc)
    if np.isnan(expected):
        assert np.isnan(value) and reason
    else:
        assert value == pytest.approx(expected) and reason is None


def test_native_values_come_from_the_aeqdsk_parameters():
    import vaft
    from vaft.data.aeqdsk import read_aeqdsk
    from vaft.data.eqdsk import read_geqdsk

    root = Path(vaft.__file__).parent / "data" / "efit"
    ods = read_geqdsk(root / "g039915.00319").to_omas()
    read_aeqdsk(root / "a039915.00319").to_omas(ods, time_index=0)
    series = separatrix_balance_series(ods)
    assert series.native_configuration == ("IN",)
    assert np.isnan(series.native_d_r_sep[0])  # EFIT's 40 cm default


def test_the_series_record_validates_its_shape():
    with pytest.raises(ValueError):
        SeparatrixBalanceSeries(np.zeros(2), np.zeros(1), ("valid", "valid"), (None, None), (None, None),
                                np.zeros(2), np.zeros(2), np.zeros(2), (None, None))
    with pytest.raises(ValueError):
        SeparatrixBalanceSeries(np.zeros(1), np.zeros(1), ("bogus",), (None,), (None,),
                                np.zeros(1), np.zeros(1), np.zeros(1), (None,))
