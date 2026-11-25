"""Issue #2270 P1: the plot layer's coordinate resolution and core-profile mapping
now live in ``vaft.process``.

The views draw exactly what they drew before (frozen copies of the pre-move
algorithms are compared bit for bit), the canonical functions are usable
without a plot, and the one deliberate change -- the kinetic overview no
longer labels a stored ``sqrt(psi_N)`` proxy as ``rho_tor_norm`` -- is pinned.
"""

from __future__ import annotations

import copy

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

import vaft.omas as vo
from vaft.data._derived import is_rho_pol_proxy, rho_tor_profile
from vaft.plot.backend import kinetic_overview
from vaft.process.equilibrium import psi_to_rz
from vaft.process.profile import (
    CORE_GRID_COORDINATES,
    CoordinateUnavailableError,
    core_profiles_grid_coordinate,
    equilibrium_rho_tor_norm_table,
    map_rho_tor_profile_to_rz,
)

PSI_N = np.linspace(0.0, 1.0, 41)


# --- frozen pre-move algorithms (develop 7edf8e026) -------------------------


def _frozen_rho_tor_table(ods, time_slice):
    base = f"equilibrium.time_slice.{time_slice}.profiles_1d"
    psi = np.asarray(ods[f"{base}.psi"], dtype=float)
    psi_norm = (psi - psi[0]) / (psi[-1] - psi[0])
    stored = np.asarray(ods[f"{base}.rho_tor_norm"], dtype=float) if f"{base}.rho_tor_norm" in ods else None
    if (
        stored is not None and stored.size == psi.size and np.all(np.isfinite(stored))
        and abs(stored[0]) < 1e-6 and abs(stored[-1] - 1.0) < 1e-6
        and np.all(np.diff(stored) >= 0.0)
        and not is_rho_pol_proxy(stored, psi_norm)
    ):
        return psi_norm, stored
    return psi_norm, rho_tor_profile(np.asarray(ods[f"{base}.q"], dtype=float), psi).rho_tor_norm


def _frozen_core_field(ods, time_slice, profile_slice, quantity):
    ts = f"equilibrium.time_slice.{time_slice}"
    r = np.asarray(ods[f"{ts}.profiles_2d.0.grid.dim1"], dtype=float)
    z = np.asarray(ods[f"{ts}.profiles_2d.0.grid.dim2"], dtype=float)
    psi_2d = np.asarray(ods[f"{ts}.profiles_2d.0.psi"], dtype=float)
    axis = float(ods[f"{ts}.global_quantities.psi_axis"])
    edge = float(ods[f"{ts}.global_quantities.psi_boundary"])
    psi_norm = (psi_2d - axis) / (edge - axis)
    if psi_norm.shape == (r.size, z.size):
        psi_norm = psi_norm.T
    table_psi, table_rho = _frozen_rho_tor_table(ods, time_slice)
    cp = f"core_profiles.profiles_1d.{profile_slice}"
    rho = np.asarray(ods[f"{cp}.grid.rho_tor_norm"], dtype=float)
    profile = np.asarray(ods[f"{cp}.electrons.{quantity}"], dtype=float)
    rho_2d = np.interp(np.clip(psi_norm, 0.0, 1.0).ravel(), table_psi, table_rho)
    values = np.interp(rho_2d, rho, profile).reshape(psi_norm.shape)
    return np.where(psi_norm <= 1.0, values, np.nan)


def _frozen_mapped_field(psi_2d, psi_1d, profile, axis, edge):
    psi_norm_2d = (psi_2d - axis) / (edge - axis)
    psi_norm_1d = (psi_1d - axis) / (edge - axis)
    order = np.argsort(psi_norm_1d)
    mapped = np.interp(psi_norm_2d, psi_norm_1d[order], profile[order])
    return np.where(psi_norm_2d <= 1.0, mapped, np.nan)


# --- fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def sample():
    """The packaged sample, whose equilibrium stores the sqrt(psi_N) proxy, plus one profile."""
    ods = vo.sample_ods()
    times = [float(ods[f"equilibrium.time_slice.{k}.time"]) for k in range(len(ods["equilibrium.time_slice"]))]
    slice_index = len(times) // 2
    rho = np.linspace(0.0, 1.0, 21)
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.asarray([times[slice_index]])
    ods["core_profiles.profiles_1d.0.time"] = times[slice_index]
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = rho
    ods["core_profiles.profiles_1d.0.electrons.temperature"] = 150.0 * (1.0 - rho**2) ** 1.5 + 2.0
    ods["core_profiles.profiles_1d.0.electrons.density"] = 1.0e19 * (1.0 - rho**2) + 1.0e17
    return ods, slice_index


def _slice(profiles_1d: dict) -> ODS:
    ods = ODS(consistency_check=False)
    base = "equilibrium.time_slice.0.profiles_1d"
    ods[f"{base}.psi"] = 0.1 * PSI_N
    for leaf, value in profiles_1d.items():
        ods[f"{base}.{leaf}"] = value
    return ods


# --- the views are unchanged ------------------------------------------------


@pytest.mark.parametrize("quantity", ["temperature", "density"])
def test_the_core_profile_field_is_bit_identical_to_the_pre_move_view(sample, quantity):
    ods, slice_index = sample
    extract = getattr(vo, f"extract_electron_{quantity}_field")
    drawn = np.asarray(extract(ods, time_slice=slice_index).values)
    expected = _frozen_core_field(ods, slice_index, 0, quantity)
    assert np.isfinite(drawn).any()
    np.testing.assert_array_equal(drawn, expected)


def test_psi_to_rz_with_edge_fill_is_the_old_flux_function_projection(sample):
    ods, slice_index = sample
    ts = ods[f"equilibrium.time_slice.{slice_index}"]
    psi_2d = np.asarray(ts["profiles_2d.0.psi"], dtype=float)
    psi_1d = np.asarray(ts["profiles_1d.psi"], dtype=float)
    pressure = np.asarray(ts["profiles_1d.pressure"], dtype=float)
    axis = float(ts["global_quantities.psi_axis"])
    edge = float(ts["global_quantities.psi_boundary"])
    mapped, psi_norm = psi_to_rz((psi_1d - axis) / (edge - axis), pressure, psi_2d, axis, edge, fill_outside="edge")
    np.testing.assert_array_equal(
        np.where(psi_norm <= 1.0, mapped, np.nan),
        _frozen_mapped_field(psi_2d, psi_1d, pressure, axis, edge),
    )


def test_the_views_do_not_write_the_callers_ods(sample):
    ods, slice_index = sample
    before = copy.deepcopy(ods)
    vo.extract_electron_temperature_field(ods, time_slice=slice_index)
    equilibrium_rho_tor_norm_table(ods, slice_index)
    core_profiles_grid_coordinate(ods, 0, "rho_tor_norm")
    assert sorted(ods.flat().keys()) == sorted(before.flat().keys())


# --- equilibrium_rho_tor_norm_table ----------------------------------------


def test_an_authentic_stored_coordinate_is_returned_as_stored():
    stored = PSI_N**0.7
    psi_norm, rho = equilibrium_rho_tor_norm_table(_slice({"rho_tor_norm": stored}), 0)
    np.testing.assert_allclose(psi_norm, PSI_N, atol=1e-15)
    np.testing.assert_array_equal(rho, stored)


def test_the_proxy_is_replaced_by_the_q_integral():
    q = 1.0 + 3.0 * PSI_N
    _, rho = equilibrium_rho_tor_norm_table(_slice({"rho_tor_norm": np.sqrt(PSI_N), "q": q}), 0)
    expected = np.sqrt((PSI_N + 1.5 * PSI_N**2) / 2.5)
    np.testing.assert_allclose(rho, expected, atol=2e-3)


def test_a_flux_of_either_sign_normalises_from_axis_to_boundary():
    ods = _slice({"rho_tor_norm": PSI_N**0.7})
    ods["equilibrium.time_slice.0.profiles_1d.psi"] = -0.1 * PSI_N - 0.3
    psi_norm, _ = equilibrium_rho_tor_norm_table(ods, 0)
    np.testing.assert_allclose(psi_norm, PSI_N, atol=1e-15)


@pytest.mark.parametrize(
    "leaves, match",
    [
        ({"rho_tor_norm": np.sqrt(PSI_N)}, "supplies no toroidal coordinate"),
        ({"rho_tor_norm": np.zeros_like(PSI_N)}, "supplies no toroidal coordinate"),
    ],
)
def test_no_toroidal_coordinate_is_refused(leaves, match):
    with pytest.raises(CoordinateUnavailableError, match=match):
        equilibrium_rho_tor_norm_table(_slice(leaves), 0)


def test_a_nonmonotone_flux_is_refused():
    ods = _slice({"rho_tor_norm": PSI_N})
    psi = 0.1 * PSI_N
    psi[10] = psi[12]
    ods["equilibrium.time_slice.0.profiles_1d.psi"] = psi
    with pytest.raises(CoordinateUnavailableError, match="not monotonic"):
        equilibrium_rho_tor_norm_table(ods, 0)


def test_a_missing_slice_is_refused_without_creating_it():
    ods = _slice({"rho_tor_norm": PSI_N})
    with pytest.raises(CoordinateUnavailableError, match="psi is required"):
        equilibrium_rho_tor_norm_table(ods, 3)
    assert len(ods["equilibrium.time_slice"]) == 1


# --- map_rho_tor_profile_to_rz ---------------------------------------------


def test_each_cell_takes_the_value_at_its_toroidal_radius():
    psi_rz = 0.1 * np.linspace(0.0, 1.2, 13)[:, None] * np.ones((1, 3))
    rho = np.linspace(0.0, 1.0, 11)
    table = (PSI_N, PSI_N)  # rho_tor_norm = psi_N
    values = map_rho_tor_profile_to_rz(psi_rz, 0.0, 0.1, rho, 100.0 * (1.0 - rho), flux_table=table)
    assert values.shape == psi_rz.shape  # orientation preserved
    np.testing.assert_allclose(values[:11, 0], 100.0 * (1.0 - np.linspace(0.0, 1.0, 11)), atol=1e-12)
    assert np.all(np.isnan(values[11:]))  # psi_N > 1 is blank


def test_the_toroidal_radius_is_not_sqrt_psi_n():
    q = 1.0 + 3.0 * PSI_N
    table = (PSI_N, rho_tor_profile(q, 0.1 * PSI_N).rho_tor_norm)
    rho = np.linspace(0.0, 1.0, 21)
    value = map_rho_tor_profile_to_rz(np.array([[0.025]]), 0.0, 0.1, rho, rho, flux_table=table)[0, 0]
    assert value == pytest.approx(np.sqrt((0.25 + 1.5 * 0.25**2) / 2.5), abs=2e-3)
    assert value != pytest.approx(0.5, abs=0.02)


def test_inverted_flux_sign_maps_the_same():
    rho = np.linspace(0.0, 1.0, 21)
    psi_rz = 0.1 * np.linspace(0.0, 1.0, 9)[:, None]
    table = (PSI_N, PSI_N**0.8)
    plus = map_rho_tor_profile_to_rz(psi_rz, 0.0, 0.1, rho, rho**2, flux_table=table)
    minus = map_rho_tor_profile_to_rz(-psi_rz, 0.0, -0.1, rho, rho**2, flux_table=table)
    np.testing.assert_array_equal(plus, minus)


def test_degenerate_inputs_are_refused():
    rho = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="equal"):
        map_rho_tor_profile_to_rz(np.zeros((2, 2)), 0.1, 0.1, rho, rho, flux_table=(rho, rho))
    with pytest.raises(ValueError, match="flux_table"):
        map_rho_tor_profile_to_rz(np.zeros((2, 2)), 0.0, 0.1, rho, rho, flux_table=(rho, rho[:-1]))


# --- core_profiles_grid_coordinate -----------------------------------------


def _core_grid(**leaves) -> ODS:
    ods = ODS(consistency_check=False)
    for leaf, value in leaves.items():
        ods[f"core_profiles.profiles_1d.0.grid.{leaf}"] = value
    return ods


def test_the_three_coordinates_are_never_substituted():
    rho_pol = np.linspace(0.0, 1.0, 11)
    rho_tor = rho_pol**1.3
    ods = _core_grid(rho_pol_norm=rho_pol, rho_tor_norm=rho_tor)
    assert CORE_GRID_COORDINATES == ("psi_norm", "rho_pol_norm", "rho_tor_norm")
    np.testing.assert_array_equal(core_profiles_grid_coordinate(ods, 0, "rho_tor_norm"), rho_tor)
    np.testing.assert_array_equal(core_profiles_grid_coordinate(ods, 0, "rho_pol_norm"), rho_pol)
    np.testing.assert_array_equal(core_profiles_grid_coordinate(ods, 0, "psi_norm"), rho_pol**2)


def test_the_grid_psi_is_normalised_by_the_matched_equilibrium():
    ods = _core_grid(psi=-0.3 - 0.1 * PSI_N)
    np.testing.assert_allclose(
        core_profiles_grid_coordinate(ods, 0, "psi_norm", psi_axis=-0.3, psi_boundary=-0.4), PSI_N
    )
    np.testing.assert_allclose(
        core_profiles_grid_coordinate(ods, 0, "rho_pol_norm", psi_axis=-0.3, psi_boundary=-0.4),
        np.sqrt(PSI_N),
    )
    with pytest.raises(CoordinateUnavailableError, match="neither rho_pol_norm nor psi"):
        core_profiles_grid_coordinate(ods, 0, "psi_norm")


@pytest.mark.parametrize("with_psi", [True, False])
def test_a_proxy_rho_tor_grid_is_refused(with_psi):
    leaves = {"rho_tor_norm": np.sqrt(PSI_N)}
    if with_psi:
        leaves["rho_pol_norm"] = np.sqrt(PSI_N)
    with pytest.raises(CoordinateUnavailableError, match="proxy"):
        core_profiles_grid_coordinate(_core_grid(**leaves), 0, "rho_tor_norm")


def test_an_unknown_coordinate_is_a_value_error():
    with pytest.raises(ValueError, match="coordinate must be one of"):
        core_profiles_grid_coordinate(_core_grid(rho_tor_norm=PSI_N), 0, "R")


# --- the one deliberate change in the views ---------------------------------


def test_the_kinetic_overview_omits_a_proxy_fit_instead_of_relabelling_it():
    ods = _core_grid(rho_tor_norm=np.sqrt(PSI_N), rho_pol_norm=np.sqrt(PSI_N))
    assert kinetic_overview._core_coordinate(ods, 0, "rho_tor_norm", None) is None
    np.testing.assert_array_equal(
        kinetic_overview._core_coordinate(ods, 0, "psi_norm", None), np.sqrt(PSI_N) ** 2
    )
    authentic = _core_grid(rho_tor_norm=PSI_N**0.7)
    np.testing.assert_array_equal(
        kinetic_overview._core_coordinate(authentic, 0, "rho_tor_norm", None), PSI_N**0.7
    )
