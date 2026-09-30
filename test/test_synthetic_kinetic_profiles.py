"""Synthetic kinetic profiles from an equilibrium, fidelity Levels 0-3 (#122)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from vaft.data import (
    Composition,
    GradientProfile,
    ProfileSpec,
    ScalarTarget,
    SqrtPressureSplit,
    SyntheticKineticProfiles,
    SyntheticKineticSpec,
    SyntheticProfileError,
    TabulatedProfile,
    TemperatureAssumption,
)
from vaft.formula.constants import QE
from vaft.process.equilibrium import as_equilibrium, solovev_example
from vaft.process.profile import (
    evaluate_analytic_profile,
    analytic_hmode_itb_state,
    analytic_hmode_state,
    analytic_itb_state,
    analytic_lmode_state,
    compose_analytic_profile,
    core_profiles_from_eq,
    core_profiles_from_eq_ratio,
    generate_synthetic_kinetic_profiles as generate,
    spec_from_plasma_state,
    write_synthetic_core_profiles,
)
from vaft.process._synthetic_kinetic_profiles import _sqrt_pressure_split


@pytest.fixture(scope="module")
def geqdsk():
    from vaft.data.resources import sample_geqdsk

    return sample_geqdsk()


@pytest.fixture(scope="module")
def eq_ods(geqdsk):
    return geqdsk.to_omas()


@pytest.fixture(scope="module")
def solovev():
    return solovev_example("limited")


NE = compose_analytic_profile("n_e", axis_value=1.0, separatrix_value=0.2, core_alpha=1.5, core_beta=1.5)
TE = compose_analytic_profile("T_e", axis_value=1.0, separatrix_value=0.05, core_alpha=1.0, core_beta=2.0)
TE_H = compose_analytic_profile("T_e", axis_value=300.0, separatrix_value=10.0, pedestal_top_value=90.0,
                                pedestal_position=0.93, pedestal_width=0.05)


def _spec(**kwargs):
    base = dict(temperature=TemperatureAssumption(ti_over_te=0.5),
                n_e=ProfileSpec(NE, ScalarTarget("line_average", 1.0e19)),
                T_e=ProfileSpec(TE, ScalarTarget("axis", 150.0)))
    base.update(kwargs)
    return SyntheticKineticSpec(**base)


def _target(result, name):
    return next(t for t in result.targets if t.name == name)


# --- Level 0 -----------------------------------------------------------------


def _legacy_reference_te0(ods, Te0_eV, rho_fit):
    """The pre-#122 core_profiles_from_eq algorithm, copied verbatim."""
    rho_src = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"], dtype=float)
    p_src = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.pressure"], dtype=float)
    order = np.argsort(rho_src)
    P_fit = np.interp(rho_fit, rho_src[order], p_src[order])
    g = np.sqrt(np.clip(P_fit / P_fit[0], 0.0, None))
    return P_fit, P_fit[0] / (2.0 * Te0_eV * 1.602176634e-19) * g, Te0_eV * g


def test_legacy_helpers_are_byte_identical_and_the_level0_kernel_reproduces_them(eq_ods):
    rho = np.linspace(0.0, 1.0, 100)
    ods = copy.deepcopy(eq_ods)
    core_profiles_from_eq(ods, Te0_eV=120.0)
    P_fit, ne_ref, te_ref = _legacy_reference_te0(eq_ods, 120.0, rho)
    ne = np.asarray(ods["core_profiles.profiles_1d.0.electrons.density"])
    te = np.asarray(ods["core_profiles.profiles_1d.0.electrons.temperature"])
    assert np.array_equal(ne, ne_ref) and np.array_equal(te, te_ref)
    ne_k, te_k = _sqrt_pressure_split(P_fit, te_axis=120.0)
    assert np.array_equal(ne_k, ne) and np.array_equal(te_k, te)
    assert np.array_equal(ods["core_profiles.profiles_1d.0.ion.0.density"], ne)

    ods = copy.deepcopy(eq_ods)
    core_profiles_from_eq_ratio(ods, C_ne_over_Te=1.0e17)
    ne_k, te_k = _sqrt_pressure_split(P_fit, ne_over_te=1.0e17)
    assert np.array_equal(ods["core_profiles.profiles_1d.0.electrons.density"], ne_k)
    assert np.array_equal(ods["core_profiles.profiles_1d.0.electrons.temperature"], te_k)


@pytest.mark.parametrize("amplitude", [SqrtPressureSplit(te_axis=120.0), SqrtPressureSplit(ne_over_te=1.0e17)])
def test_level0_generator_matches_the_legacy_slice_on_the_legacy_grid(eq_ods, amplitude):
    """Generator against core_profiles_from_eq itself, sampled on the helper's own grid.

    The helper samples p linearly in the ODS's stored rho_tor_norm and the
    generator linearly in psi_norm, so on the same points they agree to that
    interpolation difference, not bit for bit.
    """
    ods = copy.deepcopy(eq_ods)
    (core_profiles_from_eq(ods, Te0_eV=amplitude.te_axis) if amplitude.te_axis
     else core_profiles_from_eq_ratio(ods, C_ne_over_Te=amplitude.ne_over_te))
    ts = "equilibrium.time_slice.0"
    rho_src = np.asarray(eq_ods[f"{ts}.profiles_1d.rho_tor_norm"], float)
    psi_src = np.asarray(eq_ods[f"{ts}.profiles_1d.psi"], float)
    psi_n_src = (psi_src - eq_ods[f"{ts}.global_quantities.psi_axis"]) / (
        eq_ods[f"{ts}.global_quantities.psi_boundary"] - eq_ods[f"{ts}.global_quantities.psi_axis"])
    order = np.argsort(rho_src)
    rho_legacy = np.asarray(ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"], float)
    grid = np.interp(rho_legacy, rho_src[order], psi_n_src[order])
    grid[0], grid[-1] = 0.0, 1.0
    spec = SyntheticKineticSpec(temperature=TemperatureAssumption(ti_over_te=1.0), pressure_constraint="equilibrium",
                                closure="sqrt_split", sqrt_split=amplitude, psi_norm=grid)
    result = generate(eq_ods, spec)
    for leaf, values in (("electrons.density", result.n_e), ("electrons.temperature", result.T_e),
                         ("ion.0.density", result.n_i), ("ion.0.temperature", result.T_i)):
        legacy = np.asarray(ods[f"core_profiles.profiles_1d.0.{leaf}"], float)
        assert values[0] == legacy[0], leaf  # the axis samples the same pressure: identical
        inner = grid < 0.9
        np.testing.assert_allclose(values[inner], legacy[inner], rtol=2e-3, err_msg=leaf)
        np.testing.assert_allclose(values, legacy, rtol=0, atol=0.02 * legacy.max(), err_msg=leaf)


@pytest.mark.parametrize("amplitude", [SqrtPressureSplit(te_axis=120.0), SqrtPressureSplit(ne_over_te=1.0e17)])
def test_level0_generator_uses_the_legacy_kernel(eq_ods, amplitude):
    spec = SyntheticKineticSpec(temperature=TemperatureAssumption(ti_over_te=1.0), pressure_constraint="equilibrium",
                                closure="sqrt_split", sqrt_split=amplitude)
    result = generate(eq_ods, spec)
    assert result.ok and result.fidelity_level == 0
    ne_k, te_k = _sqrt_pressure_split(result.p_eq, te_axis=amplitude.te_axis, ne_over_te=amplitude.ne_over_te)
    assert np.array_equal(result.n_e, ne_k) and np.array_equal(result.T_e, te_k)
    assert np.array_equal(result.n_i, result.n_e) and np.array_equal(result.T_i, result.T_e)
    np.testing.assert_allclose(result.p_total, result.p_eq, rtol=1e-12, atol=1e-12 * result.p_eq.max())
    # the axis value is the legacy helper's own, bit for bit
    ods = copy.deepcopy(eq_ods)
    (core_profiles_from_eq(ods, Te0_eV=amplitude.te_axis) if amplitude.te_axis
     else core_profiles_from_eq_ratio(ods, C_ne_over_Te=amplitude.ne_over_te))
    assert result.n_e[0] == ods["core_profiles.profiles_1d.0.electrons.density"][0]


# --- composition ----------------------------------------------------------------


def test_single_hydrogenic_species_has_ni_equal_ne(geqdsk):
    result = generate(geqdsk, _spec(composition=Composition("D")))
    assert result.ok
    assert np.array_equal(result.n_i, result.n_e) and not np.any(result.n_impurity)
    assert np.all(result.z_eff == 1.0)
    assert _target(result, "quasineutrality").achieved == 0.0


def test_one_impurity_gives_the_analytic_fraction_quasineutrality_and_zeff(geqdsk):
    result = generate(geqdsk, _spec(composition=Composition("D", "C", z_eff=2.0)))
    assert result.ok
    np.testing.assert_allclose(result.n_impurity / result.n_e, 1.0 / 30.0, rtol=1e-13)
    np.testing.assert_allclose(result.n_i + 6.0 * result.n_impurity, result.n_e, rtol=1e-14)
    np.testing.assert_allclose(result.z_eff, 2.0, rtol=1e-13)
    assert np.all(result.n_i < result.n_e) and np.all(result.n_i > 0)  # diluted, not negative
    assert [s.label for s in result.species] == ["D", "C"] and result.species[1].z == 6.0


# --- pressure closure --------------------------------------------------------------


@pytest.mark.parametrize("closure,channels,temperature", [
    ("temperature", {"n_e": ProfileSpec(NE, ScalarTarget("volume_average", 8e18))}, TemperatureAssumption(ti_over_te=0.4)),
    ("temperature", {"n_e": ProfileSpec(NE, ScalarTarget("line_average", 8e18))},
     TemperatureAssumption(electron_pressure_fraction=0.7)),
    ("density", {"T_e": ProfileSpec(TE_H)}, TemperatureAssumption(ti_over_te=0.5)),
    ("density", {"T_e": ProfileSpec(TE_H)}, TemperatureAssumption(electron_pressure_fraction=0.6)),
])
def test_equilibrium_closure_reproduces_p_eq_locally_and_integrally(geqdsk, closure, channels, temperature):
    spec = SyntheticKineticSpec(temperature=temperature, composition=Composition("D", "C", 1.5),
                                pressure_constraint="equilibrium", closure=closure, **channels)
    result = generate(geqdsk, spec)
    assert result.ok, result.message
    eq = as_equilibrium(geqdsk)
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    np.testing.assert_array_equal(result.p_eq, np.interp(result.psi_norm, psi_n, eq.pressure))
    edge = result.pressure.closure_exact_up_to_psi_n
    exact = result.psi_norm <= edge
    assert 0.8 < edge < 1.0  # p_eq -> 0 at this g-file's separatrix: an edge region exists
    assert np.all(result.p_eq[~exact] < spec.edge_floor * result.p_eq.max())
    assert np.all(result.p_eq[exact] >= spec.edge_floor * result.p_eq.max())
    np.testing.assert_allclose(result.p_total[exact], result.p_eq[exact], rtol=1e-12, atol=1e-12 * result.p_eq.max())
    assert result.pressure.locally_consistent and result.pressure.max_relative_residual < 1e-12
    # beyond it the residual is reported, bounded by the floor it lies under, and never hidden
    assert 0.0 < result.pressure.edge_max_relative_residual < 0.05
    assert abs(result.pressure.thermal_energy_relative_difference) < 1e-2  # the edge region is only ~0.3 %
    assert np.all(np.isfinite(result.z_eff)) and np.all(result.n_e > 0) and np.all(result.T_e > 0)
    if temperature.electron_pressure_fraction:
        np.testing.assert_allclose(result.p_e / result.p_total, temperature.electron_pressure_fraction, rtol=1e-12)
    assert set(result.provenance["solved"]) and result.provenance["pressure_constraint"] == "equilibrium"


def test_kinetic_mode_reports_the_mismatch_and_leaves_the_equilibrium_alone(geqdsk):
    before = as_equilibrium(geqdsk).pressure.copy()
    result = generate(geqdsk, _spec())
    assert result.ok and "NOT pressure-consistent" in result.message
    assert not result.pressure.locally_consistent and result.pressure.max_relative_residual > 0.05
    np.testing.assert_allclose(result.pressure.absolute_residual, result.p_total - result.p_eq)
    np.testing.assert_array_equal(as_equilibrium(geqdsk).pressure, before)
    # the requested axis T_e holds: kinetic mode never rescales to hide the mismatch
    assert _target(result, "T_e.axis").achieved == pytest.approx(150.0, rel=1e-12)


def test_thermal_energy_is_a_partial_constraint_that_never_claims_local_consistency(geqdsk):
    spec = _spec(T_e=ProfileSpec(TE), pressure_constraint="thermal_energy", closure="temperature_amplitude")
    result = generate(geqdsk, spec)
    assert result.ok and _target(result, "thermal_energy").met
    assert abs(result.pressure.thermal_energy_relative_difference) < 1e-12
    assert not result.pressure.locally_consistent and "NOT pressure-consistent" in result.message


# --- Level 1: the #1045 presets -------------------------------------------------------


@pytest.mark.parametrize("preset", [analytic_lmode_state, analytic_hmode_state, analytic_itb_state,
                                    analytic_hmode_itb_state])
def test_the_1045_presets_are_special_cases_of_the_generator(solovev, preset):
    state = preset(z_eff=1.5)
    result = generate(solovev, spec_from_plasma_state(state))
    assert result.ok and result.fidelity_level == 1
    np.testing.assert_array_equal(result.psi_norm, state.psi_norm)
    for name, other in (("n_e", "n_e"), ("T_e", "T_e"), ("T_i", "T_i"), ("n_i", "n_i"),
                        ("n_impurity", "n_impurity"), ("p_total", "p_total")):
        np.testing.assert_allclose(getattr(result, name), getattr(state, other), rtol=1e-12)
    assert result.species[1].label == "C"


def test_hmode_pedestal_keeps_its_top_under_scaling(geqdsk):
    result = generate(geqdsk, _spec(T_e=ProfileSpec(TE_H, ScalarTarget("axis", 150.0))))
    top = result.resolved["T_e"]["profile"]
    assert top.pedestal_top_value == pytest.approx(90.0 * 0.5) and top.pedestal.position == 0.93
    assert result.T_e[-1] == pytest.approx(5.0)


def test_tabulated_profile_on_a_declared_coordinate(geqdsk):
    x = np.linspace(0.0, 1.0, 11)
    table = TabulatedProfile(x, 2e19 * (1 - 0.8 * x**2), coordinate="rho_tor_norm")
    result = generate(geqdsk, _spec(n_e=ProfileSpec(table)))
    assert result.ok
    from scipy.interpolate import PchipInterpolator

    np.testing.assert_allclose(result.n_e, PchipInterpolator(x, table.values)(result.rho_tor_norm), rtol=1e-12)
    np.testing.assert_allclose(result.n_e, 2e19 * (1 - 0.8 * result.rho_tor_norm**2), rtol=2e-3)
    with pytest.raises(SyntheticProfileError, match="coordinate_mapping_failed"):
        short = TabulatedProfile(x[:-2], 2e19 * (1 - 0.8 * x[:-2] ** 2), coordinate="rho_tor_norm")
        generate(geqdsk, _spec(n_e=ProfileSpec(short)))


# --- Level 2 ---------------------------------------------------------------------------


def _independent_averages(geqdsk, result, values):
    """Volume and line averages from the 2-D map, not from the generator's operators."""
    from scipy.interpolate import RegularGridInterpolator

    from vaft.formula.equilibrium import psi_normalised
    from vaft.process.equilibrium import plasma_cell_weights, volume_average

    eq = as_equilibrium(geqdsk)
    psi_n = psi_normalised(eq.psi, eq.psi_axis, eq.psi_boundary)
    weights = plasma_cell_weights(eq.r, eq.z, psi_n, eq.lcfs.r, eq.lcfs.z)
    field = np.interp(np.clip(psi_n, 0, 1), result.psi_norm, values)
    vol, _ = volume_average(field, psi_n, eq.r, eq.z, weights=weights)
    from matplotlib.path import Path

    r = np.linspace(eq.r.min(), eq.r.max(), 4001)
    z_axis = eq.magnetic_axis[1]
    inside = Path(np.column_stack([eq.lcfs.r, eq.lcfs.z])).contains_points(np.column_stack([r, np.full_like(r, z_axis)]))
    psi_line = RegularGridInterpolator((eq.r, eq.z), psi_n)(np.column_stack([r[inside], np.full(inside.sum(), z_axis)]))
    line = float(np.mean(np.interp(np.clip(psi_line, 0, 1), result.psi_norm, values)))
    return vol, line


def test_volume_and_line_average_targets_are_met_by_normalization(geqdsk):
    for kind in ("volume_average", "line_average"):
        result = generate(geqdsk, _spec(n_e=ProfileSpec(NE, ScalarTarget(kind, 1.2e19))))
        t = _target(result, f"n_e.{kind}")
        assert t.met and t.achieved == pytest.approx(1.2e19, rel=1e-12) and t.solver_status == "linear"
        vol, line = _independent_averages(geqdsk, result, result.n_e)
        assert {"volume_average": vol, "line_average": line}[kind] == pytest.approx(1.2e19, rel=1e-2)


def test_peaking_and_average_constrain_shape_and_amplitude_separately(geqdsk):
    result = generate(geqdsk, _spec(n_e=ProfileSpec(NE, ScalarTarget("volume_average", 1e19), peaking_factor=1.8)))
    assert result.ok
    peak, avg = _target(result, "n_e.peaking_factor"), _target(result, "n_e.volume_average")
    assert peak.solver_status == "brentq" and peak.achieved == pytest.approx(1.8, rel=1e-8)
    vol, _ = _independent_averages(geqdsk, result, result.n_e)
    assert result.n_e[0] / vol == pytest.approx(1.8, rel=1e-2) and avg.achieved == pytest.approx(1e19, rel=1e-12)
    assert result.resolved["n_e"]["core_beta"] != NE.core_beta
    # unreachable peaking is reported, not forced
    far = generate(geqdsk, _spec(n_e=ProfileSpec(NE, peaking_factor=50.0)))
    assert far.status == "constraint_not_reached" and not _target(far, "n_e.peaking_factor").met


def test_greenwald_fraction_uses_the_declared_line_average(geqdsk):
    from vaft.formula.stability import greenwald_density

    result = generate(geqdsk, _spec(n_e=ProfileSpec(NE, ScalarTarget("greenwald_fraction", 0.4))))
    eq = as_equilibrium(geqdsk)
    n_g = 1e19 * greenwald_density(abs(eq.ip) / 1e6, 0.5 * np.ptp(eq.lcfs.r))
    _, line = _independent_averages(geqdsk, result, result.n_e)
    assert line / n_g == pytest.approx(0.4, rel=1e-2)
    assert "line average" in _target(result, "n_e.greenwald_fraction").definition


def test_ti_te_ratio_profile_and_partition_routes(geqdsk):
    ratio = TabulatedProfile([0.0, 1.0], [0.8, 0.3], coordinate="psi_norm", interpolation="linear")
    result = generate(geqdsk, _spec(temperature=TemperatureAssumption(ti_over_te=ratio, source="test ratio")))
    np.testing.assert_allclose(result.T_i / result.T_e, 0.8 - 0.5 * result.psi_norm, rtol=1e-12)
    assert "test ratio" in result.provenance["channels"]["T_i"]
    # an independent T_i profile is used as given
    ti = compose_analytic_profile("T_i", axis_value=80.0, separatrix_value=6.0)
    result = generate(geqdsk, _spec(temperature=TemperatureAssumption(T_i=ProfileSpec(ti)), composition=Composition("D", "C", 2.0)))
    np.testing.assert_allclose(result.T_i, evaluate_analytic_profile(ti, result.psi_norm), rtol=1e-14)
    assert result.provenance["temperature_route"] == "profile"
    # a pressure partition fixes T_i from T_e and the composition, locally
    result = generate(geqdsk, _spec(temperature=TemperatureAssumption(electron_pressure_fraction=0.65),
                                    composition=Composition("D", "C", 2.0)))
    np.testing.assert_allclose(result.p_e / result.p_total, 0.65, rtol=1e-12)
    assert _target(result, "p_e/p_total").met


def test_held_ion_temperature_now_closes_with_a_reported_edge(geqdsk):
    """The reviewer's case: T_i held 50 -> 5 eV, p_eq = 0 at the separatrix."""
    ti = ProfileSpec(compose_analytic_profile("T_i", axis_value=50.0, separatrix_value=5.0))
    spec = SyntheticKineticSpec(temperature=TemperatureAssumption(T_i=ti),
                                n_e=ProfileSpec(NE, ScalarTarget("line_average", 1e19)),
                                pressure_constraint="equilibrium", closure="temperature", edge_value=5.0)
    result = generate(geqdsk, spec)
    assert result.ok, result.message
    report = result.pressure
    exact = result.psi_norm <= report.closure_exact_up_to_psi_n
    np.testing.assert_allclose(result.p_total[exact], result.p_eq[exact], rtol=1e-12, atol=1e-12 * result.p_eq.max())
    assert result.T_e[-1] == pytest.approx(5.0) and np.all(result.T_e > 0)
    assert report.edge_max_relative_residual > 0 and "edge region" in result.message
    assert "exact for psi_N <=" in result.provenance["channels"]["T_e"]
    # T_i was held, so it is not listed as solved
    assert result.provenance["solved"] == ["T_e(psi)"] and "T_i profile" in result.provenance["held"]
    np.testing.assert_allclose(result.T_i, evaluate_analytic_profile(ti.shape, result.psi_norm), rtol=1e-14)


def test_density_closure_never_stores_a_nan_zeff(geqdsk):
    spec = SyntheticKineticSpec(temperature=TemperatureAssumption(ti_over_te=0.5), T_e=ProfileSpec(TE_H),
                                composition=Composition("D", "C", 2.0), pressure_constraint="equilibrium",
                                closure="density")
    result = generate(geqdsk, spec, time=0.319)
    assert result.ok and result.n_e[-1] > 0.0
    np.testing.assert_allclose(result.z_eff, 2.0, rtol=1e-13)
    ods = write_synthetic_core_profiles(result)
    assert np.all(np.isfinite(ods["core_profiles.profiles_1d.0.zeff"]))
    assert result.provenance["solved"] == ["n_e(psi)", "n_i", "n_impurity"]
    # the Level 0 split does reach n_e = 0 at the separatrix; Z_eff there is the composition's limit
    level0 = generate(geqdsk, SyntheticKineticSpec(
        temperature=TemperatureAssumption(ti_over_te=0.5), composition=Composition("D", "C", 2.0),
        pressure_constraint="equilibrium", closure="sqrt_split", sqrt_split=SqrtPressureSplit(te_axis=100.0)))
    assert level0.n_e[-1] == 0.0 and level0.z_eff[-1] == 2.0
    assert level0.validation["z_eff_points_at_zero_density"] == 1


# --- Level 3 ---------------------------------------------------------------------------


GRADIENT = GradientProfile(x=[0.0, 0.3, 0.8, 0.9, 1.0], a_over_L=[0.0, 2.0, 2.0, 8.0, 8.0],
                           boundary_value=20.0, boundary_position=1.0, coordinate="rho_pol_norm")


@pytest.mark.parametrize("points", [201, 801])
def test_gradient_construction_reproduces_the_prescribed_a_over_L(geqdsk, points):
    grid = np.linspace(0.0, 1.0, points) ** 2
    result = generate(geqdsk, _spec(T_e=ProfileSpec(GRADIENT), psi_norm=grid))
    assert result.ok and result.fidelity_level == 3
    x = result.rho_pol_norm
    away = np.min(np.abs(x[:, None] - GRADIENT.x[None, :]), axis=1) > 1.5 / (points - 1)
    reconstructed = result.a_over_L("T_e", "rho_pol_norm")
    np.testing.assert_allclose(reconstructed[away], GRADIENT.gradient(x[away]), atol=5e-4 * 201 / points)
    assert result.T_e[-1] == pytest.approx(20.0, rel=1e-12)
    assert "not transport-predicted" in result.provenance["channels"]["T_e"]
    assert result.provenance["transport_predicted"] is False


def test_gradient_profile_keeps_its_gradient_under_normalization(geqdsk):
    result = generate(geqdsk, _spec(T_e=ProfileSpec(GRADIENT, ScalarTarget("volume_average", 100.0))))
    assert _target(result, "T_e.volume_average").met
    exact = GRADIENT.boundary_value * np.exp(GRADIENT.integral(1.0) - GRADIENT.integral(result.rho_pol_norm))
    np.testing.assert_allclose(result.T_e / exact, result.resolved["T_e"]["scale"], rtol=1e-12)


# --- coordinates, output ---------------------------------------------------------------


def test_geqdsk_and_ods_forms_give_the_same_profiles(geqdsk, eq_ods):
    spec = _spec(pressure_constraint="equilibrium", closure="temperature", T_e=None)
    a, b = generate(geqdsk, spec), generate(eq_ods, spec)
    for name in ("n_e", "T_e", "T_i", "p_eq", "volume", "rho_tor_norm"):
        np.testing.assert_allclose(getattr(a, name), getattr(b, name), rtol=1e-6, atol=1e-9, err_msg=name)


def test_core_profiles_round_trip(geqdsk, tmp_path):
    from omas import load_omas_json, save_omas_json

    result = generate(geqdsk, _spec(composition=Composition("D", "C", 1.8)), time=0.319)
    ods = write_synthetic_core_profiles(result)
    path = tmp_path / "cp.json"
    save_omas_json(ods, str(path))
    back = load_omas_json(str(path))
    base = "core_profiles.profiles_1d.0"
    assert back[f"{base}.time"] == pytest.approx(0.319)
    for leaf, values in (("grid.rho_pol_norm", result.rho_pol_norm), ("grid.rho_tor_norm", result.rho_tor_norm),
                         ("electrons.density", result.n_e), ("electrons.temperature", result.T_e),
                         ("ion.0.density", result.n_i), ("ion.1.density", result.n_impurity),
                         ("ion.0.temperature", result.T_i), ("zeff", result.z_eff),
                         ("pressure_thermal", result.p_total)):
        np.testing.assert_allclose(back[f"{base}.{leaf}"], values, rtol=1e-12, err_msg=leaf)
    assert f"{base}.grid.psi" not in back  # COCOS 1 vs 2 ambiguous for this g-file: left out, not guessed
    assert back[f"{base}.ion.1.z_ion"] == 6.0 and back[f"{base}.ion.0.element.0.a"] == pytest.approx(2.0141)
    params = json.loads(back["core_profiles.code.parameters"])
    assert params["slices"]["0.319000000"]["profile_basis"] == "assumed"
    assert "not transport-predicted" in back["core_profiles.ids_properties.comment"]
    # a second slice appends, the same time replaces
    write_synthetic_core_profiles(result, ods, time=0.320)
    write_synthetic_core_profiles(result, ods, time=0.319)
    assert list(ods["core_profiles.time"]) == pytest.approx([0.319, 0.320])


def test_declared_cocos_gives_psi_in_weber(geqdsk):
    result = generate(as_equilibrium(geqdsk, convention=1), _spec())
    eq = as_equilibrium(geqdsk)
    assert result.psi[0] == pytest.approx(2 * np.pi * eq.psi_axis, rel=1e-12)
    ods = write_synthetic_core_profiles(result, time=0.319)
    np.testing.assert_allclose(ods["core_profiles.profiles_1d.0.grid.psi"], result.psi)


def test_kinetic_profiles_container(solovev):
    result = generate(solovev, spec_from_plasma_state(analytic_hmode_state()))
    kp = result.to_kinetic_profiles()
    np.testing.assert_array_equal(kp.n_e, result.n_e)
    np.testing.assert_array_equal(kp.field("p_eq"), result.p_eq)
    assert result.rho_tor_norm is None  # the Solov'ev example carries no q
    with pytest.raises(ValueError, match="rho_tor_norm"):
        result.coordinate("rho_tor_norm")


# --- refusals --------------------------------------------------------------------------


@pytest.mark.parametrize("build,status", [
    (lambda: Composition("D", "C", z_eff=7.0), "invalid_composition"),
    (lambda: Composition("D", z_eff=1.5), "invalid_composition"),
    (lambda: Composition("He"), "invalid_composition"),
    (lambda: ScalarTarget("line_average", -1e19), "invalid_normalization"),
    (lambda: ScalarTarget("median", 1e19), "invalid_normalization"),
    (lambda: ProfileSpec(NE, peaking_factor=float("nan")), "invalid_normalization"),
    (lambda: ProfileSpec(GRADIENT, peaking_factor=1.5), "invalid_normalization"),
    (lambda: GradientProfile([0.1, 1.0], [1.0, 2.0], 10.0), "invalid_gradient_model"),
    (lambda: GradientProfile([0.0, 1.0], [1.0, np.inf], 10.0), "invalid_gradient_model"),
    (lambda: GradientProfile([0.0, 1.0], [1.0, 2.0], 0.0), "invalid_gradient_model"),
    (lambda: TabulatedProfile([0.0, 0.5, 0.4], [1, 2, 3]), "invalid_profile_model"),
    (lambda: TabulatedProfile([0.0, 1.0], [1, 2], coordinate="r_over_a"), "invalid_profile_model"),
    (lambda: TemperatureAssumption(), "invalid_profile_model"),
    (lambda: TemperatureAssumption(ti_over_te=1.0, electron_pressure_fraction=0.5), "invalid_profile_model"),
    (lambda: TemperatureAssumption(electron_pressure_fraction=1.0), "invalid_normalization"),
    (lambda: _spec(pressure_constraint="kinetic", closure="temperature"), "invalid_profile_model"),
])
def test_contradictory_or_malformed_inputs_are_refused(build, status):
    with pytest.raises(SyntheticProfileError) as info:
        build()
    assert info.value.status == status


@pytest.mark.parametrize("kwargs", [
    {"pressure_constraint": "kinetic"},
    {"pressure_constraint": "thermal_energy", "closure": "temperature_amplitude", "T_e": ProfileSpec(TE)},
])
def test_a_hollow_equilibrium_pressure_is_refused_with_its_status(geqdsk, kwargs):
    """An axis pressure sample below ``edge_floor`` of the maximum (legal for
    a foreign g-file) used to reach a bare numpy "zero-size array" ValueError
    under every non-local constraint (cold review 0.8.0 plasma-state-and-chease F1)."""
    hollow = copy.deepcopy(geqdsk)
    pressure = np.asarray(hollow["PRES"], dtype=float).copy()
    pressure[0] = 0.5e-2 * pressure.max()
    hollow["PRES"] = pressure
    with pytest.raises(SyntheticProfileError, match="hollow") as info:
        generate(hollow, _spec(**kwargs))
    assert info.value.status == "invalid_equilibrium"
    assert kwargs["pressure_constraint"] in str(info.value)


def test_analytic_kernel_refusals_come_from_the_1045_layer():
    with pytest.raises(ValueError, match="outside"):
        compose_analytic_profile("T_e", axis_value=1.0, separatrix_value=0.1, pedestal_top_value=0.5,
                                 pedestal_position=0.9, pedestal_width=0.0)
    with pytest.raises(ValueError, match="at least one"):
        compose_analytic_profile("n_e", axis_value=1.0, separatrix_value=0.1, core_alpha=0.5)


def test_generator_refuses_what_the_closure_would_contradict(geqdsk, solovev):
    with pytest.raises(SyntheticProfileError, match="solves T_e") as info:
        generate(geqdsk, _spec(pressure_constraint="equilibrium", closure="temperature"))
    assert info.value.status == "invalid_profile_model"
    with pytest.raises(SyntheticProfileError, match="fixes the same amplitude twice"):
        generate(geqdsk, _spec(pressure_constraint="thermal_energy", closure="temperature_amplitude"))
    with pytest.raises(SyntheticProfileError, match="normalizes n_e"):
        generate(geqdsk, _spec(T_e=ProfileSpec(TE, ScalarTarget("greenwald_fraction", 0.5))))
    with pytest.raises(SyntheticProfileError) as info:
        generate(solovev, _spec(n_e=ProfileSpec(TabulatedProfile([0, 1], [2e19, 1e18]))))
    assert info.value.status == "coordinate_mapping_failed"


def test_incompatible_held_ion_temperature_fails_the_closure_and_cannot_be_stored(geqdsk):
    hot_ions = ProfileSpec(compose_analytic_profile("T_i", axis_value=2000.0, separatrix_value=500.0))
    spec = SyntheticKineticSpec(temperature=TemperatureAssumption(T_i=hot_ions),
                                n_e=ProfileSpec(NE, ScalarTarget("line_average", 1e19)),
                                pressure_constraint="equilibrium", closure="temperature")
    result = generate(geqdsk, spec)
    assert result.status == "pressure_closure_failed" and "T_e" in result.message
    with pytest.raises(SyntheticProfileError) as info:
        write_synthetic_core_profiles(result, time=0.319)
    assert info.value.status == "pressure_closure_failed"


def test_the_record_refuses_a_pressure_inconsistent_with_its_profiles(geqdsk):
    import dataclasses

    result = generate(geqdsk, _spec())
    with pytest.raises(ValueError, match="p_e"):
        dataclasses.replace(result, p_e=2 * result.p_e)
    assert isinstance(result, SyntheticKineticProfiles) and QE * result.n_e[0] * result.T_e[0] == result.p_e[0]


def test_writer_refuses_an_ods_the_legacy_helper_filled(eq_ods, geqdsk):
    ods = copy.deepcopy(eq_ods)
    core_profiles_from_eq(ods, Te0_eV=100.0)
    before = copy.deepcopy(ods["core_profiles"])
    result = generate(geqdsk, _spec(), time=0.319)
    for t in (0.319, 1.319):  # neither beside it nor over its slice
        with pytest.raises(ValueError, match="another producer"):
            write_synthetic_core_profiles(result, ods, time=t)
    assert "core_profiles.ids_properties.comment" not in ods  # never relabelled
    assert len(ods["core_profiles.profiles_1d"]) == len(before["profiles_1d"])
    np.testing.assert_array_equal(ods["core_profiles.profiles_1d.0.electrons.density"],
                                  before["profiles_1d.0.electrons.density"])


def test_legacy_and_measured_writers_refuse_a_synthetic_ids(geqdsk, eq_ods):
    from vaft.process.profile import core_profiles

    ods = write_synthetic_core_profiles(generate(geqdsk, _spec(), time=0.319))
    ods["equilibrium"] = copy.deepcopy(eq_ods["equilibrium"])
    params = ods["core_profiles.code.parameters"]
    for call in (lambda: core_profiles_from_eq(ods, Te0_eV=100.0),
                 lambda: core_profiles_from_eq_ratio(ods, C_ne_over_Te=1e17),
                 lambda: core_profiles(ods, 319.0, n_e_function=lambda x: 1e19 + 0 * x,
                                       T_e_function=lambda x: 100 + 0 * x, coordinate="psi_norm")):
        with pytest.raises(ValueError, match="synthetic kinetic-profile generator"):
            call()
    assert ods["core_profiles.code.parameters"] == params and len(ods["core_profiles.profiles_1d"]) == 1
    json.loads(params)


def test_ods_input_uses_its_declared_cocos_for_psi(geqdsk, eq_ods):
    from_ods = generate(eq_ods, _spec())
    from_g = generate(as_equilibrium(geqdsk, convention=1), _spec())
    assert "declared by the ODS" in from_ods.provenance["psi"]
    np.testing.assert_allclose(from_ods.psi, from_g.psi, rtol=1e-9)
    ods = write_synthetic_core_profiles(from_ods, time=0.319)
    np.testing.assert_allclose(ods["core_profiles.profiles_1d.0.grid.psi"], from_ods.psi)


@pytest.mark.parametrize("build", [
    lambda: _spec(T_e=ProfileSpec(NE, ScalarTarget("axis", 100.0))),          # an n_e shape in the T_e slot
    lambda: _spec(n_e=ProfileSpec(TabulatedProfile([0, 1], [2e19, 1e18], unit="cm^-3"))),
    lambda: _spec(temperature=TemperatureAssumption(T_i=ProfileSpec(TE))),     # a T_e shape as T_i
    lambda: TemperatureAssumption(ti_over_te=TabulatedProfile([0, 1], [1, 1], unit="eV")),
    lambda: _spec(edge_value=10.0),                                            # no local closure to carry
    lambda: _spec(edge_floor=0.0),
])
def test_mislabelled_channels_and_units_are_refused(build):
    with pytest.raises(SyntheticProfileError):
        build()


def test_hold_extrapolation_is_recorded(geqdsk):
    table = TabulatedProfile([0.0, 0.5, 0.9], [2e19, 1.5e19, 8e18], coordinate="rho_pol_norm", extrapolation="hold",
                             unit="m^-3")
    result = generate(geqdsk, _spec(n_e=ProfileSpec(table)))
    note = result.provenance["channels"]["n_e"]
    assert "extrapolation='hold'" in note and "held at the end values" in note
    assert result.resolved["n_e"]["extrapolation"] in note
    assert np.all(result.n_e[result.rho_pol_norm > 0.9] == 8e18)
