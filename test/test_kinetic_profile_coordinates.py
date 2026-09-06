"""The radial coordinate of a kinetic profile is a choice, made explicitly (issue #420).

Before this change the mappers returned normalized poloidal flux under the
name "rho", the fitters consumed it without saying so, and one docstring
called it ``rho_tor_norm``.  Now the mapping returns every coordinate the
equilibrium supports, the fit selects one -- ``rho_tor_norm`` by default --
and the result carries which.  These tests pin that the three coordinates
are genuinely different, that nothing conflates them, that the legacy path
is reachable only by name, and that the old numbers are exactly reproduced
when asked for.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

pytest.importorskip("omas")
from omas import ODS

from vaft.data import read_geqdsk
from vaft.data.resources import data_path
from vaft.process import profile as P
from vaft.process._equilibrium_parametric import derive_radial_coordinates

DATA = data_path("kineticEfit")
GFILE = DATA / "g048224.00300"
TS_MAT = DATA / "NeTe_48224.mat"
ION_MAT = DATA / "IDS_48224.mat"
FIXTURE = data_path("..") / ".." / "test" / "data" / "profile_psi_norm_fixture.npz"

pytestmark = pytest.mark.skipif(
    not (GFILE.exists() and TS_MAT.exists() and ION_MAT.exists()),
    reason="vaft/data/kineticEfit sample not present",
)

SHOT, TIME_MS = 48224, 300.0


@pytest.fixture(scope="module")
def shot_ods():
    from vaft.machine_mapping.charge_exchange import charge_exchange
    from vaft.machine_mapping.dataset_description import dataset_description
    from vaft.machine_mapping.thomson_scattering import thomson_scattering

    geq = read_geqdsk(GFILE)
    ods = geq.to_omas()
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    dataset_description(ods, source=SHOT, options={"source_type": "shot", "description": "coordinate test"})
    thomson_scattering(ods, SHOT, str(TS_MAT))
    charge_exchange(ods, shotnumber=SHOT, options="ids", mat_file=str(ION_MAT))
    return ods, geq


def _fixture():
    import pathlib

    path = pathlib.Path(__file__).parent / "data" / "profile_psi_norm_fixture.npz"
    return np.load(path)


# --- mapping ---------------------------------------------------------------------


def test_mapping_returns_every_coordinate(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)

    assert isinstance(mapped, P.MappedPositions)
    assert mapped.source == "geqdsk"
    assert mapped.available() == ("rho_tor_norm", "rho_pol_norm", "psi_norm")
    assert mapped.rho_tor_norm is not None and mapped.rho_tor_norm_unavailable is None
    assert mapped.n_channels == len(ods["thomson_scattering.channel"])


def test_rho_pol_norm_is_the_square_root_of_psi_norm(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_charge_exchange(ods, geq)
    finite = np.isfinite(mapped.psi_norm)

    np.testing.assert_allclose(mapped.rho_pol_norm[finite], np.sqrt(mapped.psi_norm[finite]))
    assert np.all(np.isnan(mapped.rho_pol_norm[~finite]))
    assert np.all(np.isnan(mapped.rho_tor_norm[~finite]))


def test_rho_tor_norm_comes_from_the_equilibrium_not_from_psi_norm(shot_ods):
    """The table is the equilibrium's own; and it is visibly not sqrt(psi_N)."""
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    table = derive_radial_coordinates(geq)
    x = np.asarray(table["psi_n"].value)
    y = np.asarray(table["rho_tor_n"].value)
    finite = np.isfinite(mapped.psi_norm)

    np.testing.assert_allclose(mapped.rho_tor_norm[finite], np.interp(mapped.psi_norm[finite], x, y))
    inner = finite & (mapped.psi_norm < 0.5)
    assert inner.any()
    assert np.max(np.abs(mapped.rho_tor_norm[inner] - mapped.rho_pol_norm[inner])) > 0.02


def test_select_refuses_an_unknown_coordinate(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    with pytest.raises(ValueError, match="coordinate must be one of"):
        mapped.select("rho")


def _legacy_flux_surfaces(r0=0.4, a_out=0.3):
    theta = np.linspace(0, 2 * np.pi, 181)
    flux = {i: {"R": r0 + a * np.cos(theta), "Z": a * np.sin(theta)} for i, a in enumerate([a_out / 2, a_out])}
    return {"fluxSurfaces": {"levels": [0.5, 1.0], "flux": flux}}


def test_legacy_flux_surfaces_cannot_supply_rho_tor_norm():
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.channel.0.position.r"] = 0.45
    ods["thomson_scattering.channel.0.position.z"] = 0.0
    mapped = P.equilibrium_mapping_thomson_scattering(ods, _legacy_flux_surfaces())

    assert mapped.source == "legacy_flux_surfaces"
    assert mapped.rho_tor_norm is None
    assert "no q profile" in mapped.rho_tor_norm_unavailable
    assert mapped.available() == ("rho_pol_norm", "psi_norm")
    with pytest.raises(P.CoordinateUnavailableError, match="no q profile"):
        mapped.select("rho_tor_norm")
    assert mapped.select("rho_pol_norm")[0] == pytest.approx(np.sqrt(mapped.psi_norm[0]))


# --- fitting -----------------------------------------------------------------------


def test_default_fit_is_in_rho_tor_norm(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped, time_tolerance_ms=3.0)

    assert isinstance(ne_fit, P.FittedProfile) and isinstance(te_fit, P.FittedProfile)
    assert ne_fit.coordinate == te_fit.coordinate == "rho_tor_norm"
    assert te_fit.method == "polynomial" and te_fit.order in (1, 2, 3)
    assert "coordinate=rho_tor_norm" in te_fit.parameters_text()
    assert np.all(np.isfinite(te_fit(np.linspace(0, 1, 7))))


@pytest.mark.parametrize("coordinate", P.COORDINATES)
def test_each_coordinate_can_be_selected_explicitly(shot_ods, coordinate):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_charge_exchange(ods, geq)
    vt, ti, *_ = P.profile_fitting_charge_exchange(
        ods, TIME_MS, mapped, ion_index=0, time_tolerance_ms=3.0, coordinate=coordinate
    )
    assert vt.coordinate == ti.coordinate == coordinate
    assert ti.span is not None and 0.0 <= ti.span[0] < ti.span[1] <= 1.0


def test_the_three_coordinates_give_three_different_fits(shot_ods):
    """A cubic in psi_N is not a cubic in rho_tor_norm; nothing may pretend otherwise."""
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    x = np.linspace(0.1, 0.9, 9)
    curves = {}
    for coordinate in P.COORDINATES:
        _, te_fit, *_ = P.profile_fitting_thomson_scattering(
            ods, TIME_MS, mapped, time_tolerance_ms=3.0, coordinate=coordinate
        )
        curves[coordinate] = te_fit(x)
    for a, b in (("rho_tor_norm", "psi_norm"), ("rho_tor_norm", "rho_pol_norm"), ("rho_pol_norm", "psi_norm")):
        assert not np.allclose(curves[a], curves[b], rtol=1e-3), (a, b)


def test_a_bare_array_needs_its_meaning_stated(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    bare = mapped.psi_norm.copy()

    with pytest.raises(TypeError, match="bare array"):
        P.profile_fitting_thomson_scattering(ods, TIME_MS, bare, time_tolerance_ms=3.0)
    with pytest.warns(DeprecationWarning, match="deprecated"):
        _, te_legacy, *_ = P.profile_fitting_thomson_scattering(
            ods, TIME_MS, bare, time_tolerance_ms=3.0, coordinate="psi_norm"
        )
    _, te_explicit, *_ = P.profile_fitting_thomson_scattering(
        ods, TIME_MS, mapped, time_tolerance_ms=3.0, coordinate="psi_norm"
    )
    grid = np.linspace(0, 1, 33)
    np.testing.assert_allclose(te_legacy(grid), te_explicit(grid))


def test_the_old_keyword_name_still_works_with_a_warning(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    with pytest.warns(DeprecationWarning, match="mapped_rho_position"):
        ne_fit, *_ = P.profile_fitting_thomson_scattering(
            ods, TIME_MS, mapped_rho_position=mapped, time_tolerance_ms=3.0
        )
    assert ne_fit.coordinate == "rho_tor_norm"


def test_default_fit_fails_loudly_when_rho_tor_norm_is_unavailable():
    ods = ODS()
    ods["thomson_scattering.ids_properties.homogeneous_time"] = 1
    ods["thomson_scattering.time"] = np.array([0.3])
    for i, r in enumerate([0.42, 0.48, 0.55, 0.62]):
        ods[f"thomson_scattering.channel.{i}.position.r"] = r
        ods[f"thomson_scattering.channel.{i}.position.z"] = 0.0
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = np.array([100.0 - 100 * i])
        ods[f"thomson_scattering.channel.{i}.t_e.data_error_upper"] = np.array([5.0])
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = np.array([1e19 * (1 - 0.2 * i)])
        ods[f"thomson_scattering.channel.{i}.n_e.data_error_upper"] = np.array([1e18])
    mapped = P.equilibrium_mapping_thomson_scattering(ods, _legacy_flux_surfaces(a_out=0.3))

    with pytest.raises(P.CoordinateUnavailableError, match="rho_tor_norm is not available"):
        P.profile_fitting_thomson_scattering(ods, 300.0, mapped, Te_order=1, Ne_order=1)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(
        ods, 300.0, mapped, Te_order=1, Ne_order=1, coordinate="rho_pol_norm"
    )
    assert te_fit.coordinate == "rho_pol_norm"


# --- core_profiles ------------------------------------------------------------------


def test_core_profiles_evaluates_each_fit_in_its_own_coordinate(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    stored = {}
    for coordinate in P.COORDINATES:
        work = ODS()
        work["thomson_scattering"] = ods["thomson_scattering"]
        ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(
            ods, TIME_MS, mapped, time_tolerance_ms=3.0, coordinate=coordinate
        )
        P.core_profiles(work, TIME_MS, mapped, ne_fit, te_fit, geq=geq, time_tolerance_ms=3.0)
        cp = "core_profiles.profiles_1d.0"
        grid_rt = np.asarray(work[f"{cp}.grid.rho_tor_norm"])
        grid_rp = np.asarray(work[f"{cp}.grid.rho_pol_norm"])
        x = {"rho_tor_norm": grid_rt, "rho_pol_norm": grid_rp, "psi_norm": grid_rp**2}[coordinate]
        # the stored profile IS the fit evaluated on the grid in the fit's coordinate
        np.testing.assert_allclose(work[f"{cp}.electrons.temperature"], te_fit(x), rtol=1e-12)
        assert f"coordinate={coordinate}" in str(work[f"{cp}.electrons.temperature_fit.parameters"])
        assert f"coordinate={coordinate}" in str(work["core_profiles.code.parameters"])
        assert str(work["core_profiles.code.name"]) == "vaft.process.profile"
        stored[coordinate] = np.asarray(work[f"{cp}.electrons.temperature"])
    assert not np.allclose(stored["rho_tor_norm"], stored["psi_norm"], rtol=1e-3)


def test_core_profiles_refuses_a_relabelled_fit(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped, time_tolerance_ms=3.0)
    with pytest.raises(ValueError, match="cannot be re-labelled"):
        P.core_profiles(ODS(), TIME_MS, mapped, ne_fit, te_fit, coordinate="psi_norm")


def test_core_profiles_refuses_plain_callables_without_a_coordinate(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    with pytest.raises(TypeError, match="plain callable"):
        P.core_profiles(ods, TIME_MS, mapped, lambda x: x, lambda x: x, time_tolerance_ms=3.0)


def test_fit_metadata_uses_the_mappings_rho_tor_norm_directly(shot_ods):
    ods, geq = shot_ods
    mapped_ts = P.equilibrium_mapping_thomson_scattering(ods, geq)
    mapped_cx = P.equilibrium_mapping_charge_exchange(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped_ts, time_tolerance_ms=3.0)
    vt_fit, ti_fit, *_ = P.profile_fitting_charge_exchange(ods, TIME_MS, mapped_cx, ion_index=0, time_tolerance_ms=3.0)
    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    work["charge_exchange"] = ods["charge_exchange"]
    P.core_profiles(work, TIME_MS, mapped_ts, ne_fit, te_fit, T_i_function=ti_fit, V_tor_function=vt_fit,
                    ti_mapped_positions=mapped_cx, geq=geq, time_tolerance_ms=3.0)
    cp = "core_profiles.profiles_1d.0"
    finite = np.isfinite(mapped_ts.psi_norm)
    np.testing.assert_allclose(work[f"{cp}.electrons.temperature_fit.rho_tor_norm"], mapped_ts.rho_tor_norm[finite])
    finite_cx = np.isfinite(mapped_cx.psi_norm)
    np.testing.assert_allclose(work[f"{cp}.ion.0.temperature_fit.rho_tor_norm"], mapped_cx.rho_tor_norm[finite_cx])
    assert "measured_span=" in str(work[f"{cp}.ion.0.temperature_fit.parameters"])


def test_ratio_fallback_records_its_provenance(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped, time_tolerance_ms=3.0)
    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    P.core_profiles(work, TIME_MS, mapped, ne_fit, te_fit, geq=geq, time_tolerance_ms=3.0,
                    ti_te_ratio=0.17, ti_te_ratio_record="ti_te_ratio=0.17; status=inferred; source=test")
    cp = "core_profiles.profiles_1d.0"
    assert str(work[f"{cp}.ion.0.temperature_fit.parameters"]) == "ti_te_ratio=0.17; status=inferred; source=test"
    assert "status=inferred" in str(work["core_profiles.code.parameters"])
    np.testing.assert_allclose(work[f"{cp}.ion.0.temperature"], 0.17 * np.asarray(work[f"{cp}.electrons.temperature"]))


def test_fallback_grid_is_stored_in_the_fit_coordinate():
    ods = ODS()
    ods["thomson_scattering.time"] = np.array([0.3])
    for i in range(4):
        ods[f"thomson_scattering.channel.{i}.n_e.data"] = np.array([1e19])
        ods[f"thomson_scattering.channel.{i}.t_e.data"] = np.array([100.0])
    positions = P.MappedPositions(np.linspace(0.1, 0.7, 4) ** 2, np.linspace(0.1, 0.7, 4), np.linspace(0.12, 0.75, 4), None, "test")
    fn = lambda x: 100.0 * (1 - np.asarray(x, float) ** 2)
    for coordinate, key in (("rho_tor_norm", "rho_tor_norm"), ("rho_pol_norm", "rho_pol_norm"), ("psi_norm", "rho_pol_norm")):
        work = ODS()
        work["thomson_scattering"] = ods["thomson_scattering"]
        P.core_profiles(work, 300.0, positions, fn, fn, coordinate=coordinate, rho_points=11)
        grid = work[f"core_profiles.profiles_1d.0.grid"]
        assert key in grid and len(grid) == 1, (coordinate, list(grid.keys()))
        expected = np.linspace(0, 1, 11) if coordinate != "psi_norm" else np.sqrt(np.linspace(0, 1, 11))
        np.testing.assert_allclose(grid[key], expected)


# --- the legacy numbers are reproducible, on request --------------------------------


def test_explicit_psi_norm_reproduces_the_pre_change_pipeline(shot_ods):
    fx = _fixture()
    ods, geq = shot_ods
    mapped_ts = P.equilibrium_mapping_thomson_scattering(ods, geq)
    mapped_cx = P.equilibrium_mapping_charge_exchange(ods, geq)
    np.testing.assert_allclose(mapped_ts.psi_norm, fx["psi_norm_ts"], equal_nan=True)
    np.testing.assert_allclose(mapped_cx.psi_norm, fx["psi_norm_cx"], equal_nan=True)

    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped_ts, time_tolerance_ms=3.0, coordinate="psi_norm")
    vt_fit, ti_fit, *_ = P.profile_fitting_charge_exchange(ods, TIME_MS, mapped_cx, ion_index=0, time_tolerance_ms=3.0, coordinate="psi_norm")
    grid = fx["grid"]
    np.testing.assert_allclose(ne_fit(grid), fx["ne_fit"], rtol=1e-10)
    np.testing.assert_allclose(te_fit(grid), fx["te_fit"], rtol=1e-10)
    np.testing.assert_allclose(ti_fit(grid), fx["ti_fit"], rtol=1e-10)
    np.testing.assert_allclose(vt_fit(grid), fx["vtor_fit"], rtol=1e-10)

    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    work["charge_exchange"] = ods["charge_exchange"]
    P.core_profiles(work, TIME_MS, mapped_ts, ne_fit, te_fit, T_i_function=ti_fit, V_tor_function=vt_fit,
                    ti_mapped_positions=mapped_cx, geq=geq, time_tolerance_ms=3.0, ti_te_ratio=None)
    cp = "core_profiles.profiles_1d.0"
    np.testing.assert_allclose(work[f"{cp}.grid.rho_tor_norm"], fx["cp_rho_tor_norm"])
    np.testing.assert_allclose(work[f"{cp}.grid.psi"], fx["cp_psi"])
    np.testing.assert_allclose(work[f"{cp}.electrons.density"], fx["cp_ne"], rtol=1e-10)
    np.testing.assert_allclose(work[f"{cp}.electrons.temperature"], fx["cp_te"], rtol=1e-10)
    np.testing.assert_allclose(work[f"{cp}.ion.0.temperature"], fx["cp_ti"], rtol=1e-10)
    np.testing.assert_allclose(work[f"{cp}.pressure_thermal"], fx["cp_pth"], rtol=1e-10)


def test_the_default_pipeline_differs_from_the_pre_change_one_on_purpose(shot_ods):
    fx = _fixture()
    ods, geq = shot_ods
    mapped_ts = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped_ts, time_tolerance_ms=3.0)
    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    P.core_profiles(work, TIME_MS, mapped_ts, ne_fit, te_fit, geq=geq, time_tolerance_ms=3.0)
    te = np.asarray(work["core_profiles.profiles_1d.0.electrons.temperature"])
    assert te.shape == fx["cp_te"].shape
    assert not np.allclose(te, fx["cp_te"], rtol=1e-3)
    assert np.all(np.isfinite(te)) and te[0] > 0


# --- cold-review regressions ------------------------------------------------------


def test_a_stored_sqrt_psi_proxy_grid_is_re_derived_not_trusted(shot_ods):
    """A pre-#276 equilibrium stores sqrt(psi_N) under rho_tor_norm.

    Evaluating a rho_tor_norm fit on that proxy would put the profile at the
    wrong radius.  The grid builder re-derives the coordinate from the slice's
    own ``q`` instead.
    """
    ods, geq = shot_ods
    n = len(np.asarray(ods["equilibrium.time_slice.0.profiles_1d.psi"]))
    psi = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.psi"], dtype=float)
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])

    work = ODS()
    work["equilibrium"] = ods["equilibrium"]
    work["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = np.sqrt(np.clip(psi_n, 0, 1))

    from vaft.data._derived import is_rho_pol_proxy

    assert is_rho_pol_proxy(work["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"], psi_n)

    grid = P._equilibrium_grid_at_time(work, float(work["equilibrium.time"][0]), 1.0)
    rho, psi_n_grid = grid.rho_tor_norm, grid.psi_norm
    assert grid.rho_tor_norm_trusted and rho.size == n
    assert not np.allclose(rho, np.sqrt(np.clip(psi_n_grid, 0, 1)), atol=1e-3)
    expected = derive_radial_coordinates(geq)["rho_tor_n"].value
    np.testing.assert_allclose(rho, expected, atol=2e-3)


def test_a_proxy_grid_without_q_refuses_rho_tor_norm_rather_than_substituting(shot_ods):
    ods, geq = shot_ods
    psi = np.asarray(ods["equilibrium.time_slice.0.profiles_1d.psi"], dtype=float)
    psi_n = (psi - psi[0]) / (psi[-1] - psi[0])

    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    work["equilibrium.time"] = np.asarray(ods["equilibrium.time"], dtype=float)
    work["equilibrium.ids_properties.homogeneous_time"] = 1
    work["equilibrium.time_slice.0.profiles_1d.psi"] = psi
    work["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = np.sqrt(np.clip(psi_n, 0, 1))

    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped, time_tolerance_ms=3.0)
    with pytest.raises(P.CoordinateUnavailableError, match="rho_pol_norm"):
        P.core_profiles(work, TIME_MS, mapped, ne_fit, te_fit, time_tolerance_ms=3.0)

    P.core_profiles(work, TIME_MS, mapped, ne_fit.function, te_fit.function,
                    coordinate="rho_pol_norm", time_tolerance_ms=3.0)
    assert "core_profiles.profiles_1d.0.grid.rho_pol_norm" in work


def test_a_bare_ion_array_is_refused_not_downgraded_to_a_warning(shot_ods):
    """The CX refusal is a caller error; it must not be swallowed by the
    metadata try/except that reports 'could not attach'."""
    ods, geq = shot_ods
    mapped_ts = P.equilibrium_mapping_thomson_scattering(ods, geq)
    mapped_cx = P.equilibrium_mapping_charge_exchange(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped_ts, time_tolerance_ms=3.0)
    _, ti_fit, *_ = P.profile_fitting_charge_exchange(ods, TIME_MS, mapped_cx, ion_index=0, time_tolerance_ms=3.0)

    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    work["charge_exchange"] = ods["charge_exchange"]
    with pytest.raises(TypeError, match="ti_mapped_rho_position"):
        P.core_profiles(work, TIME_MS, mapped_ts, ne_fit, te_fit, T_i_function=ti_fit,
                        ti_mapped_positions=np.asarray(mapped_cx.psi_norm), geq=geq,
                        time_tolerance_ms=3.0)


def test_mapping_and_fit_records_are_usable_in_a_set(shot_ods):
    """ndarray fields make the generated __eq__ raise; the records opt out."""
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped, time_tolerance_ms=3.0)

    assert mapped == mapped and mapped != P.equilibrium_mapping_charge_exchange(ods, geq)
    assert len({mapped, ne_fit, te_fit}) == 3


def test_an_existing_producer_name_is_not_overwritten(shot_ods):
    ods, geq = shot_ods
    mapped = P.equilibrium_mapping_thomson_scattering(ods, geq)
    ne_fit, te_fit, *_ = P.profile_fitting_thomson_scattering(ods, TIME_MS, mapped, time_tolerance_ms=3.0)

    work = ODS()
    work["thomson_scattering"] = ods["thomson_scattering"]
    work["core_profiles.code.name"] = "vaft.code.efit.kinetic"
    P.core_profiles(work, TIME_MS, mapped, ne_fit, te_fit, geq=geq, time_tolerance_ms=3.0)
    assert work["core_profiles.code.name"] == "vaft.code.efit.kinetic"
