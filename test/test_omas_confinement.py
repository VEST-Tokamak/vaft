"""The ODS-facing experimental Ohmic confinement chain."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from vaft.omas.confinement import W_SOURCES, ohmic_confinement_ods
from vaft.process.confinement import ohmic_confinement_series


@pytest.fixture(scope="module")
def ods():
    from vaft.omas import sample_ods

    return sample_ods(39915)


@pytest.fixture(scope="module")
def result(ods):
    return ohmic_confinement_ods(ods, require_positive_energy=True)


def test_the_entry_records_what_it_read(result):
    assert result.loop_name == "Flux Loop - #10" and result.loop_r_m == pytest.approx(0.091)
    assert result.voltage_source == "measured voltage" and result.w_source == "efit_pressure"
    a = result.chain.assumptions
    assert "Flux Loop - #10" in a["v_loop"] and "hybrid" in a["i_p"] and a["w"] == W_SOURCES["efit_pressure"]
    assert "not R_geo" in a["r0"] and "inferred, not measured" in a["v_res"]
    assert result.chain.settings["rate_window_s"] == 3e-3


def test_the_entry_adds_no_physics_of_its_own(result):
    """Agreement with the process kernel for identical inputs and settings."""
    c = result.chain
    again = ohmic_confinement_series(c.time, c.i_p, c.dip_dt, c.v_loop, c.li_3, c.r0, c.w,
                                     rate_window_s=3e-3, rate_polyorder=1)
    for name in ("v_ind", "v_res", "p_ohmic", "dwdt", "p_net", "tau_e_net"):
        np.testing.assert_array_equal(getattr(again, name), getattr(c, name), err_msg=name)


def test_the_ods_is_not_written():
    """A fresh sample (the module fixture has been read before), checked for empty nodes too."""
    from vaft.omas import sample_ods

    fresh = sample_ods(39915)
    before = copy.deepcopy(fresh)
    ohmic_confinement_ods(fresh)
    ohmic_confinement_ods(fresh, w_source="core_profiles_thermal")
    assert set(fresh.flat()) == set(before.flat())
    for k in range(len(fresh["equilibrium.time_slice"])):
        for leaf in ("global_quantities.li_3", "global_quantities.volume", "boundary"):
            path = f"equilibrium.time_slice.{k}.{leaf}"
            assert (path in fresh) == (path in before), path


def test_time_slices_must_be_distinct_in_range_indices(ods):
    for bad in ([-1], [0, 0], [10_000]):
        with pytest.raises(ValueError, match="time_slices"):
            ohmic_confinement_ods(ods, time_slices=bad)


def test_slices_are_selected_and_ordered_by_physical_time(ods, result):
    chosen = list(result.eq_index[:5])
    forward = ohmic_confinement_ods(ods, time_slices=chosen)
    shuffled = ohmic_confinement_ods(ods, time_slices=chosen[::-1])
    np.testing.assert_array_equal(forward.eq_index, chosen)
    np.testing.assert_array_equal(shuffled.eq_index, chosen)
    np.testing.assert_array_equal(shuffled.chain.tau_e_net, forward.chain.tau_e_net)
    assert np.all(np.diff(forward.chain.time) > 0)


def test_non_positive_energy_slices_are_dropped_only_when_asked(ods, result):
    every = ohmic_confinement_ods(ods)
    assert every.eq_index.size >= result.eq_index.size
    assert np.all(np.isfinite(result.chain.w) & (result.chain.w > 0))


def test_a_missing_measured_voltage_falls_back_to_the_flux_and_says_so(ods, result):
    work = copy.deepcopy(ods)
    del work[f"magnetics.flux_loop.{result.loop_index}.voltage"]
    derived = ohmic_confinement_ods(work, loop_index=result.loop_index, require_positive_energy=True)
    assert derived.voltage_source.startswith("derived") and "derived" in derived.chain.assumptions["v_loop"]


def test_refusals_name_what_is_missing(ods):
    with pytest.raises(ValueError, match="w_source"):
        ohmic_confinement_ods(ods, w_source="global")
    work = copy.deepcopy(ods)
    del work["magnetics.ip"]
    with pytest.raises(ValueError, match="plasma current"):
        ohmic_confinement_ods(work)
    with pytest.raises(ValueError, match="no equilibrium slice left"):
        ohmic_confinement_ods(ods, w_source="core_profiles_thermal", require_positive_energy=True)  # no core_profiles


def test_the_summary_columns_are_the_shared_chain(ods, result):
    from vaft.database import _summary

    columns = _summary._ohmic_columns(ods, 39915)
    for j, k in enumerate(result.eq_index):
        row = columns[int(k)]
        assert row["tau_e_net_s"] == pytest.approx(result.chain.tau_e_net[j], nan_ok=True)
        assert row["p_net_MW"] == pytest.approx(result.chain.p_net[j] / 1e6, nan_ok=True)
        assert row["w_mhd_kJ"] == pytest.approx(result.chain.w[j] / 1e3)
    assert set(_summary.CORE_PROFILES_DEFINITIONS) >= {"p_loss_MW", "tau_e_s", *next(iter(columns.values()))}


def test_the_legacy_experimental_tau_warns_where_its_definition_differs():
    from vaft.omas import formula_wrapper

    with pytest.warns(FutureWarning, match="ohmic_confinement_ods"):
        with pytest.raises(Exception):
            formula_wrapper.compute_tau_E_exp({"equilibrium.time_slice": []}, 0)
