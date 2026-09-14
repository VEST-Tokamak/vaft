"""Resonant-response reductions: windows, magnitudes, ratios."""

from __future__ import annotations

import numpy as np
import pytest
from resonant_table_fixtures import PedestalStub, q_profile, resonant_table

from vaft.process.equilibrium import find_rational_surfaces
from vaft.process.perturbation import (
    LEGACY_WINDOWS,
    RESONANT_STATISTICS,
    ResonantWindow,
    amplification_ratio,
    reduce_resonant,
    resonant_metrics,
    resonant_windows,
    rms_resonant_field,
)


# --- where a region begins ----------------------------------------------------


def test_the_core_edge_boundary_comes_from_the_pedestal():
    """D-05: the boundary is a property of the plasma, not a constant. The
    code this replaces hard-coded 0.8, and seven different window sets were
    in use across the notebooks that used it."""
    windows = resonant_windows(PedestalStub(position=0.93, width=0.06))
    assert windows["core"].high == pytest.approx(0.90)
    assert windows["edge"].low == pytest.approx(0.90)
    assert windows["edge"].high == pytest.approx(1.0)
    assert windows["core"].high != LEGACY_WINDOWS["core"][1]


def test_a_window_records_where_its_boundary_came_from():
    """A boundary from a resolved fit and one from the fallback are different
    claims about the same plasma, and the numbers look identical."""
    fitted = resonant_windows(PedestalStub(method="eped_fit"))
    assert "eped_fit" in fitted["edge"].source

    fell_back = resonant_windows(
        PedestalStub(method="fallback", reason="width 0.2154 above the maximum",
                     inner_edge=None, position=0.85)
    )
    assert "fallback" in fell_back["edge"].source
    assert "width 0.2154" in fell_back["edge"].source
    assert fell_back["core"].high == pytest.approx(0.85)


def test_the_legacy_windows_are_available_but_never_the_default():
    legacy = resonant_windows(legacy=True)
    assert (legacy["core"].low, legacy["core"].high) == LEGACY_WINDOWS["core"]
    assert "legacy" in legacy["edge"].source
    with pytest.raises(ValueError, match="needs a PedestalTop"):
        resonant_windows()


def test_a_pedestal_on_another_coordinate_is_refused():
    with pytest.raises(ValueError, match="needs an equilibrium"):
        resonant_windows(PedestalStub(coordinate="rho_tor"))


def test_a_surface_on_the_boundary_belongs_to_both_neighbours():
    """Rather than being dropped by one of them."""
    windows = resonant_windows(PedestalStub(position=0.93, width=0.06))
    psi = np.array([0.90])
    assert windows["core"].mask(psi)[0]
    assert windows["edge"].mask(psi)[0]


# --- reducing a column --------------------------------------------------------


def test_a_complex_column_reduces_on_its_magnitude():
    """A complex mean depends on a gauge the file does not fix, so it would
    change with a convention rather than with the plasma."""
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    got = reduce_resonant(table["psi_n_rational"], table["Phi_res"], window=window)
    expected = np.sqrt(np.mean(np.abs(table["Phi_res"]) ** 2))
    assert got == pytest.approx(expected)
    # And it is not the magnitude of the complex mean, which the phase moves.
    assert got != pytest.approx(abs(np.mean(table["Phi_res"])))


def test_the_resonant_field_is_tesla_and_not_gauss():
    """The code this replaces multiplied by 1e4 inside the reduction, so the
    unit changed where nothing said so."""
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    value = rms_resonant_field(table["psi_n_rational"], table["Phi_res"], window=window)
    assert value == pytest.approx(np.sqrt(np.mean(np.abs(table["Phi_res"]) ** 2)))
    assert value < 1e-3  # tesla; the gauss figure would be order 1


def test_an_empty_window_is_not_an_error():
    """A run with four rational surfaces has regions that are legitimately
    empty, and that is a result rather than a failure."""
    table = resonant_table()
    empty = ResonantWindow("inner", 0.0, 0.2, "test")
    assert np.isnan(reduce_resonant(table["psi_n_rational"], table["Phi_res"], window=empty))


def test_each_statistic_reduces_as_it_says(  ):
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    magnitude = np.abs(table["I_res"])
    for statistic, expected in (
        ("max", magnitude.max()),
        ("mean", magnitude.mean()),
        ("sum", magnitude.sum()),
        ("rms", np.sqrt(np.mean(magnitude**2))),
    ):
        assert statistic in RESONANT_STATISTICS
        assert reduce_resonant(
            table["psi_n_rational"], table["I_res"], window=window, statistic=statistic
        ) == pytest.approx(expected)


def test_an_unknown_statistic_is_refused():
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    with pytest.raises(ValueError, match="statistic must be one of"):
        reduce_resonant(table["psi_n_rational"], table["Phi_res"],
                        window=window, statistic="median")


def test_a_column_of_the_wrong_length_is_refused():
    window = ResonantWindow("all", 0.0, 1.0, "test")
    with pytest.raises(ValueError, match="one entry per rational surface"):
        reduce_resonant(np.linspace(0, 1, 4), np.ones(3), window=window)


def test_non_finite_surfaces_are_dropped_rather_than_poisoning_the_result():
    psi = np.array([0.3, 0.5, 0.7])
    values = np.array([1.0, np.nan, 3.0])
    window = ResonantWindow("all", 0.0, 1.0, "test")
    assert reduce_resonant(psi, values, window=window, statistic="max") == pytest.approx(3.0)


# --- the metric table ---------------------------------------------------------


def test_the_metric_table_covers_every_column_and_window():
    table = resonant_table()
    windows = resonant_windows(legacy=True)
    metrics = resonant_metrics(table, windows=windows, columns=["Phi_res", "K_isl"])
    assert set(metrics) == {(c, w) for c in ("Phi_res", "K_isl") for w in windows}


def test_the_coordinates_are_not_reduced_as_if_they_were_quantities():
    """psi_n_rational, q_rational and m_rational say where a row is, not what
    it holds; an RMS of them is a number with no meaning."""
    table = resonant_table()
    metrics = resonant_metrics(table, windows=resonant_windows(legacy=True))
    reduced = {column for column, _ in metrics}
    assert not reduced & {"psi_n_rational", "q_rational", "m_rational"}
    assert "Phi_res" in reduced and "w_isl_v_crit" in reduced


def test_a_table_without_its_coordinate_is_refused():
    table = resonant_table()
    del table["psi_n_rational"]
    with pytest.raises(KeyError, match="no 'psi_n_rational'"):
        resonant_metrics(table, windows=resonant_windows(legacy=True))


# --- amplification ------------------------------------------------------------


def test_amplification_is_the_ratio_of_two_independent_reductions():
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    ratio = amplification_ratio(
        table["psi_n_rational"], table["Phi_res"],
        table["psi_n_rational"], table["Phi_res_v"], window=window,
    )
    case = rms_resonant_field(table["psi_n_rational"], table["Phi_res"], window=window)
    reference = rms_resonant_field(table["psi_n_rational"], table["Phi_res_v"], window=window)
    assert ratio == pytest.approx(case / reference)


def test_two_runs_need_not_share_a_rational_surface_set():
    """Two equilibria resonate in different places, and interpolating one onto
    the other's surfaces would invent a resonance where there is none."""
    case = resonant_table()
    reference = resonant_table(surfaces=(0.55, 0.79, 0.91))
    window = ResonantWindow("all", 0.0, 1.0, "test")
    ratio = amplification_ratio(
        case["psi_n_rational"], case["Phi_res"],
        reference["psi_n_rational"], reference["Phi_res"], window=window,
    )
    assert np.isfinite(ratio) and ratio > 0


def test_a_ratio_with_nothing_to_divide_by_is_not_a_ratio_of_one():
    """The code this replaces fell back to the vacuum column when the plasma
    one was missing, returning 1.0 and calling it no amplification."""
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    assert np.isnan(amplification_ratio(
        table["psi_n_rational"], table["Phi_res"],
        table["psi_n_rational"], np.zeros_like(table["Phi_res"]), window=window,
    ))
    empty = ResonantWindow("inner", 0.0, 0.2, "test")
    assert np.isnan(amplification_ratio(
        table["psi_n_rational"], table["Phi_res"],
        table["psi_n_rational"], table["Phi_res_v"], window=empty,
    ))


# --- rational surfaces from the equilibrium alone -----------------------------


def test_rational_surfaces_are_found_where_q_equals_m_over_n():
    psi, q = q_profile()
    found = find_rational_surfaces(psi, q, 1)
    np.testing.assert_allclose(found["q_rational"], found["m"], rtol=1e-12)
    for position, target in zip(found["psi_n_rational"], found["q_rational"]):
        assert np.interp(position, psi, q) == pytest.approx(target, abs=1e-6)


def test_the_surfaces_come_back_ordered_outward():
    psi, q = q_profile()
    found = find_rational_surfaces(psi, q, 2)
    assert np.all(np.diff(found["psi_n_rational"]) > 0)


def test_a_higher_toroidal_mode_resonates_more_often():
    psi, q = q_profile()
    assert len(find_rational_surfaces(psi, q, 2)["m"]) > len(
        find_rational_surfaces(psi, q, 1)["m"]
    )


def test_reversed_shear_resonates_twice_on_the_same_mode():
    """Both crossings are real surfaces; returning one would lose a resonance."""
    psi, q = q_profile(reversed_shear=True)
    found = find_rational_surfaces(psi, q, 2)
    modes, counts = np.unique(found["m"], return_counts=True)
    assert (counts == 2).any(), dict(zip(modes.tolist(), counts.tolist()))


def test_the_mode_range_can_be_narrowed():
    psi, q = q_profile()
    found = find_rational_surfaces(psi, q, 1, m_range=(3, 4))
    assert set(found["m"].tolist()) <= {3, 4}


def test_a_coordinate_that_does_not_increase_is_refused():
    psi, q = q_profile()
    with pytest.raises(ValueError, match="psi_norm must increase"):
        find_rational_surfaces(psi[::-1], q, 1)


def test_a_zero_toroidal_mode_is_refused():
    psi, q = q_profile()
    with pytest.raises(ValueError, match="q = m / n is undefined"):
        find_rational_surfaces(psi, q, 0)


def test_mismatched_lengths_are_refused():
    psi, q = q_profile()
    with pytest.raises(ValueError, match="q values"):
        find_rational_surfaces(psi, q[:-1], 1)
