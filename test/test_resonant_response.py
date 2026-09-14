"""Resonant-response reductions: windows, magnitudes, ratios."""

from __future__ import annotations

import numpy as np
import pytest
from resonant_table_fixtures import PedestalStub, q_profile, resonant_table

from vaft.process.equilibrium import find_rational_surfaces
from vaft.process.perturbation import (
    LEGACY_WINDOWS,
    RESONANT_RESPONSE_COLUMNS,
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
    """The bounds are written out here rather than compared with the constant
    they came from: reproducing an old number is the only reason they exist,
    and a test that checks the function against its own table would not
    notice them changing."""
    legacy = resonant_windows(legacy=True)
    assert (legacy["core"].low, legacy["core"].high) == (0.0, 0.8)
    assert (legacy["edge"].low, legacy["edge"].high) == (0.8, 0.95)
    assert (legacy["total"].low, legacy["total"].high) == (0.0, 1.0)
    assert "legacy" in legacy["edge"].source
    with pytest.raises(ValueError, match="needs a PedestalTop"):
        resonant_windows()


def test_a_pedestal_on_another_coordinate_is_refused():
    with pytest.raises(ValueError, match="needs an equilibrium"):
        resonant_windows(PedestalStub(coordinate="rho_tor"))


def test_an_object_that_never_declared_a_coordinate_is_refused(tmp_path):
    """Defaulting to psi_norm would turn "it did not say" into "it is
    psi_norm" -- the conflation pedestal_top raises rather than make."""

    class Bare:
        inner_edge, position, method, reason = 0.8, 0.85, "?", ""

    with pytest.raises(ValueError, match="declares no radial coordinate"):
        resonant_windows(Bare())


@pytest.mark.parametrize("boundary", [-0.35, 1.3, float("nan")])
def test_a_boundary_outside_the_plasma_is_refused(boundary):
    """It builds windows that still look fine: a boundary of 1.3 makes "core"
    the whole plasma and "edge" empty, so a caller gets a finite, plausible
    number labelled edge that is the total reduction."""
    with pytest.raises(ValueError, match="outside the normalized flux range"):
        resonant_windows(PedestalStub(inner_edge=boundary))


def test_the_windows_work_with_the_real_pedestal_class():
    """The stub reads four attributes; this checks the contract against the
    class that actually produces them."""
    from vaft.process.profile import PedestalTop

    pedestal = PedestalTop(position=0.93, method="eped_fit", quantity="p_total",
                           coordinate="psi_norm", width=0.06)
    windows = resonant_windows(pedestal)
    assert windows["core"].high == pytest.approx(pedestal.inner_edge)
    assert "eped_fit" in windows["edge"].source


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


def test_a_signed_real_column_keeps_its_sign():
    """A rotation profile that changes sign has a mean near zero; rectifying
    it would report a large positive rotation the plasma does not have."""
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    omega = table["omega_E_rational"]
    assert omega.min() < 0 < omega.max()
    assert reduce_resonant(
        table["psi_n_rational"], omega, window=window, statistic="mean"
    ) == pytest.approx(float(np.mean(omega)))
    assert reduce_resonant(
        table["psi_n_rational"], omega, window=window, statistic="min"
    ) == pytest.approx(float(np.min(omega)))


def test_the_metric_table_passes_its_statistic_through():
    table = resonant_table()
    windows = resonant_windows(legacy=True)
    for statistic in ("max", "min", "mean"):
        metrics = resonant_metrics(
            table, windows=windows, columns=["K_isl"], statistic=statistic
        )
        direct = reduce_resonant(
            table["psi_n_rational"], table["K_isl"],
            window=windows["total"], statistic=statistic,
        )
        assert metrics[("K_isl", "total")] == pytest.approx(direct)
    assert len({
        resonant_metrics(table, windows=windows, columns=["K_isl"], statistic=s)[
            ("K_isl", "total")
        ]
        for s in ("max", "min", "mean")
    }) == 3


def test_amplification_passes_its_statistic_through():
    table = resonant_table()
    window = ResonantWindow("all", 0.0, 1.0, "test")
    ratios = {
        s: amplification_ratio(
            table["psi_n_rational"], table["Phi_res"],
            table["psi_n_rational"], table["Phi_res_v"], window=window, statistic=s,
        )
        for s in ("rms", "max", "mean")
    }
    assert len(set(ratios.values())) == 3
    assert ratios["max"] == pytest.approx(
        reduce_resonant(table["psi_n_rational"], table["Phi_res"], window=window, statistic="max")
        / reduce_resonant(table["psi_n_rational"], table["Phi_res_v"], window=window, statistic="max")
    )


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


def test_only_the_response_columns_are_reduced_by_default():
    """A real table carries twenty-five columns and most are not a response:
    rho_rational is a coordinate, T_e_rational is the equilibrium sampled at
    the surfaces. An RMS of either is a number that reads like a metric."""
    table = resonant_table()
    metrics = resonant_metrics(table, windows=resonant_windows(legacy=True))
    reduced = {column for column, _ in metrics}
    assert reduced == set(RESONANT_RESPONSE_COLUMNS) & set(table)
    assert not reduced & {
        "psi_n_rational", "q_rational", "m_rational", "rho_rational",
        "rho1_rational", "q1_rational", "T_e_rational", "n_e_rational",
        "area_rational", "dqdpsi_n_rational", "omega_E_rational",
    }
    assert "Phi_res" in reduced and "w_isl_v_crit" in reduced


def test_a_column_that_is_not_a_response_can_still_be_asked_for():
    table = resonant_table()
    metrics = resonant_metrics(
        table, windows=resonant_windows(legacy=True), columns=["rho_rational"]
    )
    assert set(metrics) == {("rho_rational", w) for w in ("core", "edge", "total")}


def test_a_table_with_rational_surfaces_but_no_response_is_refused():
    """A reconstruction run carries q_rational and nothing perturbed; reducing
    what it does have returns a plausible table holding no resonant physics."""
    table = resonant_table()
    reconstruction = {
        k: table[k] for k in ("psi_n_rational", "q_rational", "dqdpsi_n_rational")
    }
    with pytest.raises(KeyError, match="none of the resonant response columns"):
        resonant_metrics(reconstruction, windows=resonant_windows(legacy=True))


def test_a_column_that_is_not_there_is_named():
    table = resonant_table()
    with pytest.raises(KeyError, match=r"no \['nonesuch'\]"):
        resonant_metrics(
            table, windows=resonant_windows(legacy=True), columns=["nonesuch"]
        )


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
    """On reversed shear, where generation order is by mode number and the
    two crossings of one mode sit at opposite ends of the plasma, so the sort
    is the only thing putting them in radial order."""
    psi, q = q_profile(reversed_shear=True)
    # n=4 so several modes resonate twice and their crossings interleave:
    # generated by mode, m=5 gives 0.19 and 0.71 before m=6 gives 0.06, so
    # generation order is not radial order and the sort has to do the work.
    found = find_rational_surfaces(psi, q, 4)
    assert np.all(np.diff(found["psi_n_rational"]) > 0)
    assert found["m"][0] > found["m"][1], "the modes must interleave for this to bite"
    monotonic_psi, monotonic_q = q_profile()
    assert np.all(np.diff(find_rational_surfaces(monotonic_psi, monotonic_q, 2)["psi_n_rational"]) > 0)


def test_a_root_exactly_on_a_grid_point_is_one_surface():
    """It shows up as two sign changes, +1 -> 0 and 0 -> -1, both
    interpolating to the same point; counted twice it would double its weight
    in every reduction downstream."""
    found = find_rational_surfaces([0.0, 0.25, 0.5, 0.75, 1.0],
                                   [1.0, 1.5, 2.0, 2.5, 3.0], 1)
    assert found["m"].tolist() == [1, 2, 3]
    np.testing.assert_allclose(found["psi_n_rational"], [0.0, 0.5, 1.0])


def test_a_tangent_root_is_found_once():
    psi = np.linspace(0.0, 1.0, 101)
    found = find_rational_surfaces(psi, 2.0 + (psi - 0.5) ** 2, 1)
    assert found["m"].tolist() == [2]
    assert found["psi_n_rational"][0] == pytest.approx(0.5)


def test_a_flat_resonant_interval_is_one_surface():
    found = find_rational_surfaces(np.linspace(0.0, 1.0, 7),
                                   [1.0, 1.5, 2.0, 2.0, 2.0, 2.5, 3.0], 1)
    assert found["m"].tolist() == [1, 2, 3]


def test_accuracy_degrades_on_a_grid_that_is_not_refined_at_the_surfaces():
    """The 1e-9 agreement with GPEC is a property of GPEC's own adaptively
    refined grid, not of this routine; on a uniform grid the error is that of
    linear interpolation and is worst where q is steepest."""
    fine = np.linspace(0.0, 1.0, 20001)
    q_fine = 1.05 + 4.55 * fine**2
    truth = find_rational_surfaces(fine, q_fine, 1)["psi_n_rational"]
    errors = []
    for points in (129, 513):
        coarse = np.linspace(0.0, 1.0, points)
        got = find_rational_surfaces(coarse, 1.05 + 4.55 * coarse**2, 1)["psi_n_rational"]
        errors.append(np.abs(got - truth).max())
    assert errors[0] > errors[1]          # finer is better
    assert errors[1] < 0.25 * errors[0]   # and roughly quadratically so


def test_an_inverted_mode_range_is_refused():
    psi, q = q_profile()
    with pytest.raises(ValueError, match="which is empty"):
        find_rational_surfaces(psi, q, 1, m_range=(5, 2))


def test_a_two_dimensional_profile_is_refused():
    with pytest.raises(ValueError, match="one dimensional"):
        find_rational_surfaces(np.ones((2, 3)), np.ones((2, 3)), 1)


def test_a_fractional_mode_number_is_refused():
    psi, q = q_profile()
    with pytest.raises(ValueError, match="whole toroidal mode number"):
        find_rational_surfaces(psi, q, 2.7)


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
