"""Lane D figure functions (workflow/confinement_scaling/figures.py) on synthetic inputs."""

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

HERE = pathlib.Path(__file__).resolve().parents[1] / "workflow/confinement_scaling"


def _figures():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("lane_d_figures", HERE / "figures.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


def _closures():
    rows = [{"data": "primary:A", "model": "free", "mu_rho_imposed": np.nan, "a_i_p": 1.95, "a_b_t": 0.61,
             "a_p_net": -0.90, "se_i_p": 0.3, "se_b_t": 0.31, "se_p_net": 0.11,
             "mu_rho_completed": -13.0, "mu_rho_completed_se": 17.0, "one_plus_aP_over_se": 0.87}]
    for name, mu in (("bohm", -2.0), ("gyro_bohm", -3.0)):
        rows.append({"data": "primary:A", "model": name, "mu_rho_imposed": mu, "a_i_p": 1.6, "a_b_t": 0.1,
                     "a_p_net": -0.78, "se_i_p": 0.2, "se_b_t": 0.1, "se_p_net": 0.05})
    rows.append({"data": "primary:B_with_n_e", "model": "free", "mu_rho_imposed": np.nan, "a_i_p": 2.0,
                 "a_b_t": 0.2, "a_p_net": -1.0})
    nstx = pd.DataFrame([{"scaling": "NSTX2006L (Kaye 2006)", "a_i_p": 1.01, "a_b_t": 0.7, "a_p_net": -0.37,
                          "mu_rho_completed": -4.03},
                         {"scaling": "VEST primary:A", "a_i_p": 1.95, "a_b_t": 0.61, "a_p_net": -0.9,
                          "mu_rho_completed": -13.0}])
    return pd.DataFrame(rows), nstx


def test_exponent_table_keeps_one_data_set_and_flags_the_undetermined_index():
    figures = _figures()
    closures, nstx = _closures()
    table = figures.exponent_table(closures, nstx)
    assert list(table["kind"]) == ["vest_free", "vest_closure", "vest_closure", "reference", "reference"]
    assert table.loc[0, "mu_rho_undetermined"]
    assert table.loc[1, "mu_rho"] == -2.0  # an imposed closure reports its imposed value
    assert "VEST primary:A" not in set(table["label"])  # VEST rows of the NSTX file are not references


def test_exponent_figure_draws_without_error():
    figures = _figures()
    fig, axes = figures.exponent_figure(figures.exponent_table(*_closures()))
    assert len(axes) == 2 and "ASSUMED" in axes[1].get_xlabel()
    matplotlib.pyplot.close(fig)


def test_population_table_labels_vest_and_spherical_tokamaks():
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    figures = _figures()
    base = {c: np.nan for c in CONFINEMENT_COLUMNS}
    db5 = pd.DataFrame([{**base, "machine": m, "selected": sel, "tau_e_th_s": 0.05, "i_p_A": 1e6}
                        for m, sel in (("NSTX", True), ("JET", True), ("JET", False))])
    vest = pd.DataFrame([{**base, "machine": "VEST", "shot": 1, "time_s": 0.32, "tau_e_th_s": 1e-3,
                          "i_p_A": 8e4, "quality_status": "evaluated", "ip_rate_1_s": 1.0,
                          "ip_change_per_tau": 0.01, "dwdt_fraction": 0.1, "rule_finite": True},
                         {**base, "machine": "VEST", "shot": 1, "time_s": 0.33, "tau_e_th_s": 1e-3,
                          "i_p_A": 8e4, "quality_status": "evaluated", "ip_rate_1_s": 1.0,
                          "ip_change_per_tau": 0.5, "dwdt_fraction": 0.1, "rule_finite": True}])
    table = figures.population_table(db5, vest)
    assert list(table["population"]) == ["NSTX (DB5)", "conventional tokamaks (DB5)", figures.VEST_LABEL]


def _extra():
    sys.path.insert(0, str(HERE))
    try:
        spec = importlib.util.spec_from_file_location("lane_d_extra", HERE / "extra_scalings.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(HERE))
    return module


ROW = pd.DataFrame([{"i_p_A": 1.0e5, "b_t_T": 0.2, "r_geo_m": 0.4, "a_m": 0.25, "epsilon": 0.625,
                     "kappa": 1.6, "kappa_area": 1.5, "n_e_line_avg_m3": 1.0e19, "p_loss_W": 2.0e5,
                     "m_eff_amu": 1.0}])


def test_extra_scalings_match_their_si_forms():
    """The CGS forms of the papers against hand-converted SI coefficients."""
    x = _extra()
    q = 5e6 * 0.2 * 0.4 * 0.625**2 * 1.5 / 1.0e5
    assert x.q_cyl(ROW)[0] == pytest.approx(q)
    # eq. (3): 7.1e-22 * 1e-6 * 100^(1.04 + 2.04) = 1.0263e-21 in SI.
    assert x.neo_alcator(ROW)[0] == pytest.approx(1.0263e-21 * 1e19 * 0.25**1.04 * 0.4**2.04 * q**0.5, rel=1e-3)
    # eq. (6): 6.4e-8 * 1e6 * 1e-3 * 100^1.38 = 0.03683 with I_p in MA and P in MW.
    gl = 0.03683 * 0.1 * 0.2**-0.5 * 0.4**1.75 * 0.25**-0.37 * 1.6**0.5
    assert x.goldston_l(ROW)[0] == pytest.approx(gl, rel=1e-3)
    combined = (x.neo_alcator(ROW)[0] ** -2 + gl**-2) ** -0.5
    assert x.goldston_ohmic_l(ROW)[0] == pytest.approx(combined, rel=1e-3)
    assert combined < min(x.neo_alcator(ROW)[0], gl)
    it97 = 0.023 * 0.1**0.96 * 0.2**0.03 * 0.4**1.83 * 0.625**-0.06 * 1.6**0.64 * 1.0**0.4 * 1.0**0.2 * 0.2**-0.73
    assert x.iter97_l(ROW)[0] == pytest.approx(it97)


def test_extra_scalings_return_nan_for_missing_inputs():
    x = _extra()
    row = ROW.assign(n_e_line_avg_m3=np.nan)
    assert np.isnan(x.predict(row, "NeoAlcator")[0]) and np.isnan(x.predict(row, "ITER97L")[0])
    assert np.isfinite(x.predict(row, "Goldston84L")[0])  # no density term
    with pytest.raises(KeyError):
        x.predict(ROW, "nope")


def test_population_table_by_machine_still_highlights_vest():
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    figures = _figures()
    base = {c: np.nan for c in CONFINEMENT_COLUMNS}
    db5 = pd.DataFrame([{**base, "machine": "JET", "selected": True, "tau_e_th_s": 0.05}])
    vest = pd.DataFrame([{**base, "machine": "VEST", "shot": 1, "time_s": 0.32, "tau_e_th_s": 1e-3,
                          "i_p_A": 8e4, "quality_status": "evaluated", "ip_rate_1_s": 1.0,
                          "ip_change_per_tau": 0.01, "dwdt_fraction": 0.1, "rule_finite": True}])
    table = figures.population_table(db5, vest, grouping="machine", scope="all")
    assert list(table["population"]) == ["JET", figures.VEST_LABEL]


def _population_with_verdict():
    from vaft.data.public.schema import CONFINEMENT_COLUMNS

    base = {c: np.nan for c in CONFINEMENT_COLUMNS}
    rows = [{**base, "machine": "JET", "tau_e_th_s": 0.3, "i_p_A": 2e6, "p_loss_W": 5e6, "population": "JET"}]
    for i, verdict in enumerate((True, False, np.nan, "False")):
        rows.append({**base, "machine": "VEST", "tau_e_th_s": 1e-3 * (i + 1), "i_p_A": 1e5 * (i + 1),
                     "p_loss_W": 2e5, "population": "VEST (ohmic, this work)", "thomson_consistent": verdict})
    return pd.DataFrame(rows)


def test_thomson_inconsistent_marks_only_vest_rows_with_a_false_verdict():
    figures = _figures()
    table = _population_with_verdict()
    # True and NaN (no Thomson) are not marked; False and its CSV spelling are.
    assert list(figures.thomson_inconsistent(table)) == [False, False, True, False, True]
    assert not figures.thomson_inconsistent(table.drop(columns="thomson_consistent")).any()


def test_rings_sit_on_the_vest_points_vaft_plot_draws():
    from vaft.plot.population import confinement_population

    figures = _figures()
    table = _population_with_verdict()
    fig, ax = matplotlib.pyplot.subplots()
    confinement_population(table, x="i_p_A", y="tau_e_th_s", by="population",
                           highlight="VEST (ohmic, this work)", ax=ax)
    before = len(ax.collections)
    figures.mark_thomson_inconsistent(ax, table, "i_p_A", "tau_e_th_s")
    rings = ax.collections[-1].get_offsets()
    assert len(ax.collections) == before + 1
    # Every ring is centred on a point vaft.plot drew (it draws I_p in MA, not A).
    drawn = np.vstack([c.get_offsets() for c in ax.collections[:-1]])
    for ring in np.asarray(rings):
        assert np.isclose(drawn, ring).all(axis=1).any()
    assert ax.collections[-1].get_label() == f"{figures.THOMSON_RING_LABEL} (2)"
    matplotlib.pyplot.close(fig)


def test_slide_figures_use_the_slide_format():
    figures = _figures()
    table = _population_with_verdict()
    before = dict(matplotlib.rcParams)
    out = figures.slide_figures(table, figures.exponent_table(*_closures()))
    assert set(out) == {"tau_population", "tau_predicted_vs_measured", "exponents", "vest_h_factor"}
    for fig in out.values():
        width, height = fig.get_size_inches()
        assert width == pytest.approx(11.0) and height <= 5.8 + 1e-9
        matplotlib.pyplot.close(fig)
    # The rc context is gone afterwards: nothing of the slide format leaks.
    assert {k: v for k, v in matplotlib.rcParams.items() if before.get(k) != v} == {}
    with pytest.raises(ValueError, match="presentation format"):
        figures.slide_figures(table, figures.exponent_table(*_closures()), fmt=None)


def test_every_scaling_is_tagged_and_the_vest_h_figure_draws():
    figures = _figures()
    assert set(figures.SCALING_TAGS) == set(figures.ALL_SCALINGS)
    assert {db for db, _ in figures.SCALING_TAGS.values()} <= set(figures.DATABASE_COLOURS)
    assert figures.tagged_label("NSTX2006L").endswith("[single ST | L]")
    vest = pd.DataFrame([{**ROW.iloc[0].to_dict(), "tau_e_th_s": 1.5e-3 * k} for k in (1.0, 1.2, 0.8)])
    fig, ax = figures.vest_h_factor_figure(vest, ("ITER97L", "H98y2"))
    by_position = {tick: label.get_text() for tick, label in zip(ax.get_yticks(), ax.get_yticklabels())}
    # Drawn top to bottom in the order given: the first scaling sits highest.
    assert by_position[max(by_position)].startswith("ITER97-L") and "[multi | L]  n=3" in by_position[max(by_position)]
    assert by_position[min(by_position)].startswith("IPB98") and "[multi | H]" in by_position[min(by_position)]
    # Only the database classes drawn are in the legend.
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["multi-machine fit"]
    matplotlib.pyplot.close(fig)
