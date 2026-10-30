"""Slice pairing and paired statistics for the legacy-vs-current pipeline comparison."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vaft.validation.pipeline_comparison import (
    METRICS,
    metric_direction,
    pair_slices,
    paired_summary,
    population_summary,
)


def test_pairs_by_time_not_by_position():
    """8 legacy slices against 17 current ones: zipping would pair 0.316 with 0.311."""
    legacy = [0.316, 0.317, 0.318, 0.319, 0.320, 0.322, 0.325, 0.327]
    current = [0.311, 0.312, 0.313, 0.315, 0.316, 0.317, 0.318, 0.319, 0.320,
               0.321, 0.322, 0.323, 0.324, 0.325, 0.326, 0.327, 0.328]
    pairing = pair_slices(legacy, current)

    assert len(pairing.pairs) == 8
    for pair in pairing.pairs:
        assert pair.current_time == pytest.approx(pair.legacy_time)
        assert current[pair.current_index] == pytest.approx(legacy[pair.legacy_index])
    assert pairing.unpaired_legacy == ()
    assert len(pairing.unpaired_current) == 9


def test_unpaired_slices_are_kept_and_named():
    pairing = pair_slices([0.300, 0.310], [0.3002, 0.320])
    assert [(p.legacy_index, p.current_index) for p in pairing.pairs] == [(0, 0)]
    assert pairing.unpaired_legacy == (1,)
    assert pairing.unpaired_current == (1,)


def test_a_current_slice_goes_to_the_nearer_legacy_slice():
    """0.3004 is within tolerance of both 0.300 and 0.3006; the closer pair wins, once."""
    pairing = pair_slices([0.3000, 0.3006], [0.3004])
    assert [(p.legacy_index, p.current_index) for p in pairing.pairs] == [(1, 0)]
    assert pairing.unpaired_legacy == (0,)


def test_non_finite_times_never_pair():
    pairing = pair_slices([float("nan"), 0.3], [0.3, float("nan")])
    assert [(p.legacy_index, p.current_index) for p in pairing.pairs] == [(1, 0)]


def test_negative_tolerance_is_refused():
    with pytest.raises(ValueError):
        pair_slices([0.3], [0.3], tolerance=-1.0)


def test_paired_summary_reports_an_improvement_as_a_negative_shift():
    legacy = np.linspace(1.0, 2.0, 20)
    current = legacy * 0.5
    summary = paired_summary(legacy, current, better="lower")

    assert summary["n_pairs"] == 20
    assert summary["shift_median"] < 0
    low, high = summary["shift_ci"]
    assert low <= summary["shift_median"] <= high < 0
    assert summary["improved_fraction"] == 1.0
    assert summary["wilcoxon_p"] < 1e-3


def test_zero_directed_metric_compares_magnitudes():
    """ln ratios of -0.8 -> +0.1 are an improvement although the value went up."""
    legacy = np.full(10, -0.8) + np.linspace(0, 0.01, 10)
    current = np.full(10, 0.1)
    summary = paired_summary(legacy, current, better="zero")
    assert summary["shift_median"] < 0
    assert summary["improved_fraction"] == 1.0


def test_non_finite_pairs_are_dropped_and_counted():
    summary = paired_summary([1.0, float("nan"), 3.0], [0.5, 1.0, float("inf")], better="lower")
    assert summary["n_pairs"] == 1
    assert summary["n_dropped"] == 2
    assert math.isnan(summary["wilcoxon_p"])


def test_directionless_metric_reports_no_improvement_fraction():
    summary = paired_summary(np.arange(8.0), np.arange(8.0) + 1, better=None)
    assert summary["improved_fraction"] is None
    assert summary["shift_median"] == pytest.approx(1.0)


def test_paired_lengths_must_match():
    with pytest.raises(ValueError):
        paired_summary([1.0, 2.0], [1.0])


def test_population_summary_on_independent_samples():
    rng = np.random.default_rng(1)
    summary = population_summary(rng.normal(1.0, 0.1, 200), rng.normal(0.5, 0.1, 50), better="lower")
    assert summary["n_legacy"] == 200 and summary["n_current"] == 50
    assert summary["current_median"] < summary["legacy_median"]
    assert summary["mannwhitney_p"] < 1e-6


def test_every_metric_declares_a_known_direction():
    for name, spec in METRICS.items():
        assert spec["better"] in ("lower", "zero", None), name
        assert metric_direction(name) == spec["better"]
    with pytest.raises(KeyError):
        metric_direction("not_a_metric")


# --- renderers ------------------------------------------------------------------

def _table():
    import pandas as pd

    rows = []
    for k in range(6):
        pair = f"1:{k}"
        rows.append({"side": "legacy", "pair_id": pair, "probe": 0.2 + 0.01 * k, "ratio": -0.7})
        rows.append({"side": "current", "pair_id": pair, "probe": 0.07 + 0.01 * k, "ratio": 0.01})
    rows.append({"side": "current", "pair_id": None, "probe": 0.5, "ratio": 0.2})  # unpaired, still counted
    return pd.DataFrame(rows)


def test_distribution_counts_every_slice_of_a_side():
    import matplotlib.pyplot as plt

    from vaft.plot.pipeline_comparison import pipeline_comparison_distribution

    fig, ax = pipeline_comparison_distribution(_table(), "probe", log=True)
    texts = " ".join(t.get_text() for t in ax.texts)
    assert "n = 7" in texts and "n = 6" in texts
    plt.close(fig)


def test_paired_view_uses_magnitudes_for_a_zero_metric():
    import matplotlib.pyplot as plt

    from vaft.plot.pipeline_comparison import pipeline_comparison_paired

    fig, ax = pipeline_comparison_paired(_table(), "ratio", better="zero", log=True)
    offsets = ax.collections[0].get_offsets()
    assert len(offsets) == 6  # the unpaired slice is not a pair
    assert np.all(np.asarray(offsets)[:, 0] == pytest.approx(0.7))
    assert any("100% of 6 pairs" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_paired_view_without_pairs_draws_an_empty_axes():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_paired

    fig, ax = pipeline_comparison_paired(pd.DataFrame({"side": ["legacy"], "pair_id": [None], "x": [1.0]}), "x")
    assert len(ax.collections[0].get_offsets()) == 0
    plt.close(fig)


def test_scorecard_skips_metrics_without_a_direction():
    import matplotlib.pyplot as plt

    from vaft.plot.pipeline_comparison import pipeline_comparison_scorecard

    rows = [
        {"label": "probe", "better": "lower", "legacy_median": 0.2, "current_median": 0.07,
         "improved_fraction": 1.0, "n_pairs": 11, "wilcoxon_p": 1e-4},
        {"label": "W_t", "better": None, "legacy_median": 5.0, "current_median": 4.0},
    ]
    fig, ax = pipeline_comparison_scorecard(rows)
    assert [t.get_text() for t in ax.get_yticklabels()] == ["probe"]
    assert any("p < 1e-3" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_measured_vs_reconstructed_reports_each_side_in_display_units():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_measured_vs_reconstructed

    points = pd.DataFrame({"side": ["legacy", "legacy", "current"], "family": ["probe"] * 3,
                           "measured": [0.010, 0.020, 0.010], "reconstructed": [0.015, 0.025, 0.010]})
    fig, ax = pipeline_comparison_measured_vs_reconstructed(points, family="probe", format="legacy")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert "Legacy archive (n=2, MAE 5.00 mT)" in labels
    assert "Current pipeline (n=1, MAE 0.00 mT)" in labels
    assert ax.get_title() == "Poloidal probes"
    plt.close(fig)


def test_single_channel_family_reads_the_fit_table():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_measured_vs_reconstructed

    fit = pd.DataFrame({"side": ["legacy", "current"], "ip_measured": [80e3, 80e3],
                        "ip_reconstructed": [80e3, 84e3]})
    fig, ax = pipeline_comparison_measured_vs_reconstructed(fit, family="ip", format="legacy")
    assert "Current pipeline (n=1, MAE 4.00 kA)" in [t.get_text() for t in ax.get_legend().get_texts()]
    plt.close(fig)


def test_virial_view_counts_slices_it_cannot_show():
    """A non-positive beta_p on a log axis is counted, not silently dropped."""
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_virial

    table = pd.DataFrame({"side": ["legacy", "current", "current", "current"],
                          "beta_p_integral": [0.01, 0.2, -0.5, float("nan")],
                          "beta_p_pair_13": [0.02, 0.21, 0.2, 0.1]})
    fig, ax = pipeline_comparison_virial(table, quantity="beta_p", limits=(1e-3, 2.0), format="legacy")
    assert ax.get_xscale() == "log"
    assert "Current pipeline (n=3, 2 not shown)" in [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("not shown: legacy 0, current 2" in t.get_text() for t in ax.texts)
    plt.close(fig)


def test_virial_band_is_the_qa_criterion_only_where_one_exists():
    """beta_p shades the virial criterion (ln 2 of CRITERIA); l_i has no criterion and no band."""
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.collections import PolyCollection

    from vaft.plot.pipeline_comparison import pipeline_comparison_virial
    from vaft.validation._efit_study_criteria import CRITERIA

    table = pd.DataFrame({"side": ["legacy", "current"], "beta_p_integral": [0.01, 0.2],
                          "beta_p_pair_13": [0.02, 0.2], "li_volume": [0.7, 0.6], "li_pair_13": [0.65, 0.6]})
    fig, ax = pipeline_comparison_virial(table, quantity="beta_p", limits=(1e-3, 2.0), format="legacy")
    band = [c for c in ax.collections if isinstance(c, PolyCollection)]
    assert len(band) == 1 and band[0].get_label().startswith("QA:")
    vertices = band[0].get_paths()[0].vertices
    x0 = vertices[:, 0].min()
    assert np.isclose(vertices[vertices[:, 0] == x0, 1].max() / x0, np.exp(CRITERIA["virial_log_ratio_max"]))
    plt.close(fig)
    fig, ax = pipeline_comparison_virial(table, quantity="li", limits=(0.0, 1.6), format="legacy")
    assert not [c for c in ax.collections if isinstance(c, PolyCollection)]
    plt.close(fig)


def test_overview_is_a_two_by_two_with_one_legend():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_overview

    points = pd.DataFrame({"side": ["legacy", "current"] * 2, "family": ["probe", "probe", "loop", "loop"],
                           "measured": [0.01, 0.01, -0.002, -0.002], "reconstructed": [0.012, 0.01, -0.0021, -0.002]})
    virial = pd.DataFrame({"side": ["legacy", "current"], "beta_p_integral": [0.01, 0.2],
                           "beta_p_pair_13": [0.02, 0.2], "li_volume": [0.7, 0.6], "li_pair_13": [0.65, 0.6]})
    fig, axes = pipeline_comparison_overview(points, virial)
    assert axes.shape == (2, 2)
    assert all(ax.get_legend() is None for ax in axes.ravel())
    assert len(fig.legends) == 1
    with pytest.raises(ValueError):
        pipeline_comparison_overview(points, virial, limits_by_panel={"thomson": (0, 1)})
    plt.close(fig)


def test_stability_histogram_draws_one_normalised_step_per_pipeline_and_mode():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_stability

    table = pd.DataFrame({"side": ["legacy"] * 4 + ["current"] * 3, "n": [1, 1, 2, 2, 1, 2, 2],
                          "value": [-2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]})
    fig, ax = pipeline_comparison_stability(table, quantity="w_t")
    steps = {p.get_label(): p for p in ax.patches if p.get_label().count(":") == 1}
    assert set(steps) == {"legacy:1", "legacy:2", "current:1", "current:2"}
    assert steps["legacy:1"].get_edgecolor() != steps["current:1"].get_edgecolor()  # pipeline = colour
    assert steps["legacy:1"].get_linestyle() == steps["current:1"].get_linestyle()  # mode = line style
    assert steps["legacy:1"].get_linestyle() != steps["legacy:2"].get_linestyle()
    assert ax.get_xlabel() == r"$\delta W$"
    heights = [p.get_path().vertices[:, 1].max() for p in steps.values()]
    assert max(heights) <= 1.0 + 1e-9  # fractions, not counts
    assert ax.get_yscale() == "linear" and ax.get_xscale() == "symlog"
    assert ax.get_title() == "Ideal MHD stability (DCON)"
    plt.close(fig)


def test_stability_overview_keys_pipeline_and_mode_once():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_stability_overview

    w = pd.DataFrame({"side": ["legacy", "current", "legacy"], "n": [1, 1, 2], "value": [1.0, -1.0, 2.0]})
    d = pd.DataFrame({"side": ["legacy", "current"], "n": [1, 1], "value": [10.0, -10.0]})
    fig, axes = pipeline_comparison_stability_overview(w, d)
    assert all(ax.get_legend() is None for ax in axes) and len(fig.legends) == 2
    texts = [t.get_text() for legend in fig.legends for t in legend.get_texts()]
    assert sorted(texts) == sorted(["Legacy archive", "Current pipeline", "toroidal mode", "1", "2"])
    assert [ax.get_title() for ax in axes] == ["Ideal MHD stability (DCON)", "Resistive MHD stability (RDCON)"]
    assert not any("n=" in t or "n =" in t for t in texts)
    plt.close(fig)


def test_thomson_channels_coloured_by_pipeline_with_band_and_references():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import SIDE_STYLE, pipeline_comparison_thomson

    channels = pd.DataFrame({
        "side": ["legacy", "legacy", "current", "current", "current"],
        "pair_id": ["1:0", "1:0", "1:0", "1:0", None],
        "p_e": [100.0, 200.0, 100.0, 200.0, 300.0],
        "p_recon": [20.0, 30.0, 180.0, -5.0, 500.0],
    })
    fig, ax = pipeline_comparison_thomson(channels, format="legacy")
    offsets = {c.get_label(): c.get_offsets() for c in ax.collections if c.get_label() in ("Legacy archive", "Current pipeline")}
    assert len(offsets["Legacy archive"]) == 2
    assert len(offsets["Current pipeline"]) == 1  # the -5 Pa channel is counted, the unpaired one dropped
    colors = {c.get_label(): tuple(c.get_facecolor()[0][:3]) for c in ax.collections if c.get_label() in offsets}
    assert colors["Legacy archive"] != colors["Current pipeline"]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert r"$p_{EFIT}=2p_e$" in labels and r"$p_{EFIT}=p_e$" in labels
    assert any("not shown: legacy 0, current 1" in t.get_text() for t in ax.texts)
    assert ax.get_xscale() == "log" and ax.get_title() == "Matched Thomson channels"
    plt.close(fig)
    fig, ax = pipeline_comparison_thomson(channels, paired_only=False, format="legacy")
    assert len([c for c in ax.collections if c.get_label() == "Current pipeline"][0].get_offsets()) == 2
    plt.close(fig)


def test_thomson_error_bars_are_two_sigma_of_p_e_clipped_at_the_axis():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_thomson

    channels = pd.DataFrame({"side": ["legacy", "current"], "pair_id": ["1:0", "1:0"],
                             "n_e": [1e19, 1e19], "t_e": [50.0, 50.0], "n_e_error": [3e18, 9e18],
                             "t_e_error": [20.0, 45.0], "p_e": [100.0, 100.0], "p_recon": [40.0, 150.0]})
    fig, ax = pipeline_comparison_thomson(channels, format="legacy", limits=(10.0, 1e3))
    bars = {tuple(c.get_color()[0][:3]): c for c in ax.collections if type(c).__name__ == "LineCollection"}
    assert len(bars) == 2 and all(np.isclose(c.get_alpha(), 0.25) for c in bars.values())
    spans = sorted(tuple(segment[:, 0]) for c in bars.values() for segment in c.get_segments())
    legacy = 200.0 * np.hypot(0.3, 0.4)  # 2 p_e times sqrt((sn/n)^2 + (sT/T)^2) = 100 Pa
    assert np.allclose(spans[1], (200.0 - legacy, 200.0 + legacy))
    current = 200.0 * np.hypot(0.9, 0.9)  # larger than 2 p_e: the lower arm stops at the axis
    assert np.allclose(spans[0], (10.0, 200.0 + current))
    plt.close(fig)
    fig, ax = pipeline_comparison_thomson(channels, format="legacy", errorbars=False)
    assert not [c for c in ax.collections if type(c).__name__ == "LineCollection"]
    plt.close(fig)


def _onset_tables():
    import pandas as pd

    t = np.arange(-2.0, 2.01, 1.0)
    rows, traces = [], []
    for source, offset in (("public", 0.2), ("main", 0.1)):
        for shot in (1, 2):
            for family in ("probe", "loop"):
                rows += [{"source": source, "shot": shot, "family": family, "t_ms": x,
                          "value": offset * shot * (1 + (x > 0))} for x in t]
            rows += [{"source": source, "shot": shot, "family": "ip", "t_ms": x, "value": max(x, 0) * 10} for x in t]
        for family, channel, size in (("outboard", "B_a", 0.01), ("side", "B_b", 0.05),
                                      ("inboard_flux_loop", "Flux Loop - #1", 0.002), ("ip", "ip", 0.0)):
            traces += [{"source": source, "shot": 7, "family": family, "channel": channel, "t_ms": x,
                        "measured": size * (1 + offset), "vacuum": size} for x in t]
    return pd.DataFrame(rows), pd.DataFrame(traces)


def test_onset_population_draws_the_median_over_shots_per_pipeline():
    import matplotlib.pyplot as plt

    from vaft.plot.pipeline_comparison import SIDE_STYLE, pipeline_comparison_onset_population

    onset, _ = _onset_tables()
    fig, axes = pipeline_comparison_onset_population(onset)
    assert [ax.get_title() for ax in axes] == ["Plasma current", "Poloidal probes", "Flux loops"]
    assert axes[1].get_yscale() == "log" and axes[0].get_yscale() == "linear"
    lines = {line.get_label(): line for line in axes[1].get_lines() if "shots" in line.get_label()}
    legacy = lines["Legacy archive (2 shots)"]
    assert legacy.get_color() == SIDE_STYLE["legacy"]["color"]
    assert np.allclose(legacy.get_ydata(), [0.3, 0.3, 0.3, 0.6, 0.6])  # median of 0.2 and 0.4, doubled after onset
    assert all(any(line.get_xdata()[0] == 0.0 == line.get_xdata()[1] for line in ax.get_lines()) for ax in axes)
    plt.close(fig)


def test_onset_traces_pick_the_largest_channel_both_sides_carry():
    import matplotlib.pyplot as plt

    from vaft.plot.pipeline_comparison import pipeline_comparison_onset_traces

    _, traces = _onset_tables()
    fig, axes = pipeline_comparison_onset_traces(traces, shot=7)
    assert any(text.get_text() == "B_b" for text in axes[1].texts)
    assert any(text.get_text() == "#1" for text in axes[2].texts)  # instrument prefix dropped
    dashed = [line for line in axes[1].get_lines() if line.get_linestyle() == "--"]
    assert len(dashed) == 2 and np.allclose(dashed[0].get_ydata(), 50.0)  # vacuum in mT
    with pytest.raises(ValueError):
        pipeline_comparison_onset_traces(traces, shot=8)
    plt.close(fig)


def test_onset_scatter_puts_measured_against_vacuum_per_family():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_onset_scatter

    points = pd.DataFrame({
        "source": ["public", "main"] * 3,
        "family": ["ip", "ip", "probe", "probe", "loop", "loop"],
        "t_ms": [-4.0] * 6,
        "measured": [0.5, -0.2, 0.012, 0.010, -0.003, -0.002],  # ip in kA, the rest SI
        "vacuum": [0.0, 0.0, 0.010, 0.010, -0.002, -0.002],
    })
    fig, axes = pipeline_comparison_onset_scatter(points)
    assert [ax.get_title() for ax in axes] == ["Plasma current", "Poloidal probes", "Flux loops"]
    assert all(ax.get_ylabel().startswith("vacuum (coil + eddy)") for ax in axes)
    ip = [c for c in axes[0].collections if c.get_offsets().shape[0]]
    assert sorted(o[0] for c in ip for o in c.get_offsets()) == [-0.2, 0.5]  # kA stays kA
    probe = [c.get_offsets() for c in axes[1].collections]
    assert any(np.allclose(o, [[12.0, 10.0]]) for o in probe)  # mT
    assert len(fig.legends) == 1 and all(ax.get_legend() is None for ax in axes)
    with pytest.raises(ValueError):
        pipeline_comparison_onset_scatter(points, limits_by_panel={"dia": (0, 1)})
    plt.close(fig)


def test_onset_scatter_crosses_what_the_input_qa_rejects_at_one_instant():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_onset_scatter

    points = pd.DataFrame({
        "source": ["public", "public", "main", "main", "public"],
        "family": ["probe", "probe", "probe", "ip", "probe"],
        "t_ms": [-1.0, -1.0, -1.0, -1.0, -4.0],
        "measured": [0.010, 0.300, 0.011, 0.2, 0.05],
        "vacuum": [0.010, 0.002, 0.010, 0.0, 0.05],
        "qa_code": [0, -2, 0, np.nan, -2],
    })
    fig, axes = pipeline_comparison_onset_scatter(points, t_ms=-1.0)
    crosses = [c for c in axes[1].collections if c.get_offsets().shape[0] and
               c.get_paths()[0].vertices.shape[0] < 10 and c.get_alpha() == 0.7]
    assert len(crosses) == 1 and np.allclose(crosses[0].get_offsets(), [[300.0, 2.0]])  # the -4 ms row is gone
    assert any(t.get_text() == "input QA rejects: legacy 1, current 0" for t in axes[1].texts)
    assert "rejected by the input QA" in [t.get_text() for t in fig.legends[0].get_texts()]
    assert "-1 ms" in fig._suptitle.get_text()
    with pytest.raises(ValueError):
        pipeline_comparison_onset_scatter(points, t_ms=-8.0)
    plt.close(fig)


def test_quality_checks_are_four_named_panels_with_one_key():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import QUALITY_CHECK_TITLES, pipeline_comparison_quality_checks

    onset = pd.DataFrame({"source": ["public", "main", "main"], "family": ["probe"] * 3, "t_ms": [-1.0] * 3,
                          "measured": [0.01, 0.011, 0.5], "vacuum": [0.01, 0.01, 0.0], "qa_code": [0, 0, -2]})
    points = pd.DataFrame({"side": ["legacy", "current"], "family": ["probe", "probe"],
                           "measured": [0.01, 0.01], "reconstructed": [0.012, 0.01]})
    virial = pd.DataFrame({"side": ["legacy", "current"], "beta_p_integral": [0.01, 0.2],
                           "beta_p_pair_13": [0.02, 0.2]})
    channels = pd.DataFrame({"side": ["legacy", "current"], "pair_id": ["1:0", "1:0"],
                             "p_e": [100.0, 100.0], "p_recon": [40.0, 150.0]})
    fig, axes = pipeline_comparison_quality_checks(onset, points, virial, channels,
                                                   limits_by_panel={"beta_p": (1e-3, 2.0)})
    assert axes.shape == (2, 2)
    assert [ax.get_title() for ax in axes.ravel()] == list(QUALITY_CHECK_TITLES)
    assert all(ax.title.get_fontweight() == "bold" for ax in axes.ravel())
    assert axes[0, 0].get_ylabel() == "vacuum field model [mT]"
    assert all(ax.get_legend() is None for ax in axes.ravel()) and len(fig.legends) == 1
    keys = [t.get_text() for t in fig.legends[0].get_texts()]
    assert "QA band (c, d)" in keys and "rejected by the input QA (a)" not in keys
    assert not [t for t in axes[0, 0].texts if "input QA" in t.get_text()]  # left out, not counted
    drawn = np.vstack([c.get_offsets() for c in axes[0, 0].collections if c.get_offsets().shape[0]])
    assert not np.any(np.isclose(drawn[:, 0], 500.0))
    plt.close(fig)
    fig, axes = pipeline_comparison_quality_checks(onset, points, virial, channels, show_rejected=True,
                                                   limits_by_panel={"beta_p": (1e-3, 2.0)})
    assert "rejected by the input QA (a)" in [t.get_text() for t in fig.legends[0].get_texts()]
    assert any(t.get_text().startswith("input QA rejects: legacy 0, current 1") for t in axes[0, 0].texts)
    with pytest.raises(ValueError):
        pipeline_comparison_quality_checks(onset, points, virial, channels, limits_by_panel={"li": (0, 1)})
    plt.close(fig)


def test_quality_checks_carry_no_not_shown_counts():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_quality_checks

    onset = pd.DataFrame({"source": ["public", "main"], "family": ["probe"] * 2, "t_ms": [-1.0] * 2,
                          "measured": [0.01, 0.9], "vacuum": [0.01, 0.0]})
    points = pd.DataFrame({"side": ["legacy", "current"], "family": ["probe", "probe"],
                           "measured": [0.01, 0.01], "reconstructed": [0.012, 0.01]})
    virial = pd.DataFrame({"side": ["legacy", "current"], "beta_p_integral": [0.01, -0.2],
                           "beta_p_pair_13": [0.02, 0.2]})
    channels = pd.DataFrame({"side": ["legacy", "current"], "pair_id": ["1:0", "1:0"],
                             "p_e": [100.0, 100.0], "p_recon": [40.0, -1.0]})
    fig, axes = pipeline_comparison_quality_checks(onset, points, virial, channels,
                                                   limits_by_panel={"onset": (-20, 20), "beta_p": (1e-3, 2.0)})
    assert not [t for ax in axes.ravel() for t in ax.texts if t.get_text().startswith("not shown")]
    plt.close(fig)
