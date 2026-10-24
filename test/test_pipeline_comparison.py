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


def test_stability_histogram_normalises_each_side_and_reports_the_unstable_share():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_stability

    table = pd.DataFrame({"side": ["legacy"] * 4 + ["current"] * 2,
                          "value": [-2.0, 3.0, 4.0, 5.0, 6.0, 7.0]})
    fig, ax = pipeline_comparison_stability(table, quantity="w_t")
    heights = [patch.get_path().vertices[:, 1].max() for patch in ax.patches if hasattr(patch, "get_path")]
    assert max(heights) <= 1.0 + 1e-9  # fractions, not counts
    assert ax.get_yscale() == "linear" and ax.get_xscale() == "symlog"
    assert "legacy 25% (n=4)" in ax.get_title(loc="right")
    assert "current 0% (n=2)" in ax.get_title(loc="right")
    plt.close(fig)


def test_stability_overview_has_one_shared_legend():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_stability_overview

    w = pd.DataFrame({"side": ["legacy", "current"], "value": [1.0, -1.0]})
    d = pd.DataFrame({"side": ["legacy", "current"], "value": [10.0, -10.0]})
    fig, axes = pipeline_comparison_stability_overview(w, d)
    assert all(ax.get_legend() is None for ax in axes) and len(fig.legends) == 1
    plt.close(fig)
