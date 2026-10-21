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


def test_measured_vs_reconstructed_reports_each_side():
    import matplotlib.pyplot as plt
    import pandas as pd

    from vaft.plot.pipeline_comparison import pipeline_comparison_measured_vs_reconstructed

    points = pd.DataFrame({"side": ["legacy", "legacy", "current"], "family": ["probe"] * 3,
                           "measured": [1.0, 2.0, 1.0], "reconstructed": [1.5, 2.5, 1.0]})
    fig, ax = pipeline_comparison_measured_vs_reconstructed(points, "probe", scale=1.0, unit="T")
    texts = [t.get_text() for t in ax.texts]
    assert any("legacy archive: 2 channels, RMS(r - m) 0.5 T" in t for t in texts)
    assert any("current pipeline: 1 channels, RMS(r - m) 0 T" in t for t in texts)
    plt.close(fig)
