from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest

from vaft.plot import plot_parameter_history


def test_plot_parameter_history_supports_multiple_columns_and_existing_axes():
    frame = pd.DataFrame({"shot": [1, 2], "ip_A": [3.0, 4.0], "q_95": [5.0, 6.0]})
    figure, axes = plt.subplots()

    returned_figure, returned_axes = plot_parameter_history(
        frame, y=("ip_A", "q_95"), ax=axes
    )

    assert returned_figure is figure
    assert returned_axes is axes
    assert len(axes.lines) == 2
    assert axes.get_legend() is not None
    plt.close(figure)


def test_plot_parameter_history_validates_columns_and_types():
    frame = pd.DataFrame({"shot": [1], "label": ["bad"]})
    with pytest.raises(ValueError, match="missing"):
        plot_parameter_history(frame, y="q_95")
    with pytest.raises(TypeError, match="numeric"):
        plot_parameter_history(frame, y="label")


def test_plot_parameter_history_accepts_empty_canonical_frame():
    frame = pd.DataFrame(columns=["shot", "q_95"])
    figure, axes = plot_parameter_history(frame, y="q_95")
    assert len(axes.lines) == 1
    plt.close(figure)


def test_shot_axis_dates_label_observed_positions_without_repositioning_rows():
    frame = pd.DataFrame({
        "shot": [10, 10, 11, 12],
        "pulse_time_begin": ["2024-01-01T10:00:00", "2024-01-01T10:00:00", None, "2024-05-01T10:00:00"],
        "q_95": [1.0, 2.0, 3.0, 4.0],
    })
    figure, axes = plot_parameter_history(frame, y="q_95", secondary_x="date")
    top = axes.child_axes[0]
    figure.canvas.draw()
    assert axes.lines[0].get_xdata().tolist() == [10, 10, 11, 12]
    assert axes.lines[0].get_ydata().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert top.get_xticks().tolist() == [10, 12]
    assert [label.get_text() for label in top.get_xticklabels()] == ["Jan 2024", "May 2024"]
    plt.close(figure)


def test_date_axis_uses_actual_irregular_intervals_and_shot_labels():
    frame = pd.DataFrame({
        "shot": [10, 11, 12],
        "pulse_time_begin": ["2024-01-01", "2024-01-02", "2024-04-01"],
        "q_95": [1.0, 2.0, 3.0],
    })
    figure, axes = plot_parameter_history(frame, y="q_95", x="date", secondary_x="shot")
    positions = np.asarray(axes.lines[0].get_xdata(), dtype=float)
    np.testing.assert_allclose(positions, mdates.date2num(pd.to_datetime(frame["pulse_time_begin"])))
    assert positions[2] - positions[1] > 80 * (positions[1] - positions[0])
    top = axes.child_axes[0]
    figure.canvas.draw()
    np.testing.assert_allclose(top.get_xticks(), positions)
    assert [label.get_text() for label in top.get_xticklabels()] == ["10", "11", "12"]
    plt.close(figure)


def test_date_axis_skips_missing_dates_without_aggregating():
    frame = pd.DataFrame({
        "shot": [10, 10, 11, 12],
        "pulse_time_begin": ["2024-01-01", "2024-01-01", None, "2024-04-01"],
        "q_95": [1.0, 2.0, 3.0, 4.0],
    })
    figure, axes = plot_parameter_history(frame, y="q_95", x="date")
    assert axes.lines[0].get_ydata().tolist() == [1.0, 2.0, 4.0]
    plt.close(figure)


def test_date_axis_requires_recorded_dates_and_rejects_conflicting_labels():
    frame = pd.DataFrame({"shot": [10, 11], "pulse_time_begin": [None, None], "q_95": [1.0, 2.0]})
    with pytest.raises(ValueError, match="pulse_time_begin"):
        plot_parameter_history(frame, y="q_95", secondary_x="date")
    with pytest.raises(ValueError, match="pulse_time_begin"):
        plot_parameter_history(frame, y="q_95", x="date")

    frame["pulse_time_begin"] = ["2024-01-01", "2024-01-02"]
    frame["shot"] = [10, 10]
    with pytest.raises(ValueError, match="different secondary-axis values"):
        plot_parameter_history(frame, y="q_95", secondary_x="date")


def test_arbitrary_numeric_x_date_column_and_show(monkeypatch):
    frame = pd.DataFrame({
        "shot": [10, 11], "time_s": [0.2, 0.4],
        "acquired": ["2024-01-01", "2024-01-05"], "q_95": [1.0, 2.0],
    })
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    figure, axes = plt.subplots()
    returned_figure, returned_axes = plot_parameter_history(
        frame, y="q_95", x="time_s", secondary_x="date", date_column="acquired", ax=axes, show=True
    )
    assert (returned_figure, returned_axes) == (figure, axes)
    assert axes.lines[0].get_xdata().tolist() == [0.2, 0.4]
    assert shown == [True]
    plt.close(figure)


def test_repeated_numeric_x_across_shots_labels_every_shot_at_that_position():
    """A fixed-time preset puts every shot at the same time_s: legal, not contradictory."""
    frame = pd.DataFrame({
        "shot": [39915, 39916], "time_s": [0.30, 0.30], "q_95": [5.0, 6.0],
        "pulse_time_begin": ["2022-03-02T10:00:00", "2022-03-02T11:00:00"],
    })
    figure, axes = plot_parameter_history(frame, y="q_95", x="time_s", secondary_x="shot")
    top = axes.child_axes[0]
    figure.canvas.draw()
    assert top.get_xticks().tolist() == [0.30]
    assert [label.get_text() for label in top.get_xticklabels()] == ["39915, 39916"]
    plt.close(figure)

    figure, axes = plot_parameter_history(frame, y="q_95", x="time_s", secondary_x="date")
    top = axes.child_axes[0]
    figure.canvas.draw()
    assert axes.lines[0].get_xdata().tolist() == [0.30, 0.30]
    assert [label.get_text() for label in top.get_xticklabels()] == ["Mar 02 10:00, Mar 02 11:00"]
    plt.close(figure)


def test_mixed_timezone_offsets_preserve_actual_elapsed_time():
    frame = pd.DataFrame({
        "shot": [10, 11],
        "pulse_time_begin": ["2024-01-01T12:00:00+09:00", "2024-01-01T05:00:00+00:00"],
        "q_95": [1.0, 2.0],
    })
    figure, axes = plot_parameter_history(frame, y="q_95", x="date", secondary_x="shot")
    positions = np.asarray(axes.lines[0].get_xdata(), dtype=float)
    assert (positions[1] - positions[0]) * 24 == pytest.approx(2.0)
    plt.close(figure)
