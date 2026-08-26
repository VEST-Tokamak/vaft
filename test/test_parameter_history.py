from __future__ import annotations

import matplotlib.pyplot as plt
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
