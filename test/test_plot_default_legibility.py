"""Defaults that made a correct plot unreadable.

Found by plotting real published data rather than a fixture, and about the
*unspecified* case: an explicit argument always decided correctly, so nothing
in the suite noticed.

* A geometry view went straight to `axes.legend`, bypassing the shared policy
  that replaces a legend with a count note past `LEGEND_MAX_ENTRIES`. Forty
  soft X-ray sight lines drew forty legend entries over the drawing (#764).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.plot.models import GeometryLayer, GeometryLayers
from vaft.plot.renderers.geometry import render_geometry_layers
from vaft.plot.style import LEGEND_MAX_ENTRIES, apply_legend


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _layers(count: int) -> GeometryLayers:
    return GeometryLayers(
        layers=[
            GeometryLayer(
                r=np.array([0.0, 1.0]), z=np.array([float(i), float(i) + 1.0]),
                label=f"channel {i}",
            )
            for i in range(count)
        ],
        title="geometry",
    )


def _count_note(axes) -> str | None:
    for text in axes.texts:
        if "traces" in text.get_text():
            return text.get_text()
    return None


class TestGeometryLegendFollowsThePolicy:
    def test_a_many_layer_view_summarises_instead_of_covering_itself(self):
        count = LEGEND_MAX_ENTRIES + 4
        _, axes = render_geometry_layers(_layers(count))
        assert axes.get_legend() is None
        assert _count_note(axes) == f"{count} traces"

    def test_a_few_layer_view_still_gets_its_legend(self):
        _, axes = render_geometry_layers(_layers(3))
        assert axes.get_legend() is not None
        assert _count_note(axes) is None

    def test_a_single_labelled_layer_still_gets_its_legend(self):
        """A lone labelled layer among unlabelled ones still needs naming.

        This is where geometry departs from the line policy, which drops the
        legend for a lone trace because the title already names it.
        """
        _, axes = render_geometry_layers(_layers(1))
        assert axes.get_legend() is not None

    def test_legend_true_still_forces_one_past_the_threshold(self):
        count = LEGEND_MAX_ENTRIES + 4
        _, axes = render_geometry_layers(_layers(count), legend=True)
        legend = axes.get_legend()
        assert legend is not None
        assert len(legend.get_texts()) == count

    def test_legend_false_still_suppresses(self):
        _, axes = render_geometry_layers(_layers(3), legend=False)
        assert axes.get_legend() is None
        assert _count_note(axes) is None


class TestTheLoneEntryDistinction:
    def test_a_lone_trace_gets_no_legend_by_default(self):
        _, axes = plt.subplots()
        axes.plot([0, 1], [0, 1], label="only")
        apply_legend(axes, legend=None)
        assert axes.get_legend() is None

    def test_lone_entry_keeps_it(self):
        _, axes = plt.subplots()
        axes.plot([0, 1], [0, 1], label="only")
        apply_legend(axes, legend=None, lone_entry=True)
        assert axes.get_legend() is not None
