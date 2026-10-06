"""The GUI player walks every kind of sequence through its generic control (issue #1380).

The Panel player is built for any slice-group control, so a plot that gains
one -- the spatial magnetics' dense ``time_index`` (phase 1) -- plays with no
GUI change.  These tests hold that seam for the dense-sample case; the stored
slices and the camera frames are held in ``test_gui_app.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

pn = pytest.importorskip("panel")

from vaft.gui.widgets import panel_controls  # noqa: E402
from vaft.omas import render_plot, sample_ods  # noqa: E402

#: Every flux loop of 39915 is flagged valid up to this sample and not after it.
LAST_VALID = 1999


@pytest.fixture(scope="module")
def ods():
    return sample_ods(39915)


@pytest.fixture
def drawn(ods):
    result = render_plot(
        "flux_loop_spatial_flux", ods, interactive=True, interaction_backend="none", selection="all",
    )
    yield result
    plt.close(result.figure)


def _player(laid_out, label_start="Play: Time sample"):
    box = next(w for w in laid_out if getattr(w, "name", "").startswith(label_start))
    return box.objects[0], box.objects[1]


def _texts(figure):
    return [text.get_text() for text in figure.findobj(matplotlib.text.Text)]


def test_the_player_starts_where_the_static_plot_is_drawn(drawn):
    sequence = next(c for c in drawn.state.controls if c.name == "time_index")
    player, _ = _player(panel_controls(drawn.state))
    low, high, _ = sequence.options
    assert (player.start, player.end) == (low, high)
    assert player.value == drawn.state["time_index"] == sequence.default == (high + 1) // 2


def test_playing_steps_the_samples_and_leaves_the_other_controls(drawn, ods):
    laid_out = panel_controls(drawn.state)
    player, interval = _player(laid_out)
    before = {name: drawn.state[name] for name in ("coordinate", "selection", "validity")}
    seen = []
    drawn.state.subscribe(lambda state: seen.append(state["time_index"]))
    for position in (100, 101, 102):
        player.value = position
    assert seen == [100, 101, 102]
    stamp = float(np.asarray(ods["magnetics.time"])[102]) * 1e3
    assert any(f"t = {stamp:.2f} ms (sample 102)" in text for text in _texts(drawn.figure))
    assert {name: drawn.state[name] for name in before} == before
    interval.value = 250
    assert player.interval == 250


def test_a_flagged_sample_is_played_and_drawn_by_the_validity_mode(ods):
    """Content validity, not state usability: the samples after 0.34 s stay in the playback."""
    result = render_plot(
        "flux_loop_spatial_flux", ods, interactive=True, interaction_backend="none",
        selection="all", validity="mask",
    )
    try:
        player, _ = _player(panel_controls(result.state))

        def points():
            return sum(
                int(np.isfinite(line.get_ydata()).sum())
                for axis in np.atleast_1d(result.axes).ravel() for line in axis.lines
            )

        player.value = LAST_VALID
        assert points() == 11
        player.value = LAST_VALID + 2
        assert result.state["time_index"] == LAST_VALID + 2 and points() == 0
        player.value = LAST_VALID
        assert points() == 11
    finally:
        plt.close(result.figure)


def test_a_pinned_instant_offers_no_player(ods):
    """time= chosen by the caller pins the state: no slider, so nothing to play."""
    result = render_plot("flux_loop_spatial_flux", ods, interactive=True, interaction_backend="none", time=0.31)
    try:
        laid_out = panel_controls(result.state)
        assert not any(getattr(w, "name", "").startswith("Play: ") for w in laid_out)
    finally:
        plt.close(result.figure)
