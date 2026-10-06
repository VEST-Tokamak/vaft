"""``vaft.diagram.magnetic_island(..., animation=True)``: a phase scan or a rigid rotation (#1053).

The animated form is the static diagram evaluated at every state of one
trajectory -- the same ``IslandModel``, so the same topology and width -- and
it returns the animation result of ``plot_*(..., animation=True)`` (#1049).
These are the contract tests for a **phase-driven** and a **time-driven**
diagram; no PyAV is needed (frames and ``.gif`` only).
"""

from __future__ import annotations

import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from vaft.diagram import magnetic_island

DPI = 30


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_a_phase_scan_has_one_state_per_phase_in_order():
    phases = np.linspace(0.0, 2 * np.pi, 6, endpoint=False)
    movie = magnetic_island(m=2, n=1, phase=phases, animation=True, dpi=DPI)
    assert len(movie) == 6
    driver = movie.metadata["driver"]
    assert (driver["name"], driver["coordinate"], driver["unit"]) == ("phase", "phase", "rad")
    assert driver["values"] == pytest.approx(list(phases))
    assert movie.metadata["plot"] == "magnetic_island"


def test_every_state_is_the_static_diagram_at_that_phase():
    phases = [0.0, 0.9, 2.1]
    movie = magnetic_island(m=3, n=2, width=0.16, phase=phases, projection="top", animation=True)
    for position, phase in enumerate(phases):
        state = movie._model(position)
        assert state == magnetic_island(m=3, n=2, width=0.16, phase=phase, projection="top").scene


def test_a_rigid_rotation_advances_the_phase_from_the_physics_not_the_picture():
    times = np.linspace(1e-3, 2e-3, 5)
    frequency, phase0 = 2.5e3, 0.4
    movie = magnetic_island(
        m=2, n=1, phase=phase0, time=times, rotation_frequency=frequency, animation=True, dpi=DPI,
    )
    driver = movie.metadata["driver"]
    assert (driver["name"], driver["unit"]) == ("time", "s")
    assert driver["values"] == pytest.approx(list(times))
    assert movie.metadata["options"]["rotation_frequency"] == frequency
    for position, t in enumerate(times):
        expected = (phase0 + 2 * math.pi * frequency * (t - times[0])) % (2 * math.pi)
        assert movie._model(position) == magnetic_island(m=2, n=1, phase=expected).scene


@pytest.mark.parametrize("projection", ["poloidal", "top", "3d"])
def test_frames_are_one_per_state_and_one_size(projection):
    movie = magnetic_island(
        m=2, n=1, phase=np.linspace(0, np.pi, 3), projection=projection, animation=True, dpi=DPI,
    )
    frames = list(movie.frames())
    assert len(frames) == 3 and {f.shape for f in frames} == {frames[0].shape}
    assert frames[0].dtype == np.uint8


def test_a_gif_and_its_sidecar(tmp_path):
    from PIL import Image

    movie = magnetic_island(m=2, n=1, phase=np.linspace(0, np.pi, 4), animation=True, fps=4, dpi=DPI)
    path = movie.save(tmp_path / "island.gif")
    with Image.open(path) as image:
        assert image.n_frames == 4
    sidecar = json.loads((tmp_path / "island.gif.json").read_text())
    assert sidecar["driver"]["name"] == "phase" and sidecar["presentation"]["fps"] == 4


def test_the_static_call_is_unchanged():
    diagram = magnetic_island(m=2, n=1, phase=0.3)
    assert type(diagram).__name__ == "Diagram" and diagram.name == "magnetic_island_poloidal"


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"phase": np.linspace(0, 1, 3)}, "pass animation=True"),
        ({"time": [0.0, 1e-3], "rotation_frequency": 1e3}, "pass animation=True"),
        ({"fps": 10}, "pass animation=True"),
        ({"phase": 0.0, "animation": True}, "one phase is a static diagram"),
        ({"phase": [0.5], "animation": True}, "one phase is a static diagram"),
        ({"phase": [], "animation": True}, "trajectory is empty"),
        ({"phase": [0.0, np.nan], "animation": True}, "non-finite"),
        ({"phase": np.linspace(0, 1, 3), "time": [0.0, 1e-3], "animation": True}, "two trajectories"),
        ({"time": [0.0, 1e-3], "animation": True}, "needs rotation_frequency"),
        ({"phase": np.linspace(0, 1, 3), "rotation_frequency": 1e3, "animation": True}, "give time="),
        ({"phase": np.linspace(0, 1, 3), "animation": True, "fps": 0}, "fps must be a positive"),
        ({"phase": np.linspace(0, 1, 3), "animation": True, "fps": 5, "duration": 1}, "not both"),
        ({"phase": np.linspace(0, 1, 3), "m": 2, "n": 2, "animation": True}, "lowest terms"),
        ({"phase": np.zeros((2, 2)), "animation": True}, "one-dimensional"),
        ({"time": [0.0, np.nan], "rotation_frequency": 1e3, "animation": True}, "time trajectory holds a non-finite"),
        ({"time": [0.0, 1e-3], "rotation_frequency": "fast", "animation": True}, "frequency in Hz"),
        ({"frame_label": False}, "pass animation=True"),
    ],
)
def test_ambiguous_or_invalid_requests_are_refused(options, match):
    with pytest.raises(ValueError, match=match):
        magnetic_island(**options)


def test_the_frame_stamp_names_the_phase():
    movie = magnetic_island(m=2, n=1, phase=[0.0, 1.0], animation=True, dpi=DPI)
    assert movie._stamp(1) == "phase 1 rad  (phase 1)"
    assert "2 frames of phase (phase 0-1 rad)" in repr(movie)


def test_frames_differ_from_state_to_state():
    movie = magnetic_island(m=2, n=1, phase=[0.0, 1.5], animation=True, dpi=DPI)
    first, second = movie.frames()
    assert not np.array_equal(first, second)


def _o_point_angle(scene):
    """Poloidal angle of the first O-point marker about the section's centre."""
    from vaft.diagram._scene import Marker, Polyline

    lcfs = next(item for item in scene.items if isinstance(item, Polyline) and item.style == "lcfs")
    centre = np.asarray(lcfs.points, dtype=float).mean(axis=0)
    marker = next(item for item in scene.items if isinstance(item, Marker) and item.style == "opoint")
    return math.atan2(marker.at[1] - centre[1], marker.at[0] - centre[0])


def test_a_positive_frequency_turns_the_o_points_towards_positive_theta():
    movie = magnetic_island(
        m=2, n=1, time=[0.0, 1e-5], rotation_frequency=1e3, animation=True, dpi=DPI,
    )
    before, after = _o_point_angle(movie._model(0)), _o_point_angle(movie._model(1))
    assert 0.0 < (after - before) % (2 * math.pi) < 0.5


def test_the_toroidal_direction_arc_keeps_its_arrowhead():
    from matplotlib.patches import FancyArrowPatch

    from vaft.diagram._animation import draw_scene

    figure, axes = draw_scene(magnetic_island(m=2, n=1, projection="top").scene)
    heads = [p for p in axes.patches if isinstance(p, FancyArrowPatch)]
    leaders = sum(1 for item in magnetic_island(m=2, n=1, projection="top").scene.items
                  if type(item).__name__ == "Arrow")
    assert len(heads) > leaders, "the axis polyline (toroidal direction) draws a head too"


@pytest.mark.parametrize(
    ("latex", "expected"),
    [
        (r"$\mathcal{H}=\tfrac12x^2$", r"$\mathcal{H}=\frac{1}{2}x^2$"),
        ("LCFS", "LCFS"),
        (r"two\\lines", "two\nlines"),
        (r"$\notamathtextcommand{\xi}$", "notamathtextcommandxi"),
    ],
)
def test_labels_become_mathtext_or_plain_words(latex, expected):
    from vaft.diagram._animation import _mathtext

    assert _mathtext(latex) == expected
