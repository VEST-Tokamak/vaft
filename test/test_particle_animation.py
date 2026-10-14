"""One physical coordinate across static scenes, lazy frames and quantitative views."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

import vaft.diagram as diagram
from vaft.plot import ParticleOrbitView, particle_overview
from vaft.process.particle import particle_motion_case


@pytest.fixture(scope="module")
def experiment():
    return particle_motion_case("exb_drift", duration=10, dt=0.05)


@pytest.mark.parametrize(
    "name",
    [
        "gyration",
        "grad_b_drift",
        "magnetic_mirror",
        "exb_drift",
        "curvature_drift",
        "toroidal_drift",
        "guiding_center_invariants",
        "canonical_toroidal_momentum",
        "toroidal_symmetry_breaking",
        "trapped_and_passing_orbits",
        "ripple_well_formation",
        "stochastic_ripple_orbit",
    ],
)
def test_frames_reproduce_static_scientific_scenes(name, experiment):
    builder = getattr(diagram, name)
    movie = builder(
        trajectory=experiment, time=[0.0, 1.0, 3.0], animation=True, labels=False
    )
    assert movie.metadata["driver"]["values"] == [0.0, 1.0, 3.0]
    assert movie.metadata["driver"]["unit"] == "s"
    assert movie.metadata["options"]["interpolation"] == "linear; no extrapolation"
    for i, t in enumerate([0.0, 1.0, 3.0]):
        static = builder(trajectory=experiment, time=t, labels=False)
        assert movie._model(i) == static.scene
        np.testing.assert_allclose(
            static.model.particle_position, experiment.full.sample(t)["position"][0]
        )


def test_rendering_does_not_reintegrate(experiment, monkeypatch):
    import vaft.process.particle as physics

    def forbidden(*args, **kwargs):
        raise AssertionError("frame rendering must not integrate")

    monkeypatch.setattr(physics, "trace_particle_orbit", forbidden)
    monkeypatch.setattr(physics, "trace_guiding_center", forbidden)
    movie = diagram.gyration(
        trajectory=experiment, time=[0.0, 1.0, 3.0], animation=True, dpi=30
    )
    frames = list(movie.frames())
    assert len(frames) == 3
    assert frames[0].shape == frames[-1].shape
    assert not np.array_equal(frames[0], frames[-1])


def test_gif_and_provenance_are_deterministic(experiment, tmp_path):
    movie = diagram.exb_drift(
        trajectory=experiment, time=[0.0, 1.0, 3.0], animation=True, dpi=30, fps=5
    )
    first = movie.save(tmp_path / "first.gif")
    second = movie.save(tmp_path / "second.gif")
    assert first.read_bytes() == second.read_bytes()
    with Image.open(first) as image:
        assert image.n_frames == 3
    metadata = json.loads((tmp_path / "first.gif.json").read_text())
    assert metadata["driver"]["values"] == [0.0, 1.0, 3.0]
    assert metadata["options"]["native_full_time_s"] == experiment.full.time.tolist()
    assert metadata["options"]["field"]["units"] == "SI"


@pytest.mark.parametrize(
    "options",
    [
        dict(time=[0, 1]),
        dict(time=[1, 0], animation=True),
        dict(time=[0, float("nan")], animation=True),
        dict(time=[-1, 1], animation=True),
        dict(time=[0, 11], animation=True),
        dict(time=1, animation=True),
        dict(fps=5),
        dict(time=[], animation=True),
    ],
)
def test_invalid_drivers_are_refused(options, experiment):
    with pytest.raises(ValueError):
        diagram.gyration(trajectory=experiment, **options)


def test_quantitative_view_reuses_prepared_diagnostics(experiment, monkeypatch):
    view = ParticleOrbitView.from_comparison(experiment)
    import vaft.process.particle as physics

    monkeypatch.setattr(
        physics,
        "analyze_particle_orbit",
        lambda *a, **k: pytest.fail("reduced during frame rendering"),
    )
    fig, axes = particle_overview(view, time=1.0)
    assert len(axes) == 9
    movie = particle_overview(view, time=[0.0, 1.0, 3.0], animation=True, dpi=25)
    assert len(list(movie.frames())) == 3
    assert movie.metadata["driver"]["values"] == [0.0, 1.0, 3.0]
    plt.close(fig)


def test_stationary_well_and_particle_marker_share_time():
    c = particle_motion_case("magnetic_mirror", duration=60, dt=0.06)
    s = np.linspace(-9.99, 9.99, 101)
    view = ParticleOrbitView.from_comparison(
        c, bounce_coordinate=lambda x: x[2], well_coordinate=s, well_B=1 + (s / 10) ** 2
    )
    fig, axes = particle_overview(view, time=20)
    marker = axes[1].lines[1]
    assert marker.get_xdata()[0] == pytest.approx(
        c.guiding_center.sample(20)["position"][0, 2]
    )
    np.testing.assert_allclose(axes[1].lines[0].get_ydata(), 1 + (s / 10) ** 2)
    plt.close(fig)


def test_static_only_parameters_are_not_silently_ignored(experiment):
    with pytest.raises(ValueError, match="mass_ratio"):
        diagram.exb_drift(mass_ratio=9, trajectory=experiment, time=1)
    with pytest.raises(ValueError, match="phase"):
        diagram.canonical_toroidal_momentum(phase=0.2, trajectory=experiment, time=1)
