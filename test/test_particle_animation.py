"""One physical coordinate across static scenes, lazy frames and quantitative views."""

import json
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pytest

import vaft.diagram as diagram
from vaft.data.particle import (
    TIMEBASE_SCHEMA,
    decode_timebase,
    encode_timebase,
    particle_provenance_record,
    native_timebases,
)
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
    full_time, gc_time = native_timebases(metadata["options"])
    assert np.array_equal(full_time, experiment.full.time)
    assert np.array_equal(gc_time, experiment.guiding_center.time)
    assert "native_full_time_s" not in metadata["options"]  # one copy, under experiment
    assert metadata["options"]["experiment"]["timebase_schema"] == TIMEBASE_SCHEMA
    assert metadata["options"]["field"]["units"] == "SI"
    assert metadata["options"]["full_orbit"]["charge_C"] == experiment.full.charge
    assert metadata["options"]["guiding_center"]["solver"]["method"] == "DOP853"


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


@pytest.mark.parametrize("builder", [diagram.gyration, particle_overview])
def test_custom_orbit_animation_exports_complete_provenance(builder):
    from vaft.process.particle import (trace_particle_orbit, trace_guiding_center,
                                       particle_to_guiding_center, compare_particle_orbits)
    from vaft.process.particle_fields import uniform_particle_field

    records = []
    field = uniform_particle_field()
    for factor in (1., 2.):
        full = trace_particle_orbit(factor, factor, [0, 0, 0], [1, 0, 0],
                                   field, duration=8, dt=.1)
        initial = particle_to_guiding_center(factor, factor, [0, 0, 0], [1, 0, 0], field)
        gc = trace_guiding_center(factor, factor, initial["position"],
                                 initial["p_parallel"], initial["mu"], field,
                                 duration=8, dt=.2, rtol=1e-8, atol=1e-10)
        comparison = compare_particle_orbits(full, gc)
        source = (
            {"model": ParticleOrbitView.from_comparison(comparison)}
            if builder is particle_overview else {"trajectory": comparison}
        )
        movie = builder(**source, time=[0, 1], animation=True)
        options = json.loads(json.dumps(movie.metadata, allow_nan=False))["options"]
        assert options["full_orbit"]["charge_C"] == factor
        assert options["full_orbit"]["mass_kg"] == factor
        assert options["full_orbit"]["initial_state"]["velocity_m_per_s"] == [1, 0, 0]
        assert options["full_orbit"]["solver"]["dt_s"] == .1
        assert options["guiding_center"]["solver"]["method"] == "DOP853"
        assert options["guiding_center"]["solver"]["rtol"] == 1e-8
        assert options["guiding_center"]["solver"]["atol"] == 1e-10
        records.append(options)
    assert records[0] != records[1]


def test_tilted_uniform_field_lines_follow_the_integrated_field():
    from vaft.diagram._projection import project
    from vaft.process.particle import (trace_particle_orbit, trace_guiding_center,
                                       particle_to_guiding_center, compare_particle_orbits)
    from vaft.process.particle_fields import uniform_particle_field

    field = uniform_particle_field(B=[1, 0, 0])
    full = trace_particle_orbit(1, 1, [0, 0, 0], [.2, 1, 0], field, duration=2, dt=.03)
    initial = particle_to_guiding_center(1, 1, [0, 0, 0], [.2, 1, 0], field)
    gc = trace_guiding_center(1, 1, initial["position"], initial["p_parallel"],
                             initial["mu"], field, duration=2, dt=.1)
    comparison = compare_particle_orbits(full, gc)
    scene = diagram.gyration(trajectory=comparison, time=1, labels=False).scene
    lines = [item for item in scene.items if item.role == "field_line"]
    assert len(lines) == 3
    expected = project(np.array([[0, 0, 0], [1, 0, 0]]))
    expected = expected[1] - expected[0]
    expected /= np.linalg.norm(expected)
    for line in lines:
        delta = np.asarray(line.points[-1]) - line.points[0]
        np.testing.assert_allclose(delta / np.linalg.norm(delta), expected, atol=1e-12)


def _gyrations(velocity):
    cross = np.linalg.norm(np.cross(velocity[:-1], velocity[1:]), axis=1)
    return np.arctan2(cross, np.sum(velocity[:-1] * velocity[1:], axis=1)).sum() / (2 * np.pi)


@pytest.mark.parametrize("case", ["tokamak_trapped", "symmetry_breaking"])
def test_long_full_orbit_trails_resolve_the_gyration(case):
    """A fixed 1200-point grid drew ~3 points per turn on these ~400-gyration
    orbits, so the trail zigzagged by ~0.5 rho."""
    from vaft.diagram._particle_states import _trail

    full = particle_motion_case(case).full
    trail = _trail(full, full.time[-1])
    assert len(trail) / _gyrations(full.velocity) >= 20


@pytest.mark.parametrize("case", ["gyration", "grad_b_drift", "magnetic_mirror"])
def test_guiding_center_trail_does_not_follow_adaptive_nodes(case):
    """DOP853 node placement differs between platforms; the GC trail must not,
    or committed reference SVGs hash differently on Linux CI. Curved GC paths
    agree to well below the 5e-5 rounding of the TikZ coordinates."""
    from vaft.diagram._particle_states import _trail

    gc = particle_motion_case(case).guiding_center
    # Move one node along the same path, as a different step-size ramp would.
    time = np.r_[gc.time[0], gc.time[0] + 1.01 * (gc.time[1] - gc.time[0]), gc.time[2:]]
    sampled = gc.sample(time)
    shifted = replace(gc, time=time, position=sampled["position"], p_parallel=sampled["p_parallel"])
    end = float(gc.time[-1])
    np.testing.assert_allclose(_trail(gc, end), _trail(shifted, end), rtol=0, atol=5e-6)


def test_equation_rows_stay_single_height():
    """The rows sit 0.45 units apart; a stacked \\frac in either collides with
    its neighbour. Fails if a catalog edit defeats the
    inline rewrite."""
    from vaft.diagram._particle_states import _equation_labels

    assert all(r"\frac" not in row for row in _equation_labels())


@pytest.mark.parametrize(
    "case", ["gyration", "grad_b_drift", "magnetic_mirror", "tokamak_trapped", "symmetry_breaking"]
)
def test_fixed_step_timebase_is_stored_as_its_exact_recurrence(case):
    """Lossless only: the compact form is accepted when it replays the Boris
    timestamps bit for bit (t0 + i*dt does not; the integrator accumulates)."""
    experiment = particle_motion_case(case)
    record = json.loads(json.dumps(particle_provenance_record(experiment)))
    stored = record["native_full_time_s"]
    assert isinstance(stored, dict) and stored["rule"] == "accumulate"
    assert np.array_equal(decode_timebase(stored), experiment.full.time)
    assert record["native_gc_time_s"] == experiment.guiding_center.time.tolist()


def test_a_timebase_the_recurrence_cannot_reproduce_is_kept_in_full():
    time = 0.05 * np.arange(10)  # affine, not the accumulated sums
    accumulated = decode_timebase({"rule": "accumulate", "t0": 0.0, "dt": 0.05, "end": 0.45, "count": 10})
    assert not np.array_equal(time, accumulated)
    assert encode_timebase(time, dt=0.05, end=0.45) == time.tolist()
    assert encode_timebase(time) == time.tolist()


@pytest.mark.parametrize("layout", ["both", "top_level", "experiment", "schema_2"])
def test_old_and_new_sidecar_layouts_give_the_same_timebases(layout):
    experiment = particle_motion_case("gyration")
    full, gc = experiment.full.time.tolist(), experiment.guiding_center.time.tolist()
    legacy = {"native_full_time_s": full, "native_gc_time_s": gc}
    options = {
        "both": {**legacy, "experiment": dict(legacy)},
        "top_level": {**legacy, "experiment": {}},
        "experiment": {"experiment": dict(legacy)},
        "schema_2": {"experiment": particle_provenance_record(experiment)},
    }[layout]
    full_time, gc_time = native_timebases(json.loads(json.dumps(options)))
    assert np.array_equal(full_time, experiment.full.time)
    assert np.array_equal(gc_time, experiment.guiding_center.time)
