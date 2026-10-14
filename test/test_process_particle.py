"""Independent Lorentz/GC limits, field geometry and failure contracts (#2003)."""

import numpy as np
import pytest

from vaft.data.particle import FieldEvaluationError, ParticleField
from vaft.formula.particle import (
    boris_orbit,
    bounce_action,
    effective_parallel_potential,
    guiding_center_rhs,
    mirror_force,
    mirror_loss_cone,
)
from vaft.process.particle import (
    analyze_particle_orbit,
    compare_particle_orbits,
    particle_to_guiding_center,
    trace_guiding_center,
    trace_particle_orbit,
)
from vaft.process.particle_fields import (
    filament_particle_field,
    mirror_particle_field,
    slab_particle_field,
    toroidal_particle_field,
    uniform_particle_field,
)


def test_centered_boris_has_second_order_position_and_phase():
    errors = []
    for dt in (0.2, 0.1, 0.05):
        x, v = boris_orbit(
            1,
            1,
            [0, 0, 0],
            [1, 0, 0.2],
            lambda x: np.zeros(3),
            lambda x: np.array([0, 0, 1]),
            dt,
            round(4 / dt),
            scheme="centered",
        )
        expected = [np.sin(4), np.cos(4) - 1, 0.8]
        errors.append(np.linalg.norm(x[-1] - expected))
        np.testing.assert_allclose(np.sum(v * v, axis=1), 1.04, atol=2e-14)
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8


@pytest.mark.parametrize("charge", [-1.0, 1.0])
def test_full_orbit_and_gc_match_uniform_exact_solution(charge):
    f = uniform_particle_field()
    full = trace_particle_orbit(
        charge, 1, [0, 0, 0], [1, 0, 0.2], f, duration=12, dt=0.03
    )
    initial = particle_to_guiding_center(
        charge, 1, full.position[0], full.velocity[0], f
    )
    gc = trace_guiding_center(
        charge,
        1,
        initial["position"],
        initial["p_parallel"],
        initial["mu"],
        f,
        duration=12,
        dt=0.2,
    )
    assert full.time[-1] == 12 and gc.time[-1] == 12
    comparison = compare_particle_orbits(full, gc)
    assert max(comparison.separation) < 5e-4
    assert comparison.metadata["initial_position_mismatch_m"] == 0
    assert len(comparison.metadata["gyroaverage_time_s"]) > 0
    np.testing.assert_allclose(
        gc.position[-1] - gc.position[0], [0, 0, 2.4], atol=1e-12
    )


@pytest.mark.parametrize("charge", [-1.0, 1.0])
def test_gc_crossed_field_has_charge_independent_drift(charge):
    f = uniform_particle_field(E=[0.02, 0, 0])
    gc = trace_guiding_center(charge, 1, [0, 0, 0], 0.3, 0.5, f, duration=20, dt=0.5)
    np.testing.assert_allclose(gc.position[-1], [0, -0.4, 6], atol=1e-10)
    assert max(abs(analyze_particle_orbit(gc)["energy_residual"])) < 1e-12


@pytest.mark.parametrize("charge", [-1.0, 1.0])
def test_gc_slab_drift_is_analytic(charge):
    f = slab_particle_field(length=40)
    gc = trace_guiding_center(charge, 1, [0, 0, 0], 0.0, 0.5, f, duration=40, dt=0.5)
    np.testing.assert_allclose(gc.position[-1], [0, 0.5 / charge, 0], atol=1e-9)


def test_pure_curvature_has_no_gradient_drift():
    f = toroidal_particle_field(R0=5, poloidal=0, constant_magnitude=True)
    grad, _, divergence, _ = f.derivatives([5, 0, 0])
    np.testing.assert_allclose(grad, 0, atol=1e-8)
    assert abs(divergence) < 1e-8
    gc = trace_guiding_center(1, 1, [5, 0, 0], 0.5, 0.0, f, duration=10, dt=0.1)
    assert gc.position[-1, 2] == pytest.approx(0.5, rel=1e-6)


def test_mirror_bounce_matches_harmonic_period_and_action():
    # On-axis GC motion: z''=-2 mu B0 z/(m L^2), exact harmonic oscillator.
    f = mirror_particle_field(length=10)
    gc = trace_guiding_center(1, 1, [0, 0, 0], 0.3, 0.5, f, duration=150, dt=0.15)
    d = analyze_particle_orbit(gc, bounce_coordinate=lambda x: x[2])
    expected_period = 2 * np.pi * 10
    assert d["bounce_status"] == "complete_resolved_cycles"
    for cycle in d["bounce_cycles"]:
        assert cycle["period_s"] == pytest.approx(expected_period, rel=1e-5)
        assert cycle["action_J_s"] == pytest.approx(np.pi * 0.3**2 / 0.1, rel=1e-4)
    assert max(abs(d["energy_residual"])) < 1e-10
    assert d["mu_evidence"] == "fixed model parameter"
    assert np.max(gc.position[:, 2]) == pytest.approx(3, rel=1e-4)


def test_passing_mirror_terminates_at_domain_without_inventing_bounce():
    f = mirror_particle_field()
    gc = trace_guiding_center(1, 1, [0, 0, 0], 1.0, 0.05, f, duration=100, dt=0.1)
    assert gc.termination == "outside_domain" and gc.position[-1, 2] < 10
    d = analyze_particle_orbit(gc, bounce_coordinate=lambda x: x[2])
    assert d["bounce_cycles"] == [] and d["bounce_status"] == "incomplete_cycle"


@pytest.mark.parametrize(
    "field",
    [
        slab_particle_field(),
        mirror_particle_field(),
        toroidal_particle_field(),
        toroidal_particle_field(perturbation=0.02),
    ],
)
def test_fields_are_solenoidal_and_curl_A_matches_B(field):
    point = (
        np.array([3.2, 0.1, 0.2])
        if "toroidal" in field.name
        else np.array([0.1, 0.2, 0.3])
    )
    h = 1e-4
    J = np.column_stack(
        [
            (field.magnetic(point + h * e) - field.magnetic(point - h * e)) / (2 * h)
            for e in np.eye(3)
        ]
    )
    assert abs(np.trace(J)) < 1e-8
    A = np.column_stack(
        [(field.A(point + h * e) - field.A(point - h * e)) / (2 * h) for e in np.eye(3)]
    )
    curl = [A[2, 1] - A[1, 2], A[0, 2] - A[2, 0], A[1, 0] - A[0, 1]]
    np.testing.assert_allclose(curl, field.magnetic(point), rtol=1e-7, atol=1e-9)


def test_axisymmetric_gc_canonical_momentum_converges():
    f = toroidal_particle_field(poloidal=0.3)
    gc = trace_guiding_center(
        1, 1, [3.4, 0, 0], 0.08, 0.005, f, duration=300, dt=1, rtol=1e-10
    )
    assert gc.termination == "completed"
    d = analyze_particle_orbit(gc)
    assert max(abs(d["P_phi_residual"])) < 2e-7
    assert max(abs(d["energy_residual"])) < 2e-7


def test_filament_adapter_uses_same_quadrature_for_A_and_B():
    theta = np.linspace(0, 2 * np.pi, 81)
    loop = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(len(theta))])
    loop[-1] = loop[0]
    f = filament_particle_field(
        [loop], [1e5], derivative_step=1e-4, exclusion_radius=0.01
    )
    point = np.array([0.2, 0.1, 0.3])
    J = np.column_stack(
        [(f.A(point + 1e-5 * e) - f.A(point - 1e-5 * e)) / (2e-5) for e in np.eye(3)]
    )
    np.testing.assert_allclose(
        [J[2, 1] - J[1, 2], J[0, 2] - J[2, 0], J[1, 0] - J[0, 1]],
        f.magnetic(point),
        rtol=1e-7,
    )
    with pytest.raises(FieldEvaluationError, match="outside"):
        f.magnetic([1, 0, 0])


def test_formula_mirror_and_action_limits():
    assert mirror_loss_cone(1, 2) == pytest.approx(np.pi / 4)
    assert mirror_force(0.5, 0.2) == pytest.approx(-0.1)
    assert effective_parallel_potential(0.5, 2, -1, 0.3) == pytest.approx(0.7)
    phase = np.linspace(0, 2 * np.pi, 1001)
    assert bounce_action(np.sin(phase), np.cos(phase)) == pytest.approx(np.pi, rel=1e-5)
    with pytest.raises(ValueError, match="closed"):
        bounce_action(np.sin(phase[:800]), np.cos(phase[:800]))
    with pytest.raises(ValueError):
        mirror_loss_cone(2, 1)


def test_gc_singularity_and_field_null_are_input_errors():
    with pytest.raises(ValueError, match="singularity"):
        guiding_center_rhs(1, 1, 1, 0.5, [0, 0, 1], [0, 0, 0], [0, 0, -1], [0, 0, 0])
    f = ParticleField(lambda x: np.zeros(3), jacobian=lambda x: np.zeros((3, 3)))
    with pytest.raises(ValueError, match="floor"):
        trace_particle_orbit(1, 1, [0, 0, 0], [1, 0, 0], f, duration=1, dt=0.1)


def test_domain_exit_is_recorded_and_no_extrapolation():
    f = ParticleField(
        lambda x: np.array([0, 0, 1]),
        contains=lambda x: x[2] < 0.5,
        jacobian=lambda x: np.zeros((3, 3)),
    )
    orbit = trace_particle_orbit(1, 1, [0, 0, 0], [0, 0, 1], f, duration=1, dt=0.1)
    assert orbit.termination == "outside_domain"
    assert orbit.position[-1, 2] < 0.5
    with pytest.raises(ValueError, match="outside"):
        orbit.sample([1])
    with pytest.raises(ValueError):
        orbit.position[0, 0] = 4


def test_numerical_derivatives_never_probe_outside_domain():
    seen = []

    def B(x):
        assert x[0] > 0
        seen.append(x.copy())
        return np.array([0, 0, 1 + x[0]])

    f = ParticleField(B, contains=lambda x: x[0] > 0, derivative_step=0.1)
    with pytest.raises(FieldEvaluationError, match="outside"):
        f.derivatives([0.01, 0, 0])


def test_inconsistent_E_and_potential_are_rejected():
    f = ParticleField(
        lambda x: np.array([0, 0, 1]),
        E=lambda x: np.array([1, 0, 0]),
        potential=lambda x: 0.0,
        jacobian=lambda x: np.zeros((3, 3)),
        derivative_step=0.01,
    )
    with pytest.raises(ValueError, match="inconsistent"):
        trace_guiding_center(1, 1, [0, 0, 0], 1, 0.1, f, duration=1, dt=0.1)


def test_partial_history_does_not_create_action():
    f = mirror_particle_field()
    orbit = trace_guiding_center(1, 1, [0, 0, 0], 0.3, 0.5, f, duration=10, dt=0.1)
    assert (
        analyze_particle_orbit(orbit)["bounce_status"]
        == "undefined_without_periodic_arc_coordinate"
    )
    assert (
        analyze_particle_orbit(orbit, bounce_coordinate=lambda x: x[2])["bounce_status"]
        == "incomplete_cycle"
    )


def test_ordering_scan_reduces_full_gc_error_and_mu_variation():
    separation, variation = [], []
    for mass in (1.0, 0.5, 0.25):
        field = mirror_particle_field(length=6)
        x, v = [mass, 0, 0], [0, -1, 0.3]
        state = particle_to_guiding_center(1, mass, x, v, field)
        full = trace_particle_orbit(1, mass, x, v, field, duration=50, dt=0.01 * mass)
        gc = trace_guiding_center(
            1,
            mass,
            state["position"],
            state["p_parallel"],
            state["mu"],
            field,
            duration=50,
            dt=0.1,
        )
        comparison = compare_particle_orbits(full, gc, time=np.linspace(0, 50, 100))
        separation.append(max(comparison.separation))
        variation.append(max(abs(analyze_particle_orbit(full)["mu_residual"])))
    # Leading local transformation leaves O(rho^2/L) error in this benchmark.
    assert separation[0] / separation[1] > 3.5 and separation[1] / separation[2] > 3.5
    assert variation[0] / variation[1] > 3.5 and variation[1] / variation[2] > 3.5


def test_nonadiabatic_control_is_flagged_with_real_mu_variation():
    from dataclasses import replace

    field = replace(
        mirror_particle_field(length=3),
        contains=lambda x: abs(x[2]) < 12 and np.hypot(x[0], x[1]) < 5,
    )
    x, v = [1.5, 0, 0], [0, -1.5, 1.0]
    initial = particle_to_guiding_center(1, 1, x, v, field)
    full = trace_particle_orbit(1, 1, x, v, field, duration=25, dt=0.015)
    gc = trace_guiding_center(
        1,
        1,
        initial["position"],
        initial["p_parallel"],
        initial["mu"],
        field,
        duration=25,
        dt=0.1,
    )
    comparison = compare_particle_orbits(full, gc, time=np.linspace(0, 25, 80))
    assert comparison.metadata["ordering_warning"]
    assert comparison.metadata["rho_over_L_B_max"] > 0.3
    assert max(abs(analyze_particle_orbit(full)["mu_residual"])) > 0.05
    assert max(comparison.separation) > 0.5


def test_tokamak_bounce_matches_independent_fieldline_quadrature():
    from scipy.integrate import quad

    R0, r, B0, c, m, vperp, vpar = 8.0, 0.5, 3.0, 1.35, 0.001, 0.12, 0.003
    f = toroidal_particle_field(R0=R0, B0=B0, poloidal=c)
    B = np.linalg.norm(f.magnetic([R0 + r, 0, 0]))
    mu = m * vperp**2 / (2 * B)
    speed2 = vperp**2 + vpar**2
    K = np.hypot(B0 * R0, 2 * c * r)
    mirror_B = B * speed2 / vperp**2
    theta_tip = np.arccos((K / mirror_B - R0) / r)

    # Along a circular flux surface ds/dtheta = K/(2c). Remove the endpoint
    # square-root singularity using theta=theta_tip sin(u).
    def integral(u):
        theta = theta_tip * np.sin(u)
        velocity = np.sqrt(
            max(speed2 - 2 * mu / m * K / (R0 + r * np.cos(theta)), 1e-25)
        )
        return theta_tip * np.cos(u) / velocity

    period = 4 * K / (2 * c) * quad(integral, 0, np.pi / 2, epsabs=1e-7)[0]
    gc = trace_guiding_center(
        1, m, [R0 + r, 0, 0], m * vpar, mu, f, duration=period * 2.5, dt=period / 500
    )
    d = analyze_particle_orbit(gc)
    assert d["orbit_class"] == "trapped_candidate"
    assert d["bounce_period_s"][0] == pytest.approx(period, rel=0.003)


@pytest.mark.parametrize("aspect_ratio", [3.0, 1.5])
def test_tokamak_passing_wraps_angles_and_conserves_invariants(aspect_ratio):
    f = toroidal_particle_field(R0=aspect_ratio, B0=3.0, poloidal=1.35)
    gc = trace_guiding_center(
        1, 0.1, [aspect_ratio + 0.5, 0, 0], 0.01, 0.0001, f, duration=700, dt=1.0
    )
    d = analyze_particle_orbit(gc)
    assert gc.termination == "completed"
    assert d["orbit_class"] == "passing"
    assert len(d["transit_period_s"]) >= 2
    assert max(abs(d["P_phi_residual"])) < 1e-6
    assert max(abs(np.diff(d["phi_unwrapped"]))) < np.pi


def test_full_axisymmetric_momentum_error_converges_under_step_refinement():
    f = toroidal_particle_field(B0=3.0, poloidal=1.35)
    errors = []
    for step in (0.08, 0.04, 0.02):
        orbit = trace_particle_orbit(
            1, 1, [3.5, 0, 0], [0.08, 0.03, 0.07], f, duration=40, dt=step
        )
        errors.append(max(abs(analyze_particle_orbit(orbit)["P_phi_residual"])))
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


def test_equilibrium_adapter_keeps_cocos_and_cartesian_component_order():
    from vaft.data.equilibrium import EquilibriumData, EquilibriumConvention
    from vaft.process.particle_fields import equilibrium_particle_field
    from vaft.process.equilibrium import make_equilibrium_field_interpolator

    r, z = np.linspace(1, 4, 17), np.linspace(-1, 1, 19)
    R, Z = np.meshgrid(r, z, indexing="ij")
    record = EquilibriumData(
        r=r,
        z=z,
        psi=0.1 * ((R - 2.5) ** 2 + Z**2),
        psi_1d=np.linspace(0, 1, 20),
        f=np.full(20, 3.0),
        convention=EquilibriumConvention(cocos=11, psi_per_radian=False),
    )
    field = equilibrium_particle_field(record)
    native = make_equilibrium_field_interpolator(record)
    br, bz, bt = native(2.8, 0.2)
    np.testing.assert_allclose(field.magnetic([0, 2.8, 0.2]), [-bt, br, bz])
    assert field.metadata()["provenance"]["cocos"] == 11
    with pytest.raises(FieldEvaluationError):
        field.magnetic([0, 4.1, 0])


def test_runtime_gc_singularity_cannot_be_crossed_between_samples():
    field = ParticleField(
        lambda x: np.array([np.cos(x[2]), np.sin(x[2]), 0.0]),
        E=lambda x: np.array([0.1, 0.0, 0.0]),
        potential=lambda x: -0.1 * x[0],
        jacobian=lambda x: np.array(
            [[0, 0, -np.sin(x[2])], [0, 0, np.cos(x[2])], [0, 0, 0]]
        ),
        derivative_step=1e-4,
    )
    orbit = trace_guiding_center(1, 1, [0, 0, 0], 0.5, 0.1, field, duration=10, dt=0.3)
    assert orbit.termination == "gc_singularity"
    assert orbit.p_parallel[-1] < 1


def test_physical_si_particle_scales_have_finite_gc_and_energy():
    from vaft.formula.constants import QE

    mass = 3.3435837724e-27
    field = uniform_particle_field(B=[0, 0, 1.0])
    full = trace_particle_orbit(
        QE, mass, [0, 0, 0], [1e5, 0, 2e4], field, duration=1e-6, dt=1e-10
    )
    initial = particle_to_guiding_center(
        QE, mass, full.position[0], full.velocity[0], field
    )
    gc = trace_guiding_center(
        QE,
        mass,
        initial["position"],
        initial["p_parallel"],
        initial["mu"],
        field,
        duration=1e-6,
        dt=1e-8,
    )
    d = analyze_particle_orbit(gc)
    assert max(abs(d["energy_residual"])) < 1e-12
    np.testing.assert_allclose(
        gc.position[-1] - gc.position[0], [0, 0, 0.02], atol=1e-13
    )
