"""Static-field particle tracing, guiding-center reduction and orbit diagnostics.

Numerics are machine-independent and use SI units. No ODS, solver installation,
plotting backend, collisions or transport model is required. Native integration
times are preserved; interpolation is only an explicitly recorded comparison or
presentation operation. Fields and records live in :mod:`vaft.data.particle`.
"""

from __future__ import annotations

import numpy as np

from vaft.data.particle import (
    FieldEvaluationError,
    ParticleComparison,
    ParticleField,
    ParticleTrajectory,
)
from vaft.formula.particle import (
    _boris_step,
    bounce_action,
    guiding_center_rhs,
    gyration_offset,
    magnetic_moment,
)

__all__ = [
    "trace_particle_orbit",
    "trace_guiding_center",
    "analyze_particle_orbit",
    "compare_particle_orbits",
    "particle_to_guiding_center",
    "particle_motion_case",
]


def _initial(q, m, x0, duration, dt, t0):
    if (
        not np.all(np.isfinite([q, m, duration, dt, t0]))
        or q == 0
        or m <= 0
        or dt <= 0
        or duration <= 0
    ):
        raise ValueError("finite q != 0, m > 0, duration > 0, dt > 0 and t0 required")
    x = np.asarray(x0, float)
    if x.shape != (3,) or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite Cartesian 3-vector")
    if not np.isfinite(t0 + duration) or t0 + duration <= t0:
        raise ValueError(
            "duration must produce a representable finite end time after t0"
        )
    if min(dt, duration) < abs(np.spacing(max(abs(t0), abs(t0 + duration)))):
        raise ValueError("dt must resolve the floating-point timestamp spacing at t0")
    return x.copy()


def _field_preflight(field, x, *, derivatives=False):
    if not isinstance(field, ParticleField):
        raise TypeError("field must be a ParticleField")
    field.magnetic(x)
    field.electric(x)
    field.phi(x)
    if derivatives or field.E is not None:
        field.derivatives(x)


def _record(field, **kwargs):
    return {
        "field": field.metadata(),
        "units": {
            "time": "s",
            "position": "m",
            "velocity": "m/s",
            "p_parallel": "kg m/s",
            "mu": "J/T",
        },
        **kwargs,
    }


def trace_particle_orbit(q, m, x0, v0, field, *, duration, dt, t0=0.0):
    """Integrate a full Lorentz trajectory using the shared centered Boris step.

    Parameters
    ----------
    q : float
        Signed nonzero particle charge [C].
    m : float
        Positive mass [kg].
    x0 : array_like
        Initial Cartesian position [m].
    v0 : array_like
        Initial simultaneous Cartesian velocity [m/s].
    field : ParticleField
        Static magnetic and electrostatic field in Cartesian SI units [-].
    duration : float
        Positive physical duration [s].
    dt : float
        Positive integration step; the last step is shortened to duration [s].
    t0 : float, optional
        Initial physical timestamp; zero is a coordinate origin [s].

    Returns
    -------
    ParticleTrajectory
        Native simultaneous states and explicit termination reason [-].

    Processing steps
    ----------------
    Validate initial data, then use drift-kick-drift Boris updates. Check field
    domains at every sampled position. Preserve only valid completed steps.

    Applicability
    -------------
    Machine-independent. Nonrelativistic collisionless motion in static fields.

    Limitations
    -----------
    Fixed-step integration must resolve the largest gyrofrequency encountered.
    A boundary event retains the last valid step, not an inferred wall impact.
    Steps smaller than the timestamp spacing are rejected; shift t0 toward zero
    when resolving very short motion at a large absolute time origin.

    Provenance
    ----------
    .. [1] J. P. Boris, Proc. 4th Conf. Numer. Sim. Plasmas, NRL (1970), 3.
    """
    x = _initial(q, m, x0, duration, dt, t0)
    v = np.asarray(v0, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("v0 must be a finite Cartesian 3-vector")
    _field_preflight(field, x)
    times, positions, velocities = [float(t0)], [x.copy()], [v.copy()]
    end = t0 + duration
    termination = "completed"
    while times[-1] < end:
        next_time = min(end, times[-1] + dt)
        # Advance by the interval actually represented in the stored timestamps.
        h = next_time - times[-1]
        try:
            new_x, new_v = _boris_step(q, m, x, v, field.electric, field.magnetic, h)
            _field_preflight(field, new_x)
            if not np.all(np.isfinite(new_v)):
                raise FieldEvaluationError("nonfinite_state", "nonfinite velocity")
        except FieldEvaluationError as error:
            termination = error.reason
            break
        x, v = new_x, new_v
        times.append(next_time)
        positions.append(x.copy())
        velocities.append(v.copy())
    return ParticleTrajectory(
        np.array(times),
        np.array(positions),
        q,
        m,
        field,
        "full_orbit",
        velocity=np.array(velocities),
        termination=termination,
        metadata=_record(
            field,
            method="centered_boris",
            dt_s=dt,
            requested_end_s=end,
            time_staggering="simultaneous",
        ),
    )


def trace_guiding_center(
    q, m, X0, p_parallel0, mu, field, *, duration, dt, t0=0.0, rtol=1e-9, atol=1e-11
):
    """Integrate the static Hamiltonian GC equations with adaptive DOP853.

    Parameters
    ----------
    q : float
        Signed nonzero charge [C].
    m : float
        Positive mass [kg].
    X0 : array_like
        Initial guiding-center Cartesian position [m].
    p_parallel0 : float
        Initial signed parallel kinetic momentum [kg m/s].
    mu : float
        Nonnegative magnetic moment, held constant by the model [J/T].
    field : ParticleField
        Static field with Jacobian or explicit numerical derivative step [-].
    duration : float
        Positive physical duration [s].
    dt : float
        Maximum accepted integration step [s].
    t0 : float, optional
        Initial timestamp, a coordinate origin [s].
    rtol : float, optional
        Relative numerical tolerance, a numerical convenience [-].
    atol : float, optional
        Absolute tolerance on dimensionless scaled state, a numerical convenience [-].

    Returns
    -------
    ParticleTrajectory
        Native accepted steps with momentum, model-constant mu and termination [-].

    Processing steps
    ----------------
    Scale position by a local gyroradius and momentum by characteristic kinetic
    momentum, integrate the Hamiltonian B-star equations, and preserve accepted
    states. A field boundary or B-star singularity terminates the trace.

    Convention
    ----------
    Static SI Cartesian fields; this is a weak-flow, nonrelativistic GC model.
    Numerical energy conservation is tested, not enforced by projection.

    Applicability
    -------------
    Machine-independent. Smooth magnetized static fields, small gyroradius
    ordering and weak electrostatic drift.

    Limitations
    -----------
    DOP853 is not symplectic. No long-time invariant guarantee is claimed.
    mu conservation is imposed; assess adiabaticity against full Lorentz runs.

    Provenance
    ----------
    .. [1] J. R. Cary and A. J. Brizard, Rev. Mod. Phys. 81 (2009) 693.
    """
    from scipy.integrate import DOP853

    X = _initial(q, m, X0, duration, dt, t0)
    if (
        not np.all(np.isfinite([p_parallel0, mu, rtol, atol]))
        or mu < 0
        or rtol <= 0
        or atol <= 0
    ):
        raise ValueError(
            "finite momentum, mu >= 0 and positive finite tolerances required"
        )
    _field_preflight(field, X, derivatives=True)
    B0 = np.linalg.norm(field.magnetic(X))
    momentum_scale = max(abs(p_parallel0), np.sqrt(2 * m * mu * B0))
    if momentum_scale == 0:
        # Resting particles still have a physical scale from m and local gyrofrequency.
        momentum_scale = m
    length_scale = momentum_scale / (abs(q) * B0)
    scale = np.array([length_scale] * 3 + [momentum_scale])
    origin = np.r_[X, 0.0]

    def rhs(t, scaled):
        state = origin + scaled * scale
        position, momentum = state[:3], state[3]
        B = field.magnetic(position)
        grad, curl, _, _ = field.derivatives(position)
        b = B / np.linalg.norm(B)
        if abs(b @ (B + momentum / q * curl)) <= 1e-10 * np.linalg.norm(B):
            raise FieldEvaluationError("gc_singularity", "B-star parallel singularity")
        velocity, force = guiding_center_rhs(
            q, m, momentum, mu, B, grad, curl, field.electric(position)
        )
        return np.r_[velocity, force] / scale

    initial = (np.r_[X, p_parallel0] - origin) / scale
    # Invalid initial fields/GC states are input errors, not runtime termination.
    rhs(t0, initial)

    def star_parallel(state):
        B = field.magnetic(state[:3])
        _, curl, _, _ = field.derivatives(state[:3])
        return float((B / np.linalg.norm(B)) @ (B + state[3] / q * curl))

    previous_star = star_parallel(np.r_[X, p_parallel0])
    times, states = [float(t0)], [np.r_[X, p_parallel0]]
    termination = "completed"
    try:
        solver = DOP853(
            rhs, t0, initial, t0 + duration, max_step=dt, rtol=rtol, atol=atol
        )
        while solver.status == "running":
            solver.step()
            if solver.status == "failed":
                termination = "integration_failed"
                break
            state = origin + solver.y * scale
            if not np.all(np.isfinite(state)):
                termination = "nonfinite_state"
                break
            current_star = star_parallel(state)
            if current_star * previous_star <= 0:
                termination = "gc_singularity"
                break
            previous_star = current_star
            times.append(float(solver.t))
            states.append(state.copy())
    except FieldEvaluationError as error:
        termination = error.reason
    states = np.array(states)
    return ParticleTrajectory(
        np.array(times),
        states[:, :3],
        q,
        m,
        field,
        "guiding_center",
        p_parallel=states[:, 3],
        mu=float(mu),
        termination=termination,
        metadata=_record(
            field,
            method="DOP853",
            max_step_s=dt,
            rtol=rtol,
            atol=atol,
            state_scale=scale.tolist(),
            requested_end_s=t0 + duration,
            approximation="static Hamiltonian GC, weak flow",
            mu_status="fixed model parameter; not independent evidence",
        ),
    )


def particle_to_guiding_center(q, m, x, v, field):
    """Convert simultaneous particle initial data to a leading-order GC state.

    Parameters
    ----------
    q : float
        Signed nonzero charge [C].
    m : float
        Positive mass [kg].
    x : array_like
        Particle Cartesian position [m].
    v : array_like
        Particle simultaneous Cartesian velocity [m/s].
    field : ParticleField
        Static Cartesian field [-].

    Returns
    -------
    dict
        GC position, p_parallel, mu and declared transformation order [-].

    Convention
    ----------
    Subtract local E-cross-B velocity before estimating gyromotion. The
    magnetic moment and parallel momentum are leading-order drift-frame
    estimates, not a first-order Lie-transform particle/GC transformation.

    Applicability
    -------------
    Machine-independent. Magnetized weak-flow initial states.

    Limitations
    -----------
    Initial transformation error remains part of a full-orbit/GC comparison.

    Provenance
    ----------
    .. [1] F. F. Chen, Introduction to Plasma Physics and Controlled Fusion,
       3rd ed. (2016), Sec. 2.2.
    """
    x = _initial(q, m, x, 1.0, 1.0, 0.0)
    v = np.asarray(v, float)
    if v.shape != (3,) or not np.all(np.isfinite(v)):
        raise ValueError("v must be a finite 3-vector")
    B = field.magnetic(x)
    magnitude = np.linalg.norm(B)
    b = B / magnitude
    drift = np.cross(field.electric(x), B) / magnitude**2
    relative = v - drift
    parallel = relative @ b
    X = x - gyration_offset(q, m, B, relative)
    field.check_position(X)
    return {
        "position": X,
        "p_parallel": float(m * parallel),
        "mu": float(
            magnetic_moment(m, np.linalg.norm(relative - parallel * b), magnitude)
        ),
        "transformation": "leading-order local drift-frame gyration offset",
        "approximation_order": "zeroth-order GC initialization; no Lie-transform correction",
    }


def _residual(values, scale):
    return (values - values[0]) / scale


def _bounce_cycles(time, parallel):
    """Turning times from sign changes; same-direction turns delimit full cycles."""
    events = []
    for i in range(len(time) - 1):
        if parallel[i] * parallel[i + 1] < 0 or (
            parallel[i + 1] == 0 and parallel[i] != 0
        ):
            fraction = -parallel[i] / (parallel[i + 1] - parallel[i])
            events.append(
                (
                    float(time[i] + fraction * (time[i + 1] - time[i])),
                    np.sign(parallel[i]),
                )
            )
    cycles = []
    # Nonoverlapping cycles: three alternating turns delimit one complete bounce.
    for i in range(0, len(events) - 2, 2):
        if events[i][1] == events[i + 2][1] and events[i][1] != events[i + 1][1]:
            cycles.append((events[i][0], events[i + 2][0]))
    return events, cycles


def analyze_particle_orbit(trajectory, *, bounce_coordinate=None):
    """Evaluate field, energy, invariants, ordering and defined bounce cycles.

    Parameters
    ----------
    trajectory : ParticleTrajectory
        Native computed particle or GC states [-].
    bounce_coordinate : callable, optional
        Signed field-line arc coordinate s(position) for a known periodic
        bounce. None leaves action undefined; it is never inferred from an
        arbitrary 3-D history [m].

    Returns
    -------
    dict
        Native-time arrays, dimensional scales, residuals and bounce evidence [-].

    Processing steps
    ----------------
    Sample fields; resolve parallel and perpendicular energy in the local
    drift frame; evaluate model-appropriate invariants and dimensionless
    ordering. Detect turns, then evaluate action only on a supplied periodic
    arc coordinate whose sampled cycle closes within resolution.

    Convention
    ----------
    Full-orbit total energy uses laboratory kinetic energy plus q Phi. GC
    energy uses p_parallel^2/(2m) + mu B + q Phi. Canonical toroidal momentum
    uses x p_y - y p_x + q (x A_y - y A_x), in the field's declared gauge.

    Applicability
    -------------
    Machine-independent. Static fields with adequate numerical resolution.

    Limitations
    -----------
    Full-orbit mu is an instantaneous estimate; its gyrovariation is physical.
    Derivative diagnostics are NaN with an unavailable audit when their stencil
    crosses the domain boundary; local state diagnostics remain available.
    Fixed GC mu is not an adiabaticity test. Finite orbit momentum changes
    under nonaxisymmetry alone do not demonstrate transport or NTV torque.

    Provenance
    ----------
    .. [1] J. R. Cary and A. J. Brizard, Rev. Mod. Phys. 81 (2009) 693.
    """
    tr = trajectory
    field, q, m = tr.field, tr.charge, tr.mass
    B = np.array([field.magnetic(x) for x in tr.position])
    strength = np.linalg.norm(B, axis=1)
    b = B / strength[:, None]
    E = np.array([field.electric(x) for x in tr.position])
    phi = np.array([field.phi(x) for x in tr.position])
    drift = np.cross(E, B) / strength[:, None] ** 2
    if tr.kind == "full_orbit":
        relative = tr.velocity - drift
        vpar = np.sum(relative * b, axis=1)
        vperp = np.linalg.norm(relative - vpar[:, None] * b, axis=1)
        mu = magnetic_moment(m, vperp, strength)
        kinetic = 0.5 * m * np.sum(tr.velocity**2, axis=1)
        mechanical = m * tr.velocity
        center = tr.position - np.array(
            [gyration_offset(q, m, B[i], relative[i]) for i in range(len(B))]
        )
    else:
        vpar = tr.p_parallel / m
        mu = np.full(len(B), tr.mu)
        vperp = np.sqrt(2 * mu * strength / m)
        kinetic = tr.p_parallel**2 / (2 * m) + mu * strength
        mechanical = tr.p_parallel[:, None] * b
        center = tr.position
    energy = kinetic + q * phi
    rho = m * vperp / (abs(q) * strength)
    grad, curls, divergence, audits = [], [], [], []
    for x in tr.position:
        try:
            g, c, d, audit = field.derivatives(x)
        except FieldEvaluationError as error:
            if error.reason != "outside_domain":
                raise
            # The orbit sample is valid even if its central stencil is not.
            # Preserve local energy/center diagnostics without extrapolation.
            g, c, d = np.full(3, np.nan), np.full(3, np.nan), np.nan
            audit = {"method": "unavailable", "reason": "stencil_outside_domain"}
        grad.append(g)
        curls.append(c)
        divergence.append(d)
        audits.append(audit)
    grad, curls = np.array(grad), np.array(curls)
    epsilon = rho * np.linalg.norm(grad, axis=1) / strength
    curvature = -np.cross(b, curls)
    epsilon_parallel = (m * np.abs(vpar) / (abs(q) * strength)) * np.linalg.norm(
        curvature, axis=1
    )
    speed = np.sqrt(vpar**2 + vperp**2)
    flow_ordering = np.divide(
        np.linalg.norm(drift, axis=1), speed, out=np.zeros_like(speed), where=speed > 0
    )
    flow_ordering[(speed == 0) & (np.linalg.norm(drift, axis=1) > 0)] = np.inf
    total_scale = max(
        float(np.max(np.abs(kinetic))), float(np.ptp(q * phi)), np.finfo(float).tiny
    )
    mu_scale = max(float(np.max(np.abs(mu))), np.finfo(float).tiny)
    omega = abs(q) * strength / m
    result = {
        "time": tr.time,
        "B": B,
        "B_magnitude": strength,
        "E": E,
        "potential": phi,
        "v_parallel": vpar,
        "v_perp": vperp,
        "energy_parallel": 0.5 * m * vpar**2,
        "energy_perpendicular": 0.5 * m * vperp**2,
        "kinetic_energy": kinetic,
        "energy": energy,
        "mu": mu,
        "energy_residual": _residual(energy, total_scale),
        "energy_scale_J": total_scale,
        "mu_residual": _residual(mu, mu_scale),
        "mu_scale_J_per_T": mu_scale,
        "rho": rho,
        "rho_over_L_B": epsilon,
        "parallel_curvature_ordering": epsilon_parallel,
        "exb_over_speed": flow_ordering,
        "omega_c_magnitude": omega,
        "div_B": np.array(divergence),
        "guiding_center_estimate": center,
        "derivative_audit": audits,
        "mu_evidence": "instantaneous drift-frame estimate"
        if tr.kind == "full_orbit"
        else "fixed model parameter",
        "termination": tr.termination,
    }
    if len(tr.time) > 1:
        result["dt_over_gyroperiod"] = (
            np.diff(tr.time) * np.maximum(omega[:-1], omega[1:]) / (2 * np.pi)
        )
    if field.vector_potential is not None:
        A = np.array([field.A(x) for x in tr.position])
        mechanical_phi = (
            tr.position[:, 0] * mechanical[:, 1] - tr.position[:, 1] * mechanical[:, 0]
        )
        magnetic_phi = q * (tr.position[:, 0] * A[:, 1] - tr.position[:, 1] * A[:, 0])
        momentum = mechanical_phi + magnetic_phi
        momentum_scale = max(
            float(np.max(np.abs(mechanical_phi))),
            float(np.max(np.abs(magnetic_phi))),
            np.finfo(float).tiny,
        )
        result.update(
            P_phi=momentum,
            P_phi_mechanical=mechanical_phi,
            P_phi_magnetic=magnetic_phi,
            P_phi_scale=momentum_scale,
            P_phi_residual=_residual(momentum, momentum_scale),
        )
    if len(tr.time) > 1:
        result["signed_parallel_distance"] = np.r_[
            0.0, np.cumsum(0.5 * (vpar[1:] + vpar[:-1]) * np.diff(tr.time))
        ]
    else:
        result["signed_parallel_distance"] = np.zeros(1)
    events, cycles = _bounce_cycles(tr.time, vpar)
    result["turning_times"] = np.array([e[0] for e in events])
    result["orbit_class"] = (
        "trapped_candidate"
        if len(events) >= 3
        else ("reflected" if events else "no_resolved_turn")
    )
    result["bounce_cycles"] = []
    reference_R = field.provenance.get("R0_m")
    if reference_R is not None:
        R = np.hypot(tr.position[:, 0], tr.position[:, 1])
        toroidal_angle = np.unwrap(np.arctan2(tr.position[:, 1], tr.position[:, 0]))
        poloidal_angle = np.unwrap(np.arctan2(tr.position[:, 2], R - reference_R))
        result.update(
            R=R,
            Z=tr.position[:, 2],
            phi_unwrapped=toroidal_angle,
            theta_unwrapped=poloidal_angle,
            radial_excursion_m=float(np.ptp(R)),
            toroidal_advance_rad=float(toroidal_angle[-1] - toroidal_angle[0]),
        )
        if not events and abs(poloidal_angle[-1] - poloidal_angle[0]) >= 2 * np.pi:
            result["orbit_class"] = "passing"
        # anti-alias: scalar sampling at turning instants of a solver trajectory,
        # not a rate change -- the unwrapped angle is read at two times per cycle.
        result["bounce_precession_rad_s"] = [
            float(
                (
                    np.interp(end, tr.time, toroidal_angle)
                    - np.interp(start, tr.time, toroidal_angle)
                )
                / (end - start)
            )
            for start, end in cycles
        ]
        if len(cycles):
            result["bounce_period_s"] = [end - start for start, end in cycles]
        # Count complete same-direction poloidal revolutions, never confuse them with bounce.
        excursion = poloidal_angle - poloidal_angle[0]
        direction = np.sign(excursion[-1])
        if (
            not events
            and direction != 0
            and np.all(direction * np.diff(poloidal_angle) > 0)
        ):
            count = int(abs(excursion[-1]) / (2 * np.pi))
            # anti-alias: inverse lookup of the monotonic poloidal angle to find the
            # 2*pi*k crossing times; no time series is resampled.
            crossings = [float(tr.time[0])] + [
                float(np.interp(2 * np.pi * k, direction * excursion, tr.time))
                for k in range(1, count + 1)
            ]
            result["transit_period_s"] = np.diff(crossings).tolist()
    result["bounce_status"] = "undefined_without_periodic_arc_coordinate"
    if bounce_coordinate is not None:
        s = np.array([bounce_coordinate(x) for x in tr.position], float)
        if s.shape != tr.time.shape or not np.all(np.isfinite(s)):
            raise ValueError(
                "bounce_coordinate must return finite scalar arc coordinates"
            )
        result["bounce_status"] = "incomplete_cycle"
        for start, end in cycles:
            times = np.r_[start, tr.time[(tr.time > start) & (tr.time < end)], end]
            # anti-alias: `times` is the native samples inside the cycle plus its two
            # turning endpoints, so this only inserts points -- never a rate reduction.
            arc = np.interp(times, tr.time, s)
            momentum = m * np.interp(times, tr.time, vpar)
            # Turning endpoints have zero p by construction, not by snapping arbitrary samples.
            momentum[[0, -1]] = 0
            closure = abs(arc[-1] - arc[0]) / max(
                float(np.ptp(arc)), np.finfo(float).tiny
            )
            velocity_arc = np.diff(arc) / np.diff(times)
            expected = 0.5 * (momentum[:-1] + momentum[1:]) / m
            arc_error = np.linalg.norm(velocity_arc - expected) / max(
                np.linalg.norm(expected), np.finfo(float).tiny
            )
            if closure > 1e-3 or arc_error > 0.02:
                result["bounce_status"] = "unresolved_or_nonperiodic_coordinate"
                continue
            if len(arc) < 5:
                result["bounce_status"] = "unresolved_cycle"
                continue
            action = bounce_action(arc, momentum, closure_tolerance=1e-3)
            if action <= 0:
                result["bounce_status"] = "invalid_arc_orientation"
                continue
            result["bounce_cycles"].append(
                {
                    "start_s": start,
                    "end_s": end,
                    "period_s": end - start,
                    "action_J_s": action,
                    "closure_relative": closure,
                    "arc_velocity_relative_error": arc_error,
                }
            )
        if result["bounce_cycles"]:
            result["bounce_status"] = "complete_resolved_cycles"
    return result


def compare_particle_orbits(full, guiding_center, *, time=None):
    """Pair full and GC trajectories on common physical times without extrapolation.

    Parameters
    ----------
    full : ParticleTrajectory
        Full Lorentz orbit [-].
    guiding_center : ParticleTrajectory
        Independently integrated GC orbit of the same particle and field [-].
    time : array_like, optional
        Strictly increasing comparison times; defaults to full samples in
        the common valid interval [s].

    Returns
    -------
    ParticleComparison
        Local-center separation, time pairing and approximation evidence [-].

    Processing steps
    ----------------
    Check particle, field and time compatibility; estimate the local full-orbit
    center in the drift frame; linearly interpolate both positions on a shared
    physical coordinate. Record initialization and ordering separately.

    Applicability
    -------------
    Machine-independent. Matched initial conditions in a controlled GC regime.

    Limitations
    -----------
    Local center subtraction is not higher-order gyroaveraging. Separation
    includes initialization, phase and GC approximation error; numerical and
    ordering scans are needed to attribute it. A one-gyroperiod time-window
    average is supplied separately, with incomplete edge windows omitted.

    Provenance
    ----------
    .. [1] F. F. Chen, Introduction to Plasma Physics and Controlled Fusion,
       3rd ed. (2016), Sec. 2.2.
    """
    if full.kind != "full_orbit" or guiding_center.kind != "guiding_center":
        raise ValueError("comparison requires a full orbit followed by a GC orbit")
    if (
        full.field is not guiding_center.field
        or full.charge != guiding_center.charge
        or full.mass != guiding_center.mass
    ):
        raise ValueError(
            "comparison requires the same field object, mass and signed charge"
        )
    lo, hi = (
        max(full.time[0], guiding_center.time[0]),
        min(full.time[-1], guiding_center.time[-1]),
    )
    if time is None:
        time = full.time[(full.time >= lo) & (full.time <= hi)]
    time = np.asarray(time, float)
    if (
        time.ndim != 1
        or len(time) == 0
        or not np.all(np.isfinite(time))
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError(
            "comparison time must be nonempty, finite and strictly increasing"
        )
    if time[0] < lo or time[-1] > hi:
        raise ValueError(
            "comparison would extrapolate outside the common valid interval"
        )
    diagnostics = analyze_particle_orbit(full)
    # anti-alias: evaluates the deterministic solver trajectory at comparison
    # instants, not acquired data; the estimate already removes the gyration to
    # leading order, and a low-pass would bias the residual being compared.
    center = np.column_stack(
        [
            np.interp(time, full.time, diagnostics["guiding_center_estimate"][:, j])
            for j in range(3)
        ]
    )
    gc = guiding_center.sample(time)["position"]
    period = 2 * np.pi / diagnostics["omega_c_magnitude"]
    average_time, averaged = [], []
    # Integrate piecewise-linear position in a centered one-local-gyroperiod window.
    average_selection = np.unique(
        np.linspace(0, len(time) - 1, min(256, len(time))).astype(int)
    )
    for t in time[average_selection]:
        # anti-alias: scalar sampling of the local gyroperiod at one instant.
        width = float(np.interp(t, full.time, period))
        a, z = t - width / 2, t + width / 2
        if a < full.time[0] or z > full.time[-1]:
            continue
        grid = np.r_[a, full.time[(full.time > a) & (full.time < z)], z]
        points = full.sample(grid)["position"]
        averaged.append(
            np.sum((points[1:] + points[:-1]) * np.diff(grid)[:, None] / 2, axis=0)
            / width
        )
        average_time.append(t)
    transformed = particle_to_guiding_center(
        full.charge, full.mass, full.position[0], full.velocity[0], full.field
    )
    initial_mismatch = float(
        np.linalg.norm(transformed["position"] - guiding_center.position[0])
    )
    return ParticleComparison(
        full,
        guiding_center,
        time,
        center,
        gc,
        np.linalg.norm(center - gc, axis=1),
        {
            "interpolation": "linear; no extrapolation",
            "native_full_time_s": full.time.tolist(),
            "native_gc_time_s": guiding_center.time.tolist(),
            "initial_position_mismatch_m": initial_mismatch,
            "initial_p_parallel_mismatch": float(
                transformed["p_parallel"] - guiding_center.p_parallel[0]
            ),
            "initial_mu_mismatch": float(transformed["mu"] - guiding_center.mu),
            "initialization": transformed["transformation"],
            "rho_over_L_B_max": (
                float(np.nanmax(diagnostics["rho_over_L_B"]))
                if np.any(np.isfinite(diagnostics["rho_over_L_B"]))
                else None
            ),
            "parallel_curvature_ordering_max": (
                float(np.nanmax(diagnostics["parallel_curvature_ordering"]))
                if np.any(np.isfinite(diagnostics["parallel_curvature_ordering"]))
                else None
            ),
            "derivative_diagnostics_complete": bool(
                np.all(np.isfinite(diagnostics["rho_over_L_B"]))
            ),
            "exb_over_speed_max": (
                float(np.max(diagnostics["exb_over_speed"]))
                if np.all(np.isfinite(diagnostics["exb_over_speed"]))
                else None
            ),
            "dt_over_gyroperiod_max": float(
                np.max(diagnostics.get("dt_over_gyroperiod", [0.0]))
            ),
            "gyroaverage_separation_m": (
                np.linalg.norm(
                    np.asarray(averaged)
                    - guiding_center.sample(average_time)["position"],
                    axis=1,
                ).tolist()
                if len(averaged)
                else []
            ),
            "ordering_warning": bool(
                not np.all(np.isfinite(diagnostics["rho_over_L_B"]))
                or max(
                    np.max(diagnostics["rho_over_L_B"]),
                    np.max(diagnostics["parallel_curvature_ordering"]),
                    np.max(diagnostics["exb_over_speed"]),
                )
                >= 0.1
            ),
            "ordering_warning_threshold": "0.1 educational flag; not a universal validity boundary",
            "gyroaverage_time_s": average_time,
            "gyroaverage_position_m": np.asarray(averaged).reshape(-1, 3).tolist(),
            "error_interpretation": "local-center separation; not an isolated GC truncation error",
        },
    )


def particle_motion_case(
    kind="gyration",
    *,
    charge=1.0,
    mass=1.0,
    speed=None,
    pitch_angle=None,
    aspect_ratio=3.0,
    perturbation=0.02,
    duration=None,
    dt=0.05,
):
    """Compute a reproducible matched full-orbit/GC reference experiment.

    Parameters
    ----------
    kind : str, optional
        gyration, exb_drift, grad_b_drift, curvature_drift, toroidal_drift,
        magnetic_mirror, mirror_passing, tokamak_trapped, tokamak_passing or
        symmetry_breaking; gyration is the simplest reference default [-].
    charge : float, optional
        Signed nonzero charge; 1 C is an illustrative SI reference particle [C].
    mass : float, optional
        Positive mass; 1 kg is an illustrative SI reference particle [kg].
    speed : float, optional
        Drift-frame launch speed; None chooses a documented case default [m/s].
    pitch_angle : float, optional
        Launch angle from B in [0,pi]; None chooses trapped/passing defaults [rad].
    aspect_ratio : float, optional
        Major/minor radius ratio of the toroidal reference; default three [-].
    perturbation : float, optional
        Signed 3-D harmonic amplitude for symmetry_breaking; default 0.02 [-].
    duration : float, optional
        Physical integration duration; None uses the case reference interval [s].
    dt : float, optional
        Full-orbit step, GC maximum step is four times this numerical default [s].

    Returns
    -------
    ParticleComparison
        Matched integrated trajectories with reference case and validity metadata [-].

    Processing steps
    ----------------
    Build a solenoidal static reference field, initialize a physical particle,
    apply the documented leading-order GC conversion, and integrate both models
    to the same physical end time. Pair only their common valid interval.

    Defaults
    --------
    Charge and mass define an illustrative SI particle, not an electron/ion
    mass ratio. Geometry, speed, pitch and duration are educational benchmark
    choices; all resolved values are recorded. dt is numerical convenience
    and must be reduced when charge/mass increases.

    Applicability
    -------------
    Machine-independent. Controlled reference experiments, not discharge predictions.

    Limitations
    -----------
    Tokamak references use an analytic circular field, not force-balanced VEST
    equilibrium. Symmetry-breaking output alone establishes no transport rate.
    Time-varying fields and the second invariant under slow variation are deferred.

    Provenance
    ----------
    .. [1] Particle-motion work packages #2004 through #2009.
    """
    from vaft.process.particle_fields import (
        mirror_particle_field,
        slab_particle_field,
        toroidal_particle_field,
        uniform_particle_field,
    )

    allowed = (
        "gyration",
        "exb_drift",
        "grad_b_drift",
        "curvature_drift",
        "toroidal_drift",
        "magnetic_mirror",
        "mirror_passing",
        "tokamak_trapped",
        "tokamak_passing",
        "symmetry_breaking",
    )
    if kind not in allowed:
        raise ValueError(f"kind must be one of {allowed}")
    if not np.isfinite(aspect_ratio) or aspect_ratio <= 1:
        raise ValueError("aspect_ratio must be finite and greater than one")
    torus = kind.startswith("tokamak") or kind == "symmetry_breaking"
    mirror = kind in ("magnetic_mirror", "mirror_passing")
    speed = float(
        speed if speed is not None else (0.12 if torus else (0.6 if mirror else 1.0))
    )
    pitch_angle = float(
        pitch_angle
        if pitch_angle is not None
        else (
            0.25
            if kind.endswith("passing")
            else (1.35 if torus else (1.2 if mirror else 1.1))
        )
    )
    if (
        not np.isfinite(speed)
        or speed <= 0
        or not np.isfinite(pitch_angle)
        or not 0 <= pitch_angle <= np.pi
    ):
        raise ValueError(
            "speed must be positive and pitch_angle must be in [0,pi], both finite"
        )
    if kind in ("gyration", "exb_drift"):
        f = uniform_particle_field(E=[0.03, 0, 0] if kind == "exb_drift" else [0, 0, 0])
        X = np.zeros(3)
        span = 24.0
    elif kind == "grad_b_drift":
        f, X, span = slab_particle_field(), np.zeros(3), 100.0
    elif kind in ("curvature_drift", "toroidal_drift"):
        f = toroidal_particle_field(
            R0=6.0,
            poloidal=0,
            minor_radius=3.0,
            constant_magnitude=kind == "curvature_drift",
        )
        X, span = np.array([6.0, 0.0, 0.0]), 30.0
    elif mirror:
        f, X, span = mirror_particle_field(), np.zeros(3), 160.0
    else:
        f = toroidal_particle_field(
            R0=aspect_ratio,
            B0=3.0,
            poloidal=1.35,
            perturbation=perturbation if kind == "symmetry_breaking" else 0.0,
            minor_radius=0.95,
        )
        X, span = np.array([aspect_ratio + 0.5, 0.0, 0.0]), 1000.0
    duration = span if duration is None else float(duration)
    _initial(charge, mass, X, duration, dt, 0.0)
    B = f.magnetic(X)
    b = B / np.linalg.norm(B)
    axis = np.array([1.0, 0.0, 0.0])
    if abs(axis @ b) > 0.9:
        axis = np.array([0.0, 0.0, 1.0])
    perpendicular = axis - (axis @ b) * b
    perpendicular /= np.linalg.norm(perpendicular)
    # Radially displaced gyroposition in a mirror keeps the converted GC on axis.
    if mirror:
        perpendicular = np.array([0.0, -np.sign(charge), 0.0])
    velocity = (
        speed * np.cos(pitch_angle) * b + speed * np.sin(pitch_angle) * perpendicular
    )
    velocity += np.cross(f.electric(X), B) / (B @ B)
    x = X + gyration_offset(
        charge, mass, B, speed * np.sin(pitch_angle) * perpendicular
    )
    initial = particle_to_guiding_center(charge, mass, x, velocity, f)
    if abs(charge) * np.linalg.norm(B) / mass * dt > 0.3:
        raise ValueError(
            "dt does not resolve the initial gyrofrequency; require abs(Omega_c)*dt <= 0.3"
        )
    full = trace_particle_orbit(charge, mass, x, velocity, f, duration=duration, dt=dt)
    gc = trace_guiding_center(
        charge,
        mass,
        initial["position"],
        initial["p_parallel"],
        initial["mu"],
        f,
        duration=duration,
        dt=4 * dt,
    )
    comparison = compare_particle_orbits(full, gc)
    from dataclasses import replace

    return replace(
        comparison,
        metadata={
            **comparison.metadata,
            "case": kind,
            "resolved_parameters": {
                "charge_C": charge,
                "mass_kg": mass,
                "speed_m_s": speed,
                "pitch_angle_rad": pitch_angle,
                "aspect_ratio": aspect_ratio,
                "perturbation": perturbation,
                "duration_s": duration,
                "dt_s": dt,
            },
            "fidelity": "independent centered Boris and Hamiltonian GC integration; analytic benchmark field",
            "scientific_driver": "physical time [s]",
            "time_dependent_fields": "deferred",
        },
    )
