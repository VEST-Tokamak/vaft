"""Self-consistent equilibrium / kinetic-profile iteration through CHEASE (#123).

The loop, its states, its detectors and its failure states are tested offline
with a stand-in for :func:`vaft.code.chease.refine_equilibrium` that writes
the CHEASE input for real (``prepare_chease_inputs`` needs no binary) and
returns an "equilibrium" whose pressure is a scripted multiple of the one it
was handed.  The real loop on the packaged 39915 g-file needs a CHEASE binary
and is skipped without one, like the other CHEASE integration tests.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

import vaft.code.chease_kinetic_iteration as module
from vaft.code.chease import CHEASEConfig, CHEASEResult, _copy_geqdsk, find_chease_executable, prepare_chease_inputs
from vaft.code.chease_kinetic_iteration import (
    PressureSourceError,
    build_consistent_state,
    pressure_roundtrip,
    pressure_source_from_kinetic,
)
from vaft.data import (
    ConvergenceCriteria,
    CurrentPolicy,
    EquilibriumKineticSpec,
    ProfileSpec,
    ScalarTarget,
    SyntheticKineticSpec,
    SyntheticProfileError,
    TabulatedProfile,
    TemperatureAssumption,
)
from vaft.data.eqdsk import write_geqdsk
from vaft.data.resources import sample_geqdsk
from vaft.process.profile import compose_analytic_profile, generate_synthetic_kinetic_profiles

needs_chease = pytest.mark.skipif(
    find_chease_executable() is None,
    reason="CHEASE kinetic-iteration integration test needs a CHEASE executable (CHEASEHOME, CHEASE or CHEASE_EXEC_DIR)",
)

TIME = 0.319


def _kinetic(**overrides) -> SyntheticKineticSpec:
    fields = dict(
        n_e=ProfileSpec(compose_analytic_profile("n_e", axis_value=1.0, separatrix_value=0.2, core_beta=1.5),
                        ScalarTarget("volume_average", 6e18)),
        T_e=ProfileSpec(compose_analytic_profile("T_e", axis_value=150.0, separatrix_value=10.0, core_beta=2.0)),
        temperature=TemperatureAssumption(ti_over_te=0.5),
        pressure_constraint="kinetic",
    )
    fields.update(overrides)
    return SyntheticKineticSpec(**fields)


def _spec(**overrides) -> EquilibriumKineticSpec:
    fields = dict(kinetic=_kinetic(), convergence=ConvergenceCriteria(max_iterations=6))
    fields.update(overrides)
    return EquilibriumKineticSpec(**fields)


class FakeCHEASE:
    """Writes the real CHEASE input, then "solves" it from the written ``PPRIME`` and ``FFPRIM``.

    The output pressure is ``factor(n) * (PRES(1) - int PPRIME dpsi)`` -- the
    written derivative integrated with the file's own span, so a wrong
    ``PPRIME`` shows up offline -- with ``PPRIME`` scaled alike and ``FFPRIM``
    passed through.  ``n`` is read from the ``iteration_NN`` directory, so a
    resumed run sees the same sequence; ``mutate(g, n)`` edits the output
    further; ``returncode`` != 0 writes nothing.
    """

    def __init__(self, factor=lambda n: 1.0, mutate=None, returncode=0, fail_at=None):
        self.factor, self.mutate, self.returncode, self.fail_at = factor, mutate, returncode, fail_at
        self.calls: list[tuple] = []

    def __call__(self, geqdsk, config):
        inputs = prepare_chease_inputs(geqdsk, config)
        workdir = Path(config.workdir)
        match = re.search(r"iteration_(\d+)", workdir.name)
        n = int(match.group(1)) if match else 0
        self.calls.append((geqdsk, config))
        if self.returncode != 0 or n == self.fail_at:
            return CHEASEResult(returncode=self.returncode or 1, workdir=workdir, materialized=inputs.materialized)
        from scipy.integrate import cumulative_trapezoid

        out = _copy_geqdsk(geqdsk)
        pprime = np.asarray(geqdsk["PPRIME"], float)
        span = float(geqdsk["SIBRY"]) - float(geqdsk["SIMAG"])
        psi = float(geqdsk["SIMAG"]) + np.linspace(0.0, 1.0, pprime.size) * span
        edge = float(np.asarray(geqdsk["PRES"], float)[-1])
        solved = edge + cumulative_trapezoid(pprime[::-1], psi[::-1], initial=0.0)[::-1]
        out["PRES"] = solved * self.factor(n)
        out["PPRIME"] = pprime * self.factor(n)
        out["FFPRIM"] = np.asarray(geqdsk["FFPRIM"], float)
        if self.mutate is not None:
            self.mutate(out, n)
        refined = workdir / "input_chease.geqdsk"
        write_geqdsk(out, refined)
        write_geqdsk(out, workdir / "EQDSK_COCOS_02.OUT")
        return CHEASEResult(returncode=0, workdir=workdir, refined_geqdsk=refined, materialized=inputs.materialized)


@pytest.fixture
def fake(monkeypatch):
    def install(**kwargs):
        solver = FakeCHEASE(**kwargs)
        monkeypatch.setattr(module, "refine_equilibrium", solver)
        return solver
    return install


# --- the pressure source ---------------------------------------------------------------


def test_the_pressure_source_is_regular_and_round_trips():
    g = sample_geqdsk()
    x = np.linspace(0.0, 1.0, 101) ** 2
    p = 400.0 * (1.0 - x**1.5) ** 2 + 5.0
    source = pressure_source_from_kinetic(x, p, g)
    grid = source.psi_norm
    assert grid.size == int(g["NW"]) and grid[0] == 0.0 and grid[-1] == 1.0
    exact = 400.0 * (1.0 - grid**1.5) ** 2 + 5.0
    assert np.max(np.abs(source.pressure - exact)) / 405.0 < 1e-4        # p on the g-file grid
    assert np.all(np.isfinite(source.pprime)) and np.all(source.dpressure_dpsi_norm <= 0.0)
    exact_dp = -1200.0 * (1.0 - grid**1.5) * np.sqrt(grid)
    assert np.max(np.abs(source.dpressure_dpsi_norm - exact_dp)) < 0.02 * np.max(np.abs(exact_dp))
    assert abs(source.dpressure_dpsi_norm[0]) < 0.02 * np.max(np.abs(exact_dp))   # axis: bounded, near 0
    span = float(g["SIBRY"]) - float(g["SIMAG"])
    np.testing.assert_allclose(source.pprime * span, source.dpressure_dpsi_norm)
    written = _copy_geqdsk(g)
    written["PRES"], written["PPRIME"] = source.pressure, source.pprime
    assert pressure_roundtrip(written) < 1e-3                             # p -> p' -> p on the file
    written["PPRIME"] = -source.pprime                                    # a sign error is caught
    assert pressure_roundtrip(written) > 0.5
    written["PPRIME"] = 2 * np.pi * source.pprime                         # so is a per-weber span
    assert pressure_roundtrip(written) > 0.5
    assert "PCHIP" in source.method and source.relaxation == 1.0


def test_the_pressure_source_does_not_overshoot_a_steep_monotone_step():
    g = sample_geqdsk()
    x = np.linspace(0.0, 1.0, 101) ** 2
    p = np.where(x < 0.8, 300.0, 20.0) + 1e-3 * (1.0 - x)                 # a near-discontinuous pedestal
    source = pressure_source_from_kinetic(x, p, g)
    assert source.pressure.max() <= p.max() + 1e-9 and source.pressure.min() >= p.min() - 1e-9
    assert np.all(np.diff(source.pressure) <= 1e-12)


def test_a_hollow_or_negative_pressure_is_refused():
    g = sample_geqdsk()
    x = np.linspace(0.0, 1.0, 51)
    with pytest.raises(PressureSourceError, match="rises outward"):
        pressure_source_from_kinetic(x, 100.0 + 50.0 * np.sin(np.pi * x), g)
    with pytest.raises(PressureSourceError, match="non-negative"):
        pressure_source_from_kinetic(x, 100.0 * (1.0 - x) - 1.0, g)


def test_relaxation_blends_only_the_exchanged_pressure():
    g = sample_geqdsk()
    x = np.linspace(0.0, 1.0, 51)
    raw, base = 300.0 * (1.0 - x), 500.0 * (1.0 - x)
    source = pressure_source_from_kinetic(x, raw, g, previous=base, relaxation=0.25)
    np.testing.assert_allclose(source.raw_pressure, raw)
    np.testing.assert_allclose(source.relaxed_pressure, base + 0.25 * (raw - base))
    assert source.pressure[0] == pytest.approx(450.0)


# --- specification ---------------------------------------------------------------------


def test_the_closure_mode_must_match_the_kinetic_constraint():
    with pytest.raises(ValueError, match="identity"):
        EquilibriumKineticSpec(_kinetic(T_e=None, pressure_constraint="equilibrium", closure="temperature"))
    with pytest.raises(ValueError, match="pressure_constraint='equilibrium'"):
        EquilibriumKineticSpec(_kinetic(), closure="equilibrium_pressure")
    with pytest.raises(ValueError, match="NCSCAL"):
        CurrentPolicy(normalization="beta_p")
    with pytest.raises(ValueError, match="current_profile"):
        CurrentPolicy(kind="analytic_ffprime_shape")
    with pytest.raises(ValueError, match="relaxation"):
        ConvergenceCriteria(relaxation=0.0)
    with pytest.raises(ValueError, match="neutral mode"):
        EquilibriumKineticSpec(_kinetic(pressure_constraint="thermal_energy", closure="temperature_amplitude"))


# --- modes, states and convergence ----------------------------------------------------------


def test_equilibrium_pressure_mode_never_calls_chease(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("equilibrium_pressure must not solve")

    monkeypatch.setattr(module, "refine_equilibrium", forbidden)
    spec = EquilibriumKineticSpec(_kinetic(T_e=None, pressure_constraint="equilibrium", closure="temperature"),
                                  closure="equilibrium_pressure")
    state = build_consistent_state(sample_geqdsk(), spec, workdir=tmp_path, time=TIME)
    assert state.status == "converged" and state.iterations == ()
    assert state.final is state.initial and state.initial.status == "initial"
    assert state.initial.metrics["closure_max_relative"] < 1e-6          # exact region, / max p_eq
    assert "ods_note" in state.provenance
    assert state.ods["equilibrium.ids_properties.comment"] == sample_geqdsk().to_omas()[
        "equilibrium.ids_properties.comment"]                            # the input is never relabelled
    assert "equilibrium.code.name" not in state.ods
    assert "core_profiles.profiles_1d.0.electrons.temperature" in state.ods
    assert not list(tmp_path.iterdir())
    assert any(role.startswith("held (no CHEASE") for _, role in state.assumption_table())


def test_a_kinetic_pressure_loop_converges_and_keeps_every_state(fake, tmp_path):
    solver = fake()
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "converged", state.message
    assert [s.index for s in state.states] == list(range(len(state.states)))
    assert state.initial.status == "initial" and all(s.status == "accepted" for s in state.iterations)
    assert len(solver.calls) == len(state.iterations) == 2
    first = state.iterations[0]
    # The first update's source is the INITIAL state's kinetic pressure, re-extracted afterwards.
    np.testing.assert_allclose(first.pressure_source.raw_pressure, state.initial.p_kin)
    assert first.metrics["pressure_max_relative"] < 1e-3 < state.initial.metrics["pressure_max_relative"]
    assert state.iterations[-1].metrics["p_eq_change"] < 1e-12
    for s in state.iterations:
        assert s.chease_workdir.is_dir() and s.geqdsk_path.exists()
        assert s.validation["chease"]["boundary_preserved"]["ok"]
        assert s.validation["kinetic"]["status"] == "success"
    history = state.convergence_history
    assert history["pressure_max_relative"].shape == (3,)
    assert np.isnan(history["p_kin_change"][0])                         # the initial state has no predecessor
    assert state.ods["equilibrium.code.name"] == "vaft.code.chease_kinetic_iteration"
    assert "not transport-predicted" in state.ods["equilibrium.ids_properties.comment"]
    assert state.provenance["state_basis"] == "assumption-driven self-consistent"
    assert state.provenance["transport_predicted"] is False


def test_kinetic_profiles_are_regenerated_from_the_spec_not_from_arrays(fake, monkeypatch, tmp_path):
    fake()
    seen = []

    def spy(equilibrium, spec, **kwargs):
        seen.append((equilibrium, spec))
        return generate_synthetic_kinetic_profiles(equilibrium, spec, **kwargs)

    monkeypatch.setattr(module, "generate_synthetic_kinetic_profiles", spy)
    spec = _spec()
    state = build_consistent_state(sample_geqdsk(), spec, workdir=tmp_path, time=TIME)
    assert len(seen) == len(state.states)
    assert all(s is spec.kinetic for _, s in seen)                        # one spec, re-applied
    for (equilibrium, _), st in zip(seen[1:], state.iterations, strict=True):
        assert equilibrium is st.equilibrium                              # the NEW solved equilibrium


def test_the_current_policy_reaches_chease_and_is_recorded(fake, tmp_path):
    solver = fake()
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path / "ip", time=TIME)
    _, config = solver.calls[0]
    assert config.ncscal == 2 and config.q_constraint_psi_norm is None and config.target_psin == 1.0
    assert not config.edge_zero
    assert state.policy["held_target"] == pytest.approx(abs(float(sample_geqdsk()["CURRENT"])))
    assert state.policy["never"].startswith("q is never written")
    assert solver.calls[0][0]["CURRENT"] == pytest.approx(float(sample_geqdsk()["CURRENT"]))

    solver = fake()
    state = build_consistent_state(sample_geqdsk(), _spec(current_policy=CurrentPolicy(normalization="q95")),
                                   workdir=tmp_path / "q95", time=TIME)
    _, config = solver.calls[0]
    assert config.ncscal == 1 and config.q_constraint_psi_norm == 0.95
    assert state.iterations[0].validation["chease"]["normalization"]["target"] == pytest.approx(
        state.policy["held_target"])

    g = compose_analytic_profile("g", unit="", axis_value=1.0, separatrix_value=0.05, core_beta=2.0)
    solver = fake()
    state = build_consistent_state(
        sample_geqdsk(), _spec(current_policy=CurrentPolicy("analytic_ffprime_shape", current_profile=g)),
        workdir=tmp_path / "analytic", time=TIME)
    handed = np.asarray(solver.calls[0][0]["FFPRIM"], float)
    from vaft.process.profile import evaluate_analytic_profile

    x = np.linspace(0.0, 1.0, handed.size)
    shape = -np.asarray(evaluate_analytic_profile(g, x, derivative=True))
    assert abs(np.corrcoef(handed, shape)[0, 1]) > 1 - 1e-12             # FF' is exactly the declared shape
    assert "analytic current profile" in state.iterations[0].current_source.method
    assert any("-dg/dpsi_N" in quantity for quantity, _ in state.assumption_table())


def test_a_rising_analytic_current_profile_is_refused(tmp_path):
    g = compose_analytic_profile("g", unit="", axis_value=0.2, separatrix_value=1.0, core_beta=2.0)
    with pytest.raises(ValueError, match="rises outward|does not fall"):
        build_consistent_state(sample_geqdsk(), _spec(current_policy=CurrentPolicy(
            "analytic_ffprime_shape", current_profile=g)), workdir=tmp_path, time=TIME)


# --- detectors ---------------------------------------------------------------------------------


def test_an_undamped_oscillation_is_detected(fake, tmp_path):
    fake(factor=lambda n: 1.0 + 0.2 * (-1) ** n)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "oscillating", state.message
    signs = np.sign(state.history("thermal_energy_relative"))
    assert np.all(signs[1:] * signs[:-1] < 0)
    assert state.final is state.iterations[-1]                           # the last accepted state is kept


def test_a_growing_residual_is_divergence(fake, tmp_path):
    fake(factor=lambda n: 1.0 + 0.2 * n)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "diverged", state.message
    r = state.history("pressure_max_relative")
    assert r[-1] > r[-2] > r[-3] and r[-1] > r[0]


def test_a_residual_that_stops_falling_is_stagnation(fake, tmp_path):
    fake(factor=lambda n: 1.1)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "stagnated", state.message
    assert "pressure_rtol" in state.message


def test_the_iteration_cap_is_a_state(fake, tmp_path):
    fake()
    state = build_consistent_state(sample_geqdsk(), _spec(convergence=ConvergenceCriteria(max_iterations=1)),
                                   workdir=tmp_path, time=TIME)
    assert state.status == "max_iterations" and len(state.iterations) == 1
    assert state.ods is not None                                          # the last accepted pair is still written


# --- failure states ------------------------------------------------------------------------------


def test_a_failed_solve_is_chease_failed_with_its_workdir(fake, tmp_path):
    fake(returncode=1)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "chease_failed" and "returned 1" in state.message
    failed = state.iterations[-1]
    assert failed.status == "chease_failed" and failed.chease_workdir == tmp_path / "iteration_01"
    assert failed.pressure_source is not None and state.final is state.initial


def test_a_missing_executable_is_chease_failed_not_an_exception(monkeypatch, tmp_path):
    for name in ("CHEASE", "CHEASEHOME", "CHEASE_EXEC_DIR"):
        monkeypatch.delenv(name, raising=False)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME,
                                   config=CHEASEConfig(executable=str(tmp_path / "no-chease"), create_plot=False))
    assert state.status == "chease_failed" and "no CHEASE executable" in state.message


@pytest.mark.parametrize("mutate, check", [
    (lambda g, n: g.__setitem__("CURRENT", 1.1 * float(g["CURRENT"])), "normalization"),
    (lambda g, n: g.__setitem__("RBBBS", 1.05 * np.asarray(g["RBBBS"], float)), "boundary_preserved"),
    (lambda g, n: g.__setitem__("RMAXIS", 5.0), "axis_inside"),
    (lambda g, n: g.__setitem__("CURRENT", -float(g["CURRENT"])), "orientation"),
    (lambda g, n: g.__setitem__("FPOL", 1.01 * np.asarray(g["FPOL"], float)), "vacuum_field"),
    (lambda g, n: g.__setitem__("PRES", np.asarray(g["PRES"], float) - 1e3), "pressure_nonnegative"),
])
def test_a_solution_that_fails_a_check_is_validation_failed(fake, tmp_path, mutate, check):
    fake(mutate=mutate)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "validation_failed" and check in state.message
    assert not state.iterations[-1].validation["chease"][check]["ok"]


def test_a_kinetic_failure_on_the_new_equilibrium_stops_the_loop(fake, monkeypatch, tmp_path):
    fake()
    calls = []

    def flaky(equilibrium, spec, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise SyntheticProfileError("coordinate_mapping_failed", "injected")
        return generate_synthetic_kinetic_profiles(equilibrium, spec, **kwargs)

    monkeypatch.setattr(module, "generate_synthetic_kinetic_profiles", flaky)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "kinetic_generation_failed"
    assert "coordinate_mapping_failed" in state.message and "injected" in state.message
    assert state.iterations[-1].equilibrium is not None and state.final is state.initial


def test_a_hollow_kinetic_pressure_is_refused_before_any_solve(fake, tmp_path):
    solver = fake()
    rising = TabulatedProfile(np.array([0.0, 0.5, 1.0]), np.array([50.0, 200.0, 20.0]), coordinate="psi_norm",
                              unit="eV")
    spec = _spec(kinetic=_kinetic(T_e=ProfileSpec(rising)))
    state = build_consistent_state(sample_geqdsk(), spec, workdir=tmp_path, time=TIME)
    assert state.status == "invalid_pressure_source" and "rises outward" in state.message
    assert solver.calls == []


def test_a_multi_target_synthesis_result_is_accepted_as_the_initial_equilibrium(tmp_path):
    """The #1378 result carries its g-file through the solve it reports; it used
    to die in ``_coerce_geqdsk`` with "not subscriptable", and so did any other
    unrecognised object (cold review 0.8.0 plasma-state-and-chease F3)."""
    from vaft.code.chease_synthesis import SyntheticEquilibriumResult, ZeroDimensionalEquilibriumSpec
    from vaft.code.chease_synthesis_targets import MultiTargetSynthesisResult, TargetIterationRecord

    spec = ZeroDimensionalEquilibriumSpec(major_radius=0.4, minor_radius=0.235, elongation=1.7,
                                          triangularity=0.35, toroidal_field=0.18, plasma_current=1.2e5)
    path = tmp_path / "g_synth"
    write_geqdsk(sample_geqdsk(), path)
    solved = SyntheticEquilibriumResult("success", None, spec, "plasma_current", {"plasma_current": 1.2e5},
                                        refined_geqdsk=path)
    record = TargetIterationRecord(1, 1, "probe", {}, {}, {}, {}, "success", None, result=solved)
    converged = MultiTargetSynthesisResult("converged", None, spec, {"plasma_current": 1.2e5}, record=record)
    geqdsk, label, _ = module._coerce_initial(converged)
    assert geqdsk["NW"] == sample_geqdsk()["NW"] and str(path) in label

    failed = MultiTargetSynthesisResult("chease_failed", "the solver stopped", spec, {"plasma_current": 1.2e5})
    with pytest.raises(ValueError, match="chease_failed.*the solver stopped"):
        module._coerce_initial(failed)
    with pytest.raises(ValueError, match="MultiTargetSynthesisResult.*got object"):
        module._coerce_initial(object())


# --- restart -------------------------------------------------------------------------------------


def test_a_resumed_run_reproduces_the_uninterrupted_one(fake, tmp_path):
    slow = dict(factor=lambda n: 1.0 + 0.3 * 0.2 ** n)
    fake(**slow)
    whole = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path / "whole", time=TIME)
    fake(**slow)
    part = build_consistent_state(sample_geqdsk(), _spec(convergence=ConvergenceCriteria(max_iterations=2)),
                                  workdir=tmp_path / "part", time=TIME)
    assert part.status == "max_iterations"
    fake(**slow)
    resumed = build_consistent_state(None, _spec(), resume=part, time=TIME)
    assert resumed.status == whole.status == "converged"
    assert len(resumed.iterations) == len(whole.iterations)
    for a, b in zip(resumed.states, whole.states, strict=True):
        assert a.index == b.index
        np.testing.assert_allclose(a.p_eq, b.p_eq, rtol=1e-9)
        assert a.metrics["pressure_max_relative"] == pytest.approx(b.metrics["pressure_max_relative"], rel=1e-9)


# --- integration -----------------------------------------------------------------------------------


@needs_chease
def test_the_39915_kinetic_pressure_loop_converges_through_chease(tmp_path):
    config = CHEASEConfig(nw=129, ns=50, nt=50, create_plot=False)
    state = build_consistent_state(sample_geqdsk(), _spec(), config=config, workdir=tmp_path, time=TIME)
    assert state.status == "converged", state.message
    assert len(state.iterations) <= 6
    r = state.history("pressure_max_relative")
    assert r[0] > 0.1                                                     # the initial p_kin differs from p_eq
    assert r[-1] < 1e-3 and np.all(np.diff(r[1:]) < 0.5 * r[1:-1] + 1e-4)  # contracting after the first solve
    final = state.final
    np.testing.assert_allclose(final.p_eq, final.p_kin, atol=1e-3 * final.p_kin.max())
    ip = state.history("ip")[1:]
    assert np.all(np.abs(ip / state.policy["held_target"] - 1) < 1e-3)
    changes = state.history("q95_change")[2:]
    assert changes[-1] < 1e-3 and changes[-1] < changes[0]
    assert state.ods["core_profiles.code.name"].endswith("generate_synthetic_kinetic_profiles")
    for s in state.iterations:
        assert (Path(s.chease_workdir) / "EXPEQ").exists()


@needs_chease
def test_under_relaxation_keeps_raw_and_relaxed_pressures_and_still_converges(tmp_path):
    config = CHEASEConfig(nw=129, ns=50, nt=50, create_plot=False)
    spec = _spec(convergence=ConvergenceCriteria(max_iterations=12, relaxation=0.6))
    state = build_consistent_state(sample_geqdsk(), spec, config=config, workdir=tmp_path, time=TIME)
    assert state.status == "converged", state.message
    first = state.iterations[0].pressure_source
    np.testing.assert_allclose(first.relaxed_pressure,
                               state.initial.p_eq + 0.6 * (state.initial.p_kin - state.initial.p_eq))
    assert not np.allclose(first.relaxed_pressure, first.raw_pressure)


def test_a_non_finite_solution_is_refused(fake, monkeypatch, tmp_path):
    import vaft.data.eqdsk as eqdsk

    fake()
    real = eqdsk.read_geqdsk

    def poisoned(path):
        g = real(path)
        if Path(path).name == "input_chease.geqdsk":                     # the refined file only
            q = np.asarray(g["QPSI"], float).copy()
            q[5] = np.nan                                                  # a g-file cannot carry NaN as text
            g["QPSI"] = q
        return g

    monkeypatch.setattr(eqdsk, "read_geqdsk", poisoned)
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.status == "validation_failed" and "finite" in state.message


def test_a_runaway_at_a_small_pressure_residual_is_not_stagnation(fake, tmp_path):
    # The pressure closes at once, but q keeps growing by 2 % a solve: a runaway, not a floor.
    fake(mutate=lambda g, n: g.__setitem__("QPSI", 1.02 * np.asarray(g["QPSI"], float)))
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert state.iterations[-1].metrics["pressure_max_relative"] < 1e-3
    assert state.status == "drifting", state.message
    assert "q0" in state.message and "q_rtol" in state.message


def test_a_held_q95_is_requested_every_update_so_a_solver_bias_does_not_accumulate(fake, tmp_path):
    fake(mutate=lambda g, n: g.__setitem__("QPSI", (1 + 3e-4) * np.asarray(g["QPSI"], float)))
    criteria = ConvergenceCriteria(pressure_rtol=1e-12, max_iterations=6, stagnation_window=10)
    state = build_consistent_state(sample_geqdsk(), _spec(current_policy=CurrentPolicy(normalization="q95"),
                                                          convergence=criteria), workdir=tmp_path, time=TIME)
    assert state.status == "max_iterations" and len(state.iterations) == 6
    target = state.policy["held_target"]
    for s in state.iterations:
        assert s.chease_inputs["q95_input"] == pytest.approx(target, rel=1e-12)
        assert s.validation["chease"]["normalization"]["value"] == pytest.approx(3e-4, rel=1e-2)   # no n*delta


def test_cleanup_in_the_callers_config_does_not_delete_the_solutions(fake, tmp_path):
    solver = fake()
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME,
                                   config=CHEASEConfig(cleanup=True, create_plot=False))
    assert state.status == "converged"
    assert all(config.cleanup is False for _, config in solver.calls)
    assert all(s.geqdsk_path.exists() for s in state.iterations)


def test_a_reused_root_never_reads_a_stale_solution(fake, tmp_path):
    fake()
    first = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    fake(returncode=1)
    second = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert first.status == "converged"
    assert second.status == "chease_failed"                               # the old EQDSK_COCOS_02.OUT is not used
    assert second.iterations[0].chease_workdir.name == "iteration_01_attempt_2"


def test_a_resumed_run_keeps_the_failed_attempt_and_uses_a_fresh_directory(fake, tmp_path):
    fake(fail_at=2)
    failed = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME)
    assert failed.status == "chease_failed" and failed.iterations[-1].status == "chease_failed"
    fake()
    resumed = build_consistent_state(None, _spec(), resume=failed, time=TIME)
    assert resumed.status == "converged"
    assert resumed.iterations[1].status == "chease_failed"                # the failure stays in the history
    names = [s.chease_workdir.name for s in resumed.iterations]
    assert len(set(names)) == len(names)


def test_a_slightly_non_monotone_relaxation_base_is_documented_not_refused():
    g = sample_geqdsk()
    x = np.linspace(0.0, 1.0, 51)
    base = 500.0 * (1.0 - x)
    base[30] = base[29] + 2e-4                                            # a 4e-7 rise, as an EFIT p may carry
    raw = 400.0 * (1.0 - x)
    raw[30] = raw[29]                                                     # a flat step, so the blend rises
    source = pressure_source_from_kinetic(x, raw, g, previous=base, relaxation=0.5, monotone_rtol=1e-6)
    assert source.rise_points_removed >= 1 and 0.0 < source.rise_max_relative <= 1e-6
    assert np.all(np.diff(source.relaxed_pressure) <= 0.0)
    base[30] = base[29] + 5.0
    with pytest.raises(PressureSourceError, match="rises outward"):
        pressure_source_from_kinetic(x, raw, g, previous=base, relaxation=0.5, monotone_rtol=1e-6)


def test_a_weber_family_g_file_is_converted_before_anything_reads_it(tmp_path):
    import dataclasses

    from vaft.data.eqdsk import from_equilibrium
    from vaft.process._equilibrium_parametric import convert_cocos
    from vaft.process.equilibrium import as_equilibrium

    eq = as_equilibrium(sample_geqdsk())
    index = eq.convention.candidates[0]
    pinned = dataclasses.replace(eq, convention=dataclasses.replace(eq.convention, cocos=index))
    # What from_equilibrium wrote before #1292 was fixed: psi per weber under a weber CASE label. Rebuilt by
    # hand from the per-radian export, since the exporter itself now always writes per radian.
    weber = from_equilibrium(convert_cocos(pinned, index))
    assert f"COCOS={index}" in weber["CASE"]
    two_pi = 2 * np.pi
    weber.mapping["PSIRZ"] = np.asarray(weber["PSIRZ"], dtype=float) * two_pi
    for key in ("SIMAG", "SIBRY"):
        weber.mapping[key] = float(weber[key]) * two_pi
    for key in ("PPRIME", "FFPRIM"):
        weber.mapping[key] = np.asarray(weber[key], dtype=float) / two_pi
    weber.mapping["CASE"] = f"VAFT EquilibriumData COCOS={index + 10}"
    spec = EquilibriumKineticSpec(_kinetic(T_e=None, pressure_constraint="equilibrium", closure="temperature"),
                                  closure="equilibrium_pressure")
    state = build_consistent_state(weber, spec, workdir=tmp_path, time=TIME)
    assert state.status == "converged", state.message
    g, source = state.initial.equilibrium, sample_geqdsk()
    assert f"COCOS={index}" in g["CASE"] and f"COCOS={index + 10}" not in g["CASE"]
    assert float(g["SIBRY"]) - float(g["SIMAG"]) == pytest.approx(
        float(source["SIBRY"]) - float(source["SIMAG"]), rel=1e-6)
    assert "#1292" in state.provenance["initial_cocos_conversion"]
    key = "equilibrium.time_slice.0.global_quantities.psi_axis"
    assert float(state.ods[key]) == pytest.approx(float(source.to_omas()[key]), rel=1e-6)   # no stray 2*pi


def test_a_baseline_resolve_separates_the_solver_change(fake, tmp_path):
    fake()
    state = build_consistent_state(sample_geqdsk(), _spec(), workdir=tmp_path, time=TIME, baseline=True)
    assert state.baseline is not None and state.baseline.status == "accepted"
    assert state.baseline.chease_workdir.name == "baseline"
    # Same pressure as the initial equilibrium: the baseline differs from it by the solver only.
    np.testing.assert_allclose(state.baseline.p_eq, state.initial.p_eq, atol=1e-2 * state.initial.p_eq.max())
