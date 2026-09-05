"""The reduced-wall order study on the packaged VEST wall (vaft #494, vfit #10).

Pins are the study's own findings at loose factors, so a regression in the
basis, the projection or the integrator shows -- not a tuning of the wall.
The numbers behind them (shot 39915, PF programme, 62 probes + 11 flux
loops): decay-time ranking leaves 28 % probe error at 19 modes and 24 % at
76; response ranking reaches 1.2 % at 76 and 0.26 % at 152; ten
resistive-limit patterns reach 1.7 %, nineteen 0.9 % and seventy-six 1e-4;
on a step drive nineteen patterns miss 25 % where seventy-six miss 0.6 %.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.process import wall_modes as wm
from vaft.validation import wall_reduction as wr


@pytest.fixture(scope="module")
def study():
    from vaft.omas.sample import sample_ods

    ods = sample_ods()
    system = wr.wall_system(ods)
    observation = wr.observation_set(ods, n_coils=system["n_coils"])
    rows = wr.order_convergence(
        system, observation=observation,
        rules=("tau", "output_weight", "uniform", "moments"),
        orders=(19, 76, 152), drives=("shot", "step"),
    )
    return ods, system, observation, rows


def _row(rows, rule, M, drive="shot"):
    matches = [r for r in rows if r["rule"] == rule and r["M_total"] == M and r["drive"] == drive]
    assert len(matches) == 1, (rule, M, drive)
    return matches[0]


# --- the system and its observations -------------------------------------------

def test_the_observation_set_covers_magnetics_boundary_and_grid(study):
    ods, system, observation, rows = study
    assert observation["probe"].shape == (62, 950) and observation["flux_loop"].shape == (11, 950)
    assert observation["boundary_psi"].shape[1] == 950 and observation["boundary_b"].shape[0] == 2 * observation["boundary_psi"].shape[0]
    assert 400 < observation["grid_psi"].shape[0] < 800     # 25 x 33 box, inside the limiter
    assert len(observation["channels"]) == 73


def test_the_full_rank_reduced_wall_is_the_full_wall(study):
    ods, system, observation, rows = study
    from vaft.process.electromagnetics import solve_eddy_currents

    I_full = solve_eddy_currents(system["R_mat"], system["L_mat"], system["M_mat"], system["drive"], system["time"])
    ops = wm.reduced_operators(system["basis"], system["R_mat"], system["M_mat"], system["L_mat"])
    _, I_red = wm.solve_reduced_eddy(ops, system["drive"], system["time"], V=system["basis"].V())
    assert np.linalg.norm(I_red - I_full) / np.linalg.norm(I_full) < 1e-10


# --- full versus reduced ----------------------------------------------------------

def test_every_row_carries_every_metric_and_the_reference_cost(study):
    ods, system, observation, rows = study
    full = [r for r in rows if r["rule"] == "full"]
    assert {r["drive"] for r in full} == {"shot", "step"} and all(r["cost_s"] > 0 for r in full)
    for row in rows:
        if row["rule"] == "full":
            continue
        assert set(wr.METRICS) <= set(row) and "probe_peak" in row and "current_by_segment" in row
        assert sum(row["M_repr"]) == row["M_total"]


def test_decay_time_ranking_converges_badly_and_response_ranking_does_not(study):
    ods, system, observation, rows = study
    assert _row(rows, "tau", 19)["probe"] > 0.2
    assert _row(rows, "tau", 76)["probe"] > 0.15
    assert _row(rows, "output_weight", 76)["probe"] < 0.03
    assert _row(rows, "output_weight", 152)["probe"] < 5e-3
    assert _row(rows, "output_weight", 152)["grid_psi"] < 5e-3


def test_one_mode_per_segment_beats_the_nineteen_slowest_on_current_and_magnetics(study):
    """The boundary field is the one place the slowest modes hold their own
    (0.061 against 0.068); everywhere else spreading the modes over the
    segments wins by a factor of two or more."""
    ods, system, observation, rows = study
    uniform, slowest = _row(rows, "uniform", 19), _row(rows, "tau", 19)
    for metric in ("current_l2", "current_dissipation", "probe", "flux_loop", "grid_psi"):
        assert uniform[metric] < slowest[metric], metric


def test_moment_patterns_reach_percent_fidelity_with_tens_of_vectors(study):
    """The block Krylov space of the source coupling is the coil-controllable
    subspace, so with eight blocks (76 vectors) the PF programme's response
    is reproduced to 1e-4 where 152 eigenmodes reach 3e-3."""
    ods, system, observation, rows = study
    assert _row(rows, "moments", 19)["probe"] < 3e-2
    assert _row(rows, "moments", 76)["probe"] < 1e-3
    assert _row(rows, "moments", 76)["flux_loop"] < 1e-3
    assert _row(rows, "moments", 76)["probe"] < _row(rows, "output_weight", 152)["probe"]


def test_a_selection_made_for_the_pf_programme_transfers_worse_to_a_step(study):
    ods, system, observation, rows = study
    for rule, M in (("output_weight", 152), ("moments", 19), ("moments", 76)):
        assert _row(rows, rule, M, "step")["probe"] > _row(rows, rule, M, "shot")["probe"]
    assert _row(rows, "moments", 19, "step")["probe"] > 0.1      # two Krylov blocks miss the transient
    assert _row(rows, "moments", 76, "step")["probe"] < 0.02     # eight blocks carry it
    assert _row(rows, "output_weight", 152, "step")["probe"] < 0.05


def test_the_representation_order_reads_off_the_rows(study):
    ods, system, observation, rows = study
    order = wr.representation_order(rows, {"probe": 0.01, "flux_loop": 0.01, "grid_psi": 0.01})
    assert order["rules"]["tau"]["joint"] is None
    assert order["rules"]["output_weight"]["joint"]["M_total"] == 152
    assert order["rules"]["moments"]["joint"]["M_total"] <= 20
    assert order["rules"]["moments"]["by_metric"]["flux_loop"] <= 20
    with pytest.raises(wr.WallReductionError, match="unknown metrics"):
        wr.representation_order(rows, {"nonsense": 0.1})


def test_the_greedy_allocation_meets_five_percent_dissipation_well_below_full_rank(study):
    ods, system, observation, rows = study
    keep, history = wm.allocate_per_segment(
        system["basis"], system["R_mat"], system["M_mat"], system["L_mat"],
        system["drive"], system["time"], tolerance=0.05, step=4,
    )
    assert history[-1]["dissipation"] <= 0.05
    assert 40 < history[-1]["M_total"] < 200
    assert all(k.size >= 1 for k in keep)                    # every segment carries some of the response


# --- measurement versus model ------------------------------------------------------

def test_reduction_error_is_invisible_next_to_the_vessel_model_error(study):
    """A converged wall disagrees with the data exactly as the full one does,
    so what is left is the vessel model's error and not the truncation's --
    the separation vfit #10 asks for.  tau_19 is carried alongside as the
    counter-example that shows the claim is about convergence, not about
    reduction in general."""
    ods, system, observation, rows = study
    selections = {
        "tau_19": wm.select_slowest(system["basis"], 19),
        "moments_30": wm.moment_patterns(system["R_mat"], system["M_mat"], system["L_mat"], 3),
    }
    result = wr.experimental_comparison(ods, selections)
    full_median = result["full"]["measurement"]["improvement"]["median"]

    # The full wall removes about 78% of the no-wall residual on this shot
    # (0.7809, identical on Linux and Windows and across repeated runs, so the
    # bound is on the physics rather than on numerical spread), and 0.75 leaves
    # room for that to move without going quiet if the wall term collapses.
    # It sits above the 0.7 that test_eddy_vacuum_validation.py already asks of
    # the same quantity, so the two agree on what a working vessel model is.
    assert full_median > 0.75
    assert result["no_wall"]["measurement"]["improvement"]["median"] == 0.0

    # A converged reduction is indistinguishable from the full wall against the
    # data: moments_30 lands 0.0005 away, with 0.5% of the probe wall term
    # missing.  That is the separation -- measurement error is the vessel
    # model's, reduction error is the truncation's.
    moments_median = result["models"]["moments_30"]["measurement"]["improvement"]["median"]
    assert abs(moments_median - full_median) < 0.02

    # tau_19 is not a second converged model.  The module docstring records
    # decay-time ranking leaving ~28% probe error at nineteen modes, and it
    # misses 34% of the probe wall term here, so it is visibly worse against
    # the data too.  Asserting it matched the full wall was asking the study's
    # own counter-example to contradict the study.
    tau_median = result["models"]["tau_19"]["measurement"]["improvement"]["median"]
    assert full_median - tau_median > 0.05

    assert result["models"]["tau_19"]["reduction"]["wall_term_relative"]["b_field_pol_probe"] > 0.03
    assert result["models"]["moments_30"]["reduction"]["wall_term_relative"]["b_field_pol_probe"] < 0.01
    assert result["models"]["moments_30"]["current"]["relative_dissipation"] < 0.01
    assert result["window"][0] > result["interval"]["start"] and result["basis_digest"]


def test_41672_needs_the_remapped_coupling_and_then_reduces_like_39915():
    import vaft
    import vaft.omas

    try:
        path = vaft.data.sample(41672, "imas")
    except Exception:
        pytest.skip("sample 41672 is not available in this checkout")
    ods = vaft.omas.load(path)
    try:
        system = wr.wall_system(ods)
    except wm.WallModeError:
        system = wr.wall_system(ods, remap_em_coupling=True)
    V = wm.moment_patterns(system["R_mat"], system["M_mat"], system["L_mat"], 1)
    from vaft.process.electromagnetics import solve_eddy_currents

    I_full = solve_eddy_currents(system["R_mat"], system["L_mat"], system["M_mat"], system["drive"], system["time"])
    _, I_red = wm.solve_reduced_eddy(wm.combined_operators(V, system["R_mat"], system["M_mat"], system["L_mat"]),
                                     system["drive"], system["time"], V=V)
    assert np.linalg.norm(I_red - I_full) / np.linalg.norm(I_full) < 0.1
