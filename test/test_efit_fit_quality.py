"""EFIT goodness-of-fit and numerical-convergence metrics (issue #139).

Everything asserted here is computed from quantities submitted to EFIT or
produced by the EFIT run. No independent experimental comparison, cross-code
check, synthetic truth, uncertainty propagation or sensitivity study.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

import vaft.omas as vomas
from vaft.omas.efit_quality import (
    FAMILIES,
    classify_fit_role,
    constraint_table,
    convergence_metrics,
    efit_quality_metrics,
    fit_quality_metrics,
    normalized_residuals,
    run_test_z,
    sigma_unit_factor,
)
from vaft.validation import render_stage_plots, stage_plot_filenames


@pytest.fixture(scope="module")
def efit_ods():
    from vaft.omas.sample import sample_ods

    return sample_ods()


def _fit_ods(*, residuals, weight=0.01, unit_factor=1.0, dof=10, family="bpol_probe"):
    """An equilibrium whose stored chi-square encodes a chosen unit factor."""
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = 41234
    ods["equilibrium.time"] = [0.30]
    root = "equilibrium.time_slice.0"
    ods[f"{root}.time"] = 0.30
    ods[f"{root}.convergence.grad_shafranov_deviation_value"] = 1e-6
    ods[f"{root}.convergence.iterations_n"] = 12
    for index, residual in enumerate(residuals):
        base = f"{root}.constraints.{family}.{index}"
        ods[f"{base}.measured"] = 1.0 + residual
        ods[f"{base}.reconstructed"] = 1.0
        ods[f"{base}.weight"] = weight
        ods[f"{base}.chi_squared"] = (residual * weight / unit_factor) ** 2
        ods[f"{base}.source"] = f"{family}_ch{index}"
    aux = "equilibrium.code.parameters.time_slice.0.auxquantities"
    ods[f"{aux}.degrees_of_freedom"] = dof
    ods[f"{aux}.num_input_data"] = dof + 4
    ods[f"{aux}.num_fit_variables"] = 3
    ods[f"{aux}.num_hard_constraints"] = 1
    ods["equilibrium.code.parameters.time_slice.0.in1.error"] = 1e-5
    ods["equilibrium.code.parameters.time_slice.0.in1.mxiter"] = -100
    return ods


# --- A1/A2: the normalization, recovered from EFIT's own chi-square ---------

def test_unit_factor_is_recovered_from_the_stored_chi_square(efit_ods):
    factors = {}
    for family, _title, _unit, _scale, is_array in FAMILIES:
        table = constraint_table(efit_ods, time_slice=0, family=family, is_array=is_array)
        factors[family] = sigma_unit_factor(table)
    # B probes are fitted in the units the ODS stores.
    assert factors["bpol_probe"][0] == pytest.approx(1.0, rel=1e-4)
    # Flux loops are not: the ODS stores Wb, EFIT fitted Wb/rad.
    assert factors["flux_loop"][0] == pytest.approx(2 * math.pi, rel=1e-4)
    for family in ("bpol_probe", "flux_loop"):
        assert factors[family][1] < 1e-3, f"{family} unit factor is not consistent"


def test_normalized_residual_squared_is_the_stored_chi_square(efit_ods):
    for family, _t, _u, _s, is_array in FAMILIES:
        table = constraint_table(efit_ods, time_slice=0, family=family, is_array=is_array)
        k, _spread = sigma_unit_factor(table)
        if not np.isfinite(k):
            continue
        z = normalized_residuals(table, k)
        fitted = table.mask("enabled") & np.isfinite(z) & (table.chi_squared > 0)
        assert np.allclose(z[fitted] ** 2, table.chi_squared[fitted], rtol=1e-6), family


def test_the_unit_factor_is_derived_not_assumed():
    # A family fitted in some other unit is recovered just as well, so the
    # metric survives a convention change instead of silently misreporting.
    ods = _fit_ods(residuals=[1.0, -2.0, 3.0, -1.5], unit_factor=1000.0)
    table = constraint_table(ods, time_slice=0, family="bpol_probe")
    k, spread = sigma_unit_factor(table)
    assert k == pytest.approx(1000.0, rel=1e-9)
    assert spread < 1e-9


# --- A0: fitted versus prescribed -------------------------------------------

def test_pf_currents_are_prescribed_not_fitted(efit_ods):
    table = constraint_table(efit_ods, time_slice=0, family="pf_current")
    assert classify_fit_role(efit_ods, table, time_slice=0) == "prescribed"
    for family in ("bpol_probe", "flux_loop"):
        fitted = constraint_table(efit_ods, time_slice=0, family=family)
        assert classify_fit_role(efit_ods, fitted, time_slice=0) == "fitted"


def test_a_prescribed_family_is_excluded_from_the_aggregate(efit_ods):
    metrics = fit_quality_metrics(efit_ods, time_slice=0)
    assert metrics["families"]["pf_current"]["fit_role"] == "prescribed"
    assert "z_rms" not in metrics["families"]["pf_current"]
    assert "pf_current" not in metrics["chi_squared_share"]


def test_an_exact_flag_marks_a_family_prescribed():
    ods = _fit_ods(residuals=[1.0, -1.0, 2.0])
    ods["equilibrium.time_slice.0.constraints.bpol_probe.0.exact"] = 1
    table = constraint_table(ods, time_slice=0, family="bpol_probe")
    assert classify_fit_role(ods, table, time_slice=0) == "prescribed"


# --- A3/A4: reduced chi-square and share ------------------------------------

def test_reduced_chi_square_uses_efits_own_degrees_of_freedom(efit_ods):
    metrics = fit_quality_metrics(efit_ods, time_slice=0)
    dof = metrics["degrees_of_freedom"]
    inputs = metrics["degrees_of_freedom_inputs"]
    assert dof == pytest.approx(
        inputs["num_input_data"] - inputs["num_fit_variables"] - inputs["num_hard_constraints"]
    )
    assert metrics["chi_squared_reduced"] == pytest.approx(
        metrics["chi_squared_total"] / dof
    )
    # A placeholder dof of 1 would have made this equal the raw total.
    assert dof > 1


def test_chi_square_share_identifies_what_determines_the_solution(efit_ods):
    metrics = fit_quality_metrics(efit_ods, time_slice=0)
    share = metrics["chi_squared_share"]
    assert sum(v for v in share.values() if np.isfinite(v)) == pytest.approx(1.0)
    # Shot 39915's fit is carried entirely by the plasma-current constraint.
    assert share["ip"] > 0.999
    assert share["bpol_probe"] < 1e-6 and share["flux_loop"] < 1e-6


# --- A6/A9: bias and residual structure -------------------------------------

def test_a_systematic_offset_shows_as_bias_but_not_as_structure():
    rng = np.random.default_rng(11)
    scatter = rng.normal(size=60)
    offset = scatter - scatter.mean() + 1.5  # a clean constant offset
    ods = _fit_ods(residuals=offset * 100.0, weight=0.01)
    entry = fit_quality_metrics(ods, time_slice=0)["families"]["bpol_probe"]
    assert entry["z_bias"] == pytest.approx(1.5, abs=0.05)
    assert entry["z_bias_significant"] is True
    # The two metrics are independent: an offset is not spatial structure, and
    # the signs left after it are still shuffled.
    assert abs(entry["residual_structure"]["run_test_z"]) < 2.0


def test_random_scatter_shows_neither_bias_nor_structure():
    rng = np.random.default_rng(7)
    ods = _fit_ods(residuals=rng.normal(size=200) * 100.0, weight=0.01)
    entry = fit_quality_metrics(ods, time_slice=0)["families"]["bpol_probe"]
    assert entry["z_bias_significant"] is False
    assert abs(entry["residual_structure"]["run_test_z"]) < 2.0
    assert abs(entry["residual_structure"]["lag1_autocorrelation"]) < 0.3


def test_a_coherent_sign_block_fires_the_run_test():
    # Half the array positive, half negative: the classic unmodelled-field
    # signature, with zero net bias so only the structure metric can catch it.
    values = np.concatenate([np.full(40, 1.2), np.full(40, -1.2)])
    ods = _fit_ods(residuals=values * 100.0, weight=0.01)
    entry = fit_quality_metrics(ods, time_slice=0)["families"]["bpol_probe"]
    assert entry["z_bias"] == pytest.approx(0.0, abs=1e-9)
    assert entry["z_bias_significant"] is False
    assert entry["residual_structure"]["run_test_z"] < -2.0
    assert entry["residual_structure"]["lag1_autocorrelation"] > 0.9


def test_run_test_needs_both_signs_and_enough_channels():
    assert not np.isfinite(run_test_z(np.array([1.0, 1.0, 1.0])))
    assert not np.isfinite(run_test_z(np.array([1.0, -1.0])))
    assert np.isfinite(run_test_z(np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])))


def test_max_normalized_residual_names_its_channel():
    residuals = np.array([0.1, 0.2, 9.0, 0.3]) * 100.0
    entry = fit_quality_metrics(
        _fit_ods(residuals=residuals, weight=0.01), time_slice=0
    )["families"]["bpol_probe"]
    assert entry["z_abs_max"] == pytest.approx(9.0, rel=1e-6)
    assert entry["z_abs_max_channel"] == "bpol_probe_ch2"
    assert entry["outlier_fraction"]["gt_3sigma"] == pytest.approx(0.25)


# --- Part B: convergence -----------------------------------------------------

def test_the_two_tolerances_are_reported_separately(efit_ods):
    # write_a/chkerr: the iteration exits on `error`, but EFIT *accepts* the
    # slice against `errmin` when iconvr == 2. They are different tests.
    block = convergence_metrics(efit_ods, time_slice=0)["error"]
    assert block["exit_tolerance"] == pytest.approx(1e-5)
    assert block["exit_tolerance_source"] == "in1"
    assert block["iconvr"] == 2
    assert block["acceptance_tolerance_name"] == "errmin"
    # VEST never sets errmin, so acceptance rests on EFIT's own default.
    assert block["acceptance_tolerance"] == pytest.approx(1e-2)
    assert block["acceptance_tolerance_source"] == "efit_default"
    assert block["exit_ratio"] == pytest.approx(block["final_error"] / 1e-5)
    assert block["acceptance_ratio"] == pytest.approx(block["final_error"] / 1e-2)


def test_the_requested_exit_tolerance_is_inert_for_this_configuration(efit_ods):
    """`error` never gates a VEST run, so its ratio must not read as a failure.

    residu() consumes `error` only through `idone`, which breaks the inner
    `equilibrium: do ii=1,nxiter` loop; with nxiter == 1 that loop is a single
    pass regardless, and for iconvr == 2 the outer loop leaves through `ichisq`,
    which never looks at `error`.
    """
    block = convergence_metrics(efit_ods, time_slice=0)["error"]
    assert block["nxiter"] == 1
    assert block["exit_tolerance_effective"] is False
    assert "never gates this run" in block["exit_tolerance_inert_reason"]
    # The ratio is still reported, but it is not evidence of non-convergence.
    assert block["reached_exit_tolerance"] is False
    assert block["exit_ratio"] > 100


def test_a_multi_pass_inner_loop_makes_the_exit_tolerance_effective():
    ods = _fit_ods(residuals=[1.0, -1.0, 2.0])
    ods["equilibrium.code.parameters.time_slice.0.out1.nxiter"] = 5
    block = convergence_metrics(ods, time_slice=0)["error"]
    assert block["exit_tolerance_effective"] is True
    assert "exit_tolerance_inert_reason" not in block


def test_the_iconvr2_stopping_criterion_is_the_metric_with_content(efit_ods):
    """`terror <= errmin` and `chisq <= saicon` are preconditions of stopping.

    response_matrix.F90 sets ichisq only when nniter >= minite(8), errorm <=
    errmin, saisq <= saicon and the chi-square has stalled, so a run that
    stopped that way satisfies all of them by construction. What discriminates
    is whether it stopped that way at all.
    """
    ods = _with_afile(efit_ods)
    block = convergence_metrics(ods, time_slice=0)
    assert block["iterations"]["minimum_iterations"] == 8
    assert block["iterations"]["iterations"] >= 8
    assert block["iterations"]["hit_cap"] is False
    assert block["iterations"]["stopped_on_criterion"] is True
    assert block["error"]["within_acceptance_tolerance"] is True


def test_a_run_that_exhausts_its_iterations_did_not_stop_on_the_criterion(efit_ods):
    ods = _with_afile(efit_ods)
    ods["equilibrium.time_slice.0.convergence.iterations_n"] = 100
    block = convergence_metrics(ods, time_slice=0)["iterations"]
    assert block["hit_cap"] is True
    assert block["stopped_on_criterion"] is False


def test_too_few_iterations_cannot_be_a_criterion_stop(efit_ods):
    # minite is hard-coded in response_matrix.F90; ichisq cannot fire below it.
    ods = _with_afile(efit_ods)
    ods["equilibrium.time_slice.0.convergence.iterations_n"] = 3
    assert (
        convergence_metrics(ods, time_slice=0)["iterations"]["stopped_on_criterion"]
        is False
    )


def test_a_non_iconvr2_run_is_judged_against_error_not_errmin():
    ods = _fit_ods(residuals=[1.0, -1.0, 2.0])
    ods["equilibrium.code.parameters.time_slice.0.out1.iconvr"] = 3
    block = convergence_metrics(ods, time_slice=0)["error"]
    assert block["acceptance_tolerance_name"] == "error"
    assert block["acceptance_tolerance"] == pytest.approx(1e-5)
    assert block["acceptance_tolerance_source"] == "in1"


def test_iteration_cap_is_detected():
    ods = _fit_ods(residuals=[1.0, -1.0, 2.0])
    assert convergence_metrics(ods, time_slice=0)["iterations"]["hit_cap"] is False
    ods["equilibrium.time_slice.0.convergence.iterations_n"] = 100
    assert convergence_metrics(ods, time_slice=0)["iterations"]["hit_cap"] is True


def _with_afile(ods):
    from vaft.data import read_aeqdsk
    from vaft.data.resources import data_path

    copy = ods.copy()
    read_aeqdsk(data_path("efit/a039915.00319")).to_omas(copy, time_index=0)
    return copy


def _with_history(history):
    ods = _fit_ods(residuals=[1.0, -1.0, 2.0])
    ods["equilibrium.code.parameters.time_slice.0.meqdsk.variables.cerror.data"] = (
        np.asarray(history, dtype=float)
    )
    return ods


def test_a_stagnating_history_is_flagged():
    block = convergence_metrics(
        _with_history([1e-2, 1e-3, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]), time_slice=0
    )["history"]
    assert block["available"] is True
    assert block["final_decade_rate"] == pytest.approx(0.0, abs=0.05)
    assert block["stagnated"] is True


def test_a_healthy_history_converges_monotonically():
    block = convergence_metrics(
        _with_history([1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]), time_slice=0
    )["history"]
    assert block["monotonic_fraction"] == pytest.approx(1.0)
    assert block["final_decade_rate"] < -0.5
    assert block["stagnated"] is False


def test_an_oscillating_history_is_not_monotonic():
    block = convergence_metrics(
        _with_history([1e-2, 1e-3, 5e-3, 1e-4, 5e-4, 1e-5]), time_slice=0
    )["history"]
    assert block["monotonic_fraction"] < 1.0


def test_history_is_reported_unavailable_rather_than_guessed(efit_ods):
    block = convergence_metrics(efit_ods, time_slice=0)["history"]
    assert block["available"] is False
    assert "cerror" in block["reason"]
    # Everything else is still produced.
    assert np.isfinite(convergence_metrics(efit_ods, time_slice=0)["error"]["final_error"])


def test_the_efit_verdict_is_read_when_an_afile_was_parsed(efit_ods):
    without = convergence_metrics(efit_ods, time_slice=0)["verdict"]
    assert without["accepted"] is None and "no a-file" in without["reason"]

    ods = _with_afile(efit_ods)
    verdict = convergence_metrics(ods, time_slice=0)["verdict"]
    assert verdict["accepted"] is True
    assert (verdict["jflag"], verdict["lflag"]) == (1, 0)
    assert verdict["limiter_location"] == "IN"
    assert verdict["fit_type"] == "MAG"
    # The claim must not overreach: jflag says chkerr was satisfied.
    assert "chkerr" in verdict["meaning"]
    assert "converged" not in verdict


def test_acceptance_and_reaching_the_exit_tolerance_are_independent(efit_ods):
    """The heart of it: EFIT accepts a slice that never met its own tolerance."""
    ods = _with_afile(efit_ods)
    block = convergence_metrics(ods, time_slice=0)
    assert block["verdict"]["accepted"] is True
    assert block["error"]["reached_exit_tolerance"] is False
    assert block["error"]["final_error_source"] == "aeqdsk.terror"
    assert block["error"]["exit_ratio"] > 10
    assert block["error"]["within_acceptance_tolerance"] is True


def test_the_chi_square_precondition_is_reported_with_its_caveats(efit_ods):
    ods = _with_afile(efit_ods)
    block = convergence_metrics(ods, time_slice=0)["error"]
    assert block["chi_squared_limit_name"] == "saicon"
    assert block["chi_squared_limit"] == pytest.approx(80.0)
    assert block["chi_squared_limit_source"] == "efit_default"
    # Below 1 by construction for a criterion stop, so this is not a near-miss.
    assert block["chi_squared_margin"] < 1.0
    # saisq is reset to saiold at the stop, and includes saisref and chiecc,
    # which have no OMAS constraint family -- so it is not the family sum.
    assert block["chi_squared_comparable_to_family_sum"] is False


def test_the_afile_terror_takes_precedence_over_a_cerror_history(efit_ods):
    ods = _with_afile(efit_ods)
    ods["equilibrium.code.parameters.time_slice.0.meqdsk.variables.cerror.data"] = (
        np.array([1e-1, 1e-2, 5e-3], dtype=float)
    )
    block = convergence_metrics(ods, time_slice=0)
    assert block["error"]["final_error_source"] == "aeqdsk.terror"
    assert block["history"]["available"] is True
    # The two are the same quantity, so a disagreement is worth surfacing.
    assert block["history"]["agrees_with_aeqdsk_terror"] is False


# --- B7-B9: EFIT's outputs against each other --------------------------------

def test_efit_outputs_are_self_consistent_on_the_packaged_shot(efit_ods):
    for index in range(len(efit_ods["equilibrium.time_slice"])):
        block = convergence_metrics(efit_ods, time_slice=index)["self_consistency"]
        assert block["ip_relative_spread"] < 1e-6
        assert block["psi_axis_grid_offset"] < 1e-5
        # The magnetic axis is a local flux extremum, not the grid's global one.
        assert block["magnetic_axis_is_local_extremum"] is True


def test_a_perturbed_global_ip_is_caught(efit_ods):
    ods = efit_ods.copy()
    original = float(ods["equilibrium.time_slice.0.global_quantities.ip"])
    ods["equilibrium.time_slice.0.global_quantities.ip"] = original * 1.05
    block = convergence_metrics(ods, time_slice=0)["self_consistency"]
    assert block["ip_relative_spread"] == pytest.approx(0.05 / 1.05, rel=1e-3)


def test_a_perturbed_psi_axis_is_caught(efit_ods):
    ods = efit_ods.copy()
    root = "equilibrium.time_slice.0.global_quantities"
    axis = float(ods[f"{root}.psi_axis"])
    boundary = float(ods[f"{root}.psi_boundary"])
    ods[f"{root}.psi_axis"] = axis + 0.25 * (boundary - axis)
    block = convergence_metrics(ods, time_slice=0)["self_consistency"]
    assert block["psi_axis_grid_offset"] > 0.1


# --- the stage ---------------------------------------------------------------

def test_the_efit_stage_writes_both_new_figures_and_the_metrics(tmp_path, efit_ods):
    directory = tmp_path / "plot"
    manifest = render_stage_plots("efit", efit_ods, directory, shot=39915)
    written = {path.name for path in directory.iterdir()}
    assert {"efit_fit_quality.png", "efit_convergence.png"} <= written
    assert set(stage_plot_filenames("efit", required_only=True)) <= written

    metrics = manifest["metrics"]
    assert metrics["summary"]["chi_squared_reduced_median"] > 0
    # No a-file was mapped, so no verdict is claimed rather than one invented.
    assert metrics["summary"]["slices_with_verdict"] == 0
    assert metrics["summary"]["slices_reaching_exit_tolerance"] == 0
    fit = metrics["slices"][0]["fit"]
    assert fit["tier"]["chi_squared_reduced"] == "primary"
    assert fit["tier"]["residual_structure"] == "diagnostic"
    assert metrics["slices"][0]["convergence"]["tier"]["solver_settings"] == "metadata"


def test_the_new_figures_render_for_the_packaged_shot(tmp_path, efit_ods):
    from vaft.plot import save_figure

    for name in ("equilibrium_overview_fit_quality", "equilibrium_overview_convergence"):
        figure, _axes = getattr(vomas, f"plot_{name}")(efit_ods)
        target = tmp_path / f"{name}.png"
        save_figure(figure, target, dpi=100)
        assert target.stat().st_size > 10_000, name


def test_an_empty_efit_product_still_short_circuits(tmp_path):
    minimal = ODS(consistency_check=False)
    minimal["equilibrium.ids_properties.comment"] = "EFIT output unavailable"
    manifest = render_stage_plots("efit", minimal, tmp_path / "plot")
    assert manifest["status"] == "empty"
    assert not list((tmp_path / "plot").iterdir())
    # Metrics still run on an empty product; there is simply nothing in them.
    assert manifest["metrics"]["slice_count"] == 0


def test_quality_metrics_cover_every_slice(efit_ods):
    metrics = efit_quality_metrics(efit_ods)
    assert metrics["slice_count"] == len(efit_ods["equilibrium.time_slice"])
    assert all("fit" in entry and "convergence" in entry for entry in metrics["slices"])
