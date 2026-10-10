"""Locality QA for local CGYRO runs: box adequacy vs the local approximation (#1354)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from omas import ODS

from vaft.code.gacode import cgyro
from vaft.code.gacode.cgyro import collect_cgyro_outputs, locality_report, scale_lengths
from vaft.machine_mapping import gyrokinetics as gk

from test_cgyro_adapter import tglf_local, write_run


def write_kxky_phi(directory, *, n_radial, length, n_n, steps, ell):
    """|phi(kx)|^2 = exp(-(kx ell)^2) on every n >= 1 mode: C(dx) = exp(-dx^2/(4 ell^2)),
    so the 1/e correlation length is exactly 2 ell."""
    p = np.arange(n_radial) - n_radial // 2
    kx = 2 * np.pi * p / length
    amplitude = np.exp(-0.5 * (kx * ell) ** 2)
    field = np.zeros((n_radial, 1, n_n, steps), dtype=np.complex64)
    field[:, 0, 1:, :] = amplitude[:, None, None]
    field[:, 0, 0, :] = 50.0  # a zonal mode must not enter the turbulent spectrum
    field.flatten(order="F").tofile(directory / "bin.cgyro.kxky_phi")


def nonlinear_run(tmp_path, *, ell=3.0, length=200.0):
    directory = write_run(tmp_path / "run", n_n=8, n_radial=256, n_theta=4, steps=10,
                          flux=True, exit_message="Normal", length=length)
    write_kxky_phi(directory, n_radial=256, length=length, n_n=8, steps=10, ell=ell)
    return collect_cgyro_outputs(directory)


def test_the_correlation_length_is_measured_from_the_spectrum(tmp_path):
    run = nonlinear_run(tmp_path, ell=3.0)
    result = cgyro.radial_correlation_length(run)
    assert result["l_corr"] == pytest.approx(6.0, rel=0.03)
    assert result["box_rho_s"] == pytest.approx(200.0)


def test_scale_lengths_come_from_the_gradients_cgyro_was_given():
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    scales = scale_lengths(local)
    assert scales["L_Ti"] == pytest.approx(1 / 1.4)
    assert scales["L_Te"] == pytest.approx(1 / 1.5)
    assert scales["L_n"] == pytest.approx(1 / 4.9)
    assert scales["L_q"] == pytest.approx(0.7 / 0.92)


def test_a_flat_profile_has_no_finite_scale():
    local = cgyro.cgyro_input_from_tglf(tglf_local(rlts=np.array([0.0, 0.0, 0.0])))
    assert scale_lengths(local)["L_Ti"] is None


def test_box_and_locality_are_judged_separately(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())   # rho* = 1e-3/0.3
    report = locality_report(local, nonlinear_run(tmp_path, ell=3.0))
    rho_star = 1e-3 / 0.3
    assert report["rho_star"] == pytest.approx(rho_star)
    assert report["L_x_over_a"] == pytest.approx(200 * rho_star)
    assert report["l_corr_over_L_x"] == pytest.approx(6.0 / 200, rel=0.03)
    assert report["verdict"]["box"] == "box_adequate"
    # the shortest profile scale is L_n = a/4.9
    assert report["limiting_scale"] == "L_n"
    assert report["epsilon_local"] == pytest.approx(6.0 * rho_star * 4.9, rel=0.03)
    assert report["verdict"]["locality"] == "local_ok"
    assert report["thresholds"]["kind"] == "heuristic"


def test_a_box_that_truncates_the_turbulence_is_flagged_whatever_its_size_in_a(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    report = locality_report(local, nonlinear_run(tmp_path, ell=8.0, length=40.0))
    # C never falls to 1/e inside half the periodic box: only a lower bound is known
    assert report["l_corr_is_lower_bound"] and report["l_corr_lower_bound_rho_s"] == 20.0
    assert report["verdict"]["box"] == "box_too_small"


def test_large_turbulence_on_a_steep_profile_is_questionable(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local(rlns=np.array([20.0, 20.0, 20.0])))
    report = locality_report(local, nonlinear_run(tmp_path, ell=10.0, length=400.0))
    assert report["epsilon_local"] > 0.3
    assert report["verdict"]["locality"] == "local_questionable"


def test_a_linear_run_reports_scales_but_no_locality_verdict(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    report = locality_report(local, collect_cgyro_outputs(write_run(tmp_path / "lin")))
    assert report["rho_star"] is not None and report["L_x_over_a"] is not None
    assert report["l_corr_rho_s"] is None
    assert report["verdict"]["locality"] == "unknown"
    assert "not a multi-mode" in report["correlation"]["reason"]


def test_locality_travels_with_the_imas_result(tmp_path):
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    run = nonlinear_run(tmp_path, ell=3.0)
    report = locality_report(local, run)
    ods = ODS()
    gk.gyrokinetics_local_from_cgyro(
        ods, local, run, provenance={"parameters": {"NONLINEAR_FLAG": 1, "N_FIELD": 1}},
        flux_window=(2.0, 5.0), locality=report)
    parameters = ods["gyrokinetics_local.code.parameters"]
    assert "<locality_verdict>local_ok</locality_verdict>" in parameters
    assert "locality_epsilon_local" in parameters


def test_build_nonlinear_rebuilds_the_local_input_from_its_summary(tmp_path):
    import importlib.util
    import json
    from pathlib import Path

    run = nonlinear_run(tmp_path, ell=3.0)
    local = cgyro.cgyro_input_from_tglf(tglf_local())
    norm = local.normalisation
    (Path(run.directory) / "local_summary.json").write_text(json.dumps({
        "r_over_a": local.r_over_a, "geometry": dict(local.geometry),
        "species": {k: [float(x) for x in v] for k, v in local.species.items()},
        "names": list(local.names),
        "normalisation": {"gyroradius": norm.gyroradius, "minor_radius": norm.minor_radius,
                          "b_unit": norm.b_unit, "sound_speed": norm.sound_speed,
                          "electron_density": norm.electron_density,
                          "electron_temperature": norm.electron_temperature}}))
    spec = importlib.util.spec_from_file_location(
        "build_nonlinear",
        Path(__file__).parents[1] / "workflow" / "gyrokinetic" / "build_nonlinear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = module._locality(Path(run.directory), run, (2.0, 5.0))
    assert report["verdict"]["locality"] == "local_ok"
    assert report["l_corr_rho_s"] == pytest.approx(6.0, rel=0.03)


def test_an_electrostatic_low_ky_nonlinear_run_is_refused_before_anything_runs(tmp_path):
    """#1484: the ES omega_H branch grew alone at k_y=0.1 and drove the whole ES run."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "run_nonlinear",
        Path(__file__).parents[1] / "workflow" / "gyrokinetic" / "run_nonlinear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = ["--filedb", str(tmp_path), "--labels", str(tmp_path / "l.json"),
              "--out", str(tmp_path), "--state", "39915:0.317:magnetics", "--r-over-a", "0.7"]
    with pytest.raises(SystemExit, match="omega_H"):
        module.main(common + ["--field-model", "es", "--ky", "0.1"])
    assert module.ES_MIN_KY >= 0.2


def _build_nonlinear_module():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "build_nonlinear",
        Path(__file__).parents[1] / "workflow" / "gyrokinetic" / "build_nonlinear.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_batch_means_gives_the_window_mean_and_the_spread_of_block_means():
    """A periodic burst train: the mean is exact, and with one burst per block the block
    means agree, so the standard error vanishes; with half a period per block it does not."""
    module = _build_nonlinear_module()
    t = np.linspace(0.0, 400.0, 4001)
    y = 5.0 + 4.0 * np.sin(2 * np.pi * t / 100.0)
    whole = module.batch_means(t, y, 4)                  # one period per block
    assert whole["mean"] == pytest.approx(5.0, abs=1e-6)
    assert whole["block_means"] == pytest.approx([5.0] * 4, abs=1e-6)
    assert whole["standard_error"] == pytest.approx(0.0, abs=1e-6)
    half = module.batch_means(t, y, 8)                   # half a period per block
    assert half["standard_error"] > 0.5
    with pytest.raises(ValueError):
        module.batch_means(t, y, 1)


def test_the_zonal_fraction_is_one_when_only_n0_carries_power(tmp_path):
    module = _build_nonlinear_module()
    run = nonlinear_run(tmp_path, ell=3.0)
    from pathlib import Path

    grid = run.grid
    n_radial, theta, n_n = grid["n_radial"], grid["theta_plot"], grid["n_n"]
    steps = len(run.time)
    field = np.zeros((n_radial, theta, n_n, steps), dtype=np.complex64)
    field[:, :, 0, :] = 1.0
    field.reshape(-1, order="F").tofile(Path(run.directory) / "bin.cgyro.kxky_phi")
    out = module.zonal_fraction(run)
    assert np.allclose(out["fraction"], 1.0)
    field[:, :, 1, :] = 1.0
    field.reshape(-1, order="F").tofile(Path(run.directory) / "bin.cgyro.kxky_phi")
    assert np.allclose(module.zonal_fraction(run)["fraction"], 1.0 / 2.0)


def test_a_requeued_restart_leaves_stray_records_that_are_dropped_everywhere(tmp_path):
    """A node failure + requeue can interleave a few out-of-order print records and
    leave one flux record past the last time line (#1484, job 771927): the run must
    still parse, with time strictly increasing and every time-history array aligned."""
    from pathlib import Path

    from test_cgyro_adapter import write_run

    directory = write_run(tmp_path / "run", n_n=4, n_radial=8, n_theta=4, steps=8,
                          flux=True, exit_message="Normal")
    t = [1.0, 2.0, 3.0, 4.0, 2.5, 5.0, 3.5, 6.0]           # two stray records
    (directory / "out.cgyro.time").write_text(
        "\n".join(f"{x:.6e} 1e-3 1e-3 0.0" for x in t))
    flux = np.zeros((3, 3, 1, 4, 9), dtype=np.float32)      # one record past the time lines
    flux[:, 1] = np.array([10, 20, 30, 40, -1, 50, -1, 60, 99], dtype=np.float32)
    flux.flatten(order="F").tofile(directory / "bin.cgyro.ky_flux")
    run = collect_cgyro_outputs(directory)
    assert np.all(np.diff(run.time) > 0)
    assert run.time.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert run.record_index.tolist() == [0, 1, 2, 3, 5, 7]
    assert run.flux[0, 1, 0, 0].tolist() == [10, 20, 30, 40, 50, 60]
    assert run.growth_rate.shape[-1] == run.time.size
    raw = np.arange(8.0)[None, :]
    assert run.align_records(raw)[0].tolist() == [0, 1, 2, 3, 5, 7]
    back = type(run).from_dict(run.to_dict())
    assert back.record_index.tolist() == run.record_index.tolist()


def test_a_clean_run_has_no_record_index():
    run = None
    from pathlib import Path
    # the committed linear fixture is a clean single-segment run
    run = collect_cgyro_outputs(Path(__file__).parent / "data" / "gacode" / "cgyro_linear_48224_r0.7_em")
    assert run.record_index is None


def _requeued_run(tmp_path, *, times, flux_records, n_n=4):
    from test_cgyro_adapter import write_run

    directory = write_run(tmp_path / "run", n_n=n_n, n_radial=8, n_theta=4,
                          steps=len(times), flux=True, exit_message="Normal")
    (directory / "out.cgyro.time").write_text(
        "\n".join(f"{x:.6e} 1e-3 1e-3 0.0" for x in times))
    flux = np.zeros((3, 3, 1, n_n, len(flux_records)), dtype=np.float32)
    flux[:, 1] = np.asarray(flux_records, dtype=np.float32)
    flux.flatten(order="F").tofile(directory / "bin.cgyro.ky_flux")
    return directory


def test_a_short_file_after_a_requeue_gives_the_aligned_prefix_not_a_shifted_one(tmp_path):
    directory = _requeued_run(tmp_path, times=[1, 2, 3, 4, 2.5, 5, 3.5, 6],
                              flux_records=[10, 20, 30, 40, -1, 50, -1])   # cut mid-step
    run = collect_cgyro_outputs(directory)
    assert run.flux[0, 1, 0, 0].tolist() == [10, 20, 30, 40, 50]            # not lost, not shifted
    raw = np.arange(7.0)[None, :]                                          # a 7-record kxky file
    assert run.align_records(raw)[0].tolist() == [0, 1, 2, 3, 5]


def test_frequency_is_aligned_even_without_stray_records():
    from types import SimpleNamespace
    from vaft.code.gacode.cgyro.outputs import CgyroOutputs

    run = CgyroOutputs(directory=".", time=np.array([1.0, 2.0, 3.0]))
    longer = np.arange(4.0)[None, :]                                       # a trailing record
    assert run.align_records(longer)[0].tolist() == [0, 1, 2]


@pytest.mark.parametrize("n_time", [1, 3])
def test_a_trailing_flux_record_is_not_read_as_a_fourth_moment(tmp_path, n_time):
    times = list(range(1, n_time + 1))
    directory = _requeued_run(tmp_path, times=times, flux_records=list(range(10, 10 * (n_time + 2), 10)))
    run = collect_cgyro_outputs(directory)
    assert run.flux.shape[1] == 3 and run.flux.shape[-1] == n_time
    assert run.flux[0, 1, 0, 0].tolist() == [10.0 * (k + 1) for k in range(n_time)]


def test_batch_means_tile_the_window_exactly():
    module = _build_nonlinear_module()
    t = np.linspace(0.0, 10.0, 11)                                         # samples on the edges
    y = t.copy()
    out = module.batch_means(t, y, 5)
    assert out["block_means"] == pytest.approx([1.0, 3.0, 5.0, 7.0, 9.0])
    assert out["mean"] == pytest.approx(5.0)
