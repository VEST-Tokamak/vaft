"""Scientific regression checks for retained native results; no solver or SSH."""

import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omas import ODS, load_omas_json, save_omas_json
from test_cgyro_adapter import tglf_local

from vaft.code.gacode.cgyro import (
    cgyro_input_from_tglf,
    collect_cgyro_outputs,
    read_field_history,
)
from vaft.code.gacode.result_identity import encode, parameters
from vaft.code.gacode.tglf import collect_tglf_outputs
from vaft.machine_mapping.gyrokinetics import core_transport_from_cgyro
from vaft.machine_mapping.turbulence import core_transport_from_tglf
from vaft.process.gyrokinetic_comparison import (
    assess_comparison,
    prepare_diagnostic_traces,
    select_time,
)
from vaft.process.gyrokinetics import (
    field_transport_contributions,
    rebin_contributions,
    reconstruct_field,
    reduce_time_series,
    spectral_integral,
    zonal_fraction,
    zonal_turbulence_intensity,
)

DATA = Path(__file__).parent / "data/gacode"
NATIVE = DATA / "cgyro_nonlinear_39915_em"


def test_real_native_reader_and_upstream_evidence():
    h = read_field_history(NATIVE)
    assert isinstance(h.raw, np.memmap)
    assert h.read().shape == (63, 16, 21)
    assert h.metadata["partial_tail_bytes"] == 0
    assert h.metadata["version"]["commit"] == "b493397"
    assert h.metadata["native_signed_length"] < 0
    assert json.loads((NATIVE / "oracle.json").read_text())["raw_field_array_equal"]
    with pytest.raises(FileNotFoundError):
        read_field_history(NATIVE, field="a_parallel")


@pytest.mark.parametrize("high", [False, True])
def test_precision_restart_partial_tail_and_selection(tmp_path, high):
    shutil.copytree(NATIVE, tmp_path / "run")
    p = tmp_path / "run"
    h = read_field_history(p)
    cube = np.asarray(h.raw).copy()
    eq = (p / "out.cgyro.equilibrium").read_text().split()
    eq[-1] = "1" if high else "0"
    (p / "out.cgyro.equilibrium").write_text("\n".join(eq))
    binary = (
        cube.astype(np.complex128 if high else np.complex64).tobytes(order="F") + b"x"
    )
    (p / "bin.cgyro.kxky_phi").write_bytes(binary)
    table = np.loadtxt(p / "out.cgyro.time")
    table[2, 0] = table[0, 0]
    np.savetxt(p / "out.cgyro.time", table)
    result = read_field_history(p, stride=2)
    assert result.raw.dtype == np.dtype("complex128" if high else "complex64")
    assert result.metadata["partial_tail_bytes"] == 1
    assert result.metadata["restart_records_excluded"] == 1
    assert 2 not in result.records
    np.testing.assert_array_equal(result.read(), cube[1:, 0, :, :][..., result.records])
    with pytest.raises(ValueError):
        read_field_history(p, theta_indices=[1])


def test_known_fourier_mode_phase_amplitude_and_parseval():
    # CGYRO's one-sided half-sum: one nonzonal coefficient A -> Re A exp(i*x-i*y).
    c = np.zeros((3, 2), complex)
    c[2, 1] = 2 * np.exp(0.3j)
    field = reconstruct_field(c, [-1, 0, 1], [0, 1], length_x=2 * np.pi, nx=9, ny=7)
    expected = 2 * np.cos(field.x[:, None] - field.y[None, :] + 0.3)
    np.testing.assert_allclose(field.values, expected, atol=1e-14)
    assert np.mean(field.values**2) == pytest.approx(abs(c[2, 1]) ** 2 / 2)
    spectrum = np.fft.fft2(field.values) / field.values.size
    assert spectrum[1, -1] == pytest.approx(c[2, 1] / 2)
    np.testing.assert_allclose(
        np.fft.ifft2(np.fft.fft2(field.values)).real, field.values, atol=1e-14
    )
    total = reconstruct_field(c, [-1, 0, 1], [0, 1], length_x=2 * np.pi)
    zonal = reconstruct_field(
        c, [-1, 0, 1], [0, 1], length_x=2 * np.pi, component="zonal"
    )
    nonzonal = reconstruct_field(
        c, [-1, 0, 1], [0, 1], length_x=2 * np.pi, component="nonzonal"
    )
    np.testing.assert_allclose(total.values, zonal.values + nonzonal.values)


def test_fraction_ratio_weights_and_degenerate_limits():
    phi = np.ones((3, 2, 4), complex)
    r = zonal_turbulence_intensity(phi, [0, 1], weights=[1, 2])
    np.testing.assert_allclose(r.fraction, r.ratio / (1 + r.ratio))
    phi[:, 1] = 0
    np.testing.assert_allclose(zonal_fraction(phi, [0, 1]), 1)
    phi[:] = 0
    assert np.isnan(zonal_fraction(phi, [0, 1])).all()


def test_signed_spectra_and_cancellation():
    v = np.array([3.0, -2.0, 4.0])
    e = np.array([0.0, 1.0, 3.0, 4.0])
    new = rebin_contributions(v, e, [0.0, 0.5, 2.0, 4.0])
    assert new.sum() == pytest.approx(v.sum())
    assert spectral_integral(
        v / np.diff(e), representation="density", edges=e
    ) == pytest.approx(v.sum())
    with pytest.raises(ValueError):
        spectral_integral(v, representation="density")
    out = field_transport_contributions([2.0, -1.0])
    np.testing.assert_allclose(out["fractions"], [2.0, -1.0])
    assert field_transport_contributions([1.0, -1.0])["cancellation"]


def test_irregular_time_gap_missing_data_and_independent_qualification():
    t = np.array([0.0, 0.2, 0.7, 1.0, 2.0, 2.1, 3.0, 4.0])
    y = 2 * t + 1
    r = reduce_time_series(t, y, window=(0, 4), max_gap=2)
    assert r["mean"] == pytest.approx(5)
    assert r["coverage"] == 1
    assert r["qualification"]["numerical_convergence"] == "unverified"
    g = reduce_time_series([0, 1, 10, 11], [1, 1, 1, 1], window=(0, 11), max_gap=2)
    assert g["coverage"] == pytest.approx(2 / 11)
    assert g["standard_error"] is None
    y[3] = np.nan
    r = reduce_time_series(t, y, window=(0, 4), max_gap=2)
    assert r["coverage"] < 1
    z = reduce_time_series(t, np.zeros_like(t), window=(0, 4), max_gap=2)
    assert z["relative_drift"] is None


def test_comparison_axes_and_time_selection():
    a = {
        "display": {"unit": "W/m2"},
        "state": {"state_id": "a"},
        "model": {"solver": "TGLF"},
        "qualification": {"convergence": "unverified"},
    }
    b = {**a, "model": {"solver": "CGYRO"}}
    r = assess_comparison(a, b, intent="matched_baseline", controlled=("model.solver",))
    assert r.axes["display"]["status"] == "matched"
    assert r.axes["model"]["status"] == "controlled_difference"
    assert r.axes["qualification"]["status"] == "unverified"
    assert assess_comparison({}, {}).axes["state"]["status"] == "unverified"
    assert select_time([0.3, 0.1], 0.11)["status"] == "unverified"
    assert select_time([0.3, 0.1], 0.11, tolerance=0.02)["index"] == 1
    with pytest.raises(ValueError):
        select_time([0.3, 0.1], 0.11, tolerance=0.001)
    with pytest.raises(ValueError):
        select_time([0.1, 0.1], 0.1)


def test_diagnostics_have_no_future_samples_or_bridged_gaps():
    trace = prepare_diagnostic_traces(
        {"q": {"time": [0, 1, 10, 11], "values": [1, 2, 3, 4], "max_gap": 2}}, [0.5, 10]
    )["q"]
    assert trace["frames"][0][0].tolist() == [0]
    assert np.isnan(trace["frames"][1][1][-1])


def pair(sat):
    name = "tglf_vest_39915_r0.70_" + ("sat0-es" if sat == 0 else "sat2-em-bper")
    return tglf_local(), collect_tglf_outputs(DATA / name)


PROFILE = SimpleNamespace(rmin=np.array([0.0, 1.0]), rho=np.array([0.0, 1.0]))


def test_multimodel_identity_roundtrip_and_conflict_is_atomic(tmp_path):
    ods = ODS()
    p0 = pair(0)
    p2 = pair(2)
    core_transport_from_tglf(ods, [p0], PROFILE, time=0.317, state_identity="shot39915")
    core_transport_from_tglf(ods, [p2], PROFILE, time=0.317, state_identity="shot39915")
    assert len(ods["core_transport.model"]) == 2
    core_transport_from_tglf(ods, [p0], PROFILE, time=0.317, state_identity="shot39915")
    assert len(ods["core_transport.model"]) == 2
    before = ods.copy()
    with pytest.raises(ValueError):
        core_transport_from_tglf(ods, [p0], PROFILE, time=0.317, state_identity="other")
    np.testing.assert_array_equal(
        ods["core_transport.model.0.profiles_1d.0.electrons.energy.flux"],
        before["core_transport.model.0.profiles_1d.0.electrons.energy.flux"],
    )
    core_transport_from_tglf(
        ods, [p0], PROFILE, time=0.32, time_index=1, state_identity="shot39915-newtime"
    )
    out = collect_cgyro_outputs(NATIVE)
    core_transport_from_cgyro(
        ods,
        [(cgyro_input_from_tglf(p0[0]), out, (100, 104))],
        PROFILE,
        time=0.317,
        state_identity="shot39915",
    )
    assert len(ods["core_transport.model"]) == 3
    save_omas_json(ods, str(tmp_path / "ods.json"))
    loaded = load_omas_json(str(tmp_path / "ods.json"))
    for i in range(3):
        record = parameters(loaded[f"core_transport.model.{i}.code.parameters"])
        assert record["slices"]
    assert (
        len(parameters(loaded["core_transport.model.0.code.parameters"])["slices"]) == 2
    )


def test_foreign_anomalous_entries_are_preserved():
    ods = ODS()
    ods["core_transport.model.0.identifier.index"] = 6
    ods["core_transport.model.0.code.name"] = "OTHER"
    core_transport_from_tglf(ods, [pair(0)], PROFILE)
    assert ods["core_transport.model.0.code.name"] == "OTHER"
    assert ods["core_transport.model.1.code.name"] == "TGLF"


def test_parameters_raw_and_decoded():
    import xmltodict

    record = {"solver": "CGYRO", "configuration": {"N_FIELD": 2}}
    xml = encode(record)
    assert parameters(xml) == parameters(xmltodict.parse(xml)) == record


def test_native_field_plot_and_canonical_animation():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from vaft.plot._animation import Animation
    from vaft.plot.gyrokinetics import plot_gyrokinetics_field_potential

    h = read_field_history(NATIVE, stride=10)
    movie = plot_gyrokinetics_field_potential(
        h,
        animation=True,
        diagnostics={"fZ": {"time": h.time, "values": [0.1, 0.2, 0.3], "unit": "1"}},
    )
    assert isinstance(movie, Animation)
    assert movie.metadata["driver"]["values"] == h.time.tolist()
    assert "levels" in movie.normalization()
    frame = next(movie.frames())
    assert frame.ndim == 3
    plt.close("all")


def test_reapplication_preserves_other_time_slices_and_foreign_ods_on_error():
    ods = ODS()
    p = pair(0)
    core_transport_from_tglf(ods, [p], PROFILE, time=0.317, state_identity="a")
    core_transport_from_tglf(
        ods, [p], PROFILE, time=0.32, time_index=1, state_identity="b"
    )
    core_transport_from_tglf(ods, [p], PROFILE, time=0.317, state_identity="a")
    assert ods["core_transport.model.0.profiles_1d.1.time"] == 0.32
    before = ods.copy()
    with pytest.raises(ValueError):
        core_transport_from_tglf(
            ods, [p], PROFILE, time=0.33, time_index=1, state_identity="c"
        )
    assert len(ods["core_transport.model.0.profiles_1d"]) == 2
    assert (
        ods["core_transport.model.0.profiles_1d.1.time"]
        == before["core_transport.model.0.profiles_1d.1.time"]
    )


def test_input_provenance_zero_and_default_are_distinct():
    from vaft.code.gacode.result_identity import input_value_status, read_native_input

    raw, _ = read_native_input(NATIVE, "CGYRO")
    assert raw["N_FIELD"] == 2 and raw["FIELD_PRINT_FLAG"] == 0
    measured = input_value_status(
        {"GAMMA_E": 0}, {"GAMMA_E": 0}, {"gamma_e": {"kind": "measured"}}
    )
    default = input_value_status({"GAMMA_E": 0}, {}, {})
    assumed = input_value_status({"GAMMA_E": 0}, {"GAMMA_E": 0}, {})
    assert measured["GAMMA_E"]["status"] == "measured"
    assert default["GAMMA_E"]["status"] == "solver_default"
    assert assumed["GAMMA_E"]["status"] == "assumed"
    assert default["MACH"]["status"] == "unavailable"


def test_audit_covers_parsed_native_families_and_existing_transformations():
    from dataclasses import fields

    from vaft.code.gacode.cgyro.outputs import CgyroOutputs
    from vaft.code.gacode.output_audit import mapping_transformations, output_audit
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    rows = output_audit()
    for solver, typ in [("CGYRO", CgyroOutputs), ("TGLF", TglfOutputs)]:
        assert {f.name for f in fields(typ)} <= {
            r["quantity"] for r in rows if r["solver"] == solver
        }
    assert not any("review_required" in str(r) for r in rows)
    assert mapping_transformations()["CGYRO"]


def test_saturated_intensity_matches_both_real_tglf_presets():
    from vaft.process.gyrokinetics import saturated_intensity

    for sat in (0, 2):
        _, o = pair(sat)
        actual = saturated_intensity(o.ql_intensity_spectrum, o.field_spectrum[:, :, 1])
        np.testing.assert_allclose(actual, o.intensity_spectrum, rtol=1e-12, atol=1e-12)


def test_local_simulation_replacement_is_atomic():
    from vaft.machine_mapping.gyrokinetics import gyrokinetics_local_from_tglf

    ods = ODS()
    local, native = pair(0)
    gyrokinetics_local_from_tglf(ods, local, native, time=0.317)
    original = ods.copy()
    with pytest.raises(ValueError):
        gyrokinetics_local_from_tglf(ods, replace(local, q_loc=2.9), native, time=0.317)
    assert (
        ods["gyrokinetics_local.flux_surface.q"]
        == original["gyrokinetics_local.flux_surface.q"]
    )


def test_underresolved_blocks_do_not_invent_uncertainty():
    r = reduce_time_series([0, 1], [1, 1], window=(0, 1), blocks=4)
    assert r["standard_error"] is None
    assert r["qualification"]["data_sufficiency"] == "insufficient_data"


def test_real_native_manifest_is_byte_exact():
    import hashlib

    manifest = json.loads((NATIVE / "manifest.json").read_text())
    for name, info in manifest["files"].items():
        assert (
            hashlib.sha256((NATIVE / name).read_bytes()).hexdigest()
            == info["fixture_sha256"]
        )


def test_imas_roundtrip_preserves_model_identity(tmp_path):
    pytest.importorskip("imas")
    import vaft.imas

    ods = ODS()
    core_transport_from_tglf(
        ods, [pair(0)], PROFILE, time=0.317, state_identity="39915"
    )
    vaft.imas.save(ods, tmp_path / "transport.nc")
    with vaft.imas.load(tmp_path / "transport.nc") as loaded:
        converted = loaded.to_omas()
    assert parameters(
        converted["core_transport.model.0.code.parameters"]
    ) == parameters(ods["core_transport.model.0.code.parameters"])


def test_nonhermitian_zonal_plane_is_rejected():
    c = np.zeros((3, 2), complex)
    c[0, 0] = 1j
    with pytest.raises(ValueError, match="Hermitian"):
        reconstruct_field(c, [-1, 0, 1], [0, 1], length_x=2 * np.pi)


def test_same_explicit_state_id_cannot_hide_changed_physics():
    ods = ODS()
    local, native = pair(0)
    core_transport_from_tglf(
        ods, [(local, native)], PROFILE, time=0.317, state_identity="same"
    )
    with pytest.raises(ValueError, match="physical inputs"):
        core_transport_from_tglf(
            ods,
            [(replace(local, q_loc=9.0), native)],
            PROFILE,
            time=0.317,
            state_identity="same",
        )
    assert len(ods["core_transport.model"]) == 1


def test_equally_near_times_and_missing_qualified_time_are_rejected():
    with pytest.raises(ValueError, match="ambiguous"):
        select_time([0.1, 0.3], 0.2)
    with pytest.raises(ValueError):
        select_time([np.nan], 0.2, tolerance=0.1)
