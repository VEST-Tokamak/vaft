"""Routine transport states (#1428): time pairing, Ti hierarchy, readiness, identity, driver.

Built from the packaged 48224 kinetic ODS, extended to several slices whose times are
deliberately offset so that pairing by index would pick the wrong profile -- the
one-slice sample cannot show that bug.  No solver runs: the driver test replaces
``run_tglf_case`` with a fake that writes a solved native container.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from vaft.process.transport_state import (
    DEFAULT_SURFACES,
    TransportStateKey,
    assess_neo_readiness,
    assess_tglf_readiness,
    physics_parameters,
    resolve_transport_state,
    run_identity,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "vaft" / "data" / "kineticEfit" / "ods_48224_300ms.json"
REG05 = ROOT / "test" / "data" / "gacode" / "tglf_reg05"


@pytest.fixture(scope="module")
def sample():
    from omas import load_omas_json

    return load_omas_json(str(SAMPLE), consistency_check=False)


def _multi_slice(sample, eq_times=(0.300, 0.302), cp_times=(0.2990, 0.3002, 0.3021)):
    """Equilibrium and core_profiles with different slice counts and offset times.

    Each core_profiles slice scales Te by a distinct factor, so the profile a state
    resolves to says which slice it was paired with.
    """
    from omas import ODS

    ods = ODS(consistency_check=False)
    for index, t in enumerate(eq_times):
        ods[f"equilibrium.time_slice.{index}"] = copy.deepcopy(sample["equilibrium.time_slice.0"])
        ods[f"equilibrium.time_slice.{index}.time"] = t
    ods["equilibrium.time"] = np.asarray(eq_times)
    ods["equilibrium.vacuum_toroidal_field"] = copy.deepcopy(sample["equilibrium.vacuum_toroidal_field"])
    ods["equilibrium.vacuum_toroidal_field.b0"] = np.full(len(eq_times), float(
        np.atleast_1d(sample["equilibrium.vacuum_toroidal_field.b0"])[0]))
    ods["equilibrium.ids_properties.homogeneous_time"] = 1
    for index, t in enumerate(cp_times):
        ods[f"core_profiles.profiles_1d.{index}"] = copy.deepcopy(sample["core_profiles.profiles_1d.0"])
        ods[f"core_profiles.profiles_1d.{index}.time"] = t
        te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
        ods[f"core_profiles.profiles_1d.{index}.electrons.temperature"] = te * (1.0 + 0.1 * index)
    ods["core_profiles.time"] = np.asarray(cp_times)
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    if "core_profiles.vacuum_toroidal_field" in sample:
        ods["core_profiles.vacuum_toroidal_field.r0"] = sample["core_profiles.vacuum_toroidal_field.r0"]
        ods["core_profiles.vacuum_toroidal_field.b0"] = np.full(len(cp_times), float(
            np.atleast_1d(sample["core_profiles.vacuum_toroidal_field.b0"])[0]))
    ods["dataset_description.data_entry.pulse"] = 48224
    return ods


def _electron_only(ods):
    out = copy.deepcopy(ods)
    index = 0
    while f"core_profiles.profiles_1d.{index}" in out:
        del out[f"core_profiles.profiles_1d.{index}.ion"]
        index += 1
    return out


def _key(t=0.302, lineage="magnetics-only"):
    return TransportStateKey(48224, t, lineage)


# --------------------------------------------------------------------------- pairing


def test_states_pair_by_time_not_index(sample):
    ods = _multi_slice(sample)
    state = resolve_transport_state(ods, _key(0.302), efit_label="good")
    assert state.resolved, state.reasons
    assert state.times["equilibrium_index"] == 1
    # Index pairing would take core_profiles slice 1 (0.3002 s); time takes slice 2.
    assert state.times["core_profiles_index"] == 2
    assert state.times["dt_s"] == pytest.approx(1e-4, abs=1e-9)
    te_axis = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"])[0] * 1.2
    assert state.profile.te[0] * 1e3 == pytest.approx(te_axis, rel=1e-6)


def test_a_profile_beyond_the_tolerance_is_not_paired(sample):
    ods = _multi_slice(sample, cp_times=(0.2990, 0.3002, 0.3030))
    state = resolve_transport_state(ods, _key(0.302), efit_label="good")
    assert state.status == "insufficient"
    assert state.reasons == ("no_core_profiles_within_tolerance",)
    assert state.times["dt_s"] == pytest.approx(1e-3, abs=1e-9)


def test_a_state_time_off_every_slice_is_refused(sample):
    state = resolve_transport_state(_multi_slice(sample), _key(0.3010), efit_label="good")
    assert state.reasons == ("no_equilibrium_within_tolerance",)


def test_unreconstructible_slices_are_refused(sample):
    with pytest.raises(ValueError, match="Unreconstructible"):
        resolve_transport_state(sample, _key(0.3), efit_label="unreconstructible")


def test_the_lineage_vocabulary_is_closed():
    with pytest.raises(ValueError):
        TransportStateKey(48224, 0.3, "kinetic")


# --------------------------------------------------------------------------- Ti


def test_measured_ion_temperature_is_preferred(sample):
    state = resolve_transport_state(sample, _key(0.3), efit_label="good")
    assert state.ti_lineage == "measured"
    assert state.provenance["ti"]["kind"] == "measured"


def test_policy_ratio_fills_an_electron_only_state_without_touching_the_source(sample):
    ods = _electron_only(_multi_slice(sample))
    state = resolve_transport_state(ods, _key(0.302), efit_label="admissible")
    assert state.resolved, state.reasons
    assert state.ti["kind"] == "assumed"
    assert state.ti_lineage == f"assumed_ti_te_{state.ti['ratio']:g}"
    assert "vest.yaml" in state.ti["source"]
    np.testing.assert_allclose(state.profile.ti[0], state.ti["ratio"] * state.profile.te, rtol=1e-9)
    assert list(state.profile.name) == ["H+", "C6+"]
    assert "core_profiles.profiles_1d.2.ion.0" not in ods


def test_a_caller_ratio_is_recorded_as_such(sample):
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_label="good",
                                    ti_te_ratio=0.17, ti_te_ratio_sigma=0.08)
    assert state.ti["source"] == "caller argument"
    assert state.ti_lineage == "assumed_ti_te_0.17"


def test_an_inferred_profile_outranks_the_policy(sample):
    ods = _electron_only(sample)
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
    state = resolve_transport_state(ods, _key(0.3), efit_label="good",
                                    inferred_ti={"temperature": 0.5 * te, "method": "pressure_partition"})
    assert state.ti["kind"] == "inferred"
    np.testing.assert_allclose(state.profile.ti[0], 0.5 * state.profile.te, rtol=1e-9)


def test_no_fallback_means_insufficient(sample):
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_label="good",
                                    ti_te_ratio=None)
    assert state.reasons == ("no_defensible_ion_temperature",)
    assert assess_tglf_readiness(state).status == "insufficient"


# --------------------------------------------------------------------------- readiness


def test_readiness_lists_every_surface_and_the_declared_assumptions(sample):
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_label="good")
    report = assess_tglf_readiness(state)
    assert report.status == "conditional"
    assert {"ti_assumed", "composition_policy", "no_exb_shear"} <= set(report.conditions)
    assert [s.r_over_a for s in report.surfaces] == list(DEFAULT_SURFACES)
    assert all(s.ready and s.local_input is not None for s in report.surfaces)


def test_a_surface_outside_the_profile_is_reported_not_dropped(sample):
    state = resolve_transport_state(sample, _key(0.3), efit_label="good")
    report = assess_tglf_readiness(state, (0.5, 1.0))
    assert [s.status for s in report.surfaces] == ["ready", "outside_profile_domain"]
    assert report.runnable


def test_neo_readiness_is_separate(sample):
    state = resolve_transport_state(sample, _key(0.3), efit_label="good")
    assert assess_neo_readiness(state).status == "conditional"


def test_missing_shape_profiles_are_derived_and_marked(sample):
    ods = copy.deepcopy(sample)
    for name in ("elongation", "triangularity_upper", "triangularity_lower"):
        del ods[f"equilibrium.time_slice.0.profiles_1d.{name}"]
    state = resolve_transport_state(ods, _key(0.3), efit_label="good")
    assert state.resolved, state.reasons
    assert state.provenance["shape"]["kind"] == "derived"
    reference = resolve_transport_state(sample, _key(0.3), efit_label="good")
    assert reference.provenance["shape"]["kind"] == "reconstructed"
    inner = slice(len(state.profile.kappa) // 10, int(0.9 * len(state.profile.kappa)))
    np.testing.assert_allclose(state.profile.kappa[inner], reference.profile.kappa[inner], rtol=0.02)


# --------------------------------------------------------------------------- identity


def test_identity_is_deterministic_and_tracks_every_upstream_choice(sample):
    from vaft.code.gacode.tglf import TGLFConfig

    ods = _electron_only(sample)
    base = resolve_transport_state(ods, _key(0.3), efit_label="good")
    params = physics_parameters(TGLFConfig(sat_rule=3, use_bper=True))
    ident = run_identity(base, solver="tglf", parameters=params, surface=0.5)
    again = resolve_transport_state(ods, _key(0.3), efit_label="good")
    assert ident == run_identity(again, solver="tglf", parameters=params, surface=0.5)
    variants = [
        run_identity(resolve_transport_state(ods, _key(0.3), efit_label="good", ti_te_ratio=0.5),
                     solver="tglf", parameters=params, surface=0.5),
        run_identity(resolve_transport_state(ods, _key(0.3), efit_label="good", z_eff=1.5),
                     solver="tglf", parameters=params, surface=0.5),
        run_identity(base, solver="tglf", parameters=physics_parameters(TGLFConfig(sat_rule=2)),
                     surface=0.5),
        run_identity(base, solver="tglf", parameters=params, surface=0.6),
        run_identity(base, solver="tglf", parameters=params, surface=0.5, solver_revision="x"),
    ]
    assert len({ident, *variants}) == 1 + len(variants)


def test_runtime_settings_stay_out_of_identity():
    from vaft.code.gacode.tglf import TGLFConfig

    assert physics_parameters(TGLFConfig(timeout=10, n_mpi=4)) == physics_parameters(TGLFConfig())


# --------------------------------------------------------------------------- TGLF spectra


def test_spectra_parse_and_the_ky_sum_is_the_total_flux():
    from vaft.code.gacode.tglf.outputs import TglfOutputs, collect_tglf_outputs

    native = collect_tglf_outputs(REG05)
    assert native.ky_spectrum.shape == (21,)
    assert native.growth_rate.shape == native.frequency.shape == (21, 2)
    assert native.sum_flux_spectrum.shape == (2, 1, 21, 5)
    # Each row is weight * flux, so the ky sum over fields is gbflux (to its 5 digits).
    totals = native.sum_flux_spectrum[..., 1].sum(axis=(1, 2))
    np.testing.assert_allclose(totals, native.gbflux["energy"], rtol=1e-4)
    round_trip = TglfOutputs.from_dict(json.loads(json.dumps(native.to_dict())))
    np.testing.assert_array_equal(round_trip.sum_flux_spectrum, native.sum_flux_spectrum)


# --------------------------------------------------------------------------- driver


@pytest.fixture()
def driver():
    path = ROOT / "workflow" / "transport_atlas" / "run_tglf.py"
    spec = importlib.util.spec_from_file_location("transport_atlas_run_tglf", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _write_product(root: Path, stage: str, shot: int, name: str, ods, status="success"):
    from vaft.omas import save

    base = root / stage / str(shot)
    (base / "output").mkdir(parents=True)
    (base / "metadata").mkdir()
    save(ods, base / "output" / name)
    (base / "metadata" / "manifest.json").write_text(json.dumps({"status": status}))


def test_driver_runs_good_and_admissible_states_only(driver, sample, tmp_path, monkeypatch):
    from omas import ODS

    import vaft.code.gacode.tglf as tglf
    from vaft.code.gacode.tglf import TGLFResult
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    ods = _electron_only(_multi_slice(sample, eq_times=(0.300, 0.302), cp_times=(0.300, 0.302)))
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    cp["dataset_description"] = ods["dataset_description"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [
        {"shot": 48224, "time_ms": 300, "label": "good"},
        {"shot": 48224, "time_ms": 302, "label": "unreconstructible"},
    ]}))

    def fake_run(profile, rho, workdir, config=None, *, check=True):
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        if rho > 0.75:  # one failing surface: the state must come back partial
            return TGLFResult(returncode=None, runtime_status="timeout", workdir=workdir)
        flux = np.array([1.0, 2.0, 0.1]) * rho
        native = TglfOutputs(directory=str(workdir), version={"revision": "test"},
                             gbflux={"particle": 0.1 * flux, "energy": flux,
                                     "momentum": 0 * flux, "exchange": 0 * flux},
                             grid={"n_species": 3, "n_xgrid": 4})
        return TGLFResult(returncode=0, runtime_status="completed", workdir=workdir,
                          outputs_native=native)

    monkeypatch.setattr(tglf, "run_tglf_case", fake_run)
    out = tmp_path / "out"
    assert driver.main(["--filedb", str(filedb), "--labels", str(labels), "--out", str(out),
                        "--lineages", "magnetics-only", "--workers", "2"]) == 0

    rows = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert [(r["time_efit_s"], r["efit_label"], r["status"]) for r in rows] == [(0.3, "good", "partial")]
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["enumeration"]["excluded_unreconstructible"] == 1
    state = json.loads((out / "48224" / "magnetics-only" / "00300" / "state.json").read_text())
    failed = [s for s in state["surfaces"] if s["status"] == "failed"]
    assert [s["runtime_status"] for s in failed] == ["timeout"]
    mapped = state["core_transport"]["surfaces"]
    assert [round(s["r_over_a"], 2) for s in mapped] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert all(0 < s["rho_tor_norm"] < 1 for s in mapped)
    assert state["core_transport"]["code"] == {"name": "TGLF", "version": "test"}

    # A second run reuses every solved surface by identity and re-runs only the failure.
    calls = []
    monkeypatch.setattr(tglf, "run_tglf_case", lambda *a, **k: calls.append(a[1]) or fake_run(*a, **k))
    driver.main(["--filedb", str(filedb), "--labels", str(labels), "--out", str(out),
                 "--lineages", "magnetics-only", "--workers", "1"])
    assert calls == [0.8]
