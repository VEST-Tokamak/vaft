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

from vaft.machine_mapping.core_profiles import inferred_ti_text, policy_for_ods
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


def _key(t=0.302, lineage="magnetics"):
    return TransportStateKey(48224, t, lineage)


# --------------------------------------------------------------------------- pairing


def test_states_pair_by_time_not_index(sample):
    ods = _multi_slice(sample)
    state = resolve_transport_state(ods, _key(0.302), efit_quality="good")
    assert state.resolved, state.reasons
    assert state.times["equilibrium_index"] == 1
    # Index pairing would take core_profiles slice 1 (0.3002 s); time takes slice 2.
    assert state.times["core_profiles_index"] == 2
    assert state.times["dt_s"] == pytest.approx(1e-4, abs=1e-9)
    te_axis = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"])[0] * 1.2
    assert state.profile.te[0] * 1e3 == pytest.approx(te_axis, rel=1e-6)


def test_a_profile_beyond_the_tolerance_is_not_paired(sample):
    ods = _multi_slice(sample, cp_times=(0.2990, 0.3002, 0.3030))
    state = resolve_transport_state(ods, _key(0.302), efit_quality="good")
    assert state.status == "insufficient"
    assert state.reasons == ("no_core_profiles_within_tolerance",)
    assert state.times["dt_s"] == pytest.approx(1e-3, abs=1e-9)


def test_a_state_time_off_every_slice_is_refused(sample):
    state = resolve_transport_state(_multi_slice(sample), _key(0.3010), efit_quality="good")
    assert state.reasons == ("no_equilibrium_within_tolerance",)


def test_unreconstructible_slices_are_refused(sample):
    with pytest.raises(ValueError, match="Unreconstructible"):
        resolve_transport_state(sample, _key(0.3), efit_quality="unreconstructible")


def test_the_key_follows_contract_v1():
    key = TransportStateKey(48224, 0.30000004, "electron_kinetic")
    assert key.time_efit_s == 0.3 and key.as_dict()["efit_lineage"] == "electron_kinetic"


def test_the_lineage_vocabulary_is_closed():
    with pytest.raises(ValueError):
        TransportStateKey(48224, 0.3, "kinetic")


# --------------------------------------------------------------------------- Ti


def test_measured_ion_temperature_is_preferred(sample):
    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    assert state.ti_lineage == "measured"
    assert state.provenance["ti"]["kind"] == "measured"


def test_policy_ratio_fills_an_electron_only_state_without_touching_the_source(sample):
    ods = _electron_only(_multi_slice(sample))
    state = resolve_transport_state(ods, _key(0.302), efit_quality="admissible")
    assert state.resolved, state.reasons
    assert state.ti["kind"] == "assumed"
    assert state.ti_lineage == "ti_eq_te_assumed"
    assert "vest.yaml" in state.ti["source"]
    np.testing.assert_allclose(state.profile.ti[0], state.ti["ratio"] * state.profile.te, rtol=1e-9)
    assert list(state.profile.name) == ["H+", "C6+"]
    assert "core_profiles.profiles_1d.2.ion.0" not in ods


def test_a_caller_ratio_is_recorded_as_such(sample):
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_quality="good",
                                    ti_te_ratio=0.17, ti_te_ratio_sigma=0.08)
    assert state.ti["source"] == "caller argument"
    assert state.ti_lineage == "ti_te_0.17_assumed"


def test_an_inferred_profile_outranks_the_policy(sample):
    ods = _electron_only(sample)
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good",
                                    inferred_ti={"temperature": 0.5 * te, "method": "pressure_partition",
                                                 "time": 0.3})
    assert state.ti["kind"] == "inferred"
    assert state.ti_lineage == "pressure_partition_inferred"  # contract v1 spelling
    np.testing.assert_allclose(state.profile.ti[0], 0.5 * state.profile.te, rtol=1e-9)


def test_an_inferred_profile_for_another_time_is_refused(sample):
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_quality="good",
                                    inferred_ti={"temperature": 0.5 * te, "time": 0.31})
    assert state.reasons == ("inferred_ti_time_mismatch",)


def test_a_nan_placeholder_ion_temperature_is_not_a_measurement(sample):
    ods = copy.deepcopy(sample)
    te = np.asarray(ods["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float)
    ods["core_profiles.profiles_1d.0.ion.0.temperature"] = np.full_like(te, np.nan)
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good")
    assert state.ti["kind"] == "assumed"
    assert np.all(np.isfinite(state.profile.ti))


def test_no_fallback_means_insufficient(sample):
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_quality="good",
                                    ti_te_ratio=None)
    assert state.reasons == ("no_defensible_ion_temperature",)
    assert assess_tglf_readiness(state).status == "insufficient"


# --------------------------------------------------------------------------- readiness


def test_readiness_lists_every_surface_and_the_declared_assumptions(sample):
    state = resolve_transport_state(_electron_only(sample), _key(0.3), efit_quality="good")
    report = assess_tglf_readiness(state)
    assert report.status == "conditional"
    assert {"ti_assumed", "composition_policy", "no_exb_shear"} <= set(report.conditions)
    assert [s.r_over_a for s in report.surfaces] == list(DEFAULT_SURFACES)
    assert all(s.ready and s.local_input is not None for s in report.surfaces)


def test_a_surface_outside_the_profile_is_reported_not_dropped(sample):
    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    report = assess_tglf_readiness(state, (0.5, 1.0))
    assert [s.status for s in report.surfaces] == ["ready", "outside_profile_domain"]
    assert report.runnable


def test_neo_readiness_is_separate(sample):
    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    assert assess_neo_readiness(state).status == "conditional"


def test_missing_shape_profiles_are_derived_and_marked(sample):
    ods = copy.deepcopy(sample)
    for name in ("elongation", "triangularity_upper", "triangularity_lower"):
        del ods[f"equilibrium.time_slice.0.profiles_1d.{name}"]
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good")
    assert state.resolved, state.reasons
    assert state.provenance["shape"]["kind"] == "derived"
    reference = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    assert reference.provenance["shape"]["kind"] == "reconstructed"
    inner = slice(len(state.profile.kappa) // 10, int(0.9 * len(state.profile.kappa)))
    np.testing.assert_allclose(state.profile.kappa[inner], reference.profile.kappa[inner], rtol=0.02)


# --------------------------------------------------------------------------- identity


def test_identity_is_deterministic_and_tracks_every_upstream_choice(sample):
    from vaft.code.gacode.tglf import TGLFConfig

    ods = _electron_only(sample)
    base = resolve_transport_state(ods, _key(0.3), efit_quality="good")
    params = physics_parameters(TGLFConfig(sat_rule=3, use_bper=True))
    ident = run_identity(base, solver="tglf", parameters=params, surface=0.5)
    again = resolve_transport_state(ods, _key(0.3), efit_quality="good")
    assert ident == run_identity(again, solver="tglf", parameters=params, surface=0.5)
    variants = [
        run_identity(resolve_transport_state(ods, _key(0.3), efit_quality="good", ti_te_ratio=0.5),
                     solver="tglf", parameters=params, surface=0.5),
        run_identity(resolve_transport_state(ods, _key(0.3), efit_quality="good", z_eff=1.5),
                     solver="tglf", parameters=params, surface=0.5),
        run_identity(base, solver="tglf", parameters=physics_parameters(TGLFConfig(sat_rule=2)),
                     surface=0.5),
        run_identity(base, solver="tglf", parameters=params, surface=0.6),
        run_identity(base, solver="tglf", parameters=params, surface=0.5, solver_revision="x"),
    ]
    assert len({ident, *variants}) == 1 + len(variants)


def test_identity_follows_content_not_names(sample):
    from vaft.code.gacode.tglf import TGLFConfig

    params = physics_parameters(TGLFConfig())
    ods = _electron_only(sample)
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)

    def ident(o, **kw):
        return run_identity(resolve_transport_state(o, _key(0.3), efit_quality="good", **kw),
                            solver="tglf", parameters=params, surface=0.5)

    hotter = copy.deepcopy(ods)
    hotter["core_profiles.profiles_1d.0.electrons.temperature"] = 1.1 * te
    assert ident(ods) != ident(hotter)  # same key, same (absent) input hashes
    one = {"temperature": 0.5 * te, "method": "m", "time": 0.3}
    two = {"temperature": 0.6 * te, "method": "m", "time": 0.3}
    assert ident(ods, inferred_ti=one) != ident(ods, inferred_ti=two)
    moved = {"core_profiles": {"path": "/a", "sha256": "x"}}
    assert ident(ods, inputs=moved) == ident(ods, inputs={"core_profiles": {"path": "/b", "sha256": "x"}})
    assert ident(ods, ti_te_ratio_sigma=0.1) == ident(ods, ti_te_ratio_sigma=0.9)


def test_runtime_settings_stay_out_of_identity():
    from vaft.code.gacode.tglf import TGLFConfig

    assert physics_parameters(TGLFConfig(timeout=10, n_mpi=4)) == physics_parameters(TGLFConfig())
    # The memory reservation decides where a run fits, not what it computes (F7).
    assert physics_parameters(TGLFConfig(memory_mb=2048)) == physics_parameters(TGLFConfig())


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
    assert driver.main(["--filedb", str(filedb), "--labels", str(labels), "--out", str(tmp_path / "out"),
                        "--lineages", "magnetics", "--workers", "2",
                        "--sat-rule", "2", "--field-model", "es"]) == 0
    out = tmp_path / "out" / "tglf-sat2-es"

    rows = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert [(r["time_efit_s"], r["efit_quality"], r["status"]) for r in rows] == [(0.3, "good", "partial")]
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["enumeration"]["excluded_unreconstructible"] == 1
    state = json.loads((out / "48224" / "magnetics" / "00300" / "state.json").read_text())
    failed = [s for s in state["surfaces"] if s["status"] == "failed"]
    assert [s["runtime_status"] for s in failed] == ["timeout"]
    mapped = state["core_transport"]["surfaces"]
    assert [round(s["r_over_a"], 2) for s in mapped] == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert all(0 < s["rho_tor_norm"] < 1 for s in mapped)
    assert state["core_transport"]["code"] == {"name": "TGLF", "version": "test"}

    assert manifest["status_counts"] == {"partial": 1}

    # A second run reuses every solved surface by identity and re-runs only the failure.
    calls = []
    monkeypatch.setattr(tglf, "run_tglf_case", lambda *a, **k: calls.append(a[1]) or fake_run(*a, **k))
    driver.main(["--filedb", str(filedb), "--labels", str(labels), "--out", str(tmp_path / "out"),
                 "--lineages", "magnetics", "--workers", "1", "--sat-rule", "2", "--field-model", "es"])
    assert calls == [0.8]
    assert state["tglf_config"] == "tglf-sat2-es"
    assert (state["tglf_parameters"]["sat_rule"], state["tglf_parameters"]["use_bper"],
            state["tglf_parameters"]["use_bpar"]) == (2, False, False)


def test_the_tglf_configuration_has_no_default(driver, tmp_path):
    with pytest.raises(SystemExit):
        driver.main(["--filedb", str(tmp_path), "--labels", str(tmp_path / "l.json"),
                     "--out", str(tmp_path / "o")])
    assert driver.config_label(3, "em-bper") == "tglf-sat3-em-bper"
    assert driver.FIELD_MODELS["em-bper-bpar"] == {"use_bper": True, "use_bpar": True}


def test_an_infrastructure_error_is_recorded_per_surface(driver, sample, tmp_path, monkeypatch):
    import vaft.code.gacode.tglf as tglf

    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    surface = assess_tglf_readiness(state, (0.5,)).surfaces[0]

    def boom(*args, **kwargs):
        raise FileNotFoundError("no GACODE launcher")

    monkeypatch.setattr(tglf, "run_tglf_case", boom)
    job = {"workdir": str(tmp_path / "r0.50"), "r_over_a": 0.5, "identity": "i",
           "profile": state.profile, "local_input": surface.local_input, "local": {}}
    record = driver.run_surface(job, None)
    assert (record["status"], record["runtime_status"]) == ("failed", "error")
    assert "no GACODE launcher" in record["errors"][0]


def test_projection_lines_up_when_the_mapper_drops_a_surface(driver, sample, tmp_path):
    import dataclasses

    from vaft.code.gacode.tglf.outputs import TglfOutputs

    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    report = assess_tglf_readiness(state, (0.3, 0.5, 0.7))
    jobs, surfaces = {}, []
    for index, surface in enumerate(report.surfaces):
        local = surface.local_input
        if surface.r_over_a == 0.5:  # solved, but cannot be dimensionalised
            local = dataclasses.replace(local, normalisation=None)
        workdir = tmp_path / f"r{surface.r_over_a:.2f}"
        flux = np.array([1.0, 2.0, 3.0]) * (index + 1)
        TglfOutputs(directory=str(workdir), gbflux={"particle": flux, "energy": flux,
                    "momentum": 0 * flux, "exchange": 0 * flux},
                    grid={"n_species": 3, "n_xgrid": 4}).write_json(workdir / "outputs.json")
        jobs[surface.r_over_a] = {"workdir": str(workdir), "local_input": local}
        surfaces.append({"r_over_a": surface.r_over_a, "status": "solved"})
    mapped = driver.project_state(state, surfaces, jobs)
    rows = mapped["surfaces"]
    assert [row["r_over_a"] for row in rows] == [0.3, 0.7]
    norm = report.surfaces[2].local_input.normalisation
    assert rows[1]["electron_energy_flux_W_m2"] == pytest.approx(3.0 * norm.energy_flux)

    # Nothing usable: reported, not a KeyError that loses the run.
    for job in jobs.values():
        job["local_input"] = dataclasses.replace(job["local_input"], normalisation=None)
    assert driver.project_state(state, surfaces, jobs)["surfaces"] == []
# --------------------------------------------------------------------------- partition (#1431)


def _row(r=0.5, rho=0.45, qe=100.0, ge=1e19, ions=None, gions=None):
    return {"r_over_a": r, "rho_tor_norm": rho, "electron_energy_flux_W_m2": qe,
            "electron_particle_flux_m2_s": ge, "ion_energy_flux_W_m2": ions or {},
            "ion_particle_flux_m2_s": gions or {}}


def test_partition_keeps_signs_and_splits_by_magnitude():
    from vaft.process.transport_state import transport_partition

    neo = _row(qe=50.0, ions={"z=1": 300.0, "z=6": -10.0})
    tglf = _row(qe=-150.0, ions={"H+": 100.0, "C6+": 30.0})
    part = transport_partition(neo, tglf, turbulent_charges={"H+": 1, "C6+": 6})
    assert part["status"] == "available"
    qe = part["channels"]["electron_energy"]
    assert qe["model"] == pytest.approx(-100.0)  # cancellation survives in the signed sum
    assert qe["f_neo"] == pytest.approx(0.25)    # but not in the magnitude split
    assert qe["neo_over_turb"] == pytest.approx(-1 / 3)
    total = part["channels"]["ion_energy_total"]
    assert (total["neo"], total["turb"]) == (pytest.approx(290.0), pytest.approx(130.0))
    assert part["channels"]["ion_energy_z=6"]["model"] == pytest.approx(20.0)


def test_partition_refuses_rather_than_mixing_surfaces():
    from vaft.process.transport_state import transport_partition

    assert transport_partition(None, _row())["reason"] == "missing_neoclassical_component"
    assert transport_partition(_row(), _row(r=0.6))["reason"] == "surfaces_differ_in_r_over_a"
    assert transport_partition(_row(), _row(rho=0.47))["reason"] == "surfaces_differ_in_rho_tor_norm"
    with pytest.raises(ValueError, match="no charge"):
        transport_partition(_row(ions={"z=1": 1.0}), _row(ions={"H+": 1.0}))
    with pytest.raises(ValueError, match="share z=1"):
        transport_partition(_row(ions={"z=1": 1.0}), _row(ions={"H+": 1.0, "D+": 1.0}),
                            turbulent_charges={"H+": 1, "D+": 1})
    assert transport_partition(_row(), _row(), classical=_row(r=0.6))["reason"] == "classical_surface_differs"


def test_the_ion_total_needs_the_same_species_in_both_models():
    from vaft.process.transport_state import transport_partition

    part = transport_partition(_row(ions={"z=1": 1.0, "z=6": 1.0}), _row(ions={"H+": 2.0}),
                               turbulent_charges={"H+": 1})
    assert "ion_energy_total" not in part["channels"]
    assert part["channels"]["ion_energy_z=1"]["model"] == pytest.approx(3.0)


def test_partition_ratio_is_undefined_below_the_floor_and_classical_joins():
    from vaft.process.transport_state import transport_partition

    part = transport_partition(_row(qe=10.0), _row(qe=0.0), classical=_row(qe=30.0))
    qe = part["channels"]["electron_energy"]
    assert qe["neo_over_turb"] is None
    assert qe["model"] == pytest.approx(40.0)
    assert (qe["f_neo"], qe["f_turb"], qe["f_classical"]) == (
        pytest.approx(0.25), pytest.approx(0.0), pytest.approx(0.75))
    assert part["components"] == ["neo", "turb", "classical"]


# --------------------------------------------------------------------------- routine NEO (#1431)


@pytest.fixture()
def neo_driver():
    path = ROOT / "workflow" / "transport_atlas" / "run_neo.py"
    spec = importlib.util.spec_from_file_location("transport_atlas_run_neo", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)
    sys.modules.pop("transport_atlas_run_tglf", None)


def test_neo_radial_policy_reproduces_the_tglf_surfaces_or_refuses(neo_driver):
    assert neo_driver.radial_policy(DEFAULT_SURFACES) == (6, 0.3, 0.8)
    with pytest.raises(ValueError, match="evenly"):
        neo_driver.radial_policy((0.3, 0.4, 0.6))


def test_neo_driver_maps_one_profile_run_and_checks_its_surfaces(neo_driver, sample, tmp_path, monkeypatch):
    from omas import ODS

    import vaft.code.gacode.neo as neo
    from vaft.code.gacode.neo import NEOResult
    from vaft.code.gacode.neo.outputs import collect_neo_outputs

    ods = _electron_only(sample)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [{"shot": 48224, "time_ms": 300, "label": "admissible"}]}))
    fixture = ROOT / "test" / "data" / "gacode" / "neo_vest_48224_carbon"  # r/a 0.2 ... 0.8, 7 points

    def fake_run(profile, workdir, config=None, *, check=True):
        Path(workdir).mkdir(parents=True, exist_ok=True)
        return NEOResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                         outputs_native=collect_neo_outputs(fixture))

    monkeypatch.setattr(neo, "run_neo_case", fake_run)
    common = ["--filedb", str(filedb), "--labels", str(labels), "--lineages", "magnetics"]
    seven = [f"{0.2 + 0.1 * i:.1f}" for i in range(7)]
    neo_driver.main(common + ["--out", str(tmp_path / "a"), "--surfaces", *seven])
    state = json.loads((tmp_path / "a" / "48224" / "magnetics" / "00300" / "state.json").read_text())
    assert state["status"] == "solved"
    rows = state["core_transport"]["surfaces"]
    assert [round(r["r_over_a"], 2) for r in rows] == [float(v) for v in seven]
    assert set(rows[0]["ion_energy_flux_W_m2"]) == {"z=1", "z=6"}
    assert state["core_transport"]["code"]["name"] == "NEO"

    # The default six-surface request does not match what this run solved: not "solved".
    neo_driver.main(common + ["--out", str(tmp_path / "b")])
    state = json.loads((tmp_path / "b" / "48224" / "magnetics" / "00300" / "state.json").read_text())
    assert state["status"] == "partial"


# --------------------------------------------------------------------------- classical (#1435)


def test_classical_collision_times_match_the_nrl_forms(sample):
    from vaft.process.transport_state import classical_heat_fluxes, surface_toroidal_field

    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    local = assess_tglf_readiness(state, (0.5,)).surfaces[0].local_input
    d = classical_heat_fluxes(local, surface_toroidal_field(state.profile, local))
    norm = local.normalisation
    ne_cm3 = norm.electron_density * 1e-6
    te_ev = norm.electron_temperature / 1.602176634e-19
    # NRL: tau_e = 3.44e5 Te^1.5 / (n lnL) for Z = 1; Braginskii's n_i Z^2 is n_e Z_eff.
    assert d["tau_e_s"] * float(local.zeff) == pytest.approx(
        3.44e5 * te_ev ** 1.5 / (ne_cm3 * d["coulomb_logarithm"]), rel=0.01)
    # NRL tau_i for the main ion alone, then scaled by the extra field-ion collisions.
    ti_ev = te_ev * float(local.taus[1])
    zs, fr = np.asarray(local.zs)[1:], np.asarray(local.as_)[1:]
    ni_cm3 = ne_cm3 * fr[0]
    mu = float(local.mass[1]) * 3.34358e-27 / 1.67262192e-27
    self_only = 2.09e7 * ti_ev ** 1.5 * mu ** 0.5 / (ni_cm3 * d["coulomb_logarithm_ion"] * zs[0] ** 4)
    boost = np.sum(fr * zs ** 2) / (fr[0] * zs[0] ** 2)
    assert boost > 2.0  # H + C6+ at Z_eff = 2: carbon raises the ion collisionality ~2.5x
    assert d["tau_i_s"] == pytest.approx(self_only / boost, rel=0.01)
    assert d["gamma1_perp"] == pytest.approx(4.0, rel=0.01)  # Braginskii at Z = 2


def test_classical_uses_the_physical_field_not_b_unit(sample):
    from vaft.process.transport_state import surface_toroidal_field

    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    local = assess_tglf_readiness(state, (0.5,)).surfaces[0].local_input
    b = surface_toroidal_field(state.profile, local)
    r = float(local.rmaj_loc) * local.normalisation.minor_radius
    assert b == pytest.approx(abs(state.profile.bcentr) * state.profile.rcentr / r)
    assert b < abs(local.normalisation.b_unit)  # B_unit is a flux-coordinate field, larger on VEST


def test_classical_flux_runs_down_the_gradient_and_scales_as_one_over_b_squared(sample):
    from vaft.process.transport_state import classical_heat_fluxes

    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    local = assess_tglf_readiness(state, (0.6,)).surfaces[0].local_input
    d = classical_heat_fluxes(local, 0.3)
    assert np.sign(d["electron_energy_flux_W_m2"]) == np.sign(local.rlts[0])
    assert np.sign(d["ion_energy_flux_W_m2"]["z=1"]) == np.sign(local.rlts[1])
    assert classical_heat_fluxes(local, 0.6)["chi_e_m2_s"] == pytest.approx(d["chi_e_m2_s"] / 4.0)
    assert d["electron_particle_flux_m2_s"] is None  # not modelled: missing, not zero


def test_a_classical_failure_is_recorded_not_raised(driver, sample):
    import dataclasses

    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    local = assess_tglf_readiness(state, (0.5,)).surfaces[0].local_input
    record, error = driver.classical_record(state.profile, dataclasses.replace(local, normalisation=None))
    assert record is None and "normalisation" in error


# --------------------------------------------------------------------------- cold review 0.8.0 delta-absorb-9


def _classical_state(sample, r=0.5):
    state = resolve_transport_state(sample, _key(0.3), efit_quality="good")
    return state, assess_tglf_readiness(state, (r,)).surfaces[0].local_input


def test_a_classical_failure_of_any_kind_is_recorded_not_raised(driver, sample):
    """F1: a missing field or an electron-only species list is a reason, not a raise."""
    import dataclasses

    state, local = _classical_state(sample)
    # GACODEProfile's default: no bcentr/rcentr on the profile.
    record, error = driver.classical_record(dataclasses.replace(state.profile, bcentr=None), local)
    assert record is None and "TypeError" in error
    # TGLF local input carrying electrons only: there is no main ion to evaluate.
    electrons_only = dataclasses.replace(
        local, zs=np.array([-1.0]), as_=np.array([1.0]), mass=np.array([local.mass[0]]),
        taus=np.array([1.0]), rlts=np.array([local.rlts[0]]))
    record, error = driver.classical_record(state.profile, electrons_only)
    assert record is None and "IndexError" in error


def test_classical_refuses_a_non_positive_field_and_the_row_stays_standard_json(driver, sample):
    """F2: b = 0 must not become an ``Infinity`` token in state.json."""
    import dataclasses

    from vaft.process.transport_state import classical_heat_fluxes

    state, local = _classical_state(sample)
    for b in (0.0, -0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="b_tesla"):
            classical_heat_fluxes(local, b)
    record, error = driver.classical_record(dataclasses.replace(state.profile, bcentr=0.0), local)
    assert record is None and "b_tesla" in error

    def refuse(token):
        raise AssertionError(f"non-standard JSON token {token!r}")

    payload = json.dumps({"classical": record, "classical_error": error}, default=float)
    assert json.loads(payload, parse_constant=refuse)["classical"] is None


def test_classical_gamma1_table_is_braginskii_table_2(sample):
    """F3: the Z -> inf knot is Braginskii's 3.25; above Z = 4 the coefficient is held at 3.6."""
    from vaft.process.transport_state import _GAMMA1_PERP, classical_heat_fluxes

    table = dict(_GAMMA1_PERP)
    assert table[1.0] == 4.66 and table[2.0] == 4.0 and table[4.0] == 3.6
    assert max(table) > 4.0 and table[max(table)] == 3.25
    gamma = np.interp([4.0, 10.0, max(table)], *zip(*_GAMMA1_PERP))
    assert gamma[0] == 3.6 and gamma[1] == pytest.approx(3.6, abs=1e-6) and gamma[2] == 3.25
    _, local = _classical_state(sample)
    assert classical_heat_fluxes(local, 0.3)["model"]["coefficients"]["electron"].count("3.25") == 1


def test_surface_toroidal_field_docstring_tags_rcentr_as_a_length():
    """F4: rcentr is a major radius [m], not a field [T]."""
    from vaft.process.transport_state import surface_toroidal_field

    doc = surface_toroidal_field.__doc__
    assert "``bcentr`` [T] and ``rcentr`` [m]." in doc
    assert "``rcentr`` [T]" not in doc


# --------------------------------------------------------------------------- cold review 0.8.0 delta-absorb-7


def _one_state_filedb(sample, tmp_path, label="good"):
    """A FileDB with one magnetics state (48224 @ 300 ms) and its label file."""
    from omas import ODS

    ods = _electron_only(sample)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [{"shot": 48224, "time_ms": 300, "label": label}]}))
    return ["--filedb", str(filedb), "--labels", str(labels), "--lineages", "magnetics"]


def _solved_tglf_run(profile, rho, workdir, config=None, *, check=True):
    from vaft.code.gacode.tglf import TGLFResult
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    flux = np.array([1.0, 2.0, 0.1]) * rho
    native = TglfOutputs(directory=str(workdir), version={"revision": "test"},
                         gbflux={"particle": 0.1 * flux, "energy": flux,
                                 "momentum": 0 * flux, "exchange": 0 * flux},
                         grid={"n_species": 3, "n_xgrid": 4})
    return TGLFResult(returncode=0, runtime_status="completed", workdir=workdir,
                      outputs_native=native)


def test_a_tglf_state_the_mapper_refuses_is_not_solved(driver, sample, tmp_path, monkeypatch):
    """P1: every surface ran, but no flux was projected -- the row must say so."""
    import vaft.code.gacode.tglf as tglf
    import vaft.machine_mapping.turbulence as turbulence

    monkeypatch.setattr(tglf, "run_tglf_case", _solved_tglf_run)
    monkeypatch.setattr(turbulence, "core_transport_from_tglf",
                        lambda ods, pairs, profile, time: {"model": 0, "written": [],
                                                           "skipped": ["rmin is not monotone"]})
    common = _one_state_filedb(sample, tmp_path)
    assert driver.main(common + ["--out", str(tmp_path / "out"), "--workers", "1",
                                 "--sat-rule", "2", "--field-model", "es"]) == 0
    out = tmp_path / "out" / "tglf-sat2-es"
    (row,) = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert row["status"] == "failed"
    assert row["reasons"] == ["no_surface_mapped: rmin is not monotone"]
    state = json.loads((out / "48224" / "magnetics" / "00300" / "state.json").read_text())
    assert state["core_transport"]["surfaces"] == []
    assert all(s["status"] == "solved" for s in state["surfaces"])  # the runs themselves did solve
    assert json.loads((out / "run_manifest.json").read_text())["status_counts"] == {"failed": 1}


def test_a_neo_run_that_maps_nothing_is_not_solved(neo_driver, sample, tmp_path, monkeypatch):
    """F1: a clean NEO exit whose tree has no exprhon table projects no surface."""
    import dataclasses

    import vaft.code.gacode.neo as neo
    from vaft.code.gacode.neo import NEOResult
    from vaft.code.gacode.neo.outputs import collect_neo_outputs

    fixture = ROOT / "test" / "data" / "gacode" / "neo_vest_48224_carbon"

    def fake_run(profile, workdir, config=None, *, check=True):
        Path(workdir).mkdir(parents=True, exist_ok=True)
        native = dataclasses.replace(collect_neo_outputs(fixture), coordinates=None)
        return NEOResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                         outputs_native=native)

    monkeypatch.setattr(neo, "run_neo_case", fake_run)
    common = _one_state_filedb(sample, tmp_path, label="admissible")
    seven = [f"{0.2 + 0.1 * i:.1f}" for i in range(7)]
    assert neo_driver.main(common + ["--out", str(tmp_path / "out"), "--surfaces", *seven]) == 0
    (row,) = [json.loads(line) for line in (tmp_path / "out" / "states.jsonl").read_text().splitlines()]
    assert row["status"] == "failed"
    assert len(row["reasons"]) == 1 and "exprhon" in row["reasons"][0]
    state = json.loads((tmp_path / "out" / "48224" / "magnetics" / "00300" / "state.json").read_text())
    assert state["run"]["status"] == "solved" and state["core_transport"]["surfaces"] == []


def test_a_tglf_record_without_its_outputs_is_rerun_and_a_bad_one_fails_only_its_state(
        driver, sample, tmp_path, monkeypatch):
    """F5: resume after external cleanup must neither trust nor crash on a stale run."""
    import vaft.code.gacode.tglf as tglf

    calls = []
    monkeypatch.setattr(tglf, "run_tglf_case",
                        lambda *a, **k: calls.append(a[1]) or _solved_tglf_run(*a, **k))
    common = _one_state_filedb(sample, tmp_path)
    argv = common + ["--out", str(tmp_path / "out"), "--workers", "1",
                     "--sat-rule", "2", "--field-model", "es"]
    assert driver.main(argv) == 0
    out = tmp_path / "out" / "tglf-sat2-es"
    state_dir = out / "48224" / "magnetics" / "00300"
    assert sorted(calls) == [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # record.json kept, outputs.json pruned: not a cache hit.
    (state_dir / "r0.50" / "outputs.json").unlink()
    calls.clear()
    assert driver.main(argv) == 0
    assert calls == [0.5]
    (row,) = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert row["status"] == "solved"

    # An unreadable outputs.json fails that state; the batch still writes its manifest.
    (state_dir / "r0.50" / "outputs.json").write_text("not json", encoding="utf-8")
    (out / "run_manifest.json").unlink()
    calls.clear()
    assert driver.main(argv) == 0
    assert calls == []
    (row,) = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert row["status"] == "failed"
    assert row["reasons"][0].startswith("projection_error: JSONDecodeError")
    assert json.loads((out / "run_manifest.json").read_text())["status_counts"] == {"failed": 1}


def test_a_neo_record_without_its_outputs_is_rerun_and_a_bad_one_fails_only_its_state(
        neo_driver, sample, tmp_path, monkeypatch):
    """F5, NEO twin."""
    import vaft.code.gacode.neo as neo
    from vaft.code.gacode.neo import NEOResult
    from vaft.code.gacode.neo.outputs import collect_neo_outputs

    fixture = ROOT / "test" / "data" / "gacode" / "neo_vest_48224_carbon"
    calls = []

    def fake_run(profile, workdir, config=None, *, check=True):
        calls.append(Path(workdir).name)
        Path(workdir).mkdir(parents=True, exist_ok=True)
        return NEOResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                         outputs_native=collect_neo_outputs(fixture))

    monkeypatch.setattr(neo, "run_neo_case", fake_run)
    common = _one_state_filedb(sample, tmp_path, label="admissible")
    seven = [f"{0.2 + 0.1 * i:.1f}" for i in range(7)]
    argv = common + ["--out", str(tmp_path / "out"), "--surfaces", *seven]
    assert neo_driver.main(argv) == 0
    out = tmp_path / "out"
    workdir = out / "48224" / "magnetics" / "00300" / "neo"
    assert calls == ["neo"]

    (workdir / "outputs.json").unlink()
    calls.clear()
    assert neo_driver.main(argv) == 0
    assert calls == ["neo"]
    (row,) = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert row["status"] == "solved"

    (workdir / "outputs.json").write_text("not json", encoding="utf-8")
    (out / "run_manifest.json").unlink()
    calls.clear()
    assert neo_driver.main(argv) == 0
    assert calls == []
    (row,) = [json.loads(line) for line in (out / "states.jsonl").read_text().splitlines()]
    assert row["status"] == "failed"
    assert row["reasons"][0].startswith("projection_error: JSONDecodeError")
    assert json.loads((out / "run_manifest.json").read_text())["status_counts"] == {"failed": 1}


def test_a_nan_midplane_geometry_is_derived_not_passed_through(sample):
    """F2: r_inboard/r_outboard are judged finite, like the shape profiles."""
    ods = copy.deepcopy(sample)
    n = len(ods["equilibrium.time_slice.0.profiles_1d.psi"])
    ods["equilibrium.time_slice.0.profiles_1d.r_inboard"] = np.full(n, np.nan)
    ods["equilibrium.time_slice.0.profiles_1d.r_outboard"] = np.full(n, np.nan)
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good")
    assert state.resolved, state.reasons
    assert state.provenance["midplane_geometry"]["kind"] == "derived"
    assert np.all(np.isfinite(state.profile.rmin))


def test_mem_mb_is_the_configs_memory_reservation_not_a_slurm_only_flag(driver, neo_driver):
    """F7: --mem-mb lands in config.memory_mb for both drivers and both backends."""
    import argparse

    base = dict(partition="lowpri-short", account=None, max_wait=1.0, timeout=1.0, mem_mb=1536,
                sat_rule=2, field_model="es", surfaces=list(DEFAULT_SURFACES))
    for backend in ("local", "slurm"):
        args = argparse.Namespace(backend=backend, **base)
        for config in (driver.build_config(args), neo_driver.build_config(args)):
            assert config.memory_mb == 1536
            if backend == "slurm":
                # One --mem only: the backend's _sizing writes it from memory_mb.
                assert not any("--mem" in str(a) for a in config.backend.extra_args)


def test_a_product_without_a_time_vector_is_listed_missing_not_silently_empty(driver, sample, tmp_path):
    """F8: consistency_check=False products read a missing time as [], hiding the lineage."""
    from omas import ODS

    ods = _electron_only(sample)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    labels = {(48224, 300): "good"}

    # Equilibrium without `time`: that lineage is reported, the enumeration continues.
    filedb = tmp_path / "a"
    no_time = copy.deepcopy(eq)
    del no_time["equilibrium.time"]
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", no_time)
    states, counts = driver.enumerate_states(filedb, labels, [48224], ["magnetics"])
    assert states == []
    assert counts["missing_products"] == [{"shot": 48224, "stage": "magnetics", "status": "no_time"}]

    # core_profiles without `time`: a reason, not a ValueError from an empty min().
    filedb = tmp_path / "b"
    no_time = copy.deepcopy(cp)
    del no_time["core_profiles.time"]
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", no_time)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    states, counts = driver.enumerate_states(filedb, labels, [48224], ["magnetics"])
    assert states == []
    assert counts["missing_products"] == [{"shot": 48224, "stage": "core_profiles", "status": "no_time"}]


def test_surface_codes_are_the_two_a_local_conversion_can_raise():
    """F10: positivity is refused at state resolution, so no surface code names it."""
    from vaft.code.gacode.tglf.inputs import LocalConversionError
    from vaft.process.transport_state import _surface_code

    assert _surface_code(LocalConversionError("r/a = 0.9 is outside the converted profile")) == \
        "outside_profile_domain"
    assert _surface_code(LocalConversionError("ne is not positive at grid point 3")) == \
        "local_conversion_failure"


# --------------------------------------------------------------------------- atlas (#1427)


@pytest.fixture()
def atlas():
    path = ROOT / "workflow" / "transport_atlas" / "build_atlas.py"
    spec = importlib.util.spec_from_file_location("transport_atlas_build", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_spectral_descriptors_on_the_regression_case(atlas):
    from vaft.code.gacode.tglf.outputs import collect_tglf_outputs

    native = collect_tglf_outputs(REG05)
    d = atlas.spectral_descriptors(native)
    assert d["qe_gb"] == pytest.approx(native.gbflux["energy"][0])
    assert d["q_tot_gb"] == pytest.approx(native.gbflux["energy"].sum())
    gamma = native.growth_rate
    assert d["gamma_max"] == pytest.approx(gamma.max())
    ion = native.ky_spectrum <= 1.0
    assert d["gamma_max_ion_scale"] == pytest.approx(gamma[ion].max())
    assert native.ky_spectrum.min() <= d["ky_q_mean"] <= native.ky_spectrum.max()
    assert "f_em" not in d  # reg05 is electrostatic: one field, no EM share to report
    assert d["n_unstable_ky"] == int((gamma.max(axis=1) > 0).sum())


def test_species_charges_parse_or_refuse(atlas):
    assert atlas.species_charges(["e", "H+", "C6+"]) == {"H+": 1.0, "C6+": 6.0}
    assert atlas.species_charges(["e", "ion0"]) is None


def test_every_schema_column_has_a_known_category(atlas):
    known = {"key", "label", "provenance", "measured_input", "assumed_input",
             "reconstructed_input", "derived_input", "tglf_predicted", "neo_predicted",
             "classical_predicted", "model_derived"}
    assert {category for _, category, _ in atlas.SCHEMA.values()} <= known
    assert not any("itg" in name or "tem" in name.split("_") for name in atlas.SCHEMA)


def test_atlas_joins_tglf_neo_and_classical_by_state_identity(atlas, driver, neo_driver, sample,
                                                               tmp_path, monkeypatch):
    from omas import ODS

    import vaft.code.gacode.neo as neo
    import vaft.code.gacode.tglf as tglf
    from vaft.code.gacode.neo import NEOResult
    from vaft.code.gacode.neo.outputs import collect_neo_outputs
    from vaft.code.gacode.tglf import TGLFResult
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    ods = _electron_only(sample)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [{"shot": 48224, "time_ms": 300, "label": "good"}]}))

    def fake_tglf(profile, rho, workdir, config=None, *, check=True):
        flux = np.array([1.0, 2.0, 0.1])
        native = TglfOutputs(directory=str(workdir), version={"revision": "t"},
                             gbflux={"particle": 0.1 * flux, "energy": flux,
                                     "momentum": 0 * flux, "exchange": 0 * flux},
                             grid={"n_species": 3, "n_xgrid": 4})
        return TGLFResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                          outputs_native=native)

    fixture = ROOT / "test" / "data" / "gacode" / "neo_vest_48224_carbon"

    def fake_neo(profile, workdir, config=None, *, check=True):
        Path(workdir).mkdir(parents=True, exist_ok=True)
        return NEOResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                         outputs_native=collect_neo_outputs(fixture))

    monkeypatch.setattr(tglf, "run_tglf_case", fake_tglf)
    monkeypatch.setattr(neo, "run_neo_case", fake_neo)
    surfaces = [f"{0.2 + 0.1 * i:.1f}" for i in range(7)]  # the fixture's NEO grid
    common = ["--filedb", str(filedb), "--labels", str(labels), "--lineages", "magnetics",
              "--surfaces", *surfaces]
    driver.main(common + ["--out", str(tmp_path / "tglf"), "--sat-rule", "1", "--field-model", "em-bper"])
    neo_driver.main(common + ["--out", str(tmp_path / "neo")])
    assert atlas.main(["--tglf", str(tmp_path / "tglf" / "tglf-sat1-em-bper"), "--neo", str(tmp_path / "neo"),
                       "--out", str(tmp_path / "atlas")]) == 0

    import csv

    with open(tmp_path / "atlas" / "atlas.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7
    assert {r["efit_quality"] for r in rows} == {"good"}
    assert {(r["tglf_config"], r["tglf_sat_rule"], r["tglf_use_bper"], r["tglf_use_bpar"]) for r in rows} == {
        ("tglf-sat1-em-bper", "1", "True", "False")}
    assert {r["neo_status"] for r in rows} == {"solved"}
    # The NEO fixture was solved on this sample's equilibrium, so its exprhon surfaces
    # are the ones the TGLF mapping places: every surface must join.
    assert {r["partition_status"] for r in rows} == {"available"}
    for row in rows:
        assert row["qe_classical_W_m2"] != ""
        if row["partition_status"] == "available":
            parts = [float(row["qe_neo_W_m2"]), float(row["qe_tglf_W_m2"]), float(row["qe_classical_W_m2"])]
            assert float(row["qe_model_W_m2"]) == pytest.approx(sum(parts))
            assert float(row["f_neo_qe"]) == pytest.approx(abs(parts[0]) / sum(map(abs, parts)))
            ions = [float(row["qi_neo_W_m2"]), float(row["qi_tglf_W_m2"]), float(row["qi_classical_W_m2"])]
            assert float(row["qi_model_W_m2"]) == pytest.approx(sum(ions))
            assert float(row["f_classical_qi"]) == pytest.approx(abs(ions[2]) / sum(map(abs, ions)))
    schema = json.loads((tmp_path / "atlas" / "schema.json").read_text())
    assert schema["row_key"] == ["shot", "time_efit_s", "efit_lineage", "r_over_a"]
    assert set(rows[0]) == set(schema["columns"])


def test_atlas_rows_keep_failed_surfaces_and_a_missing_model(atlas, driver, sample, tmp_path, monkeypatch):
    from omas import ODS

    import vaft.code.gacode.tglf as tglf
    from vaft.code.gacode.tglf import TGLFResult
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    ods = _electron_only(sample)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [{"shot": 48224, "time_ms": 300, "label": "admissible"}]}))

    def fake(profile, rho, workdir, config=None, *, check=True):
        if rho > 0.65:
            return TGLFResult(returncode=None, runtime_status="timeout", workdir=Path(workdir))
        flux = np.array([1.0, 2.0, 0.1])
        native = TglfOutputs(directory=str(workdir), gbflux={"particle": flux, "energy": flux,
                             "momentum": 0 * flux, "exchange": 0 * flux},
                             grid={"n_species": 3, "n_xgrid": 4})
        return TGLFResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                          outputs_native=native)

    monkeypatch.setattr(tglf, "run_tglf_case", fake)
    driver.main(["--filedb", str(filedb), "--labels", str(labels), "--out", str(tmp_path / "tglf"),
                 "--lineages", "magnetics", "--sat-rule", "0", "--field-model", "es"])
    rows = atlas.build_rows(tmp_path / "tglf" / "tglf-sat0-es")
    assert [(r["r_over_a"], r["tglf_status"]) for r in rows] == [
        (0.3, "solved"), (0.4, "solved"), (0.5, "solved"), (0.6, "solved"),
        (0.7, "failed"), (0.8, "failed")]
    assert [r["tglf_reason"] for r in rows[-2:]] == ["timeout", "timeout"]
    assert {r["partition_status"] for r in rows[:4]} == {"missing_neoclassical_component"}
    assert {r["neo_status"] for r in rows} == {"missing"}
    assert all(r.get("qe_classical_W_m2") is not None for r in rows)  # no solver needed
    radial = atlas.radial_summary(rows)
    assert len(radial) == 1 and radial[0]["n_tglf_solved"] == 4 and radial[0]["n_surfaces"] == 6
    discharge = atlas.discharge_summary(radial)
    assert discharge[0]["n_states"] == 1 and discharge[0]["n_good"] == 0


def test_spectral_descriptors_survive_nan_growth_rates(atlas):
    from vaft.code.gacode.tglf.outputs import collect_tglf_outputs

    native = collect_tglf_outputs(REG05)
    spectrum = native.eigenvalue_spectrum.copy()
    spectrum[0, 0] = np.nan
    native.eigenvalue_spectrum = spectrum
    d = atlas.spectral_descriptors(native)
    assert np.isfinite(d["gamma_max"]) and d["gamma_max"] == pytest.approx(np.nanmax(native.growth_rate))



def test_f_em_and_ky_q_mean_follow_their_documented_signed_definitions(atlas):
    from types import SimpleNamespace

    ky = np.array([0.5, 1.0])
    spectrum = np.zeros((2, 2, 2, 5))          # (species, field, ky, quantity)
    spectrum[0, 0, :, 1] = [3.0, 1.0]          # electrons, phi
    spectrum[1, 0, :, 1] = [-1.0, 1.0]         # an inward ion flux cancels at ky = 0.5
    spectrum[0, 1, :, 1] = [0.5, -1.5]         # A_par: net -1.0
    native = SimpleNamespace(gbflux={"energy": np.array([1.0, 1.0]), "particle": np.zeros(2)},
                             ky_spectrum=ky, growth_rate=None, frequency=None, sum_flux_spectrum=spectrum)
    d = atlas.spectral_descriptors(native)
    # per-ky signed totals: 3 - 1 + 0.5 = 2.5 and 1 + 1 - 1.5 = 0.5
    assert d["ky_q_mean"] == pytest.approx((0.5 * 2.5 + 1.0 * 0.5) / 3.0)
    # Q_phi = 4, Q_mag = -1: f_em = 1 / (4 + 1)
    assert d["f_em"] == pytest.approx(0.2)


def test_the_atlas_refuses_a_mixed_or_unnamed_configuration_and_a_wide_dt(atlas, tmp_path, monkeypatch):
    rows = [{"tglf_config": "tglf-sat3-em-bper", "tglf_sat_rule": 3, "tglf_use_bper": True,
             "tglf_use_bpar": False, "dt_s": 0.0, "tolerance_s": 5e-4, "shot": 1, "time_efit_s": 0.3}]
    def run(rows_):
        monkeypatch.setattr(atlas, "build_rows", lambda *a, **k: [dict(r) for r in rows_])
        return atlas.main(["--tglf", str(tmp_path), "--out", str(tmp_path / "o")])
    with pytest.raises(ValueError, match="no TGLF"):
        run([])
    with pytest.raises(ValueError, match="not one named TGLF configuration"):
        run(rows + [dict(rows[0], tglf_sat_rule=2)])
    with pytest.raises(ValueError, match="not one named TGLF configuration"):
        run([dict(rows[0], tglf_config=None)])
    with pytest.raises(ValueError, match="tolerance"):
        run([dict(rows[0], dt_s=1e-3)])


def test_duplicate_neo_records_are_refused(atlas, tmp_path):
    for name in ("a", "b"):
        path = tmp_path / name / "magnetics" / "00300"
        path.mkdir(parents=True)
        (path / "state.json").write_text(json.dumps({"state_identity": "same"}))
    with pytest.raises(ValueError, match="share state_identity"):
        atlas._neo_index(tmp_path)


def test_radial_summary_numbers(atlas):
    rows = [{"shot": 1, "time_efit_s": 0.3, "efit_lineage": "magnetics", "efit_quality": "good",
             "ti_lineage": "ti_eq_te_assumed", "neo_status": "solved", "tglf_status": "solved",
             "r_over_a": r, "f_e": fe, "gamma_max_ion_scale": g, "q_tot_gb": q, "f_neo_qi": fn}
            for r, fe, g, q, fn in ((0.3, 0.2, 0.1, 1.0, 0.9), (0.5, 0.7, 0.4, 3.0, 0.5), (0.7, 0.9, 0.2, 5.0, 0.1))]
    (s,) = atlas.radial_summary(rows)
    assert s["n_tglf_solved"] == 3
    assert s["fraction_electron_dominated"] == pytest.approx(2 / 3)
    assert (s["gamma_max_ion_scale"], s["r_over_a_at_gamma_max_ion_scale"]) == (0.4, 0.5)
    assert s["median_q_tot_gb"] == pytest.approx(3.0) and s["median_f_neo_qi"] == pytest.approx(0.5)
    (d,) = atlas.discharge_summary([s])
    assert d["n_states"] == 1 and d["n_good"] == 1


# ------------------------------------------------- atlas rows from stored records (0.8.0 review)


def _mapped_surface(r, rho=0.45, qe=100.0, ge=1e19, ions=None, *, neo=False):
    # run_tglf.project_state labels ions by species name; run_neo.project by charge.
    names = ("z=1", "z=6") if neo else ("H+", "C6+")
    return {"r_over_a": r, "rho_tor_norm": rho, "electron_energy_flux_W_m2": qe,
            "electron_particle_flux_m2_s": ge,
            "ion_energy_flux_W_m2": dict(zip(names, (50.0, 5.0))) if ions is None else ions,
            "ion_particle_flux_m2_s": dict(zip(names, (1e19, 1e17)))}


def _stored_records(tmp_path, *, tglf_surfaces, tglf_mapped, neo_mapped=None,
                    species=("e", "H+", "C6+"), status="not_run"):
    """A TGLF (and optionally NEO) record tree as run_tglf.py / run_neo.py write it.

    The surfaces are ``not_run`` (an interrupted batch, run_tglf.py:517) so no
    outputs.json is needed: the mapped core_transport rows and the NEO join are
    independent of the surface status, which is what these tests exercise.
    """
    identity = "a" * 64
    base = {"shot": 48224, "time_efit_s": 0.3, "efit_lineage": "magnetics", "efit_quality": "good",
            "quality_source": "labels", "state_identity": identity}
    tglf = tmp_path / "tglf" / "48224" / "magnetics" / "00300"
    tglf.mkdir(parents=True)
    (tglf / "state.json").write_text(json.dumps({
        **base, "tglf_config": "tglf-sat1-es",
        "surfaces": [{"r_over_a": r, "status": status, "readiness": "ready",
                      "local": {"species": list(species)}} for r in tglf_surfaces],
        "core_transport": {"surfaces": tglf_mapped},
    }), encoding="utf-8")
    neo = None
    if neo_mapped is not None:
        neo = tmp_path / "neo" / "48224" / "magnetics" / "00300"
        neo.mkdir(parents=True)
        (neo / "state.json").write_text(json.dumps({
            **base, "status": "solved", "core_transport": {"surfaces": neo_mapped}}), encoding="utf-8")
    return tmp_path / "tglf", (tmp_path / "neo" if neo is not None else None)


def test_an_all_none_ion_flux_is_absent_not_zero(atlas, tmp_path):
    # The mapper writes None per ion species when it did not project the ion
    # energy (run_tglf.column(), run_neo.project); summing nothing is not 0 W/m^2.
    none_ions = {"H+": None, "C6+": None}
    tglf_root, neo_root = _stored_records(
        tmp_path, tglf_surfaces=[0.3], tglf_mapped=[_mapped_surface(0.3, ions=none_ions)],
        neo_mapped=[_mapped_surface(0.3, qe=10.0, ions={"z=1": None, "z=6": None}, neo=True)])
    (row,) = atlas.build_rows(tglf_root, neo_root)
    assert row.get("qi_tglf_W_m2") is None and row.get("qi_neo_W_m2") is None
    assert "f_e" not in row  # not 1.0: the electrons are not known to carry everything
    assert row["qe_tglf_W_m2"] == 100.0


def test_the_neo_join_uses_the_drivers_tolerance_not_a_rounding(atlas, tmp_path):
    # run_neo.py accepts a NEO surface within 1e-4 of the request (np.allclose) and
    # transport_partition joins within 1e-4; 0.30008 rounds to 0.3001, not to 0.3.
    tglf_root, neo_root = _stored_records(
        tmp_path, tglf_surfaces=[0.3, 0.5], tglf_mapped=[_mapped_surface(0.3), _mapped_surface(0.5)],
        neo_mapped=[_mapped_surface(0.30008, qe=10.0, neo=True), _mapped_surface(0.5012, qe=10.0, neo=True)])
    near, far = atlas.build_rows(tglf_root, neo_root)
    assert near["partition_status"] == "available"
    assert near["qe_neo_W_m2"] == 10.0
    assert far["partition_status"] == "missing_neoclassical_component"  # 1.2e-3 is another surface
    assert "qe_neo_W_m2" not in far


def test_a_partition_refusal_fails_its_row_not_the_build(atlas, tmp_path):
    # H+ and D+ share z=1: transport_partition refuses the species by ValueError
    # (transport_state._channels). That is one row's partition, not the atlas.
    ions = {"H+": 50.0, "D+": 20.0}
    tglf_root, neo_root = _stored_records(
        tmp_path, tglf_surfaces=[0.3, 0.5], species=("e", "H+", "D+"),
        tglf_mapped=[_mapped_surface(0.3, ions=ions), _mapped_surface(0.5, ions=ions)],
        neo_mapped=[_mapped_surface(0.3, qe=10.0, neo=True), _mapped_surface(0.5, qe=10.0, neo=True)])
    rows = atlas.build_rows(tglf_root, neo_root)
    assert len(rows) == 2
    assert all(r["partition_status"].startswith("partition_refused: ") for r in rows)
    assert all("share z=1" in r["partition_status"] for r in rows)
    assert all("f_neo_qe" not in r and r["qe_neo_W_m2"] == 10.0 for r in rows)


def test_a_not_run_surface_names_its_status_and_the_schema_lists_it(atlas, tmp_path):
    tglf_root, _ = _stored_records(tmp_path, tglf_surfaces=[0.3], tglf_mapped=[])
    (row,) = atlas.build_rows(tglf_root)
    assert row["tglf_status"] == "not_run"
    assert row["tglf_reason"] == "not_run"  # not "ready": the surface was ready and never ran
    assert "not_run" in atlas.SCHEMA["tglf_status"][2]


def test_the_native_directory_uses_the_slash_grammar_on_every_host(atlas):
    from pathlib import PurePosixPath, PureWindowsPath

    posix = atlas._native_dir(PurePosixPath("/runs/tglf-sat1-es/48224/magnetics/00300/state.json"),
                              PurePosixPath("/runs/tglf-sat1-es"), 0.3)
    windows = atlas._native_dir(PureWindowsPath(r"C:\runs\tglf-sat1-es\48224\magnetics\00300\state.json"),
                                PureWindowsPath(r"C:\runs\tglf-sat1-es"), 0.3)
    assert posix == windows == "48224/magnetics/00300/r0.30"


# --------------------------------------------------------------------------- SAT x field sensitivity (#1482)


@pytest.fixture()
def sensitivity():
    path = ROOT / "workflow" / "transport_atlas" / "build_sensitivity.py"
    spec = importlib.util.spec_from_file_location("transport_atlas_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_delta_is_bounded_and_quiet_near_zero(sensitivity):
    assert sensitivity.delta(2.0, 1.0) == pytest.approx(0.5)
    assert sensitivity.delta(-1.0, 1.0) == pytest.approx(-2.0)
    assert abs(sensitivity.delta(1e-5, -1e-5)) < 0.03  # both below eps: not blown up
    assert sensitivity.delta(None, 1.0) is None


def _tglf_weighted(ky, flux):
    """``sum_flux_spectrum`` entries exactly as write_tglf_sum_flux_spectrum forms them."""
    entries = [ky[0] * flux[0]]
    for k in range(1, len(ky)):
        k0, k1 = ky[k - 1], ky[k]
        d = np.log(k1 / k0) / (k1 - k0)
        entries.append(k0 * (k1 * d - 1.0) * flux[k - 1] + k1 * (1.0 - k0 * d) * flux[k])
    return np.asarray(entries)


def test_ky_peak_follows_tglf_interval_weighting(sensitivity):
    from types import SimpleNamespace

    ky = np.array([0.1, 0.2, 0.3, 0.4, 0.8, 1.6, 3.2])
    flux = np.array([0.1, 0.2, 0.5, 3.0, 0.4, 0.2, 0.1])   # density peaked at ky = 0.4
    spectrum = np.zeros((2, 1, ky.size, 5))
    spectrum[0, 0, :, 1] = _tglf_weighted(ky, 0.7 * flux)
    spectrum[1, 0, :, 1] = _tglf_weighted(ky, 0.3 * flux)
    peak = sensitivity.ky_peak(SimpleNamespace(ky_spectrum=ky, sum_flux_spectrum=spectrum))
    # Interval means straddle the peak point: (0.3, 0.4] -> midpoint 0.35, (0.4, 0.8] -> 0.6.
    assert peak == pytest.approx(0.35)
    # Each entry's width-mean lies between its two endpoint densities (dky0 + dky1 = width).
    widths = ky - np.concatenate(([0.0], ky[:-1]))
    mean = _tglf_weighted(ky, flux)[1:] / widths[1:]
    assert np.all(mean >= np.minimum(flux[:-1], flux[1:]) - 1e-12)
    assert np.all(mean <= np.maximum(flux[:-1], flux[1:]) + 1e-12)


def test_sensitivity_tables_compare_configurations_on_one_state(sensitivity, driver, sample,
                                                                 tmp_path, monkeypatch):
    from omas import ODS

    import vaft.code.gacode.tglf as tglf
    from vaft.code.gacode.tglf import TGLFResult
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    ods = _electron_only(sample)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [{"shot": 48224, "time_ms": 300, "label": "good"}]}))

    def fake(profile, rho, workdir, config=None, *, check=True):
        scale = (1.0 + config.sat_rule) * (2.0 if config.use_bper else 1.0)
        flux = np.array([1.0, 2.0, 0.1]) * scale
        native = TglfOutputs(directory=str(workdir), gbflux={"particle": flux, "energy": flux,
                             "momentum": 0 * flux, "exchange": 0 * flux},
                             grid={"n_species": 3, "n_xgrid": 4})
        return TGLFResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                          outputs_native=native)

    monkeypatch.setattr(tglf, "run_tglf_case", fake)
    for sat in ("0", "3"):
        for field in ("es", "em-bper"):
            driver.main(["--filedb", str(filedb), "--labels", str(labels), "--out", str(tmp_path / "runs"),
                         "--lineages", "magnetics", "--surfaces", "0.5",
                         "--sat-rule", sat, "--field-model", field])
    assert sensitivity.main(["--runs", str(tmp_path / "runs"), "--out", str(tmp_path / "s")]) == 0
    schema = json.loads((tmp_path / "s" / "schema.json").read_text())
    assert schema["configurations"] == ["tglf-sat0-em-bper", "tglf-sat0-es", "tglf-sat3-em-bper", "tglf-sat3-es"]
    import csv

    with open(tmp_path / "s" / "sensitivity_pairs.csv", newline="") as handle:
        (pair,) = list(csv.DictReader(handle))
    # q_tot scales 1 (sat0 es), 2 (sat0 em), 4 (sat3 es), 8 (sat3 em): spreads 0.75, EM effect 0.5
    assert float(pair["q_tot_gb_sat_spread_es"]) == pytest.approx(0.75)
    assert float(pair["q_tot_gb_sat_spread_em-bper"]) == pytest.approx(0.75)
    assert float(pair["q_tot_gb_em_vs_es_sat0"]) == pytest.approx(0.5)
    assert float(pair["q_tot_gb_em_vs_es_sat3"]) == pytest.approx(0.5)
    assert pair["q_tot_gb_em_vs_es_sat1"] == ""  # not run: missing, not zero



def _sens_state(root, sat, field, *, identity="s1", quality="good", label=None, energy=(1.0, 2.0)):
    params = {"sat_rule": sat, "use_bper": field != "es", "use_bpar": field == "em-bper-bpar"}
    tree = root / (label or f"tglf-sat{sat}-{field}") / "48224" / "magnetics" / "00300"
    (tree / "r0.50").mkdir(parents=True)
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    flux = np.array([energy[0], energy[1], 0.0])
    TglfOutputs(directory=str(tree), gbflux={"particle": flux, "energy": flux, "momentum": 0 * flux,
                "exchange": 0 * flux}, grid={"n_species": 3, "n_xgrid": 4}).write_json(tree / "r0.50" / "outputs.json")
    (tree / "state.json").write_text(json.dumps({
        "shot": 48224, "time_efit_s": 0.3, "efit_lineage": "magnetics", "efit_quality": quality,
        "ti_lineage": "ti_eq_te_assumed", "state_identity": identity,
        "tglf_config": label or f"tglf-sat{sat}-{field}", "tglf_parameters": params,
        "surfaces": [{"r_over_a": 0.5, "status": "solved", "run_identity": f"r{sat}{field}"}]}))
    return tree


def test_sensitivity_refuses_what_it_cannot_compare_honestly(sensitivity, tmp_path):
    _sens_state(tmp_path / "a", 0, "es")
    _sens_state(tmp_path / "b", 0, "es")           # the same configuration on the same state twice
    with pytest.raises(ValueError, match="appears twice"):
        sensitivity.build_rows(tmp_path)
    with pytest.raises(ValueError, match="does not name its settings"):
        _sens_state(tmp_path / "c", 1, "es", label="tglf-sat3-es")
        sensitivity.build_rows(tmp_path / "c")
    with pytest.raises(ValueError, match="not good/admissible"):
        _sens_state(tmp_path / "d", 1, "es", quality="unreconstructible")
        sensitivity.build_rows(tmp_path / "d")


def test_a_pruned_native_tree_is_reported_not_fatal(sensitivity, tmp_path):
    tree = _sens_state(tmp_path, 0, "es")
    (tree / "r0.50" / "outputs.json").unlink()
    (row,) = sensitivity.build_rows(tmp_path)
    assert row["status"] == "missing_outputs" and "qe_gb" not in row


def test_a_nan_flux_does_not_make_the_spread_order_dependent(sensitivity, tmp_path):
    _sens_state(tmp_path, 0, "es", energy=(float("nan"), 1.0))
    _sens_state(tmp_path, 1, "es", energy=(1.0, 1.0))
    _sens_state(tmp_path, 2, "es", energy=(2.0, 2.0))
    (pair,) = sensitivity.build_pairs(sensitivity.build_rows(tmp_path))
    assert pair["n_configs"] == 3
    assert pair["qi_gb_sat_spread_es"] == pytest.approx(0.5)   # all three finite
    assert pair["qe_gb_sat_spread_es"] == pytest.approx(0.5)   # NaN (sat0) left out
    assert sensitivity.delta(float("nan"), 1.0) is None


def test_ky_peak_reads_the_ky_grid_model_of_the_run(sensitivity):
    """cold review 0.8.0 delta-absorb-11 F1.

    ``write_tglf_sum_flux_spectrum`` only forms the log-trapezoid weights when
    KYGRID_MODEL != 0. For KYGRID_MODEL = 0 (linear grid ky_k = k * ky_1) every entry is
    ``ky_1 * Q_k``: a right-endpoint rectangle, so the flux is sampled AT ky_k, not
    averaged over the interval, and the peak is the node, not the interval midpoint.
    """
    from types import SimpleNamespace

    ky = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    flux = np.array([0.1, 0.2, 0.5, 3.0, 0.4, 0.2])  # sampled flux peaks at ky = 0.4
    spectrum = np.zeros((1, 1, ky.size, 5))
    spectrum[0, 0, :, 1] = ky[0] * flux                 # KYGRID_MODEL = 0 weighting
    native = SimpleNamespace(ky_spectrum=ky, sum_flux_spectrum=spectrum)
    # The default grid model (1) is log-trapezoid: midpoint of the peak interval.
    assert sensitivity.ky_peak(native, {}) == pytest.approx(0.35)
    assert sensitivity.ky_peak(native, {"extra_parameters": {"KYGRID_MODEL": 1}}) == pytest.approx(0.35)
    # Model 0: the entry is the sample at the node, so the peak is the node itself,
    # whichever case the user spelled the key in.
    assert sensitivity.ky_peak(native, {"extra_parameters": {"KYGRID_MODEL": 0}}) == pytest.approx(0.4)
    assert sensitivity.ky_peak(native, {"extra_parameters": {"kygrid_model": 0}}) == pytest.approx(0.4)
    # Model 0 inverts the actual weight ky_1, not the interval width, so a corrupt
    # non-uniform grid cannot bias the density towards its narrow intervals.
    ky2 = np.array([0.1, 0.2, 0.4, 0.8])
    flux2 = np.array([1.0, 1.0, 1.0, 1.5])
    spectrum2 = np.zeros((1, 1, ky2.size, 5))
    spectrum2[0, 0, :, 1] = ky2[0] * flux2
    peak = sensitivity.ky_peak(SimpleNamespace(ky_spectrum=ky2, sum_flux_spectrum=spectrum2),
                               {"extra_parameters": {"KYGRID_MODEL": 0}})
    assert peak == pytest.approx(0.8)
    # Every non-zero model takes the Fortran ``.ne.0`` branch (log-trapezoid); only a
    # value that is not an integer leaves the descriptor empty.
    assert sensitivity.ky_peak(native, {"extra_parameters": {"KYGRID_MODEL": 4}}) == pytest.approx(0.35)
    assert sensitivity.ky_peak(native, {"extra_parameters": {"KYGRID_MODEL": "auto"}}) is None


def test_ky_peak_in_the_table_uses_the_grid_model_of_the_run(sensitivity, tmp_path):
    """cold review 0.8.0 delta-absorb-11 F1: build_rows passes the run's parameters."""
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    ky = np.array([0.1, 0.2, 0.3, 0.4])
    flux = np.array([0.1, 0.2, 3.0, 0.4])
    for sat, model in ((0, 0), (1, 1)):
        tree = _sens_state(tmp_path, sat, "es")
        state = json.loads((tree / "state.json").read_text(encoding="utf-8"))
        state["tglf_parameters"]["extra_parameters"] = {"KYGRID_MODEL": model}
        (tree / "state.json").write_text(json.dumps(state), encoding="utf-8")
        spectrum = np.zeros((1, 1, ky.size, 5))
        spectrum[0, 0, :, 1] = ky[0] * flux if model == 0 else _tglf_weighted(ky, flux)
        TglfOutputs(directory=str(tree), gbflux={"particle": flux[:1], "energy": flux[:1],
                    "momentum": 0 * flux[:1], "exchange": 0 * flux[:1]},
                    grid={"n_species": 1, "n_xgrid": 4}, ky_spectrum=ky,
                    sum_flux_spectrum=spectrum).write_json(tree / "r0.50" / "outputs.json")
    rows = {r["sat_rule"]: r for r in sensitivity.build_rows(tmp_path)}
    assert rows[0]["ky_q_peak"] == pytest.approx(0.3)    # model 0: the node
    assert rows[1]["ky_q_peak"] == pytest.approx(0.25)   # model 1: the interval midpoint


def test_a_failed_surface_keeps_why_it_failed(sensitivity, tmp_path):
    """cold review 0.8.0 delta-absorb-11 F2: a #1299 timeout is not just 'failed'."""
    tree = _sens_state(tmp_path, 0, "es")
    state = json.loads((tree / "state.json").read_text(encoding="utf-8"))
    state["surfaces"] = [
        {"r_over_a": 0.3, "status": "failed", "runtime_status": "timeout", "returncode": None,
         "errors": [], "run_identity": "r0"},
        {"r_over_a": 0.4, "status": "failed", "runtime_status": "error", "returncode": None,
         "errors": ["FileNotFoundError: tglf"], "run_identity": "r1"},
        {"r_over_a": 0.5, "status": "solved", "runtime_status": "completed", "returncode": 0,
         "errors": [], "run_identity": "r2"},
        {"r_over_a": 0.6, "status": "not_run", "readiness": "ready", "run_identity": "r3"},
        {"r_over_a": 0.7, "status": "not_ready", "readiness": "outside_profile"},
    ]
    (tree / "state.json").write_text(json.dumps(state), encoding="utf-8")
    rows = {r["r_over_a"]: r for r in sensitivity.build_rows(tmp_path)}
    assert (rows[0.3]["runtime_status"], rows[0.3]["tglf_reason"]) == ("timeout", "timeout")
    assert (rows[0.4]["runtime_status"], rows[0.4]["tglf_reason"]) == ("error", "error")
    assert rows[0.4]["n_errors"] == 1
    assert rows[0.5]["tglf_reason"] is None
    assert rows[0.6]["tglf_reason"] == "not_run"
    assert rows[0.7]["tglf_reason"] == "outside_profile"
    for column in ("runtime_status", "tglf_reason", "n_errors"):
        assert column in sensitivity.SCHEMA
    assert "not_run" in sensitivity.SCHEMA["status"][2]
    # The columns reach the CSV through main.
    assert sensitivity.main(["--runs", str(tmp_path), "--out", str(tmp_path / "s")]) == 0
    import csv

    with open(tmp_path / "s" / "sensitivity.csv", newline="", encoding="utf-8") as handle:
        table = {float(r["r_over_a"]): r for r in csv.DictReader(handle)}
    assert table[0.3]["tglf_reason"] == "timeout" and table[0.6]["tglf_reason"] == "not_run"


def test_sensitivity_refuses_other_states_and_other_configurations(sensitivity, tmp_path):
    """cold review 0.8.0 delta-absorb-11 F5: the two refusals the PR claimed but never ran."""
    _sens_state(tmp_path / "a", 0, "es", identity="s1")
    _sens_state(tmp_path / "a", 1, "es", identity="s2")
    with pytest.raises(ValueError, match="different states"):
        sensitivity.build_pairs(sensitivity.build_rows(tmp_path / "a"))
    _sens_state(tmp_path / "b", 4, "es")
    with pytest.raises(ValueError, match="not in the sensitivity space"):
        sensitivity.build_rows(tmp_path / "b")


def test_pair_rows_carry_one_label_and_the_rows_surface_value(sensitivity, tmp_path):
    """cold review 0.8.0 delta-absorb-11 F4/F6: efit_quality is checked, r_over_a is not re-rounded."""
    for sat in (0, 1):
        tree = _sens_state(tmp_path, sat, "es")
        state = json.loads((tree / "state.json").read_text(encoding="utf-8"))
        state["surfaces"][0]["r_over_a"] = 0.33333
        (tree / "state.json").write_text(json.dumps(state), encoding="utf-8")
        (tree / "r0.50").rename(tree / "r0.33")
    rows = sensitivity.build_rows(tmp_path)
    (pair,) = sensitivity.build_pairs(rows)
    assert pair["r_over_a"] == rows[0]["r_over_a"] == 0.33333   # joins on the key text
    assert pair["n_configs"] == 2
    # A re-labelled slice between two configuration runs is not silently one state.
    rows[1]["efit_quality"] = "admissible"
    with pytest.raises(ValueError, match="efit_quality"):
        sensitivity.build_pairs(rows)



# --------------------------------------------------------------------------- lane K inferred Ti (#1426)


def _with_inferred_ti(sample, outer_nan_from=None, note=inferred_ti_text("equilibrium_pressure_partition")):
    ods = copy.deepcopy(sample)
    prefix = "core_profiles.profiles_1d.0"
    te = np.asarray(ods[f"{prefix}.electrons.temperature"], dtype=float)
    rho = np.asarray(ods[f"{prefix}.grid.rho_tor_norm"], dtype=float)
    ti = 1.8 * te
    if outer_nan_from is not None:
        ti = np.where(rho >= outer_nan_from, np.nan, ti)
    ods[f"{prefix}.ion.0.temperature"] = ti
    ods[f"{prefix}.ion.0.temperature_fit.parameters"] = note
    return ods


def test_a_stored_inferred_ti_is_not_taken_as_measured(sample):
    state = resolve_transport_state(_with_inferred_ti(sample), _key(0.3), use_stored_inferred_ti=True, efit_quality="good")
    assert state.resolved, state.reasons
    assert state.ti_lineage == "pressure_partition_inferred"
    assert state.ti["kind"] == "inferred"
    # Only the separatrix point, where the sample's Te (so Ti) is exactly 0, is filled.
    assert state.ti["fill"]["points"] == 1
    assert state.ti["method"] == "equilibrium_pressure_partition"
    report = assess_tglf_readiness(state)
    assert "ti_inferred" in report.conditions
    assert all(s.ready for s in report.surfaces)


def test_inferred_ti_gaps_are_filled_declared_and_never_reach_a_run_surface(sample):
    from vaft.code.gacode.tglf.inputs import prepare_tglf_input

    ods = _with_inferred_ti(sample, outer_nan_from=0.6)
    surfaces = tuple(np.round(np.arange(0.30, 0.86, 0.01), 2))
    state = resolve_transport_state(ods, _key(0.3), use_stored_inferred_ti=True, efit_quality="good")
    assert state.resolved, state.reasons
    assert state.ti["fill"]["points"] > 0 and "(policy)" in state.ti["fill"]["value"]
    assert state.fill_probe is not None
    report = assess_tglf_readiness(state, surfaces)
    ready = [s.r_over_a for s in report.surfaces if s.ready]
    assert ready and len(ready) < len(surfaces)
    assert {s.status for s in report.surfaces if not s.ready} == {"ti_not_inferred_here"}
    # The real precondition: a run surface's ion inputs do not move when only the fill does.
    hot = resolve_transport_state(ods, _key(0.3), use_stored_inferred_ti=True, efit_quality="good", ti_te_ratio=5.0)
    for r in ready:
        a, b = prepare_tglf_input(state.profile, r), prepare_tglf_input(hot.profile, r)
        np.testing.assert_allclose(a.taus[1:], b.taus[1:], rtol=1e-3)
        np.testing.assert_allclose(a.rlts[1:], b.rlts[1:], rtol=1e-3, atol=1e-5)
    # The source ODS still carries its NaNs: the fill happened on a copy.
    assert np.isnan(np.asarray(ods["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float)).any()


def test_a_caller_ti_keeps_the_caller_closure_over_a_labelled_product(sample):
    ods = _with_inferred_ti(sample)
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good",
                                    inferred_ti={"temperature": 0.9 * te, "time": 0.3})
    assert "source" not in state.composition and state.composition["impurity"] == "C"


def test_a_non_positive_fill_ratio_is_refused(sample):
    state = resolve_transport_state(_with_inferred_ti(sample, outer_nan_from=0.6), _key(0.3), use_stored_inferred_ti=True,
                                    efit_quality="good", ti_te_ratio=-1.0)
    assert state.reasons == ("non_positive_ti_te_ratio",)

def test_an_incomplete_inferred_ti_without_a_fallback_is_insufficient(sample):
    state = resolve_transport_state(_with_inferred_ti(sample, outer_nan_from=0.6), _key(0.3), use_stored_inferred_ti=True,
                                    efit_quality="good", ti_te_ratio=None)
    assert state.reasons == ("inferred_ti_incomplete",)


def test_a_core_profiles_directory_replaces_the_stage_product(driver, tmp_path):
    (tmp_path / "39916.json.gz").write_bytes(b"x")
    path, manifest = driver.core_profiles_product(tmp_path / "filedb", 39916, tmp_path)
    assert path == tmp_path / "39916.json.gz" and manifest == {}
    assert driver.core_profiles_product(tmp_path / "filedb", 1, tmp_path)[0] is None


def test_a_product_with_its_own_inferred_species_list_is_converted_as_given(sample):
    ods = _with_inferred_ti(sample)
    prefix = "core_profiles.profiles_1d.0"
    ne = np.asarray(ods[f"{prefix}.electrons.density_thermal"], dtype=float)
    ti = np.asarray(ods[f"{prefix}.ion.0.temperature"], dtype=float)
    ods[f"{prefix}.ion.0.density_thermal"] = 0.8 * ne
    ods[f"{prefix}.ion.1.label"] = "C6+"
    ods[f"{prefix}.ion.1.z_ion"] = 6.0
    ods[f"{prefix}.ion.1.element.0.z_n"] = 6.0
    ods[f"{prefix}.ion.1.element.0.a"] = 12.011
    ods[f"{prefix}.ion.1.density_thermal"] = ne / 30.0
    ods[f"{prefix}.ion.1.temperature"] = ti
    ods[f"{prefix}.ion.1.temperature_fit.parameters"] = "origin=inferred; method=equilibrium_pressure_partition"
    state = resolve_transport_state(ods, _key(0.3), use_stored_inferred_ti=True, efit_quality="good")
    assert state.resolved, state.reasons
    assert state.ti_lineage == "pressure_partition_inferred"
    assert list(state.profile.name) == ["H+", "C6+"]
    assert state.composition["source"].startswith("the core_profiles product")
    assert state.composition["z_eff"] == pytest.approx(2.0, rel=1e-3)  # from the species list
    assert state.composition["quasineutrality_error"] < 1e-6
    # Two ions with different inferred temperatures are not one inferred state.
    ods[f"{prefix}.ion.1.temperature"] = 0.5 * ti
    assert resolve_transport_state(ods, _key(0.3), use_stored_inferred_ti=True, efit_quality="good").reasons == (
        "inferred_ti_species_disagree",)


def _two_ion_inferred(sample, outer_nan_from=None):
    ods = _with_inferred_ti(sample, outer_nan_from=outer_nan_from)
    prefix = "core_profiles.profiles_1d.0"
    ne = np.asarray(ods[f"{prefix}.electrons.density_thermal"], dtype=float)
    ods[f"{prefix}.ion.0.density_thermal"] = 0.8 * ne
    for key, value in (("label", "C6+"), ("z_ion", 6.0), ("element.0.z_n", 6.0), ("element.0.a", 12.011)):
        ods[f"{prefix}.ion.1.{key}"] = value
    ods[f"{prefix}.ion.1.density_thermal"] = ne / 30.0
    ods[f"{prefix}.ion.1.temperature"] = np.asarray(ods[f"{prefix}.ion.0.temperature"], dtype=float)
    ods[f"{prefix}.ion.1.temperature_fit.parameters"] = "origin=inferred; method=equilibrium_pressure_partition"
    return ods


def test_a_caller_ti_on_a_multi_ion_product_keeps_the_product_species(sample):
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
    state = resolve_transport_state(_two_ion_inferred(sample), _key(0.3), efit_quality="good",
                                    inferred_ti={"temperature": 0.9 * te, "time": 0.3})
    assert state.resolved, state.reasons
    assert list(state.profile.name) == ["H+", "C6+"]
    assert state.composition["z_eff"] == pytest.approx(2.0, rel=1e-3)


def test_the_gate_catches_spline_reach_and_neo_has_its_own(sample):
    from vaft.process.transport_state import inferred_ti_supported

    state = resolve_transport_state(_with_inferred_ti(sample, outer_nan_from=0.6), _key(0.3), use_stored_inferred_ti=True,
                                    efit_quality="good")
    (lo, hi), = state.ti["support_rho"]
    rho_at = lambda r: float(np.interp(r, state.profile.rmin / state.profile.rmin[-1], state.profile.rho))
    grid = np.round(np.arange(0.30, 0.86, 0.01), 2)
    tglf_ok = {r: inferred_ti_supported(state, r) for r in grid}
    neo_ok = {r: inferred_ti_supported(state, r, solver="neo") for r in grid}
    # Some surfaces inside the support are refused: the fill reaches them through the spline.
    assert any(not ok and rho_at(r) < hi for r, ok in tglf_ok.items())
    # Nothing beyond the support passes either gate.
    assert not any(ok and rho_at(r) > hi for r, ok in {**tglf_ok, **neo_ok}.items())
    assert any(neo_ok.values())
    assert assess_neo_readiness(state, (0.85,)).reasons == ("no_surface_independent_of_ti_fill",)
    assert assess_neo_readiness(state).runnable


def test_the_atlas_never_partitions_a_neo_surface_the_fill_reaches(atlas, driver, neo_driver, sample,
                                                                     tmp_path, monkeypatch):
    from omas import ODS

    import vaft.code.gacode.neo as neo
    import vaft.code.gacode.tglf as tglf
    from vaft.code.gacode.neo import NEOResult
    from vaft.code.gacode.neo.outputs import collect_neo_outputs
    from vaft.code.gacode.tglf import TGLFResult
    from vaft.code.gacode.tglf.outputs import TglfOutputs

    ods = _two_ion_inferred(sample, outer_nan_from=0.6)
    eq, cp = ODS(consistency_check=False), ODS(consistency_check=False)
    eq["equilibrium"] = ods["equilibrium"]
    cp["core_profiles"] = ods["core_profiles"]
    filedb = tmp_path / "filedb"
    _write_product(filedb / "omas", "core_profiles", 48224, "core_profiles.json.gz", cp)
    _write_product(filedb / "omas", "efit/magnetic", 48224, "efit.json.gz", eq)
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"labels": [{"shot": 48224, "time_ms": 300, "label": "good"}]}))

    def fake_tglf(profile, rho, workdir, config=None, *, check=True):
        flux = np.array([1.0, 2.0, 0.1])
        native = TglfOutputs(directory=str(workdir), gbflux={"particle": flux, "energy": flux,
                             "momentum": 0 * flux, "exchange": 0 * flux}, grid={"n_species": 3, "n_xgrid": 4})
        return TGLFResult(returncode=0, runtime_status="completed", workdir=Path(workdir), outputs_native=native)

    def fake_neo(profile, workdir, config=None, *, check=True):
        Path(workdir).mkdir(parents=True, exist_ok=True)
        return NEOResult(returncode=0, runtime_status="completed", workdir=Path(workdir),
                         outputs_native=collect_neo_outputs(ROOT / "test" / "data" / "gacode" / "neo_vest_48224_carbon"))

    monkeypatch.setattr(tglf, "run_tglf_case", fake_tglf)
    monkeypatch.setattr(neo, "run_neo_case", fake_neo)
    surfaces = [f"{0.2 + 0.1 * i:.1f}" for i in range(7)]
    common = ["--filedb", str(filedb), "--labels", str(labels), "--lineages", "magnetics", "--surfaces", *surfaces]
    driver.main(common + ["--use-inferred-ti", "--out", str(tmp_path / "tglf"), "--sat-rule", "3", "--field-model", "em-bper"])
    neo_driver.main(common + ["--use-inferred-ti", "--out", str(tmp_path / "neo")])
    rows = atlas.build_rows(tmp_path / "tglf" / "tglf-sat3-em-bper", tmp_path / "neo")
    by_r = {round(r["r_over_a"], 1): r for r in rows}
    assert by_r[0.8]["tglf_status"] == "not_ready"
    assert by_r[0.8]["partition_status"] in ("ti_not_inferred_here", "missing_turbulent_component")
    assert by_r[0.3]["partition_status"] == "available"
    assert {r["ti_lineage"] for r in rows} == {"pressure_partition_inferred"}


def test_a_stored_inferred_ti_is_refused_unless_the_caller_opts_in(sample):
    """The routine atlas keeps Ti = Te (2026-10-02): a labelled product is refused by default."""
    state = resolve_transport_state(_with_inferred_ti(sample), _key(0.3), efit_quality="good")
    assert not state.resolved
    assert state.reasons == ("inferred_ti_not_enabled",)
    assert state.ti.get("lineage") != "measured"
    opted = resolve_transport_state(_with_inferred_ti(sample), _key(0.3), efit_quality="good",
                                    use_stored_inferred_ti=True)
    assert opted.ti["lineage"] == "pressure_partition_inferred"


# --------------------------------------------------------------------------- Ti record grammar (cold review 0.8.0)


_RATIO_RECORDS = [
    "ti_te_ratio=1; status=assumed; source=legacy Ti=Te fallback",  # profile.py legacy fallback
    "ti_te_ratio=1; sigma=0.3; status=assumed; source=caller argument",  # kinetic.py caller record
    "ti_te_ratio=1; status=unspecified",  # profile.py ratio_fallback without a record
]
_INFERRED_RECORDS = [
    inferred_ti_text("equilibrium_pressure_partition"),
    "origin: inferred; method: equilibrium_pressure_partition",  # issue #1426 section 4 spelling
    "origin=pressure_inferred; method=x",
]


def _with_record(sample, record):
    ods = copy.deepcopy(sample)
    ods["core_profiles.profiles_1d.0.ion.0.temperature_fit.parameters"] = record
    return ods


@pytest.mark.parametrize("record", _RATIO_RECORDS + ["policy"])
def test_a_ratio_record_resolves_as_assumed_at_the_records_ratio_never_measured(sample, record):
    """Every #1414 spelling -- legacy, caller, policy -- is an assumed Ti, with its ti_* condition."""
    if record == "policy":
        record = policy_for_ods(sample, 48224).ti_te_ratio_text()  # build_kinetic_core_profiles(ti_te_ratio="auto")
    state = resolve_transport_state(_with_record(sample, record), _key(0.3), efit_quality="good")
    assert state.resolved, state.reasons
    assert state.ti_lineage != "measured"
    assert state.ti_lineage == "ti_eq_te_assumed"
    assert state.ti["kind"] == "assumed" and state.ti["ratio"] == 1.0
    assert state.ti["source"] == record
    assert state.ti["hierarchy"][0] == {"step": "measured", "status": "unavailable",
                                        "reason": f"product record: {record}"}
    assert state.ti["hierarchy"][-1]["source"] == "product record"
    np.testing.assert_allclose(state.profile.ti[0], state.profile.te, rtol=1e-9)
    assert "ti_assumed" in assess_tglf_readiness(state).conditions


def test_a_ratio_record_keeps_its_own_ratio_and_sigma(sample):
    state = resolve_transport_state(_with_record(sample, "ti_te_ratio=0.8; sigma=0.2; status=assumed; source=x"),
                                    _key(0.3), efit_quality="good", ti_te_ratio=0.5)
    assert state.ti_lineage == "ti_te_0.8_assumed"
    assert state.ti["ratio"] == 0.8 and state.ti["sigma"] == 0.2
    np.testing.assert_allclose(state.profile.ti[0], 0.8 * state.profile.te, rtol=1e-9)
    # A caller-supplied inference still outranks the product's ratio record.
    te = np.asarray(sample["core_profiles.profiles_1d.0.electrons.temperature"], dtype=float)
    inferred = resolve_transport_state(_with_record(sample, _RATIO_RECORDS[0]), _key(0.3), efit_quality="good",
                                       inferred_ti={"temperature": 0.9 * te, "time": 0.3})
    assert inferred.ti_lineage == "pressure_partition_inferred"


@pytest.mark.parametrize("record", _INFERRED_RECORDS)
def test_every_inferred_spelling_is_refused_by_default_and_used_on_opt_in(sample, record):
    state = resolve_transport_state(_with_record(sample, record), _key(0.3), efit_quality="good")
    assert state.ti_lineage != "measured"
    assert state.reasons == ("inferred_ti_not_enabled",)
    opted = resolve_transport_state(_with_record(sample, record), _key(0.3), efit_quality="good",
                                    use_stored_inferred_ti=True)
    assert opted.resolved, opted.reasons
    assert opted.ti_lineage == "pressure_partition_inferred" and opted.ti["kind"] == "inferred"


@pytest.mark.parametrize("record", [None,
                                    "coordinate=rho_tor_norm; method=polynomial; order=2; measured_span=0.0500:0.9500",
                                    "coordinate=rho_tor_norm; method=external",
                                    "origin=measured; source=CES"])
def test_a_measured_fit_record_or_none_stays_measured(sample, record):
    ods = sample if record is None else _with_record(sample, record)
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good")
    assert state.resolved and state.ti_lineage == "measured"
    assert "ti_measured" not in assess_tglf_readiness(state).conditions


def test_the_packaged_sample_ion_temperature_is_a_measurement(sample):
    """48224 carries charge_exchange and a fit without a record: measured is the right lineage."""
    assert "charge_exchange" in sample
    fit = sample["core_profiles.profiles_1d.0.ion.0.temperature_fit"]
    assert "parameters" not in fit and "measured" in fit
    assert resolve_transport_state(sample, _key(0.3), efit_quality="good").ti_lineage == "measured"


@pytest.mark.parametrize("record", ["origin=synthetic", "free text about this profile"])
def test_a_record_in_no_known_grammar_is_refused_by_name(sample, record):
    state = resolve_transport_state(_with_record(sample, record), _key(0.3), efit_quality="good")
    assert state.reasons == ("ti_record_unrecognised",)
    assert state.ti.get("lineage") != "measured"


def test_an_incomplete_inferred_label_set_names_what_is_missing(sample):
    """F4: an unlabelled main ion beside a labelled impurity is not a disagreement."""
    ods = _two_ion_inferred(sample)
    prefix = "core_profiles.profiles_1d.0"
    del ods[f"{prefix}.ion.0.temperature_fit.parameters"]
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good", use_stored_inferred_ti=True)
    assert state.reasons == ("inferred_ti_label_missing",)
    # A labelled ion with no temperature array is missing its array, not disagreeing.
    ods = _two_ion_inferred(sample)
    del ods[f"{prefix}.ion.1.temperature"]
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good", use_stored_inferred_ti=True)
    assert state.reasons == ("inferred_ti_temperature_missing",)
    # Two labelled ions with different arrays do disagree.
    ods = _two_ion_inferred(sample)
    ods[f"{prefix}.ion.1.temperature"] = 0.5 * np.asarray(ods[f"{prefix}.ion.1.temperature"], dtype=float)
    state = resolve_transport_state(ods, _key(0.3), efit_quality="good", use_stored_inferred_ti=True)
    assert state.reasons == ("inferred_ti_species_disagree",)


def test_a_bad_ti_te_ratio_string_is_refused_at_entry(sample):
    """F6: validated once, not only when a fill is needed."""
    complete = _with_inferred_ti(sample)
    complete["core_profiles.profiles_1d.0.ion.0.temperature"] = np.maximum(
        np.asarray(complete["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float), 1.0)
    with pytest.raises(ValueError, match="ti_te_ratio must be"):
        resolve_transport_state(complete, _key(0.3), efit_quality="good", use_stored_inferred_ti=True,
                                ti_te_ratio="bogus")
    with pytest.raises(ValueError, match="ti_te_ratio must be"):
        resolve_transport_state(sample, _key(0.3), efit_quality="good", ti_te_ratio="auto")
