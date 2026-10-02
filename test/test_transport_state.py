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
