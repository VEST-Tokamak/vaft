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
