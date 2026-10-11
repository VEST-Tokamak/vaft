"""The composition stage of pipeline 2 (#1842): C/O composition, n_i and Z_eff(rho) on measured core profiles.

Synthetic ADF11 tables and synthetic products, no network and no repository data.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from omas import ODS

from vaft.data.open_adas import read_adf11
from vaft.machine_mapping.core_profiles import classify_ti_record, ti_record_fields
from vaft.omas import vest_upstream as vu
from vaft.ods_access import path_value

SHOT = 48226
LOG_NE = (10.0, 14.0)
LOG_TE = (0.0, 1.0, 2.0, 3.0)
RHO = np.linspace(0.0, 1.0, 11)
TE = 300.0 * (1.0 - 0.95 * RHO**2)
NE = 1.5e19 * (1.0 - 0.7 * RHO**2)
E = 1.602176634e-19


def _table(path, blocks):
    lines = [f"{len(blocks)} {len(LOG_NE)} {len(LOG_TE)} / synthetic", "",
             " ".join(map(str, LOG_NE)), " ".join(map(str, LOG_TE))]
    for index, values in enumerate(blocks, start=1):
        lines.append(f"---------------- /IPRT=1/IGRD=1/TYPE=TEST/Z1={index}/")
        lines.append(" ".join(f"{v:.3f}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _element(tmp_path, symbol, z):
    scd, acd = [], []
    for j in range(z):
        threshold = 0.6 + 0.35 * j
        per_te = [-8.0 - 3.0 * max(threshold - t, 0.0) for t in LOG_TE]
        scd.append([v for v in per_te for _ in LOG_NE])
        acd.append([-11.0] * (len(LOG_NE) * len(LOG_TE)))
    read_adf11.cache_clear()
    return (_table(tmp_path / f"acd96_{symbol.lower()}.dat", acd),
            _table(tmp_path / f"scd96_{symbol.lower()}.dat", scd))


@pytest.fixture
def tables(tmp_path):
    return {"C": _element(tmp_path, "C", 6), "O": _element(tmp_path, "O", 8)}


def _core_profiles(tmp_path, *, kinetic=(0,), times=(0.300, 0.301), zeff_record=None, ti_record=True,
                   rho_pol=None, thomson=False):
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = np.asarray(times, dtype=float)
    for index in range(len(times)):
        base = f"core_profiles.profiles_1d.{index}"
        ods[f"{base}.grid.rho_tor_norm"] = RHO
        ods[f"{base}.grid.rho_pol_norm"] = RHO if rho_pol is None else rho_pol
        ods[f"{base}.electrons.temperature"] = TE
        ods[f"{base}.electrons.density"] = NE
        ods[f"{base}.electrons.temperature_fit.parameters"] = (
            "coordinate=rho_tor_norm; method=polynomial; order=2"
        )
        if index in kinetic:
            ods[f"{base}.ion.0.label"] = "H+"
            ods[f"{base}.ion.0.z_ion"] = 1.0
            ods[f"{base}.ion.0.element.0.z_n"] = 1.0
            ods[f"{base}.ion.0.element.0.a"] = 1.0
            ods[f"{base}.ion.0.temperature"] = 0.8 * TE
            # what the measured stage stores: one ion per electron
            ods[f"{base}.pressure_thermal"] = E * NE * (TE + 0.8 * TE)
            if ti_record is True:
                ods[f"{base}.ion.0.temperature_fit.parameters"] = (
                    "coordinate=rho_tor_norm; method=polynomial; order=2"
                )
            elif ti_record:
                ods[f"{base}.ion.0.temperature_fit.parameters"] = ti_record
        if zeff_record is not None and index == 1:
            ods[f"{base}.zeff"] = np.full(RHO.size, 1.5)
            ods[f"{base}.zeff_fit.parameters"] = zeff_record
    if thomson:
        # five channels, 5 % one-sigma on both n_e and T_e, at the slice times
        ods["thomson_scattering.time"] = np.asarray(times, dtype=float)
        for i, r in enumerate(np.linspace(0.2, 0.6, 5)):
            channel = f"thomson_scattering.channel.{i}"
            ods[f"{channel}.position.r"] = float(r)
            ods[f"{channel}.n_e.data"] = np.full(len(times), 1.0e19)
            ods[f"{channel}.n_e.data_error_upper"] = np.full(len(times), 5.0e17)
            ods[f"{channel}.t_e.data"] = np.full(len(times), 100.0)
            ods[f"{channel}.t_e.data_error_upper"] = np.full(len(times), 5.0)
    path = tmp_path / "core_profiles.json"
    ods.save(str(path))
    return path


def _efit(tmp_path, times=(0.300, 0.301)):
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.asarray(times, dtype=float)
    for index in range(len(times)):
        base = f"equilibrium.time_slice.{index}.profiles_1d"
        ods[f"{base}.rho_tor_norm"] = RHO
        ods[f"{base}.volume"] = 0.5 * RHO**2
    path = tmp_path / "efit.json"
    ods.save(str(path))
    return path


def _build(tmp_path, tables, **kwargs):
    kwargs.setdefault("ionization", "coronal")
    kwargs.setdefault("assume_ti_eq_te", False)  # the electron-only baseline the T_i stage builds on
    # Built only when absent: an eager setdefault would rewrite a product the
    # caller already wrote to the same path.
    if "core_profiles_product" not in kwargs:
        kwargs["core_profiles_product"] = _core_profiles(tmp_path)
    if "efit_product" not in kwargs:
        kwargs["efit_product"] = _efit(tmp_path)
    return vu.build_composition_ods(shot=SHOT, tables=tables, **kwargs)


def test_an_electron_only_slice_is_given_an_assumed_ti_equal_to_te_and_a_measured_one_is_kept(tmp_path, tables):
    ods, manifest = _build(tmp_path, tables, assume_ti_eq_te=True)
    assert manifest["ion_temperature"] == {"assumed_ti_eq_te": 1}
    base = "core_profiles.profiles_1d.0"
    assert np.asarray(ods[f"{base}.ion.0.temperature"], dtype=float) == pytest.approx(0.8 * TE), "measured kept"
    base = "core_profiles.profiles_1d.1"
    te = np.asarray(ods[f"{base}.electrons.temperature"], dtype=float)
    for k in range(3):
        assert np.asarray(ods[f"{base}.ion.{k}.temperature"], dtype=float) == pytest.approx(te)
    record = ti_record_fields(ods[f"{base}.ion.0.temperature_fit.parameters"])
    assert record["status"] == "assumed" and float(record["ti_te_ratio"]) == 1.0
    assert np.all(np.asarray(ods[f"{base}.pressure_ion_total"], dtype=float) > 0)


def test_every_slice_gets_the_diluted_main_ion_the_impurities_and_a_target_zeff(tmp_path, tables):
    ods, manifest = _build(tmp_path, tables)
    assert manifest["status"] == "success", manifest.get("error")
    assert manifest["slices"] == {"composed": 2, "skipped": 0, "total": 2}
    for index in (0, 1):
        base = f"core_profiles.profiles_1d.{index}"
        labels = [str(ods[f"{base}.ion.{k}.label"]) for k in range(3)]
        assert labels[1:] == ["C", "O"]
        zeff = np.asarray(ods[f"{base}.zeff"], dtype=float)
        assert np.all(np.isfinite(zeff)) and zeff.min() >= 1.0
        main = np.asarray(ods[f"{base}.ion.0.density"], dtype=float)
        assert np.all(main < NE), "the main ion is diluted by the impurities"
        fields = ti_record_fields(ods[f"{base}.zeff_fit.parameters"])
        assert fields["origin"] in ("assumed", "derived")
        assert float(fields["target_zeff"]) == pytest.approx(2.0)


def test_a_composed_slice_records_the_equilibrium_it_was_matched_to(tmp_path, tables):
    ods, _ = _build(tmp_path, tables)
    electrons = ti_record_fields(ods["core_profiles.profiles_1d.0.electrons.temperature_fit.parameters"])
    assert electrons["equilibrium_lineage"] == "magnetics"
    assert electrons["equilibrium_occurrence"] == "0"
    assert float(electrons["equilibrium_time_s"]) == pytest.approx(0.300)
    ion = ods["core_profiles.profiles_1d.0.ion.0.temperature_fit.parameters"]
    assert ti_record_fields(ion)["equilibrium_lineage"] == "magnetics"
    assert classify_ti_record(ion) == "measured", "appending fields must not change what the record is"


def test_an_electron_only_slice_gets_no_ion_temperature_record(tmp_path, tables):
    """A record holding only lineage would classify as 'unknown' beside no temperature."""
    ods, _ = _build(tmp_path, tables)
    assert path_value(ods, "core_profiles.profiles_1d.1.ion.0.temperature_fit.parameters", None) is None
    assert path_value(ods, "core_profiles.profiles_1d.1.ion.0.temperature", None) is None


def test_without_an_efit_product_no_lineage_is_claimed(tmp_path, tables):
    ods, manifest = _build(tmp_path, tables, efit_product=None)
    assert manifest["status"] == "success"
    record = ti_record_fields(ods["core_profiles.profiles_1d.0.electrons.temperature_fit.parameters"])
    assert "equilibrium_lineage" not in record


def test_transient_ionization_without_a_plasma_age_skips_instead_of_assuming_coronal(tmp_path, tables):
    ods, manifest = _build(tmp_path, tables, ionization="transient", diagnostics_product=None)
    assert manifest["status"] == "unavailable"
    assert manifest["slices"] == {"composed": 0, "skipped": 2, "total": 2}
    assert "no positive plasma age" in manifest["error"]
    assert path_value(ods, "core_profiles.profiles_1d.0.zeff", None) is None


def test_a_measured_zeff_is_never_overwritten_and_its_slice_is_named(tmp_path, tables):
    product = _core_profiles(tmp_path, zeff_record="origin=measured; method=spectroscopy")
    ods, manifest = _build(tmp_path, tables, core_profiles_product=product)
    assert manifest["slices"] == {"composed": 1, "skipped": 1, "total": 2}
    assert manifest["status"] == "success"
    assert "never overwritten" in manifest["skipped_slices"][0]["reason"]
    assert np.allclose(np.asarray(ods["core_profiles.profiles_1d.1.zeff"], dtype=float), 1.5)


def test_a_product_without_slices_is_unavailable(tmp_path, tables):
    empty = tmp_path / "empty.json"
    ods = ODS(consistency_check=False)
    ods["dataset_description.data_entry.pulse"] = SHOT   # saveable, and no core_profiles
    ods.save(str(empty))
    _, manifest = _build(tmp_path, tables, core_profiles_product=empty)
    assert manifest["status"] == "unavailable" and "no slices" in manifest["error"]


def test_a_missing_product_path_is_a_caller_error_not_an_absence(tmp_path, tables):
    with pytest.raises(FileNotFoundError):
        _build(tmp_path, tables, core_profiles_product=tmp_path / "nope.json")


def test_an_unknown_ionization_is_refused(tmp_path, tables):
    with pytest.raises(ValueError, match="ionization"):
        _build(tmp_path, tables, ionization="frozen")


def test_the_ne_dv_mean_of_zeff_hits_the_models_target_with_the_efit_volumes(tmp_path, tables):
    """The normalisation is an n_e dV mean; this pins that the EFIT volumes reach it."""
    efit = _efit(tmp_path)
    ods, _ = _build(tmp_path, tables, efit_product=efit)
    from vaft.database._local import load_ods
    from pathlib import Path

    solution, _ = load_ods(Path(efit))
    weight = NE * vu._volume_weights_on_grid(solution, 0, RHO)
    zeff = np.asarray(ods["core_profiles.profiles_1d.0.zeff"], dtype=float)
    assert np.sum(weight * zeff) / np.sum(weight) == pytest.approx(2.0, rel=1e-3)


def test_a_measured_ion_temperature_with_no_record_stays_measured_after_the_lineage_is_added(tmp_path, tables):
    """The packaged 48224 sample has a CES ion temperature and no record for it.

    A record holding only the lineage fields classifies as 'unknown', which a
    consumer such as `infer_kinetic_closure` refuses -- so the absence has to be
    turned into the statement it always meant.
    """
    product = _core_profiles(tmp_path, ti_record=False)
    ods, _ = _build(tmp_path, tables, core_profiles_product=product)
    record = ods["core_profiles.profiles_1d.0.ion.0.temperature_fit.parameters"]
    assert classify_ti_record(record) == "measured", record
    assert ti_record_fields(record)["equilibrium_lineage"] == "magnetics"


def test_an_electron_only_slice_still_carries_the_lineage_on_its_electron_record(tmp_path, tables):
    product = _core_profiles(tmp_path, ti_record=False)
    ods, _ = _build(tmp_path, tables, core_profiles_product=product)
    record = ods["core_profiles.profiles_1d.1.electrons.temperature_fit.parameters"]
    assert ti_record_fields(record)["equilibrium_lineage"] == "magnetics"


def _charged_ion_density(ods, base):
    """n_main plus every charged state: what carries the ion pressure."""
    total = np.asarray(ods[f"{base}.ion.0.density"], dtype=float).copy()
    for k in range(1, len(ods[f"{base}.ion"])):
        for q in range(len(ods[f"{base}.ion.{k}.state"])):
            total += np.asarray(ods[f"{base}.ion.{k}.state.{q}.density"], dtype=float)
    return total


def test_a_measured_slice_gets_the_diluted_ion_pressure_not_one_ion_per_electron(tmp_path, tables):
    ods, manifest = _build(tmp_path, tables)
    base = "core_profiles.profiles_1d.0"
    ti = 0.8 * TE
    n_ion = _charged_ion_density(ods, base)
    assert np.allclose(np.asarray(ods[f"{base}.pressure_ion_total"], dtype=float),
                       E * n_ion * ti, rtol=1e-3, equal_nan=True)
    assert np.allclose(np.asarray(ods[f"{base}.electrons.pressure"], dtype=float), E * NE * TE, rtol=1e-6)
    total = np.asarray(ods[f"{base}.pressure_thermal"], dtype=float)
    legacy = E * NE * (TE + ti)
    assert np.all(total[:-1] < legacy[:-1]), "diluted ions carry less pressure than n_i = n_e"
    assert np.allclose(total, np.asarray(ods[f"{base}.electrons.pressure"], dtype=float)
                       + np.asarray(ods[f"{base}.pressure_ion_total"], dtype=float), rtol=1e-9)
    fields = ti_record_fields(ods[f"{base}.ion.0.density_fit.parameters"])
    assert fields["pressure_closure"] == "infer_kinetic_closure" and fields["dilution"] == "species"
    assert manifest["pressure"] == {
        "closure": "infer_kinetic_closure", "dilution": "species",
        "ion": 1, "electron_only": 1, "unchanged": 0,
    }


def test_an_electron_only_slice_gets_the_electron_pressure_and_no_assumed_total(tmp_path, tables):
    ods, _ = _build(tmp_path, tables)
    base = "core_profiles.profiles_1d.1"
    assert np.allclose(np.asarray(ods[f"{base}.electrons.pressure"], dtype=float), E * NE * TE, rtol=1e-6)
    assert path_value(ods, f"{base}.pressure_thermal", None) is None
    assert path_value(ods, f"{base}.pressure_ion_total", None) is None


def test_an_ion_temperature_the_product_assumed_keeps_the_pressure_it_was_written_with(tmp_path, tables):
    """Only a measurement is turned into an ion pressure here; an assumed Ti = Te is not."""
    product = _core_profiles(tmp_path, ti_record="ti_te_ratio=0.8; status=assumed; sigma=0.5")
    ods, manifest = _build(tmp_path, tables, core_profiles_product=product)
    base = "core_profiles.profiles_1d.0"
    assert np.allclose(np.asarray(ods[f"{base}.pressure_thermal"], dtype=float), E * NE * (TE + 0.8 * TE))
    assert path_value(ods, f"{base}.pressure_ion_total", None) is None
    assert manifest["pressure"]["unchanged"] == 1


# --- the kinetic EFIT stage reading the composed product --------------------------------------


def _composed(tmp_path, tables):
    ods, _ = _build(tmp_path, tables)
    return ods


def test_the_ion_density_ratio_is_the_stored_ions_per_electron_at_each_psi_n(tmp_path, tables):
    ods = _composed(tmp_path, tables)
    fn, record = vu._ion_density_ratio_function(ods, 300.0)
    assert record["applied"] is True and record["mode"] == "species"
    expected = _charged_ion_density(ods, "core_profiles.profiles_1d.0") / NE
    psi_n = RHO**2                                   # rho_pol_norm is RHO in this fixture
    inside = slice(0, -1)                            # the edge point carries neutrals
    assert np.allclose(fn(psi_n)[inside], expected[inside], rtol=2e-3)
    assert record["n_i_over_n_e_axis"] == pytest.approx(expected[0], rel=2e-3)
    assert 0.0 < record["n_i_over_n_e_axis"] < 1.0, "dilution lowers the ion density per electron"
    assert np.all(fn(np.array([1.3, 2.0])) == fn(np.array([1.0, 1.0]))), "the scrape-off layer is held, not extrapolated"


def test_a_product_with_no_composition_falls_back_and_says_why(tmp_path):
    product = _core_profiles(tmp_path)
    from vaft.database._local import load_ods
    from pathlib import Path

    measured, _ = load_ods(Path(product))
    fn, record = vu._ion_density_ratio_function(measured, 300.0)
    assert fn is None and record["applied"] is False
    assert "no composition" in record["reason"]


def test_no_slice_at_the_reconstruction_time_is_named(tmp_path, tables):
    fn, record = vu._ion_density_ratio_function(_composed(tmp_path, tables), 350.0)
    assert fn is None and "no core_profiles slice at the reconstruction time" in record["reason"]


def _kinetic_stage(tmp_path, tables, monkeypatch, *, dilution):
    """build_kinetic_efit_ods over the composed product, with a fake EFIT that records its config."""
    import json
    from types import SimpleNamespace

    import vaft.code.efit.kinetic as kinetic
    import vaft.data.eqdsk as eqdsk
    import vaft.process.equilibrium as process_equilibrium

    composed = _composed(tmp_path, tables)
    composed["thomson_scattering.time"] = np.array([0.300])
    constraints = ODS(consistency_check=False)
    constraints["equilibrium.time"] = np.array([0.300])
    solution = ODS(consistency_check=False)
    solution["equilibrium.time"] = np.array([0.300])
    solution["equilibrium.time_slice.0.global_quantities.ip"] = 1.0
    solution["equilibrium.code.parameters"] = json.dumps({"efit_collection": {"slice_statuses": [
        {"time": 0.300, "overall_status": "usable"}]}})
    paths = {n: tmp_path / f"{n}.json" for n in ("core_profiles", "constraints", "efit")}
    for path in paths.values():
        path.write_text("{}")
    products = {str(paths["core_profiles"]): composed, str(paths["constraints"]): constraints,
                str(paths["efit"]): solution}
    monkeypatch.setattr(vu, "load_ods", lambda path: (products[str(path)], {}))
    monkeypatch.setattr(process_equilibrium, "as_equilibrium", lambda sol, time_index: time_index)
    monkeypatch.setattr(eqdsk, "from_equilibrium", lambda index: index)
    monkeypatch.setattr(eqdsk, "to_omas", lambda geq, ods: ods)
    monkeypatch.setattr(vu, "read_geqdsk_file", lambda path: path)
    seen = {}

    def fake_chain(ods, geq, time_ms, efit_config):
        seen["fn"] = efit_config.ion_density_ratio
        return {"kinetic_efit": SimpleNamespace(status="ok", converged=True, scale=1.0, chi2=1.0,
                                                reason="", gfile="g")}

    monkeypatch.setattr(kinetic, "run_kinetic_chain", fake_chain)
    out, manifest = vu.build_kinetic_efit_ods(
        shot=SHOT, stage="kinetic_efit", core_profiles_product=paths["core_profiles"],
        constraints_product=paths["constraints"], efit_product=paths["efit"],
        workdir=tmp_path / "work", dilution=dilution,
    )
    return out, manifest, seen


def test_the_kinetic_stage_hands_the_ratio_to_the_efit_and_records_it(tmp_path, tables, monkeypatch):
    import json

    out, manifest, seen = _kinetic_stage(tmp_path, tables, monkeypatch, dilution="species")
    assert manifest["status"] == "success", manifest.get("error")
    assert callable(seen["fn"]) and 0.0 < float(seen["fn"](np.array([0.0]))[0]) < 1.0
    assert manifest["configuration"]["dilution"] == "species"
    assert manifest["dilution"]["applied"] is True
    parameters = json.loads(str(out["equilibrium.code.parameters"]))
    assert parameters["kinetic_efit"]["dilution"]["applied"] is True


def test_dilution_none_is_the_legacy_pressure_and_is_recorded_as_such(tmp_path, tables, monkeypatch):
    import json

    out, manifest, seen = _kinetic_stage(tmp_path, tables, monkeypatch, dilution="none")
    assert seen["fn"] is None
    assert manifest["dilution"] == {"mode": "none", "applied": False}
    assert json.loads(str(out["equilibrium.code.parameters"]))["kinetic_efit"]["dilution"]["mode"] == "none"


def test_an_unknown_dilution_is_refused(tmp_path, tables):
    with pytest.raises(ValueError, match="dilution"):
        vu.build_kinetic_efit_ods(shot=SHOT, stage="kinetic_efit", core_profiles_product="a",
                                  constraints_product="b", efit_product="c", dilution="partial")


# --- the inferred T_i stage (item 4) ----------------------------------------------------------

PSI_N = RHO ** (1.0 / 0.6)      # so that rho_tor_norm = psi_N**0.6, the equilibrium's own coordinate
TI_OVER_TE = 1.0                # what the synthetic plasma really has


def _magnetic_efit(tmp_path, pressure, *, times=(0.300, 0.301), under="efit/magnetic"):
    """A magnetic EFIT product whose pressure is the given function of the profile points."""
    ods = ODS(consistency_check=False)
    ods["equilibrium.time"] = np.asarray(times, dtype=float)
    for index in range(len(times)):
        base = f"equilibrium.time_slice.{index}"
        ods[f"{base}.global_quantities.psi_axis"] = 0.0
        ods[f"{base}.global_quantities.psi_boundary"] = 1.0
        ods[f"{base}.profiles_1d.psi"] = PSI_N
        ods[f"{base}.profiles_1d.phi"] = PSI_N ** 1.2
        ods[f"{base}.profiles_1d.pressure"] = pressure
    directory = tmp_path / "omas" / pathlib.Path(under) / str(SHOT) / "output"
    directory.mkdir(parents=True)
    path = directory / "efit.json"
    ods.save(str(path))
    return path


def _composed_electron_only(tmp_path, tables, *, thomson=True):
    """A composed product whose slices have no measured T_i, and the closure's n_i,tot/n_e on it."""
    from vaft.process.kinetic_closure import infer_kinetic_closure

    product = _core_profiles(tmp_path, kinetic=(), rho_pol=np.sqrt(PSI_N), thomson=thomson)
    efit = _efit(tmp_path)
    composed, _ = vu.build_composition_ods(
        shot=SHOT, core_profiles_product=product, efit_product=efit, ionization="coronal", tables=tables,
        assume_ti_eq_te=False)
    out = tmp_path / "composition.json"
    composed.save(str(out))
    closure = infer_kinetic_closure(composed, time=0.300, dilution="species", ti_te_ratio=1.0)
    fraction = np.sum(list(closure.ion_densities.values()), axis=0) / NE
    return out, fraction


def _ti_stage(tmp_path, tables, *, p_scale=1.0, **kwargs):
    composed, fraction = _composed_electron_only(tmp_path, tables, thomson=kwargs.pop("thomson", True))
    truth = E * NE * (TE + fraction * TI_OVER_TE * TE) * p_scale
    efit = kwargs.pop("efit", None) or _magnetic_efit(tmp_path, truth)
    ods, manifest = vu.build_ti_inferred_ods(
        shot=SHOT, core_profiles_product=composed, efit_product=efit, **kwargs)
    return ods, manifest, truth, fraction


def test_the_inferred_ti_recovers_the_temperature_the_pressure_was_made_from(tmp_path, tables):
    ods, manifest, truth, _ = _ti_stage(tmp_path, tables)
    assert manifest["status"] == "success", manifest.get("error")
    assert manifest["slices"] == {"inferred": 2, "measured": 0, "unavailable": 0, "total": 2}
    base = "core_profiles.profiles_1d.0"
    t_i = np.asarray(ods[f"{base}.ion.0.temperature"], dtype=float)
    valid = np.isfinite(t_i)
    assert valid.sum() >= 5, "most of the profile passes the significance test"
    assert np.allclose(t_i[valid], TI_OVER_TE * TE[valid], rtol=2e-3)
    assert np.allclose(np.asarray(ods[f"{base}.t_i_average"], dtype=float), t_i, equal_nan=True)
    for k in range(3):
        assert np.allclose(np.asarray(ods[f"{base}.ion.{k}.temperature"], dtype=float), t_i, equal_nan=True)


def test_the_inferred_ti_is_labelled_inferred_with_its_lineage_and_validity(tmp_path, tables):
    ods, _, _, _ = _ti_stage(tmp_path, tables)
    base = "core_profiles.profiles_1d.0"
    record = ods[f"{base}.ion.0.temperature_fit.parameters"]
    assert classify_ti_record(record) == "inferred", record
    fields = ti_record_fields(record)
    assert fields["method"] == "equilibrium_pressure_partition"
    assert fields["equilibrium_lineage"] == "magnetics" and fields["equilibrium_occurrence"] == "0"
    assert float(fields["equilibrium_time_s"]) == pytest.approx(0.300)
    assert fields["sigma_p_eq_basis"] == "floor"
    valid, total = fields["valid_points"].split("/")
    assert 0 < int(valid) <= int(total)
    assert "not measured" in record


def test_the_pressures_follow_the_inferred_ti_as_a_closure_identity(tmp_path, tables):
    ods, _, truth, _ = _ti_stage(tmp_path, tables)
    base = "core_profiles.profiles_1d.0"
    t_i = np.asarray(ods[f"{base}.ion.0.temperature"], dtype=float)
    valid = np.isfinite(t_i)
    total = np.asarray(ods[f"{base}.pressure_thermal"], dtype=float)
    assert np.allclose(total[valid], truth[valid], rtol=2e-3), "p_thermal is p_eq where T_i is inferred"
    assert np.all(np.isnan(total[~valid])), "invalid points stay empty"
    assert np.allclose(total[valid], (np.asarray(ods[f"{base}.electrons.pressure"], dtype=float)
                                      + np.asarray(ods[f"{base}.pressure_ion_total"], dtype=float))[valid])
    assert ti_record_fields(ods[f"{base}.ion.0.density_fit.parameters"])["pressure_closure"] == "infer_kinetic_closure"


def test_a_slice_with_a_measured_ion_temperature_is_left_alone(tmp_path, tables):
    from vaft.database._local import load_ods

    product = _core_profiles(tmp_path, kinetic=(0,), rho_pol=np.sqrt(PSI_N), thomson=True)
    composed, _ = vu.build_composition_ods(
        shot=SHOT, core_profiles_product=product, efit_product=_efit(tmp_path),
        ionization="coronal", tables=tables, assume_ti_eq_te=False)
    path = tmp_path / "composition.json"
    composed.save(str(path))
    efit = _magnetic_efit(tmp_path, E * NE * TE * 2.0)
    ods, manifest = vu.build_ti_inferred_ods(shot=SHOT, core_profiles_product=path, efit_product=efit)
    assert manifest["slices"]["measured"] == 1 and manifest["slices"]["inferred"] == 1
    before, _ = load_ods(path)
    assert np.allclose(np.asarray(ods["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float),
                       np.asarray(before["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float))
    assert (ods["core_profiles.profiles_1d.0.ion.0.temperature_fit.parameters"]
            == before["core_profiles.profiles_1d.0.ion.0.temperature_fit.parameters"])


@pytest.mark.parametrize("under", ["electron_efit", "efit/kinetic", "somewhere/else"])
def test_a_pressure_that_is_not_the_magnetic_lineage_is_refused_by_where_it_lives(tmp_path, tables, under):
    composed, fraction = _composed_electron_only(tmp_path, tables)
    efit = _magnetic_efit(tmp_path, E * NE * (TE + fraction * TE), under=under)
    ods, manifest = vu.build_ti_inferred_ods(shot=SHOT, core_profiles_product=composed, efit_product=efit)
    assert manifest["status"] == "unavailable"
    assert "lineage is not established" in manifest["error"]
    assert path_value(ods, "core_profiles.profiles_1d.0.ion.0.temperature", None) is None


def test_a_slice_with_no_composition_is_skipped_not_inferred_with_one_ion_per_electron(tmp_path, tables):
    product = _core_profiles(tmp_path, kinetic=(), rho_pol=np.sqrt(PSI_N), thomson=True)   # not composed
    efit = _magnetic_efit(tmp_path, E * NE * TE * 2.0)
    ods, manifest = vu.build_ti_inferred_ods(shot=SHOT, core_profiles_product=product, efit_product=efit)
    assert manifest["status"] == "unavailable"
    assert "no composition" in manifest["error"]
    assert path_value(ods, "core_profiles.profiles_1d.0.ion.0.temperature", None) is None


def test_no_equilibrium_near_the_slice_is_named(tmp_path, tables):
    composed, fraction = _composed_electron_only(tmp_path, tables)
    efit = _magnetic_efit(tmp_path, E * NE * (TE + fraction * TE), times=(0.350, 0.351))
    _, manifest = vu.build_ti_inferred_ods(shot=SHOT, core_profiles_product=composed, efit_product=efit)
    assert manifest["status"] == "unavailable" and "no magnetic equilibrium within" in manifest["error"]


def test_a_pressure_below_the_electron_pressure_gives_no_ion_temperature(tmp_path, tables):
    """p_i <= 0 everywhere: every point is flagged, never filled, and the slice says so."""
    ods, manifest, _, _ = _ti_stage(tmp_path, tables, p_scale=0.4)
    assert manifest["status"] == "unavailable"
    assert "no point has a positive, significant p_i" in manifest["error"]
    assert path_value(ods, "core_profiles.profiles_1d.0.ion.0.temperature", None) is None
    assert path_value(ods, "core_profiles.profiles_1d.0.t_i_average", None) is None


def test_without_thomson_errors_the_inference_runs_and_the_missing_sigma_is_counted(tmp_path, tables):
    _, manifest, _, _ = _ti_stage(tmp_path, tables, thomson=False)
    assert manifest["status"] == "success"
    assert manifest["inference"]["flags"]["sigma_unavailable"] > 0


def test_a_missing_composed_product_is_a_caller_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        vu.build_ti_inferred_ods(shot=SHOT, core_profiles_product=tmp_path / "no.json", efit_product=tmp_path / "e.json")


def test_the_inferred_ti_replaces_the_one_the_composition_assumed(tmp_path, tables):
    product = _core_profiles(tmp_path, kinetic=(0,), rho_pol=np.sqrt(PSI_N), thomson=True)
    composed, _ = vu.build_composition_ods(
        shot=SHOT, core_profiles_product=product, efit_product=_efit(tmp_path),
        ionization="coronal", tables=tables)
    path = tmp_path / "composition.json"
    composed.save(str(path))
    assert ti_record_fields(composed["core_profiles.profiles_1d.1.ion.0.temperature_fit.parameters"])["status"] == "assumed"
    efit = _magnetic_efit(tmp_path, E * NE * TE * 2.0)
    ods, manifest = vu.build_ti_inferred_ods(shot=SHOT, core_profiles_product=path, efit_product=efit)
    record = ti_record_fields(ods["core_profiles.profiles_1d.1.ion.0.temperature_fit.parameters"])
    assert record["origin"] == "inferred"
    assert manifest["slices"]["measured"] == 1



def test_an_assumed_or_inferred_ion_temperature_does_not_make_a_slice_kinetic(tmp_path, tables):
    """With the default T_i = T_e a Thomson-only shot stays in the electron lineage (cold review #1842)."""
    ods, _ = _build(tmp_path, tables, assume_ti_eq_te=True)
    assert ti_record_fields(ods["core_profiles.profiles_1d.1.ion.0.temperature_fit.parameters"])["status"] == "assumed"
    assert vu._has_kinetic_slice(ods), "slice 0 carries the fixture's measured T_i"
    del ods["core_profiles.profiles_1d.0.ion.0.temperature"]
    assert not vu._has_kinetic_slice(ods), "an assumed T_i alone is not kinetic"
    ods["core_profiles.profiles_1d.1.ion.0.temperature_fit.parameters"] = "origin=inferred; method=equilibrium_pressure_partition"
    assert not vu._has_kinetic_slice(ods), "an inferred T_i alone is not kinetic"
