"""The composition stage of pipeline 2 (#1842): C/O composition, n_i and Z_eff(rho) on measured core profiles.

Synthetic ADF11 tables and synthetic products, no network and no repository data.
"""

from __future__ import annotations

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


def _core_profiles(tmp_path, *, kinetic=(0,), times=(0.300, 0.301), zeff_record=None, ti_record=True):
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = np.asarray(times, dtype=float)
    for index in range(len(times)):
        base = f"core_profiles.profiles_1d.{index}"
        ods[f"{base}.grid.rho_tor_norm"] = RHO
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
    # Built only when absent: an eager setdefault would rewrite a product the
    # caller already wrote to the same path.
    if "core_profiles_product" not in kwargs:
        kwargs["core_profiles_product"] = _core_profiles(tmp_path)
    if "efit_product" not in kwargs:
        kwargs["efit_product"] = _efit(tmp_path)
    return vu.build_composition_ods(shot=SHOT, tables=tables, **kwargs)


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
