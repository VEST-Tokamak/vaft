"""The resistive Z_eff stage of pipeline 2: the Romero-balance estimate in the kinetic state.

Built on the packaged 39915 equilibrium (nine slices, weber flux) with synthetic
electron profiles at three of its times, so the real Romero balance, the real
Grad-Shafranov geometry and the real conductivity model run end to end, offline.
"""

from __future__ import annotations

import logging
import pathlib

import numpy as np
import pytest
from omas import ODS

from vaft.machine_mapping.core_profiles import ti_record_fields
from vaft.ods_access import path_value
from vaft.omas import vest_upstream as vu

pytest.importorskip("skimage")

SHOT = 39915
RHO = np.linspace(0.0, 1.0, 41)
SLICES = (1, 2, 3)


@pytest.fixture(scope="module")
def sample():
    from vaft.omas.sample import sample_ods

    logging.disable(logging.WARNING)
    try:
        return sample_ods(SHOT)
    except FileNotFoundError:
        pytest.skip("the 39915 sample is repository-only")
    finally:
        logging.disable(logging.NOTSET)


def _inputs(tmp_path, sample, *, composition=True, under="efit/magnetic", t_scale=1.0, phi=True):
    times = np.asarray(sample["equilibrium.time"], dtype=float)
    ods = ODS(consistency_check=False)
    ods["core_profiles.time"] = times[list(SLICES)]
    for j, i in enumerate(SLICES):
        base = f"core_profiles.profiles_1d.{j}"
        ods[f"{base}.time"] = times[i]
        ods[f"{base}.grid.rho_tor_norm"] = RHO
        ods[f"{base}.electrons.temperature"] = t_scale * (10.0 + 150.0 * (1.0 - RHO**2) ** 1.5)
        ods[f"{base}.electrons.density"] = 0.4e19 + 1.4e19 * (1.0 - RHO**2)
        if composition:
            ods[f"{base}.zeff"] = 1.0 + 1.2 * (1.0 - RHO**2)
            ods[f"{base}.zeff_fit.parameters"] = "origin=derived; method=openadas_coronal; target_zeff=2.0"
    product = tmp_path / "ti_inferred.json"
    ods.save(str(product))
    equilibrium = ODS(consistency_check=False)
    equilibrium["equilibrium"] = sample["equilibrium"]
    if phi:
        # The packaged equilibrium stores rho_tor_norm as the sqrt(psi_N) proxy, which the
        # cross-check refuses to use as a toroidal-flux coordinate. A toroidal flux that is
        # the stored coordinate squared gives it a real one, equal to what the geometry uses.
        for i in range(len(equilibrium["equilibrium.time_slice"])):
            base = f"equilibrium.time_slice.{i}.profiles_1d"
            equilibrium[f"{base}.phi"] = 0.01 * np.asarray(equilibrium[f"{base}.rho_tor_norm"], dtype=float) ** 2
    directory = tmp_path / "omas" / pathlib.Path(under) / str(SHOT) / "output"
    directory.mkdir(parents=True)
    efit = directory / "efit.json"
    equilibrium.save(str(efit))
    return product, efit


def _run(tmp_path, sample, **options):
    product, efit = _inputs(
        tmp_path, sample, **{k: options.pop(k) for k in ("composition", "under", "t_scale", "phi") if k in options})
    return vu.build_resistive_zeff_ods(shot=SHOT, core_profiles_product=product, efit_product=efit, **options)


def test_the_window_estimate_is_stored_at_every_usable_slice_and_nowhere_else(tmp_path, sample):
    ods, manifest = _run(tmp_path, sample)
    assert manifest["status"] == "success", manifest.get("error")
    stored = np.asarray(ods["core_profiles.global_quantities.z_eff_resistive"], dtype=float)
    assert stored.shape == (3,)
    assert np.all(np.isfinite(stored)) and np.ptp(stored) == 0.0, "one scalar for the window, not a value per slice"
    assert 1.0 < stored[0] < 8.0
    assert manifest["inference"]["status"] == "ok" and manifest["inference"]["zeff"] == pytest.approx(stored[0])
    assert manifest["stored"] == {"slices": 3, "total": 3}
    assert manifest["window"]["t_start_s"] < manifest["window"]["t_end_s"]


def test_the_provenance_is_prefixed_and_leaves_the_composition_record_alone(tmp_path, sample):
    ods, _ = _run(tmp_path, sample)
    record = ods["core_profiles.profiles_1d.0.zeff_fit.parameters"]
    fields = ti_record_fields(record)
    assert fields["origin"] == "derived" and fields["method"] == "openadas_coronal", "the composition record is untouched"
    assert fields["target_zeff"] == "2.0"
    assert fields["resistive_origin"] == "inferred" and fields["resistive_model"] == "romero_transformer"
    assert fields["resistive_scope"] == "window" and fields["resistive_status"] == "ok"
    assert fields["resistive_equilibrium_lineage"] == "magnetics" and fields["resistive_equilibrium_occurrence"] == "0"
    assert fields["resistive_conductivity"] == "redl" and fields["resistive_i_ni_a"] == "0.0"
    assert fields["resistive_smoothing"] == "none" and fields["resistive_n_states"] == "3"
    assert "resistive_sigma" in fields


def test_the_composition_zeff_profile_is_not_replaced(tmp_path, sample):
    ods, _ = _run(tmp_path, sample)
    assert np.allclose(np.asarray(ods["core_profiles.profiles_1d.0.zeff"], dtype=float), 1.0 + 1.2 * (1.0 - RHO**2))


def test_the_per_slice_solutions_are_a_labelled_diagnostic_in_the_manifest_only(tmp_path, sample):
    ods, manifest = _run(tmp_path, sample)
    diagnostic = manifest["per_slice_diagnostic"]
    assert "diagnostic" in diagnostic["note"] and len(diagnostic["zeff"]) == 3
    assert set(np.round(np.asarray(ods["core_profiles.global_quantities.z_eff_resistive"], float), 9)) == {
        round(manifest["inference"]["zeff"], 9)}, "the IDS holds the window estimate, not these"


def test_the_composition_is_cross_checked_through_the_same_estimator(tmp_path, sample):
    _, manifest = _run(tmp_path, sample)
    check = manifest["crosscheck"]
    assert check["available"] and check["states"] == 3
    assert check["resistive_zeff"] == pytest.approx(manifest["inference"]["zeff"])
    assert check["difference"] == pytest.approx(check["zeff_equivalent"] - check["resistive_zeff"])
    assert check["relative_difference"] == pytest.approx(check["difference"] / check["resistive_zeff"])


def test_without_a_composition_the_crosscheck_says_so_and_the_estimate_still_stands(tmp_path, sample):
    ods, manifest = _run(tmp_path, sample, composition=False)
    assert manifest["status"] == "success"
    assert manifest["crosscheck"]["available"] is False and "no composition" in manifest["crosscheck"]["reason"]
    assert np.all(np.isfinite(np.asarray(ods["core_profiles.global_quantities.z_eff_resistive"], float)))


def test_an_estimate_on_a_bound_is_recorded_and_not_stored(tmp_path, sample):
    """A plasma too cold for its resistance to be explained from above: Z_eff pins at the lower bound."""
    ods, manifest = _run(tmp_path, sample, t_scale=0.5)
    assert manifest["status"] == "unavailable"
    assert manifest["inference"]["status"] == "bound_hit"
    assert "not stored" in manifest["error"]
    assert path_value(ods, "core_profiles.global_quantities.z_eff_resistive", None) is None
    assert "resistive_origin" not in ti_record_fields(ods["core_profiles.profiles_1d.0.zeff_fit.parameters"])


@pytest.mark.parametrize("under", ["electron_efit", "efit/kinetic", "somewhere/else"])
def test_a_geometry_that_is_not_the_magnetic_lineage_is_refused_by_where_it_lives(tmp_path, sample, under):
    ods, manifest = _run(tmp_path, sample, under=under)
    assert manifest["status"] == "unavailable" and "lineage is not established" in manifest["error"]
    assert path_value(ods, "core_profiles.global_quantities.z_eff_resistive", None) is None


def test_an_unknown_conductivity_model_is_refused(tmp_path, sample):
    with pytest.raises(ValueError, match="model"):
        _run(tmp_path, sample, model="neo")


def test_a_missing_core_profiles_product_is_a_caller_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        vu.build_resistive_zeff_ods(shot=SHOT, core_profiles_product=tmp_path / "no.json", efit_product=tmp_path / "e.json")


def test_a_window_with_no_kinetic_slice_is_named(tmp_path, sample):
    product, efit = _inputs(tmp_path, sample)
    times = np.asarray(sample["equilibrium.time"], dtype=float)
    _, manifest = vu.build_resistive_zeff_ods(
        shot=SHOT, core_profiles_product=product, efit_product=efit,
        time_range=(float(times[7]), float(times[8])),
    )
    assert manifest["status"] == "unavailable" and manifest["error"]


def test_a_proxy_rho_tor_norm_is_not_used_to_place_the_composition(tmp_path, sample):
    """The estimate stands; only the cross-check, which needs a real toroidal-flux coordinate, is withheld."""
    ods, manifest = _run(tmp_path, sample, phi=False)
    assert manifest["status"] == "success"
    assert manifest["crosscheck"]["available"] is False
    assert "sqrt(psi_N) proxy" in manifest["crosscheck"]["reason"]
    assert np.all(np.isfinite(np.asarray(ods["core_profiles.global_quantities.z_eff_resistive"], float)))
