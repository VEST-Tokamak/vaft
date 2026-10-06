"""The impurity-composition resolver (#1565 Stage B): precedence, the VEST preset, provenance."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.process.impurity import (
    ImpurityComposition,
    ImpuritySpecies,
    composition_from_fractions,
    composition_record_origin,
    composition_record_text,
    resolve_impurity_composition,
)

RHO = np.linspace(0.0, 1.0, 5)
NE = 1e19 * (1.0 - 0.8 * RHO**2)


def _ods(*, ions=(), zeff=None, zeff_record=None, times=(0.3,)):
    """core_profiles with an electron density and optional ions / Z_eff on every slice."""
    ods = ODS()
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.asarray(times, dtype=float)
    for k, t in enumerate(times):
        base = f"core_profiles.profiles_1d.{k}"
        ods[f"{base}.time"] = t
        ods[f"{base}.grid.rho_tor_norm"] = RHO
        ods[f"{base}.electrons.density_thermal"] = NE
        for j, (label, z_n, z_ion, a, fraction, record) in enumerate(ions):
            ion = f"{base}.ion.{j}"
            ods[f"{ion}.label"] = label
            ods[f"{ion}.element.0.z_n"] = z_n
            ods[f"{ion}.element.0.a"] = a
            ods[f"{ion}.z_ion"] = z_ion
            ods[f"{ion}.density_thermal"] = fraction * NE
            if record is not None:
                ods[f"{ion}.density_fit.parameters"] = record
        if zeff is not None:
            ods[f"{base}.zeff"] = np.broadcast_to(zeff, RHO.shape).astype(float)
            if zeff_record is not None:
                ods[f"{base}.zeff_fit.parameters"] = zeff_record
    return ods


def _lane_k_like(record=None):
    """H+ and C6+ at Z_eff = 2, as an assumed product stores them (no density label)."""
    return (("H+", 1.0, 1.0, 1.008, 1 - 6 / 30, None), ("C6+", 6.0, 6.0, 12.011, 1 / 30, record))


# --- the VEST preset --------------------------------------------------------------


def test_the_vest_preset_resolves_to_the_issue_reference_case():
    r = resolve_impurity_composition(machine_preset="vest", shot=39915)
    assert r.kind == "assumed"
    assert "impurity_model" in r.source and "39915" in r.source
    np.testing.assert_allclose(r.impurity_fractions, [1 / 86, 1 / 86], rtol=1e-12)
    assert r.fraction("C") == pytest.approx(1 / 86)
    assert r.zeff == pytest.approx(2.0)
    assert (r.S1, r.S2) == (pytest.approx(7.0), pytest.approx(50.0))
    assert r.effective_charge == pytest.approx(50 / 7)
    assert r.effective_fraction == pytest.approx(49 / 2150)
    assert r.main_ion_fraction == pytest.approx(36 / 43)
    assert r.dilution_fraction == pytest.approx(7 / 43)
    assert r.provenance["composition_status"] == "assumed"


def test_nothing_is_defaulted():
    with pytest.raises(ValueError, match="nothing is defaulted"):
        resolve_impurity_composition()


# --- precedence --------------------------------------------------------------------


def test_explicit_outranks_the_machine_preset():
    c = composition_from_fractions(["C", "O", "N"], [3, 4, 3], [6, 8, 7], target_zeff=1.7)
    r = resolve_impurity_composition(composition=c, machine_preset="vest")
    assert r.kind == "explicit" and r.zeff == pytest.approx(1.7)
    np.testing.assert_allclose(r.weights, [0.3, 0.4, 0.3])
    assert r.provenance["input_record"]["fractions"] == [3.0, 4.0, 3.0]   # original input kept
    assert {c["source"]: c["outcome"] for c in r.candidates} == {
        "explicit": "chosen", "machine_preset": "outranked"
    }


def test_derived_outranks_the_preset_and_yields_to_explicit():
    derived = composition_from_fractions(["C"], [1], [4], target_zeff=1.5, status="derived", source="model")
    explicit = composition_from_fractions(["O"], [1], [8], target_zeff=2.0)
    assert resolve_impurity_composition(derived=derived, machine_preset="vest").kind == "derived"
    assert resolve_impurity_composition(composition=explicit, derived=derived).kind == "explicit"


def test_a_stored_composition_labelled_measured_outranks_everything():
    record = composition_record_text("measured", "charge_exchange")
    ods = _ods(ions=_lane_k_like(record))
    explicit = composition_from_fractions(["O"], [1], [8], target_zeff=2.0)
    r = resolve_impurity_composition(ods, composition=explicit, machine_preset="vest")
    assert r.kind == "measured"
    assert [s.label for s in r.species] == ["C6+"]
    np.testing.assert_allclose(r.impurity_fractions[..., 0], 1 / 30)
    np.testing.assert_allclose(r.zeff, 2.0)
    assert r.time == pytest.approx(0.3)


def test_an_unlabelled_stored_composition_yields_to_the_caller_but_beats_the_preset():
    """Lane K products carry an assumed H+/C6+ list with no density label."""
    ods = _ods(ions=_lane_k_like())
    explicit = composition_from_fractions(["O"], [1], [8], target_zeff=2.0)
    assert resolve_impurity_composition(ods, composition=explicit).kind == "explicit"
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.kind == "assumed" and "unlabelled" in r.source
    assert {c["source"]: c["outcome"] for c in r.candidates}["machine_preset"] == "outranked"


def test_a_zero_edge_density_leaves_that_point_undefined_not_the_slice():
    """Lane K's fitted profiles reach n_e = 0 exactly at rho = 1 (found on Tier A, 39906)."""
    ods = _ods(ions=_lane_k_like())
    for path in ("electrons.density_thermal", "ion.0.density_thermal", "ion.1.density_thermal"):
        values = np.array(ods[f"core_profiles.profiles_1d.0.{path}"], dtype=float)
        values[-1] = 0.0
        ods[f"core_profiles.profiles_1d.0.{path}"] = values
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert np.isnan(r.zeff[-1]) and np.isnan(r.dilution_fraction[-1])
    np.testing.assert_allclose(r.zeff[:-1], 2.0)
    np.testing.assert_allclose(r.effective_charge[:-1], 6.0)


def test_a_stored_derived_composition_is_derived():
    ods = _ods(ions=_lane_k_like(composition_record_text("derived", "openadas_coronal")))
    assert resolve_impurity_composition(ods, machine_preset="vest").kind == "derived"


# --- measured Z_eff, and the one that is not ---------------------------------------


def test_a_measured_zeff_profile_closes_the_composition():
    profile = 1.5 + 0.5 * RHO
    ods = _ods(zeff=profile, zeff_record=composition_record_text("measured", "bremsstrahlung"))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.kind == "derived" and "measured" in r.zeff_source
    np.testing.assert_allclose(r.zeff, profile)
    assert r.impurity_fractions.shape == (5, 2)
    assert r.provenance["target_zeff_replaced"] == 2.0
    np.testing.assert_allclose(r.rho, RHO)


@pytest.mark.parametrize("record", [None, composition_record_text("derived", "impurity_model_preset")])
def test_an_unlabelled_or_derived_zeff_is_not_a_target(record):
    """Stage C writes a derived zeff; reading it back as a target would be circular."""
    ods = _ods(zeff=3.0, zeff_record=record)
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.zeff == pytest.approx(2.0) and r.kind == "assumed"


def test_resistive_zeff_is_carried_never_used():
    r = resolve_impurity_composition(machine_preset="vest", resistive_zeff={"zeff": 1.09, "model": "redl"})
    assert r.zeff == pytest.approx(2.0)
    assert r.resistive_zeff["value"] == pytest.approx(1.09)
    assert r.resistive_zeff["kind"] == "resistive_inferred"
    assert r.as_record()["resistive_zeff"]["model"] == "redl"


# --- time matching and the ODS rule ---------------------------------------------------


def test_slices_are_matched_by_time():
    ods = _ods(ions=_lane_k_like(composition_record_text("measured", "x")), times=(0.30, 0.31))
    ods["core_profiles.profiles_1d.1.ion.1.density_thermal"] = NE / 60   # slice 1 differs
    early = resolve_impurity_composition(ods, time=0.2999, machine_preset="vest")
    late = resolve_impurity_composition(ods, time=0.3101, machine_preset="vest")
    assert (early.time, late.time) == (pytest.approx(0.30), pytest.approx(0.31))
    np.testing.assert_allclose(early.impurity_fractions[..., 0], 1 / 30)
    np.testing.assert_allclose(late.impurity_fractions[..., 0], 1 / 60)
    with pytest.raises(ValueError, match="pass the time"):
        resolve_impurity_composition(ods, machine_preset="vest")
    far = resolve_impurity_composition(ods, time=0.5, machine_preset="vest")
    assert far.kind == "assumed" and far.candidates[0]["outcome"] == "skipped"


def test_resolving_creates_no_path():
    ods = _ods(ions=_lane_k_like())
    before_slice = sorted(ods["core_profiles.profiles_1d.0"].keys())
    before_ion = sorted(ods["core_profiles.profiles_1d.0.ion.1"].keys())
    resolve_impurity_composition(ods, machine_preset="vest")
    assert sorted(ods["core_profiles.profiles_1d.0"].keys()) == before_slice
    assert sorted(ods["core_profiles.profiles_1d.0.ion.1"].keys()) == before_ion
    assert len(ods["core_profiles.profiles_1d.0.ion"]) == 2
    assert "summary" not in ods and "equilibrium" not in ods


# --- species and records ----------------------------------------------------------------


def test_the_charge_state_is_never_the_atomic_number():
    assert ImpuritySpecies("C", 4).charge_state == 4.0
    assert ImpuritySpecies("C", 4).z_n == 6
    with pytest.raises(ValueError, match="Z_n = 6"):
        ImpuritySpecies("C", 7)
    with pytest.raises(TypeError):
        ImpuritySpecies("C")  # no implicit charge state


def test_a_composition_refuses_duplicates_and_bad_status():
    c6 = ImpuritySpecies("C", 6)
    with pytest.raises(ValueError, match="twice"):
        ImpurityComposition(species=(c6, c6), target_zeff=2.0)
    with pytest.raises(ValueError, match="status"):
        ImpurityComposition(species=(c6,), target_zeff=2.0, status="guessed")


def test_record_grammar_round_trips():
    text = composition_record_text("derived", "impurity_model_preset", target_zeff=2.0)
    assert text == "origin=derived; method=impurity_model_preset; target_zeff=2"
    assert composition_record_origin(text) == "derived"
    assert composition_record_origin("origin: Measured; method: x") == "measured"
    assert composition_record_origin("origin=synthetic") == "unknown"
    assert composition_record_origin("coordinate=rho_tor_norm") is None
    assert composition_record_origin(None) is None
    with pytest.raises(ValueError):
        composition_record_text("guessed", "x")


# --- review findings (cold review, PR 1) -----------------------------------------------


def test_each_slice_s_own_time_decides_over_a_stale_homogeneous_vector():
    ods = _ods(ions=_lane_k_like(composition_record_text("measured", "x")), times=(0.30, 0.31))
    ods["core_profiles.time"] = np.array([0.30])          # stale, one entry for two slices
    ods["core_profiles.profiles_1d.1.ion.1.density_thermal"] = NE / 60
    r = resolve_impurity_composition(ods, time=0.31, machine_preset="vest")
    assert r.kind == "measured" and r.time == pytest.approx(0.31)
    np.testing.assert_allclose(r.impurity_fractions[..., 0], 1 / 60)


def test_a_measured_zeff_outranks_a_stored_assumed_composition():
    profile = 1.5 + 0.5 * RHO
    ods = _ods(ions=_lane_k_like(), zeff=profile, zeff_record=composition_record_text("measured", "x"))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.kind == "derived" and "measured" in r.zeff_source
    np.testing.assert_allclose(r.zeff, profile)
    assert [s.label for s in r.species] == ["C6+"]
    np.testing.assert_allclose(r.impurity_fractions[..., 0], (profile - 1) / 30)


def test_bad_points_in_a_measured_zeff_are_undefined_not_fatal():
    profile = np.array([2.0, np.nan, 0.95, 9.0, 1.5])
    ods = _ods(zeff=profile, zeff_record=composition_record_text("measured", "x"))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.provenance["undefined_points"] == 3
    np.testing.assert_allclose(r.zeff[[0, 4]], [2.0, 1.5])
    assert np.all(np.isnan(r.zeff[1:4])) and np.all(np.isnan(r.impurity_fractions[1:4]))


def test_negative_or_overcharged_stored_points_are_undefined():
    ods = _ods(ions=_lane_k_like())
    carbon = np.full(RHO.shape, 1 / 30)
    carbon[1] = -1e-4          # a fitted density dipping below zero
    carbon[2] = 0.5            # more impurity charge than electrons
    ods["core_profiles.profiles_1d.0.ion.1.density_thermal"] = carbon * NE
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert np.all(np.isnan(r.zeff[[1, 2]])) and np.all(np.isnan(r.dilution_fraction[[1, 2]]))
    np.testing.assert_allclose(r.zeff[[0, 3, 4]], 2.0)


def test_every_stored_ion_is_visited_not_scanned_until_a_gap():
    ods = _ods(ions=_lane_k_like(composition_record_text("measured", "x")))
    del ods["core_profiles.profiles_1d.0.ion.0.label"]    # ion.0: hydrogen with no label, no z_ion
    del ods["core_profiles.profiles_1d.0.ion.0.z_ion"]
    assert resolve_impurity_composition(ods, machine_preset="vest").kind == "measured"


def test_a_stored_slice_without_a_hydrogenic_main_ion_is_skipped_with_its_reason():
    helium = (("He2+", 2.0, 2.0, 4.0026, 0.45, None), ("C6+", 6.0, 6.0, 12.011, 0.01, None))
    r = resolve_impurity_composition(_ods(ions=helium), machine_preset="vest")
    assert r.kind == "assumed" and "impurity_model" in r.source
    assert any("hydrogenic" in c.get("reason", "") for c in r.candidates)


# --- Stage C: writing core_profiles -------------------------------------------------------

SAMPLE_48224 = __import__("pathlib").Path(__file__).resolve().parent.parent / "vaft/data/kineticEfit/ods_48224_300ms.json"


@pytest.fixture(scope="module")
def sample_48224():
    from omas import load_omas_json

    return load_omas_json(str(SAMPLE_48224), consistency_check=False)


def test_populate_writes_the_species_list_and_a_flat_zeff():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 1.0, None),))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    out = populate_impurity_profiles(ods, r)
    base = "core_profiles.profiles_1d.0"
    assert [out[f"{base}.ion.{k}.label"] for k in range(3)] == ["H+", "C6+", "O8+"]
    assert [out[f"{base}.ion.{k}.z_ion"] for k in range(3)] == [1.0, 6.0, 8.0]
    assert [out[f"{base}.ion.{k}.element.0.z_n"] for k in range(3)] == [1.0, 6.0, 8.0]
    n = [np.asarray(out[f"{base}.ion.{k}.density_thermal"]) for k in range(3)]
    np.testing.assert_allclose(n[1] / NE, 1 / 86)
    np.testing.assert_allclose(n[0] + 6 * n[1] + 8 * n[2], NE, rtol=1e-12)       # quasi-neutral
    np.testing.assert_allclose(out[f"{base}.zeff"], 2.0)
    assert composition_record_origin(out[f"{base}.zeff_fit.parameters"]) == "assumed"
    assert composition_record_origin(out[f"{base}.ion.1.density_fit.parameters"]) == "assumed"
    # the source is untouched
    assert len(ods[f"{base}.ion"]) == 1 and "zeff" not in ods[base]


def test_a_written_composition_reads_back_as_assumed_and_never_as_a_target():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 1.0, None),))
    out = populate_impurity_profiles(ods, resolve_impurity_composition(ods, machine_preset="vest"))
    again = resolve_impurity_composition(out, machine_preset="vest")
    assert again.kind == "assumed" and "origin=assumed" in again.source
    np.testing.assert_allclose(again.zeff, 2.0)
    explicit = composition_from_fractions(["C"], [1], [6], target_zeff=1.5)
    assert resolve_impurity_composition(out, composition=explicit).zeff == pytest.approx(1.5)


def test_a_measured_composition_is_never_rewritten():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(ions=_lane_k_like(composition_record_text("measured", "x")))
    with pytest.raises(ValueError, match="measured"):
        populate_impurity_profiles(ods, resolve_impurity_composition(ods, machine_preset="vest"))


def test_impurities_share_the_main_ion_temperature_and_its_record(sample_48224):
    from vaft.process.impurity import populate_impurity_profiles

    r = resolve_impurity_composition(sample_48224, time=0.3, machine_preset="vest", shot=48224)
    out = populate_impurity_profiles(sample_48224, r)
    base = "core_profiles.profiles_1d.0"
    main_t = np.asarray(out[f"{base}.ion.0.temperature"])
    for k in (1, 2):
        np.testing.assert_allclose(out[f"{base}.ion.{k}.temperature"], main_t)
    record = sample_48224[f"{base}.ion.0.temperature_fit.parameters"] if "parameters" in sample_48224[f"{base}.ion.0.temperature_fit"] else None
    if record is not None:
        assert out[f"{base}.ion.1.temperature_fit.parameters"] == record


def test_gacode_reads_the_written_species_and_labels_the_zeff(sample_48224):
    from vaft.code.gacode.inputs import prepare_gacode_profile
    from vaft.process.impurity import populate_impurity_profiles

    r = resolve_impurity_composition(sample_48224, time=0.3, machine_preset="vest", shot=48224)
    out = populate_impurity_profiles(sample_48224, r)
    profile = prepare_gacode_profile(out, time=0.3, rho_max=0.95, z_eff=None, impurity=None)
    assert profile.provenance["z_eff"]["kind"] == "policy_assumption"
    assert profile.provenance["z_eff"]["origin"] == "assumed"
    assert profile.provenance["z_eff"]["species_value"] == pytest.approx(2.0, abs=1e-6)
    assert list(profile.z) == [1.0, 6.0, 8.0]


def test_gacode_refuses_a_zeff_column_that_contradicts_the_species(sample_48224):
    from vaft.code.gacode.inputs import ProfileConversionError, prepare_gacode_profile
    from vaft.process.impurity import populate_impurity_profiles

    r = resolve_impurity_composition(sample_48224, time=0.3, machine_preset="vest", shot=48224)
    out = populate_impurity_profiles(sample_48224, r)
    out["core_profiles.profiles_1d.0.zeff"] = np.full_like(np.asarray(out["core_profiles.profiles_1d.0.zeff"]), 2.5)
    with pytest.raises(ProfileConversionError, match="contradicts the species"):
        prepare_gacode_profile(out, time=0.3, rho_max=0.95, z_eff=None, impurity=None)


def test_a_malformed_scalar_ion_leaf_is_replaced_not_fatal():
    """Campaign FileDB core_profiles store profiles_1d.N.ion as a bare float (found on Tier A)."""
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods()
    ods["core_profiles.profiles_1d.0"].setraw("ion", float("nan"))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    out = populate_impurity_profiles(ods, r)
    assert [out[f"core_profiles.profiles_1d.0.ion.{k}.label"] for k in range(3)] == ["H+", "C6+", "O8+"]


# --- Stage C cold-review findings --------------------------------------------------------


def test_a_slice_s_own_measured_zeff_is_never_overwritten():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(zeff=1.7, zeff_record=composition_record_text("measured", "x"))
    r = resolve_impurity_composition(ods, machine_preset="vest", use_measured_zeff=False)
    with pytest.raises(ValueError, match="never overwritten"):
        populate_impurity_profiles(ods, r)


def test_a_slice_s_own_measured_ions_are_never_overwritten():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(ions=_lane_k_like(composition_record_text("measured", "x")))
    r = resolve_impurity_composition(None, machine_preset="vest")
    with pytest.raises(ValueError, match="never overwritten"):
        populate_impurity_profiles(ods, r, time=0.3)


def test_the_write_time_must_be_the_resolved_time():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(times=(0.30, 0.31))
    r = resolve_impurity_composition(ods, time=0.30, machine_preset="vest")
    with pytest.raises(ValueError, match="resolved at"):
        populate_impurity_profiles(ods, r, time=0.31)


def test_non_hydrogenic_or_mixed_main_ions_are_refused_and_deuterium_is_not():
    from vaft.process.impurity import populate_impurity_profiles

    helium = _ods(ions=(("He2+", 2.0, 2.0, 4.0026, 0.45, None),))
    with pytest.raises(ValueError, match="no hydrogenic main ion"):
        populate_impurity_profiles(helium, resolve_impurity_composition(None, machine_preset="vest"), time=0.3)
    mixed = _ods(ions=(("H+", 1.0, 1.0, 1.008, 0.5, None), ("D+", 1.0, 1.0, 2.014, 0.5, None)))
    with pytest.raises(ValueError, match="2 hydrogenic"):
        populate_impurity_profiles(mixed, resolve_impurity_composition(None, machine_preset="vest"), time=0.3)
    c = composition_from_fractions(["C"], [1], [6], target_zeff=2.0, main_ion="D")
    out = populate_impurity_profiles(_ods(), resolve_impurity_composition(None, composition=c), time=0.3)
    assert out["core_profiles.profiles_1d.0.ion.1.label"] == "C6+"


def test_undefined_points_are_filled_not_written_as_nan():
    from vaft.process.impurity import populate_impurity_profiles

    profile = np.array([2.0, 1.9, 1.8, 1.6, np.nan])
    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 1.0, None),), zeff=profile,
               zeff_record=composition_record_text("measured", "x"))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    out = populate_impurity_profiles(ods, r)
    for k in range(3):
        assert np.all(np.isfinite(out[f"core_profiles.profiles_1d.0.ion.{k}.density_thermal"]))
    assert "filled_points=1" in out["core_profiles.profiles_1d.0.ion.1.density_fit.parameters"]
    # the measured zeff (NaN edge included) is left as it was
    np.testing.assert_array_equal(out["core_profiles.profiles_1d.0.zeff"], profile)


def test_rotation_is_carried_so_gacode_keeps_vtor():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 1.0, None),))
    ods["core_profiles.profiles_1d.0.ion.0.velocity.toroidal"] = 1e4 * (1 - RHO)
    out = populate_impurity_profiles(ods, resolve_impurity_composition(ods, machine_preset="vest"))
    np.testing.assert_allclose(out["core_profiles.profiles_1d.0.ion.2.velocity.toroidal"], 1e4 * (1 - RHO))


def test_without_write_zeff_a_stale_zeff_is_dropped():
    from vaft.process.impurity import populate_impurity_profiles

    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 1.0, None),), zeff=1.2)
    out = populate_impurity_profiles(ods, resolve_impurity_composition(ods, machine_preset="vest"), write_zeff=False)
    assert "zeff" not in out["core_profiles.profiles_1d.0"].keys()


def test_gacode_compares_but_does_not_refuse_an_independent_measured_zeff(sample_48224):
    import copy as _copy

    from vaft.code.gacode.inputs import prepare_gacode_profile

    ods = _copy.deepcopy(sample_48224)
    base = "core_profiles.profiles_1d.0"
    ne = np.asarray(ods[f"{base}.electrons.density_thermal"], dtype=float)
    ods[f"{base}.ion.0.density_thermal"] = 0.8 * ne
    ods[f"{base}.ion.0.density"] = 0.8 * ne
    for key, value in (("label", "C6+"), ("z_ion", 6.0), ("element.0.z_n", 6.0), ("element.0.a", 12.011)):
        ods[f"{base}.ion.1.{key}"] = value
    ods[f"{base}.ion.1.density_thermal"] = ne / 30
    ods[f"{base}.ion.1.temperature"] = np.asarray(ods[f"{base}.ion.0.temperature"])
    ods[f"{base}.zeff"] = np.full(ne.shape, 2.004)          # unlabelled: an independent measurement
    profile = prepare_gacode_profile(ods, time=0.3, rho_max=0.95, z_eff=None, impurity=None)
    assert profile.provenance["z_eff"]["kind"] == "measured"
    assert profile.provenance["z_eff"]["species_value"] == pytest.approx(2.0, abs=1e-6)


# --- PR 5: an explicit composition in the GACODE profile, per surface ---------------------------


@pytest.fixture(scope="module")
def transport_profile():
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    import test_transport_state as T
    from omas import load_omas_json

    from vaft.process.transport_state import TransportStateKey, resolve_transport_state

    sample = load_omas_json(str(T.SAMPLE), consistency_check=False)
    state = resolve_transport_state(T._multi_slice(sample), TransportStateKey(48224, 0.302, "magnetics"),
                                    efit_quality="good")
    assert state.resolved
    return state.profile


def _charge(profile):
    return np.sum(np.asarray(profile.ni) * np.asarray(profile.z)[:, None], axis=0)


def test_the_preset_in_a_gacode_profile_is_quasi_neutral_at_zeff_two(transport_profile):
    from vaft.process.impurity import surface_composition_profile

    p = transport_profile
    v = surface_composition_profile(p, resolve_impurity_composition(machine_preset="vest"), 0.6)
    assert list(v.z) == [1.0, 6.0, 8.0] and list(v.name)[1:] == ["C", "O"]
    np.testing.assert_allclose(_charge(v), p.ne, rtol=1e-12)
    np.testing.assert_allclose(v.z_eff, 2.0, rtol=1e-12)
    for name in ("ne", "te", "rmin", "q", "rho"):
        np.testing.assert_array_equal(getattr(v, name), getattr(p, name))
    np.testing.assert_array_equal(v.ti[1], np.atleast_2d(p.ti)[0])


def test_a_radial_composition_is_lumped_at_the_surface(transport_profile):
    from vaft.process.impurity import resolve_radial_composition, surface_composition_profile

    p = transport_profile
    rho = np.asarray(p.rho)
    radial = resolve_radial_composition(np.asarray(p.te) * 1e3, np.asarray(p.ne) * 1e19, rho, {"C": 1, "O": 1},
                                        ionization="transient", plasma_age_s=0.015)
    for r in (0.3, 0.8):
        v = surface_composition_profile(p, radial, r)
        rho_s = np.interp(r, np.asarray(p.rmin) / p.rmin[-1], rho)
        # charge density kept at every rho (quasi-neutral gradients) ...
        np.testing.assert_allclose(_charge(v), p.ne, rtol=1e-9)
        # ... and the Z^2 moment, i.e. the composition's Z_eff, at the surface
        assert np.interp(rho_s, rho, v.z_eff) == pytest.approx(np.interp(rho_s, rho, radial.zeff), rel=2e-3)
        assert v.provenance["ni"]["surface_r_over_a"] == r
    assert surface_composition_profile(p, radial, 0.8).z[1] < surface_composition_profile(p, radial, 0.3).z[1]


def test_surface_profiles_refuse_a_foreign_grid_or_surface(transport_profile):
    from vaft.process.impurity import resolve_radial_composition, surface_composition_profile

    p = transport_profile
    radial = resolve_radial_composition(np.asarray(p.te)[:50] * 1e3, np.asarray(p.ne)[:50] * 1e19,
                                        np.asarray(p.rho)[:50], {"C": 1})
    with pytest.raises(ValueError, match="profile.rho"):
        surface_composition_profile(p, radial, 0.5)
    with pytest.raises(ValueError, match="r_over_a"):
        surface_composition_profile(p, resolve_impurity_composition(machine_preset="vest"), 1.2)


# --- a bundled (radial-charge) product read back (cold review 0.8.0 delta-absorb-16 F2) --------


def _synthetic_tables():
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
    from _adf11_synthetic import synthetic_adf11_tables

    return synthetic_adf11_tables()


def _real_adf11_tables():
    """The OPEN-ADAS C/O 96 tables from the local cache; skipped when absent (nothing is downloaded)."""
    cache = __import__("pathlib").Path.home() / "Library/Caches/vaft/open_adas/adf11"
    tables = {s: (cache / f"acd96_{s.lower()}.dat", cache / f"scd96_{s.lower()}.dat") for s in ("C", "O")}
    if not all(p.is_file() for pair in tables.values() for p in pair):
        pytest.skip("the real ADF11 tables are not in the local OPEN-ADAS cache")
    return {k: tuple(str(p) for p in v) for k, v in tables.items()}


def _tables(which):
    return _synthetic_tables() if which == "synthetic" else _real_adf11_tables()


def _radial_product(sample, tables):
    """The package's own radial writer product on the 48224 slice: bundled C and O ions."""
    from vaft.ods_access import path_value
    from vaft.process.impurity import populate_radial_impurity_profiles, resolve_radial_composition

    base = "core_profiles.profiles_1d.0"
    rho = np.asarray(sample[f"{base}.grid.rho_tor_norm"], dtype=float)
    te = np.asarray(sample[f"{base}.electrons.temperature"], dtype=float)
    ne = path_value(sample, f"{base}.electrons.density_thermal")
    ne = np.asarray(sample[f"{base}.electrons.density"] if ne is None else ne, dtype=float)
    radial = resolve_radial_composition(te, ne, rho, {"C": 1, "O": 1}, tables=tables, time=0.3)
    return populate_radial_impurity_profiles(sample, radial), radial, ne


def _stored(out, path):
    return np.asarray(out[f"core_profiles.profiles_1d.0.{path}"], dtype=float)


@pytest.mark.parametrize("which", ["synthetic", "real_adf11"])
def test_the_resolver_rereads_a_bundled_ion_with_its_radial_charge(sample_48224, which):
    out, radial, ne = _radial_product(sample_48224, _tables(which))
    again = resolve_impurity_composition(out, machine_preset="vest")
    stored = _stored(out, "zeff")
    with np.errstate(invalid="ignore", divide="ignore"):
        # Z_eff = n_H/n_e + sum_k n_k <Z^2>_k / n_e from the stored fields themselves ...
        direct = _stored(out, "ion.0.density") / ne + sum(
            _stored(out, f"ion.{k}.density") * _stored(out, f"ion.{k}.z_ion_square_1d") / ne for k in (1, 2))
        # ... and what reading the scalar z_ion as a fixed charge state gives instead
        flattened = _stored(out, "ion.0.density") / ne + sum(
            _stored(out, f"ion.{k}.density") * _stored(out, f"ion.{k}.z_ion") ** 2 / ne for k in (1, 2))
    zeff = np.asarray(again.zeff)
    finite = np.isfinite(zeff)
    assert finite.sum() >= stored.size - 1          # the zero-density edge point stays undefined
    assert np.nanmax(np.abs(flattened - stored)) > 0.1     # the defect this pins was O(1)
    np.testing.assert_allclose(zeff[finite], direct[finite], atol=1e-6)
    np.testing.assert_allclose(zeff[finite], stored[finite], atol=1e-6)
    with np.errstate(invalid="ignore", divide="ignore"):
        main = _stored(out, "ion.0.density") / ne
    np.testing.assert_allclose(np.asarray(again.main_ion_fraction)[finite], main[finite], atol=1e-9)
    both = finite & np.isfinite(radial.S2)
    np.testing.assert_allclose(np.asarray(again.S2)[both], radial.S2[both], atol=1e-9)
    np.testing.assert_allclose(np.asarray(again.S1)[both], radial.S1[both], atol=1e-9)
    assert again.kind == "derived" and "origin=derived" in again.source
    np.testing.assert_array_equal(again.mean_charge[:, 0], _stored(out, "ion.1.z_ion_1d"))
    np.testing.assert_array_equal(again.mean_square_charge[:, 1], _stored(out, "ion.2.z_ion_square_1d"))
    assert again.provenance["charge_fields"] == {"ion.1": "z_ion_1d + z_ion_square_1d",
                                                 "ion.2": "z_ion_1d + z_ion_square_1d"}
    assert "bundled" in again.provenance["charge"]
    assert again.species[0].charge_state == pytest.approx(float(out["core_profiles.profiles_1d.0.ion.1.z_ion"]))
    __import__("json").dumps(again.as_record())


def test_a_fixed_charge_ion_is_still_read_from_its_scalar_z_ion():
    ods = _ods(ions=_lane_k_like(composition_record_text("derived", "x")))
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.mean_charge is None and r.mean_square_charge is None
    assert r.provenance["charge_fields"] == {"ion.1": "z_ion"}
    assert "charge" not in r.provenance
    np.testing.assert_allclose(r.zeff, 2.0)
    np.testing.assert_allclose(r.S2, 36.0)


def test_a_bundled_ion_without_z_ion_square_1d_takes_its_states_then_the_square_of_the_mean():
    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 0.7, None), ("C", 6.0, 5.0, 12.011, 0.05, None)))
    ion = "core_profiles.profiles_1d.0.ion.1"
    mean = np.array([6.0, 5.5, 5.0, 4.5, 4.0])
    ods[f"{ion}.z_ion_1d"] = mean
    # the charge states behind that mean: half the ions one charge below, half one above
    density = 0.05 * NE
    for q, share in ((1, 0.5), (2, 0.5)):
        ods[f"{ion}.state.{q - 1}.z_min"] = ods[f"{ion}.state.{q - 1}.z_max"] = float(q)
    # states carry the mean +- 1 split, written per point: q-1 and q+1 around each mean
    del ods[f"{ion}.state"]
    for k, q in enumerate((3.0, 7.0)):
        ods[f"{ion}.state.{k}.z_min"] = ods[f"{ion}.state.{k}.z_max"] = q
    lower = (7.0 - mean) / 4.0                          # share at charge 3 so that <Z> = mean
    ods[f"{ion}.state.0.density"] = density * lower
    ods[f"{ion}.state.1.density"] = density * (1 - lower)
    r = resolve_impurity_composition(ods, machine_preset="vest")
    mean2 = 9.0 * lower + 49.0 * (1 - lower)
    assert r.provenance["charge_fields"]["ion.1"] == "z_ion_1d + state[].density for <Z^2>"
    np.testing.assert_allclose(r.mean_square_charge[:, 0], mean2)
    np.testing.assert_allclose(r.zeff, (1 - 0.05 * mean) + 0.05 * mean2)
    np.testing.assert_allclose(r.main_ion_fraction, 1 - 0.05 * mean)
    del ods[f"{ion}.state"]
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.provenance["charge_fields"]["ion.1"].startswith("z_ion_1d + <Z^2> = <Z>^2")
    np.testing.assert_allclose(r.zeff, (1 - 0.05 * mean) + 0.05 * mean**2)


def test_a_bundled_ion_closed_at_a_measured_zeff_uses_its_radial_moments():
    ods = _ods(ions=(("H+", 1.0, 1.0, 1.008, 0.7, None), ("C", 6.0, 5.0, 12.011, 0.05, None)),
               zeff=1.8, zeff_record=composition_record_text("measured", "x"))
    ion = "core_profiles.profiles_1d.0.ion.1"
    mean = np.array([6.0, 5.5, 5.0, 4.5, 4.0])
    ods[f"{ion}.z_ion_1d"] = mean
    ods[f"{ion}.z_ion_square_1d"] = mean**2 + 1.0
    r = resolve_impurity_composition(ods, machine_preset="vest")
    assert r.kind == "derived" and "measured" in r.zeff_source
    np.testing.assert_allclose(r.zeff, 1.8)
    # closed point by point with S1 = <Z>, S2 = <Z^2>: f = (Z_eff - 1) / (<Z^2> - <Z>)
    np.testing.assert_allclose(r.impurity_fractions[:, 0], 0.8 / (mean**2 + 1.0 - mean))
    np.testing.assert_allclose(1.0 * r.main_ion_fraction + r.impurity_fractions[:, 0] * mean, 1.0)
