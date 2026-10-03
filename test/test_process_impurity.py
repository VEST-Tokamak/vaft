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
    assert resolve_impurity_composition(ods, time=0.3101, machine_preset="vest").time == pytest.approx(0.31)
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
