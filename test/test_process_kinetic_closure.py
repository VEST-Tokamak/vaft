"""Dilution-aware pressure assembly and the classical fast-ion baseline (#1606)."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.process.impurity import composition_from_fractions
from vaft.process.kinetic_closure import (
    assemble_pressure,
    fast_ion_slowing_down_estimate,
    infer_kinetic_closure,
)

E = 1.602176634e-19
RHO = np.linspace(0.0, 1.0, 6)
NE = 1e19 * (1.0 - 0.6 * RHO**2)
TE = 100.0 * (1.0 - 0.8 * RHO**2) + 5.0


def _ods(ti=None):
    ods = ODS()
    ods["core_profiles.time"] = np.array([0.3])
    base = "core_profiles.profiles_1d.0"
    ods[f"{base}.time"] = 0.3
    ods[f"{base}.grid.rho_tor_norm"] = RHO
    ods[f"{base}.electrons.density_thermal"] = NE
    ods[f"{base}.electrons.temperature"] = TE
    ods[f"{base}.ion.0.label"] = "H+"
    ods[f"{base}.ion.0.element.0.z_n"] = 1.0
    ods[f"{base}.ion.0.element.0.a"] = 1.008
    ods[f"{base}.ion.0.z_ion"] = 1.0
    ods[f"{base}.ion.0.density_thermal"] = NE
    if ti is not None:
        ods[f"{base}.ion.0.temperature"] = ti
    return ods


def test_no_dilution_reproduces_the_legacy_pressure_exactly():
    closure = infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0, ti_record="ti_te_ratio=1; status=assumed")
    np.testing.assert_allclose(closure.pressure.p_thermal, E * NE * (TE + TE), rtol=1e-12)
    assert closure.provenance["ti"]["status"] == "assumed"


def test_zeff_one_recovers_the_pure_main_ion():
    pure = composition_from_fractions(["C"], [1], [6], target_zeff=1.0)
    closure = infer_kinetic_closure(_ods(), composition=pure, ti_te_ratio=1.0)
    np.testing.assert_allclose(closure.ion_densities["H+"], NE)
    np.testing.assert_allclose(closure.ion_densities["C6+"], 0.0)
    legacy = infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0)
    np.testing.assert_allclose(closure.pressure.p_thermal, legacy.pressure.p_thermal)


def test_dilution_lowers_the_ion_pressure_by_the_missing_ions():
    closure = infer_kinetic_closure(_ods(), machine_preset="vest", ti_te_ratio=1.0)
    ions = closure.ion_densities
    assert set(ions) == {"H+", "C6+", "O8+"}
    charge = ions["H+"] + 6 * ions["C6+"] + 8 * ions["O8+"]
    np.testing.assert_allclose(charge, NE, rtol=1e-12)                      # quasi-neutral
    total_ions = sum(ions.values())
    np.testing.assert_allclose(total_ions / NE, 36 / 43 + 2 / 86)            # 0.860
    np.testing.assert_allclose(closure.pressure.p_i_thermal, E * total_ions * TE, rtol=1e-12)
    legacy = infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0)
    np.testing.assert_allclose(legacy.pressure.p_thermal - closure.pressure.p_thermal,
                               E * NE * TE * (1 - (36 / 43 + 2 / 86)), rtol=1e-12)
    assert closure.provenance["composition"]["kind"] == "assumed"


def test_a_stored_ion_temperature_wins_over_the_ratio():
    ti = 0.5 * TE
    closure = infer_kinetic_closure(_ods(ti=ti), dilution="none", ti_te_ratio=3.0)
    np.testing.assert_allclose(closure.ion_temperatures["H+"], ti)
    with pytest.raises(ValueError, match="ti_te_ratio"):
        infer_kinetic_closure(_ods(), dilution="none")


def test_thermal_and_fast_stay_separate():
    fast = fast_ion_slowing_down_estimate(NE, TE, {"H+": (NE, 1.0, 1.0)}, 1e20 * (1 - RHO**2), 20e3, A_b=1.0)
    closure = infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0, fast_ion=fast)
    p = closure.pressure
    np.testing.assert_allclose(p.p_thermal, E * NE * 2 * TE, rtol=1e-12)     # untouched by the fast ions
    np.testing.assert_allclose(p.p_total, p.p_thermal + fast.p_fast)
    assert np.all(fast.E_c_eV > 0) and np.all(fast.n_fast[:-1] > 0) and fast.n_fast[-1] == 0.0
    assert np.all(fast.tau_thermalisation >= 0) and "classical" in closure.provenance["fast_ion"]["model"]


def test_assemble_pressure_refuses_mismatched_species():
    with pytest.raises(ValueError, match="same species"):
        assemble_pressure(NE, TE, {"H+": NE}, {"D+": TE})
    with pytest.raises(ValueError, match="dilution"):
        infer_kinetic_closure(_ods(), dilution="half")


def test_reading_creates_no_path():
    for ods in (_ods(), _ods(ti=0.5 * TE)):
        before = sorted(ods.paths())
        infer_kinetic_closure(ods, machine_preset="vest", ti_te_ratio=1.0)
        assert sorted(ods.paths()) == before


def test_the_main_ion_temperature_is_found_by_nuclear_charge_not_position():
    ods = _ods()
    base = "core_profiles.profiles_1d.0"
    # carbon first, hydrogen second, each with its own temperature
    for k, (label, z_n, a, z, t) in enumerate([("C6+", 6.0, 12.011, 6.0, 3.0 * TE), ("H+", 1.0, 1.008, 1.0, 0.5 * TE)]):
        ods[f"{base}.ion.{k}.label"] = label
        ods[f"{base}.ion.{k}.element.0.z_n"] = z_n
        ods[f"{base}.ion.{k}.element.0.a"] = a
        ods[f"{base}.ion.{k}.z_ion"] = z
        ods[f"{base}.ion.{k}.density_thermal"] = NE / 100 if z_n > 1 else NE * 0.94
        ods[f"{base}.ion.{k}.temperature"] = t
    closure = infer_kinetic_closure(ods, dilution="none")
    np.testing.assert_allclose(closure.ion_temperatures["H+"], 0.5 * TE)
    assert closure.provenance["ti"]["source"].endswith("ion.1.temperature")
    ods[f"{base}.ion.0.element.0.z_n"] = 1.0                                 # two hydrogenic entries
    with pytest.raises(ValueError, match="hydrogenic"):
        infer_kinetic_closure(ods, dilution="none")


def test_a_zero_or_undefined_edge_gives_nan_not_an_error():
    ne = NE.copy()
    ne[-1] = 0.0
    te = TE.copy()
    te[-2] = np.nan
    fast = fast_ion_slowing_down_estimate(ne, te, {"H+": (ne, 1.0, 1.0)}, 1e20 * np.ones_like(ne), 20e3, A_b=1.0)
    assert np.all(np.isnan(fast.p_fast[-2:])) and np.all(np.isfinite(fast.p_fast[:-2]))
    assert fast.provenance["undefined_points"] == 2


def test_a_fast_ion_estimate_from_another_slice_is_refused():
    def estimate(**kw):
        return fast_ion_slowing_down_estimate(NE, TE, {"H+": (NE, 1.0, 1.0)}, 1e20, 20e3, A_b=1.0, **kw)

    infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0, fast_ion=estimate(time=0.3, rho=RHO))
    with pytest.raises(ValueError, match="estimated at"):
        infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0, fast_ion=estimate(time=0.31))
    with pytest.raises(ValueError, match="rho grid"):
        infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0, fast_ion=estimate(rho=RHO**2))
    short = fast_ion_slowing_down_estimate(NE[:3], TE[:3], {"H+": (NE[:3], 1.0, 1.0)}, 1e20, 20e3, A_b=1.0)
    with pytest.raises(ValueError, match="shape"):
        infer_kinetic_closure(_ods(), dilution="none", ti_te_ratio=1.0, fast_ion=short)
