"""Atomic-data charge states behind Z_eff(rho) (#1565 Sec. 8): synthetic ADF11 tables, no network."""

from __future__ import annotations

import numpy as np
import pytest
from omas import ODS

from vaft.data.open_adas import read_adf11
from vaft.formula.atomic import (
    coronal_relaxation_time,
    fractional_abundances,
    mean_charge_from_charge_state_densities,
    transient_fractional_abundances,
)
from vaft.process.impurity import (
    composition_record_origin,
    populate_radial_impurity_profiles,
    resolve_radial_composition,
)

LOG_NE = (10.0, 14.0)            # cm^-3
LOG_TE = (0.0, 1.0, 2.0, 3.0)    # 1 eV .. 1 keV


def _table(path, blocks):
    lines = [f"{len(blocks)} {len(LOG_NE)} {len(LOG_TE)} / synthetic", "",
             " ".join(map(str, LOG_NE)), " ".join(map(str, LOG_TE))]
    for index, values in enumerate(blocks, start=1):
        lines.append(f"---------------- /IPRT=1/IGRD=1/TYPE=TEST/Z1={index}/")
        lines.append(" ".join(f"{v:.3f}" for v in values))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    return path


def _element(tmp_path, symbol, z):
    """Ionisation that turns on with T_e, stage by stage; constant recombination."""
    scd, acd = [], []
    for j in range(z):
        threshold = 0.6 + 0.35 * j          # log10 T_e where stage j ionises fast
        per_te = [-8.0 - 3.0 * max(threshold - t, 0.0) for t in LOG_TE]
        scd.append([v for v in per_te for _ in LOG_NE])   # nT outer, nne inner -> nne*nT values
        acd.append([-11.0] * (len(LOG_NE) * len(LOG_TE)))
    read_adf11.cache_clear()
    return (_table(tmp_path / f"acd96_{symbol.lower()}.dat", acd),
            _table(tmp_path / f"scd96_{symbol.lower()}.dat", scd))


@pytest.fixture
def tables(tmp_path):
    return {"C": _element(tmp_path, "C", 6), "O": _element(tmp_path, "O", 8)}


RHO = np.linspace(0.0, 1.0, 11)
TE = 300.0 * (1.0 - 0.95 * RHO**2)
NE = 1.5e19 * (1.0 - 0.7 * RHO**2)


# --- formulas ----------------------------------------------------------------------------


def test_transient_starts_neutral_and_tends_to_coronal(tables):
    acd, scd = tables["C"]
    zero = transient_fractional_abundances(1e19, 50.0, acd, scd, 0.0)
    assert zero[0] == pytest.approx(1.0)
    late = transient_fractional_abundances(1e19, 50.0, acd, scd, 10.0)
    np.testing.assert_allclose(late, fractional_abundances(1e19, 50.0, acd, scd), atol=1e-8)
    profile = transient_fractional_abundances(NE, TE, acd, scd, 1e-3)
    assert profile.shape == (RHO.size, 7)
    np.testing.assert_allclose(profile.sum(axis=-1), 1.0)


def test_the_relaxation_time_bounds_the_approach_to_coronal(tables):
    acd, scd = tables["C"]
    tau = float(coronal_relaxation_time(1e19, 50.0, acd, scd))
    assert 0.0 < tau < np.inf
    coronal = mean_charge_from_charge_state_densities(fractional_abundances(1e19, 50.0, acd, scd))
    early = mean_charge_from_charge_state_densities(transient_fractional_abundances(1e19, 50.0, acd, scd, 0.1 * tau))
    late = mean_charge_from_charge_state_densities(transient_fractional_abundances(1e19, 50.0, acd, scd, 10 * tau))
    assert abs(late - coronal) < abs(early - coronal)


# --- the radial composition -----------------------------------------------------------------


def test_partially_ionised_edge_lowers_zeff_at_fixed_elemental_density(tables):
    r = resolve_radial_composition(TE, NE, RHO, {"C": 1, "O": 1}, normalization="fixed", tables=tables)
    np.testing.assert_allclose(r.elemental_fractions[:, 0], 1 / 86)
    assert r.zeff[0] <= 2.0 + 1e-9
    assert r.zeff[-1] < r.zeff[0]                    # the cold edge is less ionised
    assert r.mean_charge[-1, 0] < r.mean_charge[0, 0] <= 6.0
    np.testing.assert_allclose(r.effective_charge, r.S2 / r.S1)
    # quasi-neutrality and the Z_eff identity, point by point
    z2 = r.elemental_fractions * r.mean_square_charge
    np.testing.assert_allclose(r.main_ion_fraction + z2.sum(axis=1), r.zeff, rtol=1e-12)


@pytest.mark.parametrize("rule", ["ne_weighted_mean", "axis"])
def test_each_normalization_hits_its_target(tables, rule):
    r = resolve_radial_composition(TE, NE, RHO, {"C": 1, "O": 1}, normalization=rule, target_zeff=2.0, tables=tables)
    if rule == "axis":
        assert r.zeff[0] == pytest.approx(2.0)
    else:
        edges = np.concatenate(([0.0], 0.5 * (RHO[1:] + RHO[:-1]), [RHO[-1]]))
        dv = RHO * np.diff(edges)
        assert np.sum(NE * dv * r.zeff) / np.sum(NE * dv) == pytest.approx(2.0)
    assert r.normalization["method"] == rule and r.kind == "derived"


def test_the_resistive_closure_matches_its_projection(tables):
    def projection(zeff):
        return float(np.mean(zeff))

    r = resolve_radial_composition(TE, NE, RHO, {"C": 1, "O": 1}, normalization="resistive_closure",
                                   resistive_target=1.6, projection=projection, tables=tables)
    assert projection(r.zeff) == pytest.approx(1.6, abs=1e-9)
    assert r.kind == "inferred"
    with pytest.raises(ValueError, match="projection"):
        resolve_radial_composition(TE, NE, RHO, {"C": 1}, normalization="resistive_closure", tables=tables)


def test_the_coronal_check_flags_a_young_plasma(tables):
    acd, scd = tables["C"]
    tau = float(np.max(coronal_relaxation_time(NE, TE, acd, scd)))
    young = resolve_radial_composition(TE, NE, RHO, {"C": 1}, plasma_age_s=1e-3 * tau, tables=tables)
    old = resolve_radial_composition(TE, NE, RHO, {"C": 1}, plasma_age_s=100 * tau, tables=tables)
    assert not young.coronal_valid.all() and old.coronal_valid.all()
    assert resolve_radial_composition(TE, NE, RHO, {"C": 1}, tables=tables).coronal_valid is None
    transient = resolve_radial_composition(TE, NE, RHO, {"C": 1}, ionization="transient",
                                           plasma_age_s=1e-3 * tau, tables=tables)
    assert np.all(transient.mean_charge <= young.mean_charge + 1e-12)


def test_unreachable_targets_and_bad_tables_are_refused(tables):
    with pytest.raises(ValueError, match="more impurity charge"):
        resolve_radial_composition(TE, NE, RHO, {"C": 1}, normalization="axis", target_zeff=6.5, tables=tables)
    with pytest.raises(ValueError, match="Z_n = 8"):
        resolve_radial_composition(TE, NE, RHO, {"O": 1}, tables={"O": tables["C"]})
    with pytest.raises(ValueError, match="transient"):
        resolve_radial_composition(TE, NE, RHO, {"C": 1}, ionization="transient", tables=tables)


def test_invalid_points_are_undefined_not_fatal(tables):
    ne = NE.copy()
    ne[-1] = 0.0
    r = resolve_radial_composition(TE, ne, RHO, {"C": 1, "O": 1}, tables=tables)
    assert np.isnan(r.zeff[-1]) and np.all(np.isfinite(r.zeff[:-1]))


# --- writing the bundled ions ------------------------------------------------------------------


def test_bundled_ions_carry_charge_profiles_and_states(tables):
    ods = ODS()
    ods["core_profiles.time"] = np.array([0.3])
    ods["core_profiles.profiles_1d.0.time"] = 0.3
    ods["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = RHO
    ods["core_profiles.profiles_1d.0.electrons.density_thermal"] = NE
    r = resolve_radial_composition(TE, NE, RHO, {"C": 1, "O": 1}, tables=tables, time=0.3)
    out = populate_radial_impurity_profiles(ods, r)
    base = "core_profiles.profiles_1d.0"
    assert [out[f"{base}.ion.{k}.label"] for k in range(3)] == ["H+", "C", "O"]
    np.testing.assert_allclose(out[f"{base}.ion.1.z_ion_1d"], r.mean_charge[:, 0])
    np.testing.assert_allclose(out[f"{base}.ion.2.z_ion_square_1d"], r.mean_square_charge[:, 1])
    states = sum(np.asarray(out[f"{base}.ion.1.state.{q}.density"]) for q in range(6))
    assert np.all(states <= np.asarray(out[f"{base}.ion.1.density"]) * (1 + 1e-12))
    np.testing.assert_allclose(out[f"{base}.zeff"], r.zeff)
    assert composition_record_origin(out[f"{base}.zeff_fit.parameters"]) == "derived"
    charge = (np.asarray(out[f"{base}.ion.0.density"])
              + sum((q + 1) * np.asarray(out[f"{base}.ion.{k}.state.{q}.density"])
                    for k, z in ((1, 6), (2, 8)) for q in range(z)))
    np.testing.assert_allclose(charge, NE, rtol=1e-9)              # quasi-neutral incl. charge states


def test_the_resistive_closure_projection_must_ignore_undefined_points(tables):
    te = TE.copy()
    te[-1] = np.nan
    with pytest.raises(ValueError, match="ignore NaN"):
        resolve_radial_composition(te, NE, RHO, {"C": 1}, normalization="resistive_closure",
                                   resistive_target=1.5, projection=lambda z: float(np.mean(z)), tables=tables)
    r = resolve_radial_composition(te, NE, RHO, {"C": 1}, normalization="resistive_closure",
                                   resistive_target=1.5, projection=lambda z: float(np.nanmean(z)), tables=tables)
    assert np.nanmean(r.zeff) == pytest.approx(1.5)
