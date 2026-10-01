"""Thomson against EFIT pressure per slice, the atlas state key (#1430, #1454).

The equilibria here have several slices stored out of time order, because the
bug this layer exists to avoid -- pairing by index -- passes every one-slice test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omas import ODS

from vaft.validation import kinetic_state as ks
from vaft.validation.equilibrium import _thomson_pressure, thomson_pressure_samples

E = 1.602176634e-19
R0, A, KAPPA = 0.40, 0.25, 1.6
TIMES = (0.314, 0.312, 0.316, 0.313, 0.315)  # stored out of order on purpose


def _p0(time: float) -> float:
    """Each slice gets its own axis pressure so a wrong slice shows up."""
    return 1000.0 * (1.0 + 100.0 * (time - 0.312))


def _equilibrium(times=TIMES, *, phi=True, proxy=False, outline=True) -> ODS:
    ods = ODS(consistency_check=False)
    r = np.linspace(0.1, 0.8, 71)
    z = np.linspace(-0.6, 0.6, 81)
    rm, zm = np.meshgrid(r, z, indexing="ij")
    psi_n = ((rm - R0) ** 2 + (zm / KAPPA) ** 2) / A ** 2
    level = np.linspace(0.0, 1.0, 41)
    ods["equilibrium.time"] = np.asarray(times)
    for i, t in enumerate(times):
        root = f"equilibrium.time_slice.{i}"
        ods[f"{root}.time"] = t
        ods[f"{root}.global_quantities.psi_axis"] = -0.02
        ods[f"{root}.global_quantities.psi_boundary"] = 0.03
        ods[f"{root}.profiles_1d.psi"] = -0.02 + 0.05 * level
        ods[f"{root}.profiles_1d.pressure"] = _p0(t) * (1.0 - level)
        if phi:
            ods[f"{root}.profiles_1d.phi"] = 0.01 * level ** 1.3
        if proxy:
            ods[f"{root}.profiles_1d.rho_tor_norm"] = np.sqrt(level)
        ods[f"{root}.profiles_2d.0.grid.dim1"] = r
        ods[f"{root}.profiles_2d.0.grid.dim2"] = z
        ods[f"{root}.profiles_2d.0.psi"] = -0.02 + 0.05 * psi_n
        if outline:
            theta = np.linspace(0, 2 * np.pi, 200)
            ods[f"{root}.boundary.outline.r"] = R0 + A * np.cos(theta)
            ods[f"{root}.boundary.outline.z"] = KAPPA * A * np.sin(theta)
    return ods


def _thomson(times=(0.310, 0.311, 0.312, 0.313, 0.314, 0.315, 0.316, 0.317), radii=(0.30, 0.36, 0.42, 0.48)) -> ODS:
    ods = ODS(consistency_check=False)
    ods["thomson_scattering.time"] = np.asarray(times)
    for c, radius in enumerate(radii):
        base = f"thomson_scattering.channel.{c}"
        ods[f"{base}.position.r"] = radius
        ods[f"{base}.position.z"] = 0.0
        # n_e rises with time so each sample is distinguishable
        ods[f"{base}.n_e.data"] = 1e18 * (1.0 + np.arange(len(times)))
        ods[f"{base}.n_e.data_error_upper"] = 1e17 * np.ones(len(times))
        ods[f"{base}.t_e.data"] = 50.0 * np.ones(len(times))
        ods[f"{base}.t_e.data_error_upper"] = 5.0 * np.ones(len(times))
    return ods


def test_state_time_rounds_to_the_contract_precision():
    assert ks.state_time(0.31600000001) == 0.316
    assert ks.LINEAGES == ("magnetics", "electron_kinetic")
    assert ks.QUALITIES == ("good", "admissible")


def test_slice_is_found_by_time_not_index():
    eq = _equilibrium()
    for t in TIMES:
        index, offset = ks.slice_at_time(eq, t)
        assert TIMES[index] == t and offset == 0.0
    index, offset = ks.slice_at_time(eq, 0.3154)
    assert TIMES[index] == 0.315 and offset == pytest.approx(-0.0004)


def test_slice_beyond_tolerance_is_refused():
    eq = _equilibrium()
    # cadence 1 ms -> tolerance max(0.5 ms, 1 ms) = 1 ms
    assert ks.default_tolerance(np.asarray(TIMES)) == pytest.approx(1e-3)
    ks.slice_at_time(eq, 0.3169)
    with pytest.raises(LookupError, match="beyond"):
        ks.slice_at_time(eq, 0.3171)
    with pytest.raises(LookupError):
        ks.slice_at_time(eq, 0.3165, tolerance_s=1e-4)


def test_matched_ratios_use_the_slice_and_sample_at_that_time():
    eq, ts = _equilibrium(), _thomson()
    for t in TIMES:
        row = ks.match_thomson_pressure(eq, ts, time_s=t)
        assert row["ts_status"] == "matched"
        assert row["time_efit_s"] == t and row["time_ts_s"] == pytest.approx(t)
        assert row["dt_ts_efit_s"] == pytest.approx(0.0, abs=1e-12)
        sample = int(round((t - 0.310) * 1e3))
        for ch in row["channels"]:
            psi_n = ((ch["r"] - R0) ** 2) / A ** 2
            assert ch["psi_norm"] == pytest.approx(psi_n, abs=2e-3)
            assert ch["p_recon"] == pytest.approx(_p0(t) * (1 - psi_n), rel=5e-3)
            assert ch["p_e"] == pytest.approx(E * 1e18 * (1 + sample) * 50.0)
            assert ch["r_p"] == pytest.approx(ch["p_recon"] / ch["p_e"])
            assert ch["n_e_error"] == 1e17 and ch["t_e_error"] == 5.0


def test_r_sum_is_the_quantity_the_criteria_band_grades():
    eq, ts = _equilibrium(), _thomson()
    for t in TIMES:
        row = ks.match_thomson_pressure(eq, ts, time_s=t)
        index = TIMES.index(t)
        graded = _thomson_pressure(eq, index, ts)
        assert row["r_sum"] == pytest.approx(1.0 / graded["sum_ratio"], rel=1e-12)
        assert row["log_ratio"] == pytest.approx(graded["log_ratio"], rel=1e-12)
        assert graded["channels_inside"] == row["points"]


def test_the_split_sampler_keeps_the_check_result():
    eq, ts = _equilibrium(), _thomson()
    graded = _thomson_pressure(eq, 2, ts)
    sampled = thomson_pressure_samples(eq, 2, ts)
    p_e = sum(c["p_e"] for c in sampled["channels"])
    p_r = sum(c["p_recon"] for c in sampled["channels"])
    assert graded["log_ratio"] == pytest.approx(math.log(p_e / p_r))
    assert graded["status"] in {"pass", "warn", "fail"}
    assert graded["time_offset"] == 0.0


def test_thomson_outside_tolerance_stays_as_unmatched():
    eq = _equilibrium()
    ts = _thomson(times=(0.300, 0.301))
    row = ks.match_thomson_pressure(eq, ts, time_s=0.314)
    assert row["ts_status"] == "unmatched"
    assert "beyond" in row["reason"]
    assert "r_sum" not in row
    assert row["dt_ts_efit_s"] == pytest.approx(-0.013)


def test_thomson_outside_the_plasma_is_invalid_not_unmatched():
    eq = _equilibrium()
    ts = _thomson(radii=(0.75, 0.78))
    row = ks.match_thomson_pressure(eq, ts, time_s=0.314)
    assert row["ts_status"] == "invalid"
    assert "LCFS" in row["reason"]


def test_explicit_ts_tolerance_applies_to_a_one_slice_equilibrium():
    eq = _equilibrium(times=(0.3145,))
    ts = _thomson()
    assert ks.match_thomson_pressure(eq, ts, time_s=0.3145)["ts_status"] == "matched"
    row = ks.match_thomson_pressure(eq, ts, time_s=0.3145, ts_tolerance_s=1e-4)
    assert row["ts_status"] == "unmatched"


def test_rho_comes_from_phi_and_never_from_the_proxy():
    eq = _equilibrium()
    coordinate = ks.rho_tor_norm_of(eq, 0)
    assert coordinate["coordinate"] == "rho_tor_norm" and coordinate["source"] == "phi"
    level = np.linspace(0.0, 1.0, 41)
    np.testing.assert_allclose(coordinate["rho_tor_norm"], level ** 0.65)

    proxied = ks.rho_tor_norm_of(_equilibrium(phi=False, proxy=True), 0)
    assert proxied["coordinate"] == "unavailable" and "proxy" in proxied["reason"]
    row = ks.match_thomson_pressure(_equilibrium(phi=False, proxy=True), _thomson(), time_s=0.314)
    assert row["rho_coordinate"] == "unavailable"
    assert all(math.isnan(c["rho_tor_norm"]) for c in row["channels"])
    assert all(np.isfinite(c["psi_norm"]) for c in row["channels"])


def test_integrated_ratio_recovers_a_uniform_ratio():
    eq = _equilibrium()
    index = TIMES.index(0.315)
    coordinate = ks.rho_tor_norm_of(eq, index)
    p_eq = _p0(0.315) * (1.0 - coordinate["psi_norm"])
    full = ks.integrated_pressure_ratio(eq, index, coordinate["rho_tor_norm"], p_eq / 2.0)
    assert full["available"] and full["cell_weights"] == "outline"
    assert full["ratio"] == pytest.approx(2.0, rel=1e-3)
    assert full["volume_m3"] == pytest.approx(2 * np.pi * R0 * np.pi * A * A * KAPPA, rel=0.03)
    span = ks.integrated_pressure_ratio(eq, index, coordinate["rho_tor_norm"], p_eq / 2.0,
                                        psi_norm_span=(0.2, 0.6))
    assert span["ratio"] == pytest.approx(2.0, rel=1e-3)
    assert span["volume_m3"] < full["volume_m3"]


def test_integrated_ratio_does_not_extrapolate_p_e():
    eq = _equilibrium()
    coordinate = ks.rho_tor_norm_of(eq, 0)
    p_eq = _p0(TIMES[0]) * (1.0 - coordinate["psi_norm"])
    keep = coordinate["rho_tor_norm"] <= 0.5
    inner = ks.integrated_pressure_ratio(eq, 0, coordinate["rho_tor_norm"][keep], p_eq[keep] / 3.0)
    full = ks.integrated_pressure_ratio(eq, 0, coordinate["rho_tor_norm"], p_eq / 3.0)
    assert inner["ratio"] == pytest.approx(3.0, rel=1e-3)
    assert inner["volume_m3"] < full["volume_m3"]


def test_core_profiles_pressure_is_matched_by_time():
    cp = ODS(consistency_check=False)
    cp["core_profiles.time"] = np.array([0.316, 0.312, 0.314])
    for j, t in enumerate((0.316, 0.312, 0.314)):
        root = f"core_profiles.profiles_1d.{j}"
        cp[f"{root}.time"] = t
        cp[f"{root}.grid.rho_tor_norm"] = np.linspace(0, 1, 5)
        cp[f"{root}.electrons.density_thermal"] = np.full(5, 1e18 * (j + 1))
        cp[f"{root}.electrons.temperature"] = np.full(5, 10.0)
    got = ks.core_profiles_electron_pressure(cp, time_s=0.312)
    assert got["available"] and got["time_s"] == 0.312
    np.testing.assert_allclose(got["p_e"], E * 2e18 * 10.0)
    missing = ks.core_profiles_electron_pressure(cp, time_s=0.320)
    assert not missing["available"] and "beyond" in missing["reason"]


def test_inputs_are_not_mutated():
    eq, ts = _equilibrium(), _thomson()
    before_eq, before_ts = set(eq.flat()), set(ts.flat())
    ks.match_thomson_pressure(eq, ts, time_s=0.313)
    coordinate = ks.rho_tor_norm_of(eq, 1)
    ks.integrated_pressure_ratio(eq, 1, coordinate["rho_tor_norm"], np.ones(41))
    ks.core_profiles_electron_pressure(eq, time_s=0.313)
    assert set(eq.flat()) == before_eq and set(ts.flat()) == before_ts
