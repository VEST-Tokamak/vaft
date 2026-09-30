"""`compute_diamagnetism` normalises by the plasma volume, not by the mean of
the cumulative `profiles_1d.volume` profile (cold review 0.8.0
process-ml-and-omas-wrappers F0 / equilibrium-representation F5).

IMAS `profiles_1d.volume` is the volume enclosed by each flux surface: it
runs from 0 on the axis to the plasma volume at the LCFS, so its mean is
roughly half the plasma volume and mu_i = integral / (B_pa^2 V_p) came out
about twice too large on every ODS that carries the profile -- the packaged
48224 kinetic sample among them.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

pytest.importorskip("omas")

from vaft.omas.process_wrapper import compute_diamagnetism
from vaft.omas.sample import sample_ods


def _outline_volume(eq_ts) -> float:
    """The LCFS-outline volume the no-profile branch of the wrapper uses."""
    r = np.asarray(eq_ts["boundary.outline.r"], float)
    z = np.asarray(eq_ts["boundary.outline.z"], float)
    if r[0] != r[-1] or z[0] != z[-1]:
        r = np.append(r, r[0])
        z = np.append(z, z[0])
    r_mid = 0.5 * (r[:-1] + r[1:])
    return float(abs(-np.sum(np.pi * r_mid**2 * np.diff(z))))


def _plasma_slice_index(ods) -> int:
    n = len(ods["equilibrium.time_slice"])
    plasma = [
        i for i in range(n)
        if float(ods[f"equilibrium.time_slice.{i}.global_quantities.ip"]) != 0.0
    ]
    return plasma[len(plasma) // 2]


def test_a_known_cumulative_volume_profile_normalises_by_its_lcfs_value():
    ods = sample_ods(39915)
    i = _plasma_slice_index(ods)
    ts = ods["equilibrium.time_slice"][i]
    assert "profiles_1d.volume" not in ts, "the sample must exercise the outline branch"
    mu_outline = float(np.asarray(compute_diamagnetism(ods, time_index=i)).reshape(-1)[0])

    # A synthetic cumulative profile V(psi_N) = V_total * psi_N with a plasma
    # volume that is NOT the outline volume, so the normaliser is identifiable.
    v_total = 2.0
    synthetic = copy.deepcopy(ods)
    sts = synthetic["equilibrium.time_slice"][i]
    psi_n = np.linspace(0.0, 1.0, np.asarray(sts["profiles_1d.f"]).size)
    sts["profiles_1d.volume"] = v_total * psi_n
    mu_profile = float(np.asarray(compute_diamagnetism(synthetic, time_index=i)).reshape(-1)[0])

    # mu_i is proportional to 1 / V_p; the profile branch must divide by
    # V(psi_N = 1) = v_total. The old nanmean would have divided by v_total / 2.
    expected = mu_outline * _outline_volume(ts) / v_total
    assert mu_profile == pytest.approx(expected, rel=1e-9)
    assert mu_profile != pytest.approx(2.0 * expected, rel=1e-3)


def test_sample_48224_ships_the_lcfs_normalised_value():
    ods = sample_ods(48224)
    ts = ods["equilibrium.time_slice"][0]
    vol = np.asarray(ts["profiles_1d.volume"], float)
    assert "profiles_1d.volume" in ts and vol.size > 1, "48224 carries the profile natively"
    v_lcfs = float(np.nanmax(vol))
    v_outline = _outline_volume(ts)
    # The stored profile's LCFS value is the plasma volume the outline gives.
    assert v_lcfs == pytest.approx(v_outline, rel=0.02)
    assert float(np.nanmean(vol)) < 0.6 * v_lcfs, "cumulative: its mean is ~half"

    as_shipped = float(np.asarray(compute_diamagnetism(ods, time_index=0)).reshape(-1)[0])

    stripped = copy.deepcopy(ods)
    del stripped["equilibrium.time_slice.0.profiles_1d.volume"]
    from_outline = float(np.asarray(compute_diamagnetism(stripped, time_index=0)).reshape(-1)[0])

    # Same integral, normalised by V_lcfs instead of V_outline: the two branches
    # must agree to the (2 %) difference between the two plasma volumes.
    assert as_shipped == pytest.approx(from_outline * v_outline / v_lcfs, rel=1e-9)
    assert as_shipped == pytest.approx(from_outline, rel=0.03)
