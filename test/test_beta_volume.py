"""The volume beta and its normalized form (#1691).

``beta_B = 2 mu0 <p>_V / <B^2>_V`` divides by the total field energy inside the
plasma (Troyon 1984; Menard et al. 2004), not by the vacuum field at one radius
as the toroidal beta does. These tests pin the closed forms, the distinctness
from ``beta_tor``/``beta_normal``, and the summary columns on the packaged sample.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.constants import mu_0 as MU0

from vaft.formula.equilibrium import (
    beta_normal_from_beta_tor,
    beta_normal_from_beta_volume,
    beta_toroidal_from_p_B0,
    beta_volume_from_p_B2,
)


def test_the_volume_beta_is_the_ratio_of_the_averages():
    pressure, b2 = 2.0e3, 0.09
    assert beta_volume_from_p_B2(pressure, b2) == pytest.approx(2 * MU0 * pressure / b2)


def test_with_a_uniform_field_the_volume_beta_is_the_toroidal_beta():
    # <B^2> = B0^2: the two definitions coincide only here
    pressure, b0 = 5.0e3, 0.3
    assert beta_volume_from_p_B2(pressure, b0**2) == pytest.approx(beta_toroidal_from_p_B0(pressure, b0))


def test_the_normalized_volume_beta_uses_the_conventional_normalization():
    beta, a, b0, ip = 0.02, 0.3, 0.15, 1.0e5
    assert beta_normal_from_beta_volume(beta, a, b0, ip) == pytest.approx(beta_normal_from_beta_tor(beta, a, b0, ip))
    assert beta_normal_from_beta_volume(beta, a, -b0, -ip) == pytest.approx(100 * beta * a * b0 / 0.1)


@pytest.mark.parametrize("b2", [0.0, -1.0, float("nan")])
def test_a_non_positive_field_energy_is_refused(b2):
    with pytest.raises(ValueError):
        beta_volume_from_p_B2(1.0e3, b2)


def _sample():
    pytest.importorskip("omas")
    from vaft.omas.sample import sample_ods

    try:
        return sample_ods()
    except Exception as exc:  # pragma: no cover - sample not packaged
        pytest.skip(f"39915 sample unavailable: {exc}")


def test_the_equilibrium_toroidal_field_reduces_to_the_vacuum_one_for_a_flat_f():
    from vaft.omas.process_wrapper import compute_magnetic_energy

    ods = copy.deepcopy(_sample())
    ts = ods["equilibrium.time_slice"][0]
    # the reference the function's vacuum field uses: global b0 / major_radius, else the vacuum b0 / r0
    # (checked with `in` first: reading a missing OMAS path creates it)
    b0 = (float(ts["global_quantities.b0"]) if "global_quantities.b0" in ts
          else float(np.asarray(ods["equilibrium.vacuum_toroidal_field.b0"], float).flat[0]))
    r0 = (float(ts["global_quantities.major_radius"]) if "global_quantities.major_radius" in ts
          else float(np.asarray(ods["equilibrium.vacuum_toroidal_field.r0"], float).flat[0]))
    f_vacuum = b0 * r0
    ts["profiles_1d.f"] = np.full(len(ts["profiles_1d.psi"]), f_vacuum)
    vacuum = compute_magnetic_energy(ods, time_slice=0)
    equilibrium = compute_magnetic_energy(ods, time_slice=0, toroidal_field="equilibrium")
    assert equilibrium == pytest.approx(vacuum, rel=1e-12)
    with pytest.raises(ValueError):
        compute_magnetic_energy(ods, time_slice=0, toroidal_field="paramagnetic")


def test_the_summary_reports_the_volume_beta_as_a_separate_definition():
    from vaft.database import _summary as summary_module

    rows = summary_module.extract_equilibrium_global(_sample(), 39915)
    row = rows[0]
    for name in ("beta_volume_B2", "beta_normal_B2", "normalized_plasma_current"):
        assert name in summary_module.EQUILIBRIUM_GLOBAL_COLUMNS and np.isfinite(row[name])
    # VEST is a spherical tokamak: <B^2>_V exceeds b0^2 (the 1/R field and the
    # poloidal field), so the volume beta is below the toroidal beta -- the
    # aspect-ratio effect the definition exists to remove (Menard 2004).
    assert 0.0 < row["beta_volume_B2"] < row["beta_tor"]
    # one normalization for both: beta_N / beta_N,B is beta_tor / beta_B
    assert row["beta_normal"] / row["beta_normal_B2"] == pytest.approx(
        row["beta_tor"] / row["beta_volume_B2"], rel=0.05)
    # I_N is the abscissa of the Troyon line: beta_N = beta_tor[%] / I_N
    assert row["beta_normal"] == pytest.approx(100 * row["beta_tor"] / row["normalized_plasma_current"], rel=0.05)
