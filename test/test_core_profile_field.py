"""The core-profile 2-D maps: paired with the equilibrium by time, drawn the right way up.

Cold review plot F5: ``profiles_1d[k]`` was mapped onto ``time_slice[k]`` by
index, and the DD-ordered ``(R, Z)`` psi of a square grid was never transposed,
so on the packaged 129x129 sample the Te peak sat at (0.634, -0.516) while the
magnetic axis is at (0.427, 0.023).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from omas import ODS

import vaft.omas as vo

STORED = [0.326, 0.323, 0.316]  # core_profiles times: fewer than, and not ordered as, the equilibrium's


def _add_core_profiles(ods, times):
    rho = np.linspace(0.0, 1.0, 21)
    ods["core_profiles.ids_properties.homogeneous_time"] = 1
    ods["core_profiles.time"] = np.asarray(times, dtype=float)
    for index, instant in enumerate(times):
        base = f"core_profiles.profiles_1d.{index}"
        ods[f"{base}.time"] = float(instant)
        ods[f"{base}.grid.rho_tor_norm"] = rho
        # the peak encodes the instant: Te0 = 1000 t
        ods[f"{base}.electrons.temperature"] = 1000.0 * instant * (1.0 - rho**2) + 1.0
        ods[f"{base}.electrons.density"] = 1.0e19 * (1.0 - rho**2) + 1.0e16
    return ods


@pytest.fixture(scope="module")
def sample():
    return _add_core_profiles(vo.sample_ods(), STORED)


def _slice_at(ods, instant):
    times = [float(ods[f"equilibrium.time_slice.{k}.time"]) for k in range(len(ods["equilibrium.time_slice"]))]
    index = int(np.argmin(np.abs(np.asarray(times) - instant)))
    assert times[index] == pytest.approx(instant, abs=2e-4)
    return index


@pytest.mark.parametrize("instant", STORED)
def test_the_profile_mapped_is_the_one_stored_at_the_equilibrium_time(sample, instant):
    model = vo.extract_electron_temperature_field(sample, time_slice=_slice_at(sample, instant))
    assert float(np.nanmax(model.values)) == pytest.approx(1000.0 * instant + 1.0, rel=1e-3)


def test_time_names_the_same_map_as_its_slice(sample):
    by_time = vo.extract_electron_temperature_field(sample, time=0.326)
    assert float(np.nanmax(by_time.values)) == pytest.approx(327.0, rel=1e-3)


def test_an_equilibrium_slice_without_a_profile_is_refused_not_borrowed(sample):
    index = _slice_at(sample, 0.320)  # 3 ms from the nearest stored profile, slices 1 ms apart
    with pytest.raises(ValueError, match="no core_profiles slice is stored at equilibrium slice"):
        vo.extract_electron_temperature_field(sample, time_slice=index)


def test_the_peak_sits_at_the_magnetic_axis_on_the_square_sample_grid(sample):
    index = _slice_at(sample, 0.316)
    model = vo.extract_electron_temperature_field(sample, time_slice=index)
    values = np.asarray(model.values)
    r, z = np.asarray(model.r, dtype=float), np.asarray(model.z, dtype=float)
    assert values.shape == (z.size, r.size)
    iz, ir = np.unravel_index(np.nanargmax(values), values.shape)
    axis = f"equilibrium.time_slice.{index}.global_quantities.magnetic_axis"
    assert r[ir] == pytest.approx(float(sample[f"{axis}.r"]), abs=2 * np.diff(r).max())
    assert z[iz] == pytest.approx(float(sample[f"{axis}.z"]), abs=2 * np.diff(z).max())


@pytest.mark.parametrize("stored_order", ["rz", "zr"])
def test_a_non_square_grid_is_oriented_whichever_way_it_was_stored(stored_order):
    r = np.linspace(0.1, 0.9, 20)
    z = np.linspace(-1.0, 1.0, 33)
    r0, z0 = 0.4, 0.3  # off-centre in both directions
    psi_rz = (r[:, None] - r0) ** 2 + ((z[None, :] - z0) / 1.5) ** 2
    ods = ODS(consistency_check=False)
    base = "equilibrium.time_slice.0"
    ods["equilibrium.time"] = np.asarray([0.3])
    ods[f"{base}.time"] = 0.3
    ods[f"{base}.profiles_2d.0.grid.dim1"] = r
    ods[f"{base}.profiles_2d.0.grid.dim2"] = z
    ods[f"{base}.profiles_2d.0.psi"] = psi_rz if stored_order == "rz" else psi_rz.T
    ods[f"{base}.global_quantities.psi_axis"] = 0.0
    ods[f"{base}.global_quantities.psi_boundary"] = 0.04
    _add_core_profiles(ods, [0.3])
    model = vo.extract_electron_temperature_field(ods)
    values = np.asarray(model.values)
    assert values.shape == (z.size, r.size)
    iz, ir = np.unravel_index(np.nanargmax(values), values.shape)
    assert abs(r[ir] - r0) <= np.diff(r).max() and abs(z[iz] - z0) <= np.diff(z).max()
