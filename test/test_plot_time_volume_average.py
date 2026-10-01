"""``plot_core_profiles_time_volume_averaged`` draws what the ODS stores, ions included.

Cold review 0.8.0 (omas wrappers / plot item 9): the ion container of
``core_profiles.global_quantities`` is an OMAS struct array on a real ODS --
neither a list nor a dict -- so the trace selection listed no ion, and the
dict/getattr path walk that read the traces found nothing on an ODS at all.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from omas import ODS

from vaft.plot.time import plot_core_profiles_time_volume_averaged


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def _ods_with_averages(ions: int) -> ODS:
    ods = ODS(consistency_check=False)
    times = np.array([0.30, 0.31, 0.32])
    ods["core_profiles.time"] = times
    ods["core_profiles.global_quantities.n_e_volume_average"] = 1e19 * (1 + np.arange(3))
    ods["core_profiles.global_quantities.t_e_volume_average"] = 0.1 * (1 + np.arange(3))
    for ion in range(ions):
        ods[f"core_profiles.global_quantities.ion.{ion}.n_i_volume_average"] = 0.9e19 * (1 + np.arange(3)) / (ion + 1)
        ods[f"core_profiles.global_quantities.ion.{ion}.t_i_volume_average"] = 0.08 * (1 + np.arange(3)) / (ion + 1)
    return ods


@pytest.mark.parametrize("ions", [0, 1, 2])
def test_every_stored_average_is_drawn(ions):
    ods = _ods_with_averages(ions)
    plot_core_profiles_time_volume_averaged(ods)
    axes = plt.gcf().axes
    assert len(axes) == 2 + 2 * ions
    for axis in axes:
        (line,) = axis.get_lines()
        assert line.get_xdata().size == 3
    labels = [axis.get_ylabel() for axis in axes]
    assert sum("n_{i," in label for label in labels) == ions
    if ions:
        n_i = axes[2].get_lines()[0].get_ydata()
        np.testing.assert_allclose(n_i, ods["core_profiles.global_quantities.ion.0.n_i_volume_average"])
