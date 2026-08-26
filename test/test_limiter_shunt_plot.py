"""Plot contracts for the VEST limiter-current monitor panels."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from omas import ODS

import vaft.omas as vomas
import vaft.plot as vplot
from vaft.machine_mapping.magnetics import LIMITER_SHUNT_CHANNELS, LIMITER_SHUNT_RESISTANCE
from vaft.plot.models import LineSeries, Panels, Series


def _limiter_ods() -> ODS:
    ods = ODS(consistency_check=False)
    time = np.array([0.0, 0.1, 0.2])
    for index, channel in enumerate(LIMITER_SHUNT_CHANNELS):
        base = f"magnetics.shunt.{index}"
        ods[f"{base}.name"] = channel["name"]
        ods[f"{base}.identifier"] = channel["identifier"]
        ods[f"{base}.resistance"] = LIMITER_SHUNT_RESISTANCE
        ods[f"{base}.voltage.time"] = time
        ods[f"{base}.voltage.data"] = (index + 1) * LIMITER_SHUNT_RESISTANCE * time
    return ods


def test_omas_limiter_current_adapter_creates_lc_uc_mm_three_by_one_panels():
    ods = _limiter_ods()

    figure, axes = vomas.plot_magnetics_time_limiter_current(ods)

    assert axes.shape == (3, 1)
    assert [axis.get_title() for axis in axes.ravel()] == [
        channel["name"] for channel in LIMITER_SHUNT_CHANNELS
    ]
    for index, axis in enumerate(axes.ravel()):
        np.testing.assert_allclose(axis.lines[0].get_xdata(), [0.0, 0.1, 0.2])
        np.testing.assert_allclose(axis.lines[0].get_ydata(), (index + 1) * np.array([0.0, 0.1, 0.2]))
        assert axis.get_ylabel() == "Limiter Current [A]"
    assert "magnetics.shunt.0.current" not in ods
    assert "magnetics_time_limiter_current" in {
        row["name"] for row in vomas.available_plots(ods)
    }
    plt.close(figure)


def test_limiter_current_renderer_accepts_a_three_panel_view_model():
    panels = Panels(
        models=tuple(
            LineSeries(
                series=(Series(x=[0.0, 0.1], y=[index, index + 1]),),
                x_label="Time",
                x_unit="s",
                y_label="Limiter Current",
                y_unit="A",
            )
            for index in range(3)
        ),
        share_x=True,
    )

    figure, axes = vplot.magnetics_time_limiter_current(panels)

    assert axes.shape == (3, 1)
    assert all(len(axis.lines) == 1 for axis in axes.ravel())
    plt.close(figure)
