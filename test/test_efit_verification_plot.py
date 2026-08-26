"""Finalized-ODS EFIT verification plot coverage."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from vaft.data.eqdsk import read_geqdsk
from vaft.data.resources import data_path
from vaft.omas import plot_equilibrium_overview_verification


def _verification_ods():
    ods = read_geqdsk(data_path("efit/g039915.00319")).to_omas()
    ods["dataset_description.data_entry.pulse"] = 39915
    for family, count, scale in (
        ("bpol_probe", 4, 1e-2),
        ("flux_loop", 3, 1e-3),
        ("pf_current", 2, 1e4),
    ):
        for index in range(count):
            root = f"equilibrium.time_slice.0.constraints.{family}.{index}"
            measured = scale * (index + 1)
            ods[f"{root}.measured"] = measured
            ods[f"{root}.reconstructed"] = measured * 1.02
            ods[f"{root}.measured_error_upper"] = abs(measured) * 0.05
            ods[f"{root}.weight"] = 0.0 if index == count - 1 else 1.0
    for family, measured in (("ip", 70_000.0), ("diamagnetic_flux", 0.006)):
        root = f"equilibrium.time_slice.0.constraints.{family}"
        ods[f"{root}.measured"] = measured
        ods[f"{root}.reconstructed"] = measured * 0.99
        ods[f"{root}.measured_error_upper"] = abs(measured) * 0.02
        ods[f"{root}.weight"] = 1.0
    return ods


def test_equilibrium_verification_plot_renders_constraints_and_psi(tmp_path):
    figure, axes = plot_equilibrium_overview_verification(
        _verification_ods(),
        time_slice=0,
        show=False,
    )

    assert axes.shape == (2, 2)
    assert "shot 39915" in figure._suptitle.get_text()
    assert "Ip: measured" in figure._suptitle.get_text()
    assert "Diamagnetic flux: measured" in figure._suptitle.get_text()
    assert "relative RMS error" in axes[0, 0].get_title()
    assert len(axes[0, 0].lines) >= 2
    assert axes[1, 1].get_xlabel() == "R [m]"
    assert axes[1, 1].get_title() == "Normalized Poloidal Flux"
    output = tmp_path / "efit-verification.png"
    figure.savefig(output, dpi=100)
    assert output.stat().st_size > 10_000
