"""Renderers of the #1902 local-MHD criteria (table input, no solver)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from vaft.plot.local_mhd_criteria import (  # noqa: E402
    local_mhd_criteria_field_curvature,
    local_mhd_criteria_profile_interchange,
    local_mhd_criteria_s_alpha,
)

VEST = Path(__file__).resolve().parents[1] / "vaft" / "data" / "efit" / "g039915.00319"


@pytest.fixture(scope="module")
def inputs():
    from vaft.data.eqdsk import read_geqdsk

    m = read_geqdsk(VEST).mapping
    nw, nh = int(m["NW"]), int(m["NH"])
    grid = dict(psi_grid=np.asarray(m["PSIRZ"]), R=m["RLEFT"] + np.linspace(0, m["RDIM"], nw),
                Z=m["ZMID"] - m["ZDIM"] / 2 + np.linspace(0, m["ZDIM"], nh),
                psi_axis=float(m["SIMAG"]), psi_boundary=float(m["SIBRY"]))
    profiles = dict(profile_psi_n=np.linspace(0, 1, nw), f=np.asarray(m["FPOL"]), pressure=np.asarray(m["PRES"]))
    return grid, profiles, np.asarray(m["QPSI"]), (float(m["RMAXIS"]), float(m["ZMAXIS"])), m


def test_the_interchange_profile_draws_the_criterion_and_the_comparison(inputs):
    from vaft.process.local_mhd_criteria import interchange_comparison, mercier_ggj_profile

    grid, profiles, q, axis_rz, _ = inputs
    profile = mercier_ggj_profile(**grid, levels_norm=np.round(np.arange(0.1, 0.95, 0.05), 2), **profiles, q=q,
                                  axis_rz=axis_rz)
    comparison = interchange_comparison(profile, profile.psi_n, profile.D_I, profile.D_R, label="self")
    fig, ax = local_mhd_criteria_profile_interchange(profile, comparison=comparison, rational_surfaces=[(0.4, "5/2")])
    assert any(np.allclose(line.get_ydata(), 0.0) for line in ax.get_lines())  # the D = 0 criterion
    assert {"di, self", "dr, self"} <= {t.get_text() for t in ax.get_legend().get_texts()}
    assert ax.get_yscale() == "symlog"
    plt.close(fig)


def test_the_s_alpha_plane_names_its_model(inputs):
    from vaft.process.local_mhd_criteria import s_alpha_profile

    grid, profiles, q, axis_rz, _ = inputs
    profile = s_alpha_profile(**grid, levels_norm=np.round(np.arange(0.1, 0.951, 0.1), 2), **profiles, q=q,
                              axis_rz=axis_rz)
    fig, ax = local_mhd_criteria_s_alpha(profile)
    assert any("s–α circular, first stability" == t.get_text() for t in ax.get_legend().get_texts())
    assert not ax.patches  # no shaded "unstable" region beyond the model's boundary
    plt.close(fig)


def test_the_curvature_maps_are_centred_diverging_fields(inputs):
    from matplotlib.colors import TwoSlopeNorm

    from vaft.process.local_mhd_criteria import curvature_map

    grid, profiles, _, axis_rz, m = inputs
    maps = curvature_map(**grid, **profiles, axis_rz=axis_rz)
    fig, axes = local_mhd_criteria_field_curvature(maps, boundary=(np.asarray(m["RBBBS"]), np.asarray(m["ZBBBS"])))
    assert len(axes) == 3
    for axis in axes:
        mappable = axis.collections[0]
        assert isinstance(mappable.norm, TwoSlopeNorm) and mappable.norm.vcenter == 0.0
        assert axis.get_xlim()[1] < grid["R"].max()  # framed on the plasma, not the grid
    plt.close(fig)
    fig, axis = local_mhd_criteria_field_curvature(maps, quantity="kappa_g")
    plt.close(fig)
    with pytest.raises(ValueError, match="quantity"):
        local_mhd_criteria_field_curvature(maps, quantity="beta")
