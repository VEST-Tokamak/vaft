"""The electron-density representation figure (#2253): one time-matched sample, four views.

The figure's claims are tested on its model and on the scene it draws:
measured Thomson points carry ``data_error_upper`` as vertical bars in (a),
(b) and (c); unusable channels are flagged, never drawn as data; slices are
matched by time; ``rho_tor_norm`` is the equilibrium's table, never
``sqrt(psi_N)``; and (c) is the ``Z = 0`` cut of the 2-D state drawn in (d).
"""

import copy

import numpy as np
import pytest

from vaft.data.resources import data_path

SAMPLE = data_path("kineticEfit/ods_48224_300ms.json")

pytestmark = pytest.mark.skipif(not SAMPLE.is_file(), reason="the 48224 kinetic sample is repository-only")


@pytest.fixture(scope="module")
def sample():
    from vaft.omas import sample_ods

    return sample_ods(48224)


@pytest.fixture(scope="module")
def figure():
    from vaft.diagram import electron_density_semantics

    return electron_density_semantics()


def _model(ods, **kwargs):
    from vaft.diagram._density_semantics import density_semantics

    return density_semantics(ods, **kwargs)


def test_every_panel_reads_one_time_matched_sample(figure):
    meta = figure.model.metadata
    assert meta["shot"] == 48224 and meta["machine"] == "VEST"
    for key in ("thomson_time_s", "core_profiles_time_s", "equilibrium_time_s"):
        assert meta[key] == pytest.approx(0.300, abs=1e-9)
    assert meta["source_path"] == "vaft/data/kineticEfit/ods_48224_300ms.json"
    assert meta["source_blob_sha"] == "7b5c773028f85f7d483639eb3592c4702f63c9bd"


def test_the_thomson_sample_is_chosen_by_time_not_by_index(sample):
    ods = copy.deepcopy(sample)
    order = np.arange(len(ods["thomson_scattering.time"]))[::-1]
    ods["thomson_scattering.time"] = np.asarray(ods["thomson_scattering.time"])[order]
    for k in range(len(ods["thomson_scattering.channel"])):
        for leaf in ("data", "data_error_upper"):
            path = f"thomson_scattering.channel.{k}.n_e.{leaf}"
            ods[path] = np.asarray(ods[path])[order]
    reference = _model(sample).thomson
    reversed_ = _model(ods).thomson
    np.testing.assert_array_equal(reversed_.n_e, reference.n_e)
    np.testing.assert_array_equal(reversed_.sigma, reference.sigma)


def test_a_slice_outside_the_tolerance_is_refused(sample):
    with pytest.raises(ValueError, match="no sample within"):
        _model(sample, time=0.320)


def test_measurement_uncertainty_is_data_error_upper_at_that_time(sample, figure):
    T = figure.model.thomson
    index = int(np.argmin(np.abs(np.asarray(sample["thomson_scattering.time"]) - 0.300)))
    for k in range(len(T.name)):
        data = np.asarray(sample[f"thomson_scattering.channel.{k}.n_e.data"])[index]
        error = np.asarray(sample[f"thomson_scattering.channel.{k}.n_e.data_error_upper"])[index]
        np.testing.assert_equal(T.n_e[k], data)
        np.testing.assert_equal(T.sigma[k], error)


def test_the_unusable_channels_are_flagged_not_drawn(figure):
    T = figure.model.thomson
    assert T.status == ("usable",) * 5 + ("no_sample",) * 2
    assert figure.model.metadata["flagged_channels"] == {"Polychrometer 6": "no_sample",
                                                         "Polychrometer 7": "no_sample"}
    scene = figure.scene
    for key in ("rho", "psi", "mid"):
        assert len(scene.role(f"ts:{key}")) == 5
        assert len(scene.role(f"ts:{key}:error")) == 5
    assert len(scene.role("ts:mid:flagged")) == 2
    assert len(scene.role("map:ts")) == 5 and len(scene.role("map:ts:flagged")) == 2


def test_negative_validity_rejects_and_a_channel_outside_the_boundary_is_flagged(sample):
    ods = copy.deepcopy(sample)
    count = len(ods["thomson_scattering.time"])
    ods["thomson_scattering.channel.0.n_e.validity_timed"] = np.full(count, -2)
    ods["thomson_scattering.channel.4.position.r"] = 0.74  # outside the LCFS on the midplane
    T = _model(ods).thomson
    assert T.status[0] == "rejected" and T.status[4] == "outside_lcfs"
    assert T.usable.sum() == 3


def test_error_bars_span_two_sigma_on_the_density_axis(figure):
    from vaft.diagram._density_semantics import _PLOT_DY, _Y_RANGE

    T = figure.model.thomson
    cm_per_unit = (_PLOT_DY[1] - _PLOT_DY[0]) / (_Y_RANGE[1] - _Y_RANGE[0])
    bars = figure.scene.role("ts:psi:error")
    lengths = sorted(abs(b.points[1][1] - b.points[0][1]) for b in bars)
    expected = sorted(2 * T.sigma[T.usable] / 1e18 * cm_per_unit)
    np.testing.assert_allclose(lengths, expected, atol=1e-9)


def test_rho_tor_norm_is_the_equilibrium_table_not_sqrt_psi_n(figure):
    m = figure.model
    T = m.thomson
    usable = T.usable
    np.testing.assert_allclose(T.rho_tor_norm[usable],
                               np.interp(T.psi_norm[usable], m.table_psi_norm, m.table_rho_tor_norm))
    assert np.max(np.abs(T.rho_tor_norm[usable] - np.sqrt(T.psi_norm[usable]))) > 0.05
    # the stored profile sits on the same table: its own grid rho_tor_norm is reproduced
    assert m.metadata["stored_state"].endswith("placed by core_profiles grid.psi")


def test_the_core_profile_keeps_its_own_grid(sample, figure):
    m = figure.model
    np.testing.assert_array_equal(m.profile_n_e,
                                  np.asarray(sample["core_profiles.profiles_1d.0.electrons.density"]))
    np.testing.assert_allclose(m.profile_rho_tor_norm,
                               np.asarray(sample["core_profiles.profiles_1d.0.grid.rho_tor_norm"]), atol=1e-6)


def test_a_sqrt_psi_n_proxy_table_is_refused(sample):
    ods = copy.deepcopy(sample)
    base = "equilibrium.time_slice.0"
    psi = np.asarray(ods[base + ".profiles_1d.psi"])
    axis, edge = float(ods[base + ".global_quantities.psi_axis"]), float(ods[base + ".global_quantities.psi_boundary"])
    ods[base + ".profiles_1d.rho_tor_norm"] = np.sqrt(np.clip((psi - axis) / (edge - axis), 0, None))
    with pytest.raises(ValueError, match="proxy"):
        _model(ods)


def test_the_midplane_cut_is_the_2d_state_at_z_zero(figure):
    from vaft.process.profile import equilibrium_mapping_points

    m = figure.model
    j = int(np.flatnonzero(np.isclose(m.grid_z, 0.0, atol=1e-12))[0])
    np.testing.assert_array_equal(m.midplane_n_e, m.map_n_e[:, j])
    # independently: psi_N at (R, 0) by the channel mapper, the stored profile interpolated there
    from vaft.omas import sample_ods

    inside = np.isfinite(m.midplane_n_e)
    mapped = equilibrium_mapping_points(sample_ods(48224), m.midplane_r[inside], np.zeros(inside.sum()), time=0.3)
    expected = np.interp(mapped.psi_norm, m.profile_psi_norm, m.profile_n_e)
    np.testing.assert_allclose(m.midplane_n_e[inside], expected, rtol=1e-9, atol=1e9)


def test_the_map_is_masked_outside_the_boundary_and_never_negative(figure):
    m = figure.model
    finite = np.isfinite(m.map_n_e)
    assert finite.sum() > 1000
    assert np.nanmin(m.map_n_e) >= 0.0
    R, Z = np.meshgrid(m.grid_r, m.grid_z, indexing="ij")
    assert R[finite].min() >= m.lcfs_r.min() - 0.01 and R[finite].max() <= m.lcfs_r.max() + 0.01
    assert Z[finite].min() >= m.lcfs_z.min() - 0.01 and Z[finite].max() <= m.lcfs_z.max() + 0.01


def test_an_off_midplane_channel_is_omitted_from_the_cut_not_relabelled(sample):
    from vaft.diagram._density_semantics import _scene

    ods = copy.deepcopy(sample)
    ods["thomson_scattering.channel.3.position.z"] = 0.10
    m = _model(ods)
    assert m.metadata["off_midplane_channels"] == ["Polychrometer 4"]
    scene = _scene(m)
    assert len(scene.role("ts:mid")) == 4
    assert len(scene.role("ts:psi")) == 5


def test_the_midplane_points_sit_at_their_physical_r(figure):
    from vaft.diagram._density_semantics import _PLOT_X, _R_RANGE, _MARKER_R

    T = figure.model.thomson
    centres = sorted(np.mean([p[0] for p in dot.points]) for dot in figure.scene.role("ts:mid"))
    expected = sorted(_PLOT_X[0] + (T.r[T.usable] - _R_RANGE[0]) / (_R_RANGE[1] - _R_RANGE[0])
                      * (_PLOT_X[1] - _PLOT_X[0]))
    np.testing.assert_allclose(centres, expected, atol=_MARKER_R * 1e-3)
    # the inboard/outboard split is about the magnetic axis
    np.testing.assert_array_equal(T.outboard, T.r > figure.model.axis_r)


def test_measured_stored_and_derived_have_separate_legend_entries(figure):
    for role in ("legend:measured", "legend:stored", "legend:derived", "legend:flagged:text"):
        assert len(figure.scene.role(role)) == 1


def test_the_stored_fit_provenance_is_reported_not_assumed(figure):
    meta = figure.model.metadata
    assert meta["density_fit_measured_matches"] == "5 of 5 usable Thomson values"
    assert "not asserted" in meta["not_claimed"]


def test_the_figure_source_is_deterministic(figure):
    from vaft.diagram import electron_density_semantics

    assert electron_density_semantics().tikz == figure.tikz


def test_the_map_has_no_hole_at_the_magnetic_axis(figure):
    """Bilinear psi undershoots psi_axis next to the axis; psi_N < 0 there is the axis, not exterior."""
    m = figure.model
    i = int(np.argmin(np.abs(m.grid_r - m.axis_r)))
    j = int(np.argmin(np.abs(m.grid_z - m.axis_z)))
    block = m.map_n_e[i - 2:i + 3, j - 2:j + 3]
    assert np.isfinite(block).all()
    assert block.min() > 0.95 * m.profile_n_e[0]


def test_the_panels_are_typed_vaft_plot_view_models(figure):
    from vaft.plot.models import Field2D, Profile1D

    views = figure.model.views
    assert all(isinstance(views[k], Profile1D) for k in ("rho", "psi", "mid"))
    assert isinstance(views["map"], Field2D)
    T = figure.model.thomson
    for key in ("rho", "psi"):
        traces = [s for s in views[key].series if s.label.startswith("Thomson")]
        assert sum(int(np.sum(s.valid_mask)) for s in traces) == int(T.usable.sum())
        # measurement sigma and validity travel in the view model, not beside it
        for s, side in zip(traces, (T.outboard, ~T.outboard)):
            np.testing.assert_allclose(s.yerr, T.sigma[side] * 1e-18)
            np.testing.assert_array_equal(s.valid_mask, T.usable[side])


def test_the_display_policy_is_vaft_plot_s(figure):
    from vaft.plot.display import resolve_display

    spec = figure.model.display
    assert spec == resolve_display("m^-3", unit="10^18 m^-3", subject="electron_density")
    assert spec.scale == pytest.approx(1e-18) and spec.notation == "scaled_axis"
    assert all(v.display == spec for v in figure.model.views.values())


def test_the_stored_profile_view_is_the_vaft_plot_recipe(sample, figure):
    """(a) is what ``electron_density_profile`` builds: the same stored state on rho_tor_norm."""
    from vaft.plot.backend.recipes import build_model

    recipe = build_model("electron_density_profile", [("", sample)])
    stored = next(s for s in figure.model.views["rho"].series if s.label == "core_profiles")
    np.testing.assert_allclose(stored.x, recipe.series[0].x, atol=1e-6)
    np.testing.assert_array_equal(stored.y, recipe.series[0].y)


def test_the_map_is_the_vaft_plot_field_recipe_inside_the_boundary(sample):
    """On the equilibrium's own grid, (d) equals ``electron_density_field`` wherever both call it plasma."""
    from vaft.plot.backend.recipes import build_model

    recipe = build_model("electron_density_field", [("", sample)], time=0.3)
    mine = _model(sample, refine=1).views["map"]
    np.testing.assert_array_equal(mine.r, recipe.r)
    both = np.isfinite(mine.values) & np.isfinite(recipe.values)
    assert both.sum() == np.isfinite(mine.values).sum()      # every plasma cell of ours is one of the recipe's
    np.testing.assert_allclose(mine.values[both] / mine.display.scale, recipe.values[both], rtol=1e-12)


def test_the_architecture_records_responsibility_edges(figure):
    edges = figure.model.metadata["architecture_edges"]
    for edge in (("identity", "adapter", "input"), ("data", "adapter", "input"),
                 ("adapter", "model", "extraction"), ("model", "display", "policy"),
                 ("model", "presentation", "policy")):
        assert edge in edges
    for role in ("arch:identity", "arch:data", "arch:adapter", "arch:model", "arch:display", "arch:presentation"):
        assert len(figure.scene.role(role)) == 1


@pytest.mark.skipif(__import__("shutil").which("pdflatex") is None, reason="needs pdflatex")
def test_the_pdf_export_is_reproducible(figure, tmp_path):
    a, b = figure.save(tmp_path / "a.pdf"), figure.save(tmp_path / "b.pdf")
    assert a.read_bytes() == b.read_bytes()
