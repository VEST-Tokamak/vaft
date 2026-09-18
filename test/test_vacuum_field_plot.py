"""The interactive vacuum-field map: what it draws and what it offers to change.

Four quantities -- flux, |B_p|, the decay index, the breakdown figure of merit
-- from one cached evaluation, over the PF time base rather than a handful of
stored equilibrium slices.
"""

import numpy as np
import pytest

import vaft.omas
from vaft.omas import process_wrapper as pw
from vaft.plot.backend import recipes
from vaft.plot.backend.discovery import describe_one
from vaft.plot.controls import controls_for
from vaft.plot.models import Field2D

COARSE = 21


@pytest.fixture(scope="module")
def ods():
    return vaft.omas.sample_ods()


@pytest.fixture(scope="module")
def record(ods):
    return describe_one("vacuum_field", [("sample", ods)])


# ---------------------------------------------------------------------------
# What it draws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", recipes.VACUUM_FIELD_NAMES)
def test_each_quantity_builds_a_field_on_the_poloidal_plane(ods, field):
    model = recipes._build_vacuum_field(ods, field=field, resolution=COARSE)
    assert isinstance(model, Field2D)
    assert model.values.shape == (model.z.size, model.r.size)
    assert np.isfinite(model.values).any()


def test_the_quantities_are_read_off_one_field(ods):
    """|B_p| must be the quadrature of the same B_r and B_z the map holds --
    four separate computations is how the panels of a startup figure end up
    describing four slightly different instants."""
    model = recipes._build_vacuum_field(ods, field="b_poloidal", resolution=COARSE)
    instant = recipes._vacuum_psi_time(ods)
    result = pw.compute_vacuum_field_map(
        recipes._isolated_copy(ods, recipes._NULL_FIELD_ROOTS),
        time=instant,
        grid=(model.r, model.z),
    )
    expected = np.hypot(result["b_r"], result["b_z"]).T * 1e4  # gauss
    inside = np.isfinite(model.values)
    np.testing.assert_allclose(model.values[inside], expected[inside], rtol=1e-9)


def test_a_misspelt_quantity_names_the_vocabulary(ods):
    with pytest.raises(ValueError, match="b_poloidal"):
        recipes._build_vacuum_field(ods, field="b_pol")


def test_the_map_is_confined_to_the_limiter(ods):
    """Every current filament of the machine lies outside the limiter outline,
    so the plasma-facing region is exactly the part of a vacuum map that is not
    a conductor's own singular field."""
    model = recipes._build_vacuum_field(ods, field="b_poloidal", resolution=COARSE)
    drawn = np.isfinite(model.values)
    assert drawn.any() and not drawn.all()

    source_r, source_z, _, _ = pw._vacuum_sources(ods)
    interior = recipes._limiter_interior(ods, model.r, model.z)
    assert interior is not None
    from matplotlib.path import Path

    outline = recipes._wall_layers(ods)[0]
    polygon = Path(np.column_stack([outline.r, outline.z]))
    assert not polygon.contains_points(np.column_stack([source_r, source_z])).any()


def test_the_units_are_the_ones_a_start_up_is_read_in(ods):
    """Gauss for the poloidal field, V/m for the electric field and the
    breakdown figure, nothing at all for the decay index or the Lloyd margin,
    which are ratios -- one vocabulary per quantity."""
    units = {
        field: recipes._build_vacuum_field(ods, field=field, resolution=COARSE).display.unit
        for field in recipes.VACUUM_FIELD_NAMES
    }
    assert units == {
        "psi": "mWb",
        "b_poloidal": "G",
        "decay_index": "",
        "e_toroidal": "V/m",
        "breakdown": "V/m",
        "lloyd_margin": "",
    }


def test_out_of_range_values_saturate_rather_than_vanish(ods):
    """The levels come from a percentile, so the points it excludes must come
    out as the end colour and not as holes indistinguishable from no data."""
    for field in ("b_poloidal", "breakdown", "decay_index"):
        model = recipes._build_vacuum_field(ods, field=field, resolution=COARSE)
        assert model.extend in ("max", "both"), field
    assert recipes._build_vacuum_field(ods, field="psi", resolution=COARSE).extend == "neither"


def test_the_decay_index_marks_its_stable_band(ods):
    model = recipes._build_vacuum_field(ods, field="decay_index", resolution=COARSE)
    assert model.secondary_levels == recipes.DECAY_INDEX_STABLE_BAND


# ---------------------------------------------------------------------------
# Which instant
# ---------------------------------------------------------------------------

def test_time_index_selects_a_pf_sample(ods):
    base = np.asarray(ods["pf_active.time"], dtype=float)
    model = recipes._build_vacuum_field(ods, field="psi", time_index=1200, resolution=COARSE)
    assert f"{base[1200] * 1e3:.1f} ms" in model.title


def test_an_out_of_range_time_index_says_how_many_samples_there_are(ods):
    with pytest.raises(ValueError, match="stored PF samples"):
        recipes._build_vacuum_field(ods, field="psi", time_index=99999, resolution=COARSE)


def test_the_slider_starts_where_the_plot_does(ods, record):
    """The default instant is the breakdown onset, read from the magnetics.
    Discovery has to resolve it the same way the builder does, or opening the
    controls jumps the figure to a different time."""
    default_index = record.times["selected"]
    base = np.asarray(ods["pf_active.time"], dtype=float)
    drawn = recipes._build_vacuum_field(ods, field="psi", resolution=COARSE)
    assert f"{base[default_index] * 1e3:.1f} ms" in drawn.title


# ---------------------------------------------------------------------------
# What it offers to change
# ---------------------------------------------------------------------------

def test_a_dense_time_base_is_offered_as_a_slider(record):
    """Thousands of PF samples cannot be radio buttons, which is what a stored
    equilibrium's handful of slices gets."""
    controls = {control.name: control for control in controls_for(record)}
    assert "time_slice" not in controls
    slider = controls["time_index"]
    assert slider.kind == "range"
    assert slider.options == (0, record.times["count"] - 1, 1)
    assert slider.options[0] <= slider.default <= slider.options[1]
    assert "ms" in slider.label


def test_the_quantity_is_offered_with_its_own_vocabulary(record):
    controls = {control.name: control for control in controls_for(record)}
    assert controls["field"].options == recipes.VACUUM_FIELD_NAMES
    assert controls["field"].default == "psi"


def test_the_flux_units_are_offered_only_while_the_flux_is_drawn(record):
    """Gauss, V/m and dimensionless are three different vocabularies; a unit
    control that ignored which quantity is drawn would offer mWb for |B_p|."""
    controls = {control.name: control for control in controls_for(record)}
    assert controls["units"].applies_to == {"field": ("psi",)}


def test_a_vacuum_map_offers_no_flux_map_style(record):
    """The psi styles normalise against a separatrix this map does not have."""
    assert "style" not in {control.name for control in controls_for(record)}


def test_the_breakdown_figure_is_withheld_without_a_toroidal_field(ods):
    """A control that would raise when used is worse than one that is absent."""
    stripped = recipes._isolated_copy(ods, ("pf_active", "pf_passive", "wall", "equilibrium"))
    record = describe_one("vacuum_field", [("no tf", stripped)])
    # Both quantities that multiply by the toroidal field are withheld; |E_phi|
    # is the loop voltage over 2 pi R and never involves it, so it stays.
    assert "breakdown" not in record.fields["options"]
    assert "lloyd_margin" not in record.fields["options"]
    assert set(record.fields["options"]) == {"psi", "b_poloidal", "decay_index", "e_toroidal"}
    with pytest.raises(ValueError, match="tf.b_field_tor_vacuum_r"):
        recipes._build_vacuum_field(stripped, field="breakdown", resolution=COARSE)


def test_the_caller_s_ods_is_left_as_it_was_found(ods):
    """The evaluator solves the vessel currents; that must not land in the
    caller's ODS."""
    assert "time" not in ods["pf_passive"]
    recipes._build_vacuum_field(ods, field="psi", resolution=COARSE)
    assert "time" not in ods["pf_passive"]


def test_the_declared_ids_cover_what_the_default_instant_reads():
    """With no ``time=`` the map is drawn at the breakdown onset, and that
    timing reads H-alpha alongside the plasma current.  An adapter that loads
    only the declared IDSs then resolves a different instant than one handed
    the whole entry -- which is a difference in the drawn values, not just in
    the title.  ``equilibrium_field_psi_vacuum`` declares both for the same
    reason.
    """
    from vaft.plot.registry import get_spec

    declared = set(get_spec("vacuum_field").ids)
    assert {"magnetics", "spectrometer_uv"} <= declared
    # And the Lloyd margin's fill pressure: discovery offers that field only
    # when a gauge is present, so an undeclared barometry would hide it.
    assert "barometry" in declared
    assert declared >= set(get_spec("equilibrium_field_psi_vacuum").ids) - {"tf"}


# ---------------------------------------------------------------------------
# Reading a value off the map
# ---------------------------------------------------------------------------

def _labels_of(axes):
    """The contour labels `clabel` wrote onto an axes."""
    from matplotlib.text import Text

    return [
        child for child in axes.get_children()
        if isinstance(child, Text) and child.get_text() and child not in
        (axes.title, axes.xaxis.label, axes.yaxis.label)
    ]


def test_contour_labels_are_off_unless_asked_for(ods):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, axes = plt.subplots()
    vaft.omas.plot_vacuum_field(ods, field="b_poloidal", ax=axes, resolution=COARSE)
    assert not _labels_of(axes)
    plt.close("all")


def test_contour_labels_stay_few_enough_to_read(ods):
    """A psi map draws 40 levels; labelling every one is a smear, so only an
    evenly spaced handful gets a value written on it.

    Counted as distinct values, not as label objects: a level broken into
    several contour segments is labelled once per segment, which is correct --
    the same number reappearing on a disconnected branch is not clutter.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from vaft.plot.renderers.fields import _MAX_CONTOUR_LABELS

    _, axes = plt.subplots()
    vaft.omas.plot_vacuum_field(
        ods, field="psi", ax=axes, resolution=COARSE, label_contours=True
    )
    values = {label.get_text() for label in _labels_of(axes)}
    assert 0 < len(values) <= _MAX_CONTOUR_LABELS
    plt.close("all")


def test_a_contour_label_is_readable_on_a_filled_map(ods):
    """Left to itself a label takes the contour's colour, which on a filled map
    is the colour of what it is written on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, axes = plt.subplots()
    vaft.omas.plot_vacuum_field(
        ods, field="b_poloidal", ax=axes, resolution=COARSE, label_contours=True
    )
    labels = _labels_of(axes)
    assert labels
    for label in labels:
        assert label.get_path_effects(), "no halo behind the text"
    plt.close("all")


def test_label_contours_is_a_recognised_option(ods):
    """It reaches the renderer as a style keyword, so every field plot takes it
    and a misspelling is refused rather than swallowed."""
    from vaft.plot.backend.options import STYLE_OPTIONS

    assert "label_contours" in STYLE_OPTIONS
