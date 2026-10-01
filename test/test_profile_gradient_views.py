"""The profile gradient views on the vaft.plot backend (issue #551, PR 2).

``electron_temperature_profile_gradient``, ``electron_density_profile_gradient``
and ``ion_temperature_profile_gradient`` call
:func:`vaft.process.profile_gradients.profile_gradient` once, in the builder;
the model carries the values, the label and the resolved record, and the
renderer only draws it.  The packaged 48224 sample carries one core_profiles
slice and one equilibrium slice at 300 ms.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import vaft
import vaft.omas
import vaft.process.profile_gradients as gradients
from vaft.omas.entries import normalize_entries
from vaft.plot.backend import recipes as R
from vaft.plot.backend.options import validate_options
from vaft.plot.display import gradient_label
from vaft.plot.models import Profile1D

from _read_recorder import assert_models_equal

VIEWS = (
    "electron_temperature_profile_gradient",
    "electron_density_profile_gradient",
    "ion_temperature_profile_gradient",
)


@pytest.fixture(scope="module")
def ods():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(48224, representation="omas"))


def _extract(name, ods, **options):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(vaft.omas, f"extract_{name}")(ods, **options)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", VIEWS)
def test_plot_and_extract_produce_the_same_model(name, ods, monkeypatch):
    import vaft.plot.renderers.profiles as profiles

    drawn = []
    original = profiles.render_profile_1d

    def capture(model, **kwargs):
        drawn.append(model)
        return original(model, **kwargs)

    monkeypatch.setattr(profiles, "render_profile_1d", capture)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        getattr(vaft.omas, f"plot_{name}")(ods, coordinate="r_minor_norm")
    assert len(drawn) == 1
    assert_models_equal(drawn[0], _extract(name, ods, coordinate="r_minor_norm"), where=name)


def test_the_renderer_recomputes_nothing(ods, monkeypatch):
    model = _extract("ion_temperature_profile_gradient", ods)

    def forbidden(*args, **kwargs):
        raise AssertionError("the renderer called profile_gradient")

    monkeypatch.setattr(gradients, "profile_gradient", forbidden)
    monkeypatch.setattr(gradients, "radial_coordinate_map", forbidden)
    figure, axes = vaft.plot.ion_temperature_profile_gradient(model)
    line = axes.get_lines()[0]
    np.testing.assert_array_equal(line.get_ydata(), model.series[0].y)
    assert axes.get_ylabel() == model.y_label


def test_the_view_is_the_process_result_on_the_profiles_own_grid(ods):
    model = _extract("electron_temperature_profile_gradient", ods, coordinate="psi_norm")
    base = "core_profiles.profiles_1d.0"
    te = np.asarray(ods[f"{base}.electrons.temperature"], dtype=float)
    slice_ = ods["equilibrium.time_slice.0"]
    psi_n = (np.asarray(ods[f"{base}.grid.psi"]) - slice_["global_quantities.psi_axis"]) / (
        slice_["global_quantities.psi_boundary"] - slice_["global_quantities.psi_axis"])
    keep = te > 0
    expected = gradients.profile_gradient(
        te[keep], psi_n[keep], "psi_norm", equilibrium=ods, time=0.3,
        gradient_coordinate="r_minor", reference_length="a_minor")
    np.testing.assert_allclose(model.series[0].y, expected.values)
    np.testing.assert_allclose(model.series[0].x, expected.coordinate)
    # the LCFS sample where T_e = 0 is left out, and said to be
    assert int(np.count_nonzero(~keep)) == 1
    assert model.metadata["excluded_points"]["edge_side"] == {"non_positive": 1, "non_finite": 0}
    assert model.metadata["excluded_points"]["axis_side"] == {"non_positive": 0, "non_finite": 0}
    assert model.title.endswith("left out: 1 edge-side point with T_e <= 0")
    assert model.metadata["profile_grid"] == f"{base}.grid.psi"
    assert model.metadata["profile_coordinate"] == "psi_norm"


def test_the_record_reaches_the_xarray_attributes(ods):
    model = _extract("ion_temperature_profile_gradient", ods, coordinate="rho_tor_norm", convention="tglf")
    attrs = model.to_xarray().attrs
    record = json.loads(attrs["metadata"])
    assert record == json.loads(json.dumps(model.metadata))
    assert record["requested_convention"] == "tglf"
    assert record["resolved_convention"]["gradient_coordinate"] == "r_minor"
    assert record["reference_length"]["name"] == "a_minor"
    assert record["reference_length"]["unit"] == "m"
    assert record["mathematical_definition"] == "-a * d(log(T_i)) / d(r_minor)"
    assert record["plot_coordinate"] == "rho_tor_norm"
    assert attrs["y_label"] == r"$a/L_{T_i}$"


def test_the_tglf_convention_is_the_explicit_r_minor_and_a_minor_call(ods):
    preset = _extract("ion_temperature_profile_gradient", ods, convention="tglf")
    explicit = _extract("ion_temperature_profile_gradient", ods,
                        gradient_coordinate="r_minor", reference_length="a_minor")
    np.testing.assert_array_equal(preset.series[0].y, explicit.series[0].y)
    np.testing.assert_array_equal(preset.series[0].x, explicit.series[0].x)
    assert preset.y_label == explicit.y_label
    assert preset.metadata["requested_convention"] == "tglf"
    assert explicit.metadata["requested_convention"] is None


# ---------------------------------------------------------------------------
# notation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reference, label, unit", [
    ("a_minor", r"$a/L_{T_i}$", ""),
    ("R_major_axis", r"$R_0/L_{T_i}$", ""),
    ("none", r"$-\partial \ln T_i/\partial r$", "m^-1"),
])
def test_the_label_follows_the_record(ods, reference, label, unit):
    model = _extract("ion_temperature_profile_gradient", ods, reference_length=reference)
    assert (model.y_label, model.y_unit) == (label, unit)
    assert gradient_label(model.metadata) == (label, unit)
    assert model.metadata["unit"] == ("m^-1" if reference == "none" else "1")


def test_the_ratio_of_two_reference_lengths_is_their_ratio(ods):
    a = _extract("ion_temperature_profile_gradient", ods, reference_length="a_minor")
    r0 = _extract("ion_temperature_profile_gradient", ods, reference_length="R_major_axis")
    ratio = r0.metadata["reference_length"]["value"] / a.metadata["reference_length"]["value"]
    np.testing.assert_allclose(r0.series[0].y, ratio * a.series[0].y)


@pytest.mark.parametrize("name, subscript", [
    ("electron_temperature_profile_gradient", "T_e"),
    ("electron_density_profile_gradient", "n_e"),
    ("ion_temperature_profile_gradient", "T_i"),
])
def test_each_profile_has_its_own_subscript(ods, name, subscript):
    assert _extract(name, ods).y_label == rf"$a/L_{{{subscript}}}$"


def test_a_gradient_coordinate_other_than_r_minor_is_named():
    record = {"source_quantity": "T_e", "gradient_coordinate": "r_outboard", "unit": "1",
              "reference_length": {"name": "a_minor", "symbol": "a"}}
    assert gradient_label(record) == (r"$a/L_{T_e}$ (along $R_{out}$)", "")


# ---------------------------------------------------------------------------
# discovery and options
# ---------------------------------------------------------------------------


def test_discovery_lists_the_four_options(ods):
    from vaft.plot.backend.discovery import describe_one
    from vaft.plot.controls import controls_for

    record = describe_one("ion_temperature_profile_gradient", normalize_entries(ods))
    assert record.available
    assert record.coordinates["default"] == "rho_tor_norm"
    assert set(record.coordinates["options"]) == set(gradients.RADIAL_COORDINATES)
    assert record.choices["gradient_coordinate"]["default"] == "r_minor"
    assert set(record.choices["gradient_coordinate"]["options"]) == set(gradients.RADIAL_COORDINATES)
    assert record.choices["reference_length"]["default"] == "a_minor"
    assert record.choices["reference_length"]["options"] == tuple(
        "none" if name is None else name for name in gradients.REFERENCE_LENGTHS)
    # only the presets a plot option can carry are offered
    assert record.choices["convention"] == {"default": None, "options": ("tglf", "cgyro")}
    assert record.choices["reference_length"]["applies_to"] == {"convention": ("none",)}
    assert record.computation["backend"] == "omas"
    controls = {control.name: control for control in controls_for(record, include_style=False)}
    assert {"coordinate", "gradient_coordinate", "reference_length", "convention"} <= set(controls)
    # "none" is a value of reference_length, and the absence of a convention
    assert controls["reference_length"].keeps_none and not controls["convention"].keeps_none


def test_the_controls_send_none_only_where_it_is_a_value():
    from vaft.plot.controls import ControlSpec
    from vaft.plot.navigation import ControlState

    state = ControlState((
        ControlSpec("reference_length", "choice", "Reference length", "none", ("none", "a_minor"), keeps_none=True),
        ControlSpec("convention", "choice", "Convention", "none", ("none", "tglf")),
    ))
    assert state.as_options() == {"reference_length": "none"}


def test_the_option_vocabularies_are_validated():
    validate_options("ion_temperature_profile_gradient",
                     {"coordinate": "r_minor_norm", "gradient_coordinate": "r_outboard",
                      "reference_length": "none", "convention": "cgyro"})
    for key, value in (("coordinate", "r_major"), ("gradient_coordinate", "a_minor"),
                       ("reference_length", "minor"), ("convention", "tglf2")):
        with pytest.raises(ValueError, match=key):
            validate_options("ion_temperature_profile_gradient", {key: value})


def test_the_profile_fits_keep_their_coordinate_vocabulary():
    for name in ("thomson_scattering_profile_fit", "charge_exchange_profile_fit"):
        assert R.coordinate_options_for(name) == R.PROFILE_FIT_COORDINATES == (
            "psi_norm", "rho_tor_norm", "rho_pol_norm", "r_major")
        assert R.coordinate_default_for(name) == "psi_norm"
        validate_options(name, {"coordinate": "r_major"})
        with pytest.raises(ValueError, match="coordinate"):
            validate_options(name, {"coordinate": "r_minor"})
    # a computed view without a declaration takes no coordinate=
    assert R.coordinate_options_for("neoclassical_profile_bootstrap_current") is None


def test_a_choice_declaration_refuses_a_default_outside_its_options():
    with pytest.raises(ValueError, match="default"):
        R.ChoiceDeclaration("r_major", ("psi_norm",))
    with pytest.raises(ValueError, match="at least one"):
        R.ChoiceDeclaration(None, ())


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("options, reason", [
    ({"convention": "gs2"}, "convention must be one of tglf, cgyro"),
    ({"reference_length": "L_ref"}, "stated length and definition"),
    ({"reference_length": None}, "reference_length='none'"),
    ({"gradient_coordinate": "psi_norm"}, r"not an outward length, so reference_length='a_minor' \(the default\)"),
    ({"gradient_coordinate": "rho_tor_norm", "reference_length": "R_major_axis"}, "not an outward length"),
    ({"convention": "tglf", "reference_length": "R_major_axis"}, "contradicts"),
    ({"time_slice": 3}, "outside the 1 core_profiles.profiles_1d"),
])
def test_refusals_carry_their_reasons(ods, options, reason):
    with pytest.raises(ValueError, match=reason):
        _extract("ion_temperature_profile_gradient", ods, **options)


def test_an_electron_view_refuses_ion_index(ods):
    with pytest.raises(ValueError, match="ion_index"):
        _extract("electron_density_profile_gradient", ods, ion_index=0)


def _variant(ods, edit):
    copied = copy.deepcopy(ods)
    edit(copied)
    return copied


def test_an_equilibrium_at_another_time_is_refused_not_paired_by_index(ods):
    def shift(o):
        o["core_profiles.profiles_1d.0.time"] = 0.5
        o["core_profiles.time"] = np.array([0.5])

    with pytest.raises(ValueError, match="no equilibrium slice at the profile's t = 0.5"):
        _extract("ion_temperature_profile_gradient", _variant(ods, shift))


def test_a_gap_inside_the_profile_is_refused(ods):
    def hole(o):
        values = np.asarray(o["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float).copy()
        values[40] = 0.0
        o["core_profiles.profiles_1d.0.ion.0.temperature"] = values

    with pytest.raises(ValueError, match="inside the profile"):
        _extract("ion_temperature_profile_gradient", _variant(ods, hole))


def test_a_profile_without_a_grid_is_not_offered(ods):
    from vaft.plot.backend.recipes import missing_required_path

    def strip(o):
        for leaf in ("grid.psi", "grid.rho_tor_norm"):
            del o[f"core_profiles.profiles_1d.0.{leaf}"]

    assert missing_required_path(_variant(ods, strip), "ion_temperature_profile_gradient") is not None
    assert missing_required_path(ods, "ion_temperature_profile_gradient") is None


# ---------------------------------------------------------------------------
# OMAS and native IMAS
# ---------------------------------------------------------------------------


def test_omas_and_native_imas_give_equal_models(ods, tmp_path):
    imas = pytest.importorskip("imas")
    import vaft.imas
    from omas import ODS

    subset = ODS(consistency_check=False)
    for root in ("core_profiles", "equilibrium", "dataset_description"):
        subset[root] = copy.deepcopy(ods[root])
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vaft.imas.save(subset, tmp_path / "gradient.nc")
    with imas.DBEntry(str(tmp_path / "gradient.nc"), "r", dd_version="3.41.0") as entry:
        for name in VIEWS:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                native = getattr(vaft.imas, f"extract_{name}")(entry, coordinate="r_minor")
            assert isinstance(native, Profile1D)
            assert_models_equal(native, _extract(name, ods, coordinate="r_minor"), where=name)


@pytest.mark.parametrize("convention, reason", [("gs2", "irho, local_eq"), ("gkw", "geom_type"),
                                                ("gene", "a plot option cannot carry")])
def test_a_run_dependent_preset_is_refused_by_the_builder_with_its_reason(ods, convention, reason):
    with warnings.catch_warnings(), pytest.raises(ValueError, match=reason):
        warnings.simplefilter("ignore")
        R.build_model("ion_temperature_profile_gradient", normalize_entries(ods), convention=convention)


def test_a_flux_label_gradient_is_drawn_dimensionless_with_none(ods):
    model = _extract("ion_temperature_profile_gradient", ods, gradient_coordinate="rho_tor_norm",
                     reference_length="none")
    assert model.metadata["unit"] == "1" and model.y_unit == ""
    assert model.y_label == r"$-\partial \ln T_i/\partial \rho_{tor,N}$"


def test_a_given_title_keeps_the_record(ods):
    model = _extract("electron_temperature_profile_gradient", ods, title="mine")
    assert model.title == "mine"
    assert model.metadata["excluded_points"]["edge_side"]["non_positive"] == 1
    assert json.loads(model.to_xarray().attrs["metadata"])["excluded_points"] == model.metadata["excluded_points"]


def test_non_finite_and_axis_side_samples_are_counted_apart(ods):
    def ends(o):
        values = np.asarray(o["core_profiles.profiles_1d.0.ion.0.temperature"], dtype=float).copy()
        values[0] = np.nan
        values[1] = -1.0
        values[-1] = np.nan
        o["core_profiles.profiles_1d.0.ion.0.temperature"] = values

    model = _extract("ion_temperature_profile_gradient", _variant(ods, ends))
    assert model.metadata["excluded_points"]["axis_side"] == {"non_positive": 1, "non_finite": 1}
    assert model.metadata["excluded_points"]["edge_side"] == {"non_positive": 0, "non_finite": 1}
    assert ("left out: 1 axis-side point with T_i <= 0, 1 axis-side point non-finite, "
            "1 edge-side point non-finite") in model.title


# ---------------------------------------------------------------------------
# the profile's grid
# ---------------------------------------------------------------------------


def _psi_norm(o):
    slice_ = o["equilibrium.time_slice.0"]
    psi = np.asarray(o["core_profiles.profiles_1d.0.grid.psi"], dtype=float)
    return (psi - slice_["global_quantities.psi_axis"]) / (
        slice_["global_quantities.psi_boundary"] - slice_["global_quantities.psi_axis"])


def test_the_rho_pol_norm_grid_maps_exactly_to_psi_norm(ods):
    def to_rho_pol(o):
        o["core_profiles.profiles_1d.0.grid.rho_pol_norm"] = np.sqrt(np.clip(_psi_norm(o), 0.0, None))
        del o["core_profiles.profiles_1d.0.grid.psi"]

    model = _extract("ion_temperature_profile_gradient", _variant(ods, to_rho_pol))
    reference = _extract("ion_temperature_profile_gradient", ods)
    assert model.metadata["profile_grid"].endswith("grid.rho_pol_norm")
    assert model.metadata["profile_coordinate"] == "rho_pol_norm"
    np.testing.assert_allclose(model.series[0].y, reference.series[0].y, rtol=1e-6, atol=1e-9)


def test_a_stored_rho_tor_norm_grid_is_used_once_verified(ods):
    def drop_psi(o):
        del o["core_profiles.profiles_1d.0.grid.psi"]

    model = _extract("ion_temperature_profile_gradient", _variant(ods, drop_psi))
    reference = _extract("ion_temperature_profile_gradient", ods)
    check = model.metadata["profile_grid_check"]
    assert model.metadata["profile_grid"].endswith("grid.rho_tor_norm")
    assert check["proxy"] is False and check["tolerance"] == R.RHO_TOR_GRID_TOLERANCE
    assert 0.0 < check["max_abs_difference_from_map_rho_tor_norm"] < R.RHO_TOR_GRID_TOLERANCE
    np.testing.assert_allclose(model.series[0].y, reference.series[0].y, rtol=1e-9, atol=1e-12)


def test_a_rho_tor_norm_grid_that_is_the_sqrt_psi_proxy_is_refused(ods):
    def proxy(o):
        rho = np.sqrt(np.clip(_psi_norm(o), 0.0, None))
        o["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = rho
        o["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = rho
        del o["core_profiles.profiles_1d.0.grid.psi"]

    with pytest.raises(ValueError, match="sqrt\\(psi_N\\), the rho_pol proxy"):
        _extract("ion_temperature_profile_gradient", _variant(ods, proxy))


def test_a_rho_tor_norm_grid_far_from_the_map_is_refused_with_the_mismatch(ods):
    def shifted(o):
        rho = np.asarray(o["core_profiles.profiles_1d.0.grid.rho_tor_norm"], dtype=float)
        bent = rho ** 1.5
        o["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = bent
        o["equilibrium.time_slice.0.profiles_1d.rho_tor_norm"] = bent
        del o["core_profiles.profiles_1d.0.grid.psi"]

    with pytest.raises(ValueError, match=r"differs from the map's q-integrated rho_tor_norm by up to 0\.\d+ \(tolerance 0\.05\)"):
        _extract("ion_temperature_profile_gradient", _variant(ods, shifted))


def test_a_rho_tor_norm_grid_that_cannot_be_checked_is_refused(ods):
    def unpaired(o):
        rho = np.asarray(o["core_profiles.profiles_1d.0.grid.rho_tor_norm"], dtype=float)
        o["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = rho * 0.999
        del o["core_profiles.profiles_1d.0.grid.psi"]

    with pytest.raises(ValueError, match="cannot be checked"):
        _extract("ion_temperature_profile_gradient", _variant(ods, unpaired))


# ---------------------------------------------------------------------------
# pairing by time
# ---------------------------------------------------------------------------


def test_the_equilibrium_is_paired_by_time_in_a_multi_slice_shot(ods):
    """Slice 0 is a 10%-larger copy at 0.25 s; the profile at 0.30 s must use slice 1."""

    def two_slices(o):
        original = copy.deepcopy(o["equilibrium.time_slice.0"])
        o["equilibrium.time_slice.1"] = original
        first = "equilibrium.time_slice.0"
        o[f"{first}.time"] = 0.25
        for path in ("profiles_2d.0.grid.dim1", "global_quantities.magnetic_axis.r", "boundary.outline.r"):
            o[f"{first}.{path}"] = np.asarray(o[f"{first}.{path}"], dtype=float) * 1.1
        o["equilibrium.time"] = np.array([0.25, 0.30])
        b0 = np.asarray(o["equilibrium.vacuum_toroidal_field.b0"], dtype=float).reshape(-1)
        o["equilibrium.vacuum_toroidal_field.b0"] = np.array([b0[0], b0[0]])

    shot = _variant(ods, two_slices)
    paired = _extract("ion_temperature_profile_gradient", shot, reference_length="none")
    single = _extract("ion_temperature_profile_gradient", ods, reference_length="none")
    assert paired.metadata["equilibrium_time_slice"] == 1
    assert paired.metadata["time"] == pytest.approx(0.30)
    np.testing.assert_allclose(paired.series[0].y, single.series[0].y)

    shot["core_profiles.profiles_1d.0.time"] = 0.25
    shot["core_profiles.time"] = np.array([0.25])
    early = _extract("ion_temperature_profile_gradient", shot, reference_length="none")
    assert early.metadata["equilibrium_time_slice"] == 0
    # every length is 10% larger, so -d ln T/dr is 10% smaller
    np.testing.assert_allclose(early.series[0].y, single.series[0].y / 1.1, rtol=1e-3, atol=1e-6)


# ---------------------------------------------------------------------------
# controls
# ---------------------------------------------------------------------------


def test_choosing_a_convention_drops_the_view_defaults(ods):
    from vaft.plot.backend.discovery import describe_one
    from vaft.plot.controls import controls_for
    from vaft.plot.navigation import ControlState

    record = describe_one("ion_temperature_profile_gradient", normalize_entries(ods))
    assert record.choices["convention"]["options"] == ("tglf", "cgyro")
    controls = tuple(c for c in controls_for(record, include_style=False)
                     if c.name in ("gradient_coordinate", "reference_length", "convention"))
    state = ControlState(controls)
    assert state.as_options() == {"gradient_coordinate": "r_minor", "reference_length": "a_minor"}
    state.set("convention", "cgyro")
    assert state.as_options() == {"convention": "cgyro"}
    model = _extract("ion_temperature_profile_gradient", ods, **state.as_options())
    assert model.metadata["requested_convention"] == "cgyro"


def test_the_profile_fits_now_offer_their_coordinates_in_discovery_and_controls():
    """Additive since #551: the fits' coordinate= is declared, so discovery and controls see it."""
    import omas
    from vaft.data import data_path
    from vaft.plot.backend.discovery import describe_one
    from vaft.plot.controls import controls_for

    kinetic = omas.load_omas_json(str(data_path("kineticEfit/ods_48224_300ms.json")), consistency_check=False)
    for name in ("thomson_scattering_profile_fit", "charge_exchange_profile_fit"):
        record = describe_one(name, normalize_entries(kinetic))
        assert record.coordinates == {"default": "psi_norm", "options": R.PROFILE_FIT_COORDINATES,
                                      "declared": R.PROFILE_FIT_COORDINATES}
        control = next(c for c in controls_for(record, include_style=False) if c.name == "coordinate")
        assert control.default == "psi_norm" and control.options == R.PROFILE_FIT_COORDINATES
