"""The canonical DD path grammar and what every recipe declares (umbrella #434).

Two things are pinned here.  The grammar round-trips between the four
spellings of one path.  And every path every canonical plot declares exists
in the Data Dictionary and carries the units the recipe claims -- with an
allowlist of the known deviations, each with its reason, and a test that
fails the moment an allowlist entry stops being needed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from vaft.plot.backend import dd
from vaft.plot.backend import recipes as R
from vaft.plot.registry import canonical_names, get_spec

# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------

ROUND_TRIPS = [
    # template, canonical, info form, imas form
    ("magnetics.ip.{i}.data", "magnetics/ip(:)/data", "magnetics.ip.:.data", "ip/data"),
    (
        "equilibrium.time_slice.{i}.profiles_2d.0.psi",
        "equilibrium/time_slice(:)/profiles_2d(0)/psi",
        "equilibrium.time_slice.:.profiles_2d.:.psi",
        "time_slice/profiles_2d/psi",
    ),
    (
        "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw",
        "camera_visible/channel(:)/detector(:)/frame(:)/image_raw",
        "camera_visible.channel.:.detector.:.frame.:.image_raw",
        "channel/detector/frame/image_raw",
    ),
    (
        "wall.description_2d.0.limiter.unit.0.outline.r",
        "wall/description_2d(0)/limiter/unit(0)/outline/r",
        "wall.description_2d.:.limiter.unit.:.outline.r",
        "description_2d/limiter/unit/outline/r",
    ),
    (
        "pf_active.coil.{i}.element.0.geometry.rectangle.r",
        "pf_active/coil(:)/element(0)/geometry/rectangle/r",
        "pf_active.coil.:.element.:.geometry.rectangle.r",
        "coil/element/geometry/rectangle/r",
    ),
    (
        "pf_active.coil.{i}.element.:.turns_with_sign",
        "pf_active/coil(:)/element(:)/turns_with_sign",
        "pf_active.coil.:.element.:.turns_with_sign",
        "coil/element/turns_with_sign",
    ),
    ("magnetics.time", "magnetics/time", "magnetics.time", "time"),
]


@pytest.mark.parametrize("template, canonical, info, imas_form", ROUND_TRIPS)
def test_the_four_spellings_round_trip(template, canonical, info, imas_form):
    path = dd.from_template(template)
    assert path.canonical == canonical
    assert str(path) == canonical
    assert path.ids == template.split(".")[0]
    assert dd.to_omas_template(path) == template
    assert dd.to_info(path) == info
    assert dd.to_imas(path) == imas_form
    # The canonical spelling parses back to the same path.
    parsed = dd.parse(canonical)
    assert parsed == path
    assert dd.to_info(parsed) == info and dd.to_imas(parsed) == imas_form


def test_to_omas_fills_the_enumerated_indices():
    path = dd.from_template("camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw")
    assert dd.to_omas(path) == "camera_visible.channel.0.detector.0.frame.0.image_raw"
    assert dd.to_omas(path, 2) == "camera_visible.channel.2.detector.2.frame.2.image_raw"
    assert dd.to_omas(path, {"i": 1, "j": 2}) == "camera_visible.channel.1.detector.2.frame.0.image_raw"
    fixed = dd.from_template("equilibrium.time_slice.{i}.profiles_2d.0.psi")
    assert dd.to_omas(fixed, 4) == "equilibrium.time_slice.4.profiles_2d.0.psi"


def test_a_path_parsed_from_canonical_letters_its_template():
    path = dd.parse("camera_visible/channel(:)/detector(:)/frame(:)/image_raw")
    assert path.template == "camera_visible.channel.{i}.detector.{j}.frame.{k}.image_raw"
    assert dd.to_omas_template(dd.parse("equilibrium/time_slice(:)/profiles_2d(0)/psi")) == (
        "equilibrium.time_slice.{i}.profiles_2d.0.psi"
    )


@pytest.mark.parametrize(
    "bad",
    ["", "magnetics..ip", "magnetics.{i}.ip", "{i}.data", "Magnetics.ip", "magnetics.ip.{i}.{j}"],
)
def test_malformed_templates_are_refused(bad):
    with pytest.raises(ValueError):
        dd.from_template(bad)


@pytest.mark.parametrize("bad", ["", "magnetics(0)/ip", "magnetics/ip[:]/data", "magnetics/ip(x)/data"])
def test_malformed_canonical_paths_are_refused(bad):
    with pytest.raises(ValueError):
        dd.parse(bad)


def test_an_unknown_role_is_refused():
    with pytest.raises(ValueError, match="role"):
        dd.parse("magnetics/ip(:)/data", role="ordinate")


@pytest.mark.parametrize(
    "spelling, canonical",
    [
        ("A.m^-2", "A m^-2"), ("A/m^2", "A m^-2"), ("A m^-2", "A m^-2"),
        ("T.m", "T m"), ("T m", "T m"), ("Pa.Wb^-1", "Pa Wb^-1"), ("Pa/Wb", "Pa Wb^-1"),
        ("T^2.m^2/Wb", "T^2 Wb^-1 m^2"), ("T^2 m^2/Wb", "T^2 Wb^-1 m^2"),
        ("", ""), ("-", ""), ("1", ""), ("m.s^-1", "m s^-1"), ("m/s", "m s^-1"),
        ("A-turns", "A-turns"),
    ],
)
def test_units_normalise_to_one_spelling(spelling, canonical):
    assert dd.normalise_units(spelling) == canonical


# ---------------------------------------------------------------------------
# resolve
# ---------------------------------------------------------------------------


def test_resolve_reads_the_data_dictionary_without_data():
    info = dd.resolve("magnetics/ip(:)/data")
    assert info.units == "A"
    assert info.coordinates == ("magnetics/ip(:)/time",)
    assert info.data_type == "FLT_1D"
    assert info.dd_version == dd.DEFAULT_DD_VERSION
    assert info.lifecycle_status == "active"
    # A recipe template is accepted as well as the canonical spelling.
    assert dd.resolve("magnetics.ip.{i}.data") == info


def test_resolve_names_the_version_for_a_path_the_dd_lacks():
    with pytest.raises(KeyError, match=r"magnetics/bogus\(:\)/data is not in Data Dictionary 3\.41\.0"):
        dd.resolve("magnetics/bogus(:)/data")


def test_resolve_cross_checks_with_imas(monkeypatch):
    pytest.importorskip("imas")
    info = dd.resolve("magnetics/ip(:)/data")
    assert info.source == "omas+imas"
    assert dd.resolve("magnetics/ip(:)/data", cross_check=False).source == "omas"

    # The two Data Dictionary readers disagreeing is an error naming both.
    original = dd._omas_info

    def lying(info_path, dd_version):
        return {**original(info_path, dd_version), "units": "V"}

    monkeypatch.setattr(dd, "_omas_info", lying)
    with pytest.raises(ValueError, match="omas says units 'V', imas-python says 'A'"):
        dd.resolve("magnetics/ip(:)/data")


def test_the_grammar_module_loads_neither_data_model():
    code = (
        "import sys, vaft.plot.backend.dd as d; d.dd_paths('plasma_current_time'); "
        "print(sorted(m for m in ('omas', 'imas') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]", out.stdout


# ---------------------------------------------------------------------------
# dd_paths: what a plot declares
# ---------------------------------------------------------------------------


def _by_role(paths, role):
    return [p.canonical for p in paths if p.role == role]


def test_a_line_plot_declares_its_data_and_coordinates_with_roles():
    paths = dd.dd_paths("plasma_current_time")
    data = [p for p in paths if p.role == "data"]
    assert [p.canonical for p in data] == ["magnetics/ip(0)/data"]
    assert data[0].coordinate == "magnetics/ip(0)/time"
    assert data[0].fallback_coordinate == ("magnetics/time",)
    assert data[0].units == "A"
    assert data[0].template == "magnetics.ip.0.data"
    assert _by_role(paths, "coordinate") == ["magnetics/ip(0)/time", "magnetics/time"]
    # Data paths come first, then coordinates.
    assert [p.role for p in paths][:1] == ["data"]


def test_a_field_map_declares_its_grid_and_boundary():
    paths = dd.dd_paths("equilibrium_field_psi")
    assert _by_role(paths, "data") == ["equilibrium/time_slice(:)/profiles_2d(0)/psi"]
    assert {p.attrs["axis"] for p in paths if p.role == "coordinate"} == {"r", "z"}
    assert _by_role(paths, "boundary") == [
        "equilibrium/time_slice(:)/boundary/outline/r",
        "equilibrium/time_slice(:)/boundary/outline/z",
    ]
    assert paths[0].units == "Wb"


def test_a_profile_declares_every_coordinate_it_can_be_drawn_against():
    paths = dd.dd_paths("equilibrium_profile_q")
    assert _by_role(paths, "data") == ["equilibrium/time_slice(:)/profiles_1d/q"]
    declared = {p.attrs["coordinate"] for p in paths if p.role == "coordinate"}
    offered = set(R.coordinate_options_for("equilibrium_profile_q"))
    assert offered <= {name for key in declared for name in key.split("|")}
    assert paths[0].coordinate == "equilibrium/time_slice(:)/profiles_1d/rho_tor_norm"
    assert "equilibrium/time_slice(:)/profiles_1d/r_outboard" in _by_role(paths, "coordinate")


def test_a_field_map_declares_every_field_it_can_draw():
    paths = dd.dd_paths("equilibrium_field_2d")
    fields = {p.attrs["field"]: p.canonical for p in paths if "field" in p.attrs}
    assert set(fields) == set(R.EQUILIBRIUM_FIELD_NAMES) - {"psi"}
    assert fields["j_tor"] == "equilibrium/time_slice(:)/profiles_2d(0)/j_tor"
    assert fields["pressure"] == "equilibrium/time_slice(:)/profiles_1d/pressure"
    assert dd.dd_paths("equilibrium_field_psi") and not any(
        "field" in p.attrs for p in dd.dd_paths("equilibrium_field_psi")
    )


def test_a_composite_keeps_the_most_specific_role_and_every_member():
    # tf/coil(:)/current/data is data for tf_coil_time_current and merely
    # optional for impa_profile_field; the overview says data, and names both.
    paths = {p.canonical: p for p in dd.dd_paths("impa_overview")}
    path = paths["tf/coil(:)/current/data"]
    assert path.role == "data"
    assert set(path.attrs["members"]) >= {"tf_coil_time_current", "impa_profile_field"}


def test_a_composite_lists_its_members_paths_and_names_the_member():
    recipe = R.RECIPES["summary_time_energy"]
    assert isinstance(recipe, R.PanelRecipe)
    paths = dd.dd_paths("summary_time_energy")
    members = {p.attrs.get("member") for p in paths}
    assert set(recipe.members) <= members
    for member in recipe.members:
        assert {p.canonical for p in dd.dd_paths(member)} <= {p.canonical for p in paths}


@pytest.mark.parametrize("name, backend", [("vacuum_field", "omas"), ("pf_coil_geometry_poloidal", "neutral")])
def test_a_computed_view_declares_its_reads_beside_its_gate(name, backend):
    recipe = R.RECIPES[name]
    assert isinstance(recipe, R.CallableRecipe) and recipe.backend == backend
    paths = dd.dd_paths(name)
    by_canonical = {p.canonical: p for p in paths}
    assert {p.attrs["declared_by"] for p in paths} <= {"spec", "recipe", "spec+recipe"}
    assert "recipe" in {p.attrs["declared_by"] for p in paths}
    spec = get_spec(name)
    for template in spec.required_paths:
        assert by_canonical[dd.from_template(template).canonical].role == "required"
    for template in recipe.reads:
        path = by_canonical[dd.from_template(template).canonical]
        assert "recipe" in path.attrs["declared_by"]
        assert path.role == "input" or path.attrs["declared_by"] == "spec+recipe"
        assert path.attrs["backend"] == backend


def test_a_geometry_layer_caption_is_not_a_path():
    # Some geometry layers label themselves with a literal caption; those
    # never become paths.
    paths = dd.dd_paths("magnetics_geometry_poloidal")
    assert all("/" in p.canonical for p in paths)
    assert {p.attrs.get("axis") for p in paths if p.role == "geometry"} == {"r", "z"}


def test_every_canonical_plot_has_paths_with_one_entry_per_spelling():
    for name in canonical_names():
        paths = dd.dd_paths(name)
        assert paths, name
        spellings = [p.canonical for p in paths]
        assert len(spellings) == len(set(spellings)), name
        assert all(p.role in dd.ROLES for p in paths), name


def test_an_unknown_plot_is_refused():
    with pytest.raises(KeyError, match="no plot named"):
        dd.dd_paths("no_such_plot")


# ---------------------------------------------------------------------------
# every declared path is in the Data Dictionary, with the units it claims
# ---------------------------------------------------------------------------

#: Declared paths the Data Dictionary 3.41.0 does not define, with the reason
#: each is still declared.  A ``data`` path may only stand here as a VAFT
#: extension leaf; the rest are legacy spellings read as fallbacks.
MISSING_FROM_DD: dict[str, str] = {
    "equilibrium/time_slice(:)/global_quantities/energy_mag": (
        "legacy VEST extension leaf (magnetic energy) with no producer in vaft; "
        "absent from DD 3.41.0 and 4.1.1"
    ),
    "equilibrium/time_slice(:)/global_quantities/energy_total": (
        "legacy VEST extension leaf (total energy) with no producer in vaft; "
        "absent from DD 3.41.0 and 4.1.1"
    ),
    "equilibrium/time_slice(:)/global_quantities/qa": (
        "legacy VEST extension leaf (edge safety factor) with no producer in vaft; "
        "absent from DD 3.41.0 and 4.1.1"
    ),
    "magnetics/flux_loop/time": (
        "legacy VAFT timebase spelling read as a fallback coordinate; the DD timebase is magnetics/time"
    ),
    "magnetics/b_field_pol_probe/time": (
        "legacy VAFT timebase spelling read as a fallback coordinate; the DD timebase is magnetics/time"
    ),
    "equilibrium/time_slice(:)/profiles_1d/pprime": (
        "legacy leaf name read as a fallback for profiles_1d/dpressure_dpsi"
    ),
    "equilibrium/time_slice(:)/profiles_1d/ffprime": (
        "legacy leaf name read as a fallback for profiles_1d/f_df_dpsi"
    ),
    "pf_passive/loop(:)/identifier": (
        "VEST's passive-structure mapping writes an identifier beside name; DD 3.41.0 defines "
        "none on pf_passive.loop; read by the channel selection policy"
    ),
    "magnetics/ip(:)/validity": (
        "DD 3.41.0 carries no validity on magnetics.ip; VAFT's validation layer writes it (#253) "
        "and the plasma-onset finder reads it"
    ),
    "equilibrium/ids_properties/cocos": (
        "VAFT probe for a COCOS hint beside the DD's code.parameters; not a DD leaf"
    ),
    "magnetics/ip(:)/validity_timed": (
        "DD 3.41.0 carries no validity_timed on magnetics.ip; VAFT's validation layer writes it "
        "(#253) and the plasma-onset finder reads it"
    ),
    "equilibrium/time_slice(:)/global_quantities/major_radius": (
        "probed by vaft.process.equilibrium.as_equilibrium as a per-slice r0 fallback before "
        "vacuum_toroidal_field.r0; not a DD leaf"
    ),
    "spectrometer_uv/channel(:)/processed_line(:)/intensity/validity": (
        "DD 3.41.0 carries no validity on processed_line.intensity; VAFT's validation layer "
        "writes it (#253) and the plasma-onset finder reads it"
    ),
    "spectrometer_uv/channel(:)/processed_line(:)/intensity/validity_timed": (
        "DD 3.41.0 carries no validity_timed on processed_line.intensity; VAFT's validation "
        "layer writes it (#253) and the plasma-onset finder reads it"
    ),
    "equilibrium/time_slice(:)/global_quantities/b0": (
        "probed by vaft.process.equilibrium.as_equilibrium as a per-slice b0 fallback before "
        "vacuum_toroidal_field.b0; not a DD leaf"
    ),
}

#: ``(canonical path, declared unit)`` whose declared unit is not the Data
#: Dictionary's, with the reason.
UNIT_DEVIATIONS: dict[tuple[str, str], str] = {
    ("soft_x_rays/channel(:)/brightness/data", "a.u."): (
        "VEST soft X-ray channels are uncalibrated; the DD unit W.m^-2.sr^-1 would be a lie"
    ),
    ("soft_x_rays/channel(:)/power/data", "a.u."): "VEST soft X-ray channels are uncalibrated (DD: W)",
    ("spectrometer_uv/channel(:)/processed_line(0)/intensity/data", "a.u."): (
        "VEST UV spectrometer intensities are uncalibrated counts (DD: s^-1)"
    ),
    ("pf_active/coil(:)/current/data", "A-turns"): (
        "A-turns is the display unit of a coil current weighted by its dimensionless turn count "
        "(pf_coil_time_current_turns reads pf_active/coil(:)/element(:)/turns_with_sign as weight)"
    ),
}


def _divisor_units() -> dict[tuple[str, str], str]:
    """``(canonical data path, declared unit)`` -> DD units of the scalar it is divided by."""
    out: dict[tuple[str, str], str] = {}
    for recipe in R.RECIPES.values():
        divisor = getattr(recipe, "divide_by_path", "")
        if divisor:
            out[(dd.from_template(recipe.y_path).canonical, recipe.y_unit)] = dd.resolve(divisor).units
    return out


def _declared() -> dict[tuple[str, str], dd.DDPath]:
    """Every distinct ``(canonical spelling, declared unit)`` pair."""
    out: dict[tuple[str, str], dd.DDPath] = {}
    for name in canonical_names():
        for path in dd.dd_paths(name):
            out.setdefault((path.canonical, path.units), path)
    return out


@pytest.mark.parametrize("name", canonical_names())
def test_every_declared_path_exists_in_the_data_dictionary(name):
    for path in dd.dd_paths(name):
        if "/" not in path.canonical:
            # A bare IDS root: a computed view that deep-copies the whole IDS
            # declares it as such (issue #439); the IDS exists by construction.
            assert path.role == "input" and path.attrs["backend"] == "omas", path.canonical
            continue
        if path.canonical in MISSING_FROM_DD:
            assert path.role != "data" or "extension" in MISSING_FROM_DD[path.canonical], path.canonical
            continue
        info = dd.resolve(path)
        assert info.units is not None, path.canonical


def test_every_declared_unit_agrees_with_the_data_dictionary():
    divisors = _divisor_units()
    disagreements = []
    for (canonical, _unit), path in _declared().items():
        if not path.units or canonical in MISSING_FROM_DD:
            continue
        expected = dd.resolve(path).units
        declared = path.units
        if (canonical, path.units) in divisors:
            # The recipe divides the leaf by a scalar: declared * divisor = DD.
            declared = f"{declared} {divisors[(canonical, path.units)]}"
        if dd.normalise_units(declared) == dd.normalise_units(expected):
            continue
        if (canonical, path.units) in UNIT_DEVIATIONS:
            continue
        disagreements.append((canonical, path.units, expected))
    assert not disagreements, disagreements


def test_the_allowlists_are_still_needed():
    stale = []
    declared = _declared()
    still_declared = {canonical for canonical, _ in declared}
    for canonical, reason in MISSING_FROM_DD.items():
        if canonical not in still_declared:
            stale.append(f"{canonical} is no longer declared by any plot; remove the entry ({reason})")
            continue
        try:
            dd.resolve(canonical)
        except KeyError:
            continue
        stale.append(f"{canonical} is now in the DD; remove the entry ({reason})")
    divisors = _divisor_units()
    for (canonical, unit), reason in UNIT_DEVIATIONS.items():
        path = declared.get((canonical, unit))
        if path is None:
            stale.append(f"{canonical} no longer declares {unit!r}; remove the entry ({reason})")
            continue
        expected = dd.resolve(path).units
        declared_unit = f"{unit} {divisors[(canonical, unit)]}" if (canonical, unit) in divisors else unit
        if dd.normalise_units(declared_unit) == dd.normalise_units(expected):
            stale.append(f"{canonical} now agrees with the DD ({expected!r}); remove the entry")
    assert not stale, stale


def test_a_divided_leaf_is_compared_against_the_product():
    # tf_coil_time_b_t reads b_field_tor_vacuum_r [T.m] and divides by r0 [m]:
    # the declared T is the DD unit over the divisor's, not a deviation.
    recipe = R.RECIPES["tf_coil_time_b_t"]
    assert recipe.divide_by_path
    canonical = dd.from_template(recipe.y_path).canonical
    assert (canonical, recipe.y_unit) not in UNIT_DEVIATIONS
    assert dd.normalise_units(f"{recipe.y_unit} {dd.resolve(recipe.divide_by_path).units}") == (
        dd.normalise_units(dd.resolve(canonical).units)
    )
