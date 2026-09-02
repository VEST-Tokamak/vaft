"""Semantic plot discovery (issue #262).

``available_plots`` returns a catalog that prints as a ``subject / view /
[quantity]`` tree and iterates as records.  Discovery reports the policies
decided in #251 (taxonomy), #256 (display), #259 (selection) and #260
(layout) and decides nothing of its own: every instance-level answer comes
from the helper the adapters run, so the listing and rendering agree by
construction.  Policy: ``notebooks/plotting_sample_using_vaft_plot_module.ipynb``.
"""

import contextlib
import io
import warnings

import matplotlib

matplotlib.use("Agg")

import omas
import pytest

import vaft
import vaft.omas
from vaft.omas._plot_recipes import RECIPES, LineRecipe, diagnoses_itself
from vaft.plot import discovery
from vaft.plot.discovery import PlotCapability, PlotCatalog, match_query
from vaft.plot.selection import radial_divider


def _load(rel):
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path(rel)))


@pytest.fixture(scope="module")
def shot():
    return _load("samples/39915/omas.json.gz")


@pytest.fixture(scope="module")
def catalog(shot):
    return vaft.omas.available_plots(shot)


# ---------------------------------------------------------------------------
# The flat listing's contract survives (section 18, 19)
# ---------------------------------------------------------------------------

def test_the_catalog_still_behaves_like_the_flat_rows():
    rows = vaft.plot.available_plots()
    assert isinstance(rows, PlotCatalog)
    assert len(rows) == len(vaft.plot.canonical_names())
    assert {row["name"] for row in rows} == set(vaft.plot.canonical_names())
    first = rows[0]
    assert isinstance(first, PlotCapability)
    assert first["model"] == first.model and isinstance(first["ids"], tuple)
    assert list(first.row()) == [
        "name", "domain", "subject", "view", "quantity",
        "model", "ids", "required_paths", "description", "status",
    ]
    assert rows.rows() == tuple(row.row() for row in rows)
    assert rows[1:3] == tuple(rows)[1:3]


def test_printing_never_discards_the_structure(catalog):
    text = str(catalog)
    assert text.startswith("Available plots — #39915")
    assert len(catalog) == len(list(catalog))
    assert catalog.find("flux_loop_time_flux").function == "plot_flux_loop_time_flux()"


# ---------------------------------------------------------------------------
# Tree = subject / view / [quantity]; developer paths hidden (sections 1, 2, 17)
# ---------------------------------------------------------------------------

def test_the_tree_is_grouped_by_subject_view_and_quantity(catalog):
    text = str(catalog)
    flux_loop = text[text.index("\nflux_loop\n"):]
    assert "└─ time" in flux_loop.split("\n\n")[0]
    assert "└─ flux  plot_flux_loop_time_flux()" in flux_loop
    # A plot whose identity has no quantity hangs straight off its view.
    assert "└─ time  plot_plasma_current_time()" in text
    assert "spectrogram  plot_mirnov_spectrogram()" in str(vaft.omas.available_plots(query="mirnov"))
    assert "spectrogram\n   └─" not in str(vaft.omas.available_plots(query="mirnov"))


def test_compact_output_hides_imas_paths_and_detail_shows_them(shot):
    compact = str(vaft.omas.available_plots(shot, query="flux_loop"))
    assert "required" not in compact and "magnetics.flux_loop" not in compact
    detailed = str(vaft.omas.available_plots(shot, query="flux_loop", detail=True))
    assert "required: magnetics.flux_loop.{i}.flux.data" in detailed
    assert "model: LineSeries" in detailed and "domain: magnetics" in detailed


def test_headings_carry_the_canonical_identity_and_its_aliases(shot):
    text = str(vaft.omas.available_plots(shot, query="ip"))
    assert "plasma_current [ip, I_p]" in text
    assert "\nip\n" not in text


# ---------------------------------------------------------------------------
# Query = strict aliases (sections 3, 20, 21)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "term, subject",
    [("ip", "plasma_current"), ("bpol probe", "b_field_probe"), ("mirnov coil", "mirnov"),
     ("Electron Density", "electron_density"), ("flux", "flux_loop")],
)
def test_queries_resolve_through_the_alias_registry(term, subject):
    assert match_query(term).subjects == (subject,)
    assert {row.subject for row in vaft.plot.available_plots(query=term)} == {subject}


@pytest.mark.parametrize("term", ["Rogowski coil", "line_radiation", "no such thing"])
def test_related_but_distinct_concepts_do_not_match(term):
    assert not match_query(term)
    assert len(vaft.plot.available_plots(query=term)) == 0
    assert "(no plots match)" in str(vaft.plot.available_plots(query=term))


def test_a_quantity_family_query_narrows_to_those_quantities():
    names = vaft.plot.available_plots(query="beta").names()
    assert names and all("beta" in name for name in names)
    assert "equilibrium_profile_q" not in names


def test_developer_filters_still_work():
    assert {row.view for row in vaft.plot.available_plots(view="profile")} == {"profile"}
    assert {row.subject for row in vaft.plot.available_plots(subject="ip")} == {"plasma_current"}


# ---------------------------------------------------------------------------
# Availability equals render()'s test (sections 4, 6, 22)
# ---------------------------------------------------------------------------

def test_availability_agrees_with_render_for_every_plot(shot):
    everything = vaft.omas.available_plots(shot, available_only=False)
    assert len(everything) == len(vaft.plot.canonical_names())
    offered = {row.name for row in vaft.omas.available_plots(shot)}
    assert offered == {row.name for row in everything if row.available}
    for row in everything:
        if row.available or diagnoses_itself(row.name):
            continue
        assert row.reason.startswith("requires ") or "members" in row.reason
        with pytest.raises(ValueError, match="not available in this input"):
            getattr(vaft.omas, f"plot_{row.name}")(shot)


def test_unavailable_plots_carry_a_machine_readable_reason(shot):
    everything = vaft.omas.available_plots(shot, available_only=False)
    voltage = everything.find("flux_loop_time_voltage")
    assert voltage.available is False
    assert voltage.reason == "requires magnetics.flux_loop.{i}.voltage.data"
    assert "unavailable — requires magnetics.flux_loop.{i}.voltage.data" in str(everything)
    assert "flux_loop_time_voltage" not in str(vaft.omas.available_plots(shot))


def test_multi_shot_input_reports_availability_per_entry(shot):
    other = _load("samples/41524/imas.nc")
    odc = omas.ODC()
    odc["39915"] = shot
    odc["41524"] = other
    record = vaft.omas.available_plots(odc, query="ip").find("plasma_current_time")
    assert record.entries == {"39915": True, "41524": True}
    assert str(vaft.omas.available_plots(odc, query="ip")).startswith("Available plots — #39915, #41524")


# ---------------------------------------------------------------------------
# Channels, regions and representatives come from the selection policy (5, 6)
# ---------------------------------------------------------------------------

def test_channel_counts_distinguish_total_and_usable(shot):
    record = vaft.omas.available_plots(shot, query="flux_loop").find("flux_loop_time_flux")
    channels = record.channels
    assert channels["total"] == 11 and channels["usable"] == 11 and channels["flagged"] == 0
    assert channels["regions"] == {"inboard": 7, "outboard": 4}
    assert "channels: 11 / 11 usable · regions: inboard, outboard" in str(
        vaft.omas.available_plots(shot, query="flux_loop")
    )


def test_representatives_are_what_the_presets_resolve(shot):
    record = vaft.omas.available_plots(shot, query="flux_loop").find("flux_loop_time_flux")
    from vaft.omas._plot_recipes import _resolve_selection
    recipe = RECIPES["flux_loop_time_flux"]
    for preset, index in record.channels["representatives"].items():
        assert [index] == _resolve_selection(shot, recipe.y_path, preset)


def test_usable_excludes_flagged_channels(shot):
    record = vaft.omas.available_plots(shot, query="mirnov").find("mirnov_time_voltage")
    channels = record.channels
    assert channels["flagged"] == 6
    assert channels["usable"] == channels["with_data"] - 6
    assert record.validity["available"] and record.validity["flagged"] == 6
    assert "validity (6 flagged)" in str(vaft.omas.available_plots(shot, query="mirnov"))


def test_the_default_view_never_prints_channel_indices(shot, catalog):
    text = str(catalog)
    assert "identifiers:" not in text and "positions:" not in text
    detailed = str(vaft.omas.available_plots(shot, query="flux_loop", detail=True))
    assert "identifiers: Flux Loop - #3" in detailed
    assert "positions: [0] (59.2 cm, 68.5 cm)" in detailed


# ---------------------------------------------------------------------------
# Display, layouts, methods, overviews (sections 7, 8, 9, 13)
# ---------------------------------------------------------------------------

def test_display_comes_from_the_display_policy(shot):
    ip = vaft.omas.available_plots(shot, query="ip").find("plasma_current_time")
    assert ip.display == {"unit": "kA", "units": ("A", "kA", "MA"), "notation": "auto"}
    pressure = vaft.omas.available_plots(query="barometry").find("barometry_time_pressure")
    assert pressure.display["unit"] == "Torr" and pressure.display["notation"] == "scientific"
    assert "unit: Torr (scientific)" in str(vaft.omas.available_plots(query="barometry"))


def test_layouts_are_derived_from_the_recipe_and_the_geometry(shot):
    registry = vaft.omas.available_plots()
    assert registry.find("plasma_current_time").layouts == ("overlay",)
    assert registry.find("flux_loop_time_flux").layouts == ("overlay", "subplots", "grouped")
    assert "grouped (with a radial split)" in str(vaft.omas.available_plots(query="flux_loop"))
    # A plot that takes no layout= advertises none.
    assert registry.find("equilibrium_field_psi").layouts == ()
    with_ods = vaft.omas.available_plots(shot).find("flux_loop_time_flux")
    assert with_ods.layouts == ("overlay", "subplots", "grouped")
    assert "grouped (with" not in str(vaft.omas.available_plots(shot, query="flux_loop"))


def test_grouped_is_advertised_only_where_the_family_splits():
    ods = omas.ODS()
    ods["magnetics.time"] = [0.0, 0.1]
    for index in range(3):
        ods[f"magnetics.b_field_pol_probe.{index}.position.r"] = 0.796
        ods[f"magnetics.b_field_pol_probe.{index}.position.z"] = 0.1 * index
        ods[f"magnetics.b_field_pol_probe.{index}.voltage.data"] = [1.0, 1.0]
    record = vaft.omas.available_plots(ods, query="mirnov").find("mirnov_time_voltage")
    assert record.layouts == ("overlay", "subplots")
    assert "regions" not in record.channels


def test_analysis_methods_list_what_exists():
    registry = vaft.omas.available_plots(query="mirnov")
    assert registry.find("mirnov_spectrogram").analysis_methods == ("STFT",)
    assert registry.find("mirnov_spectrum").analysis_methods == ("Welch PSD",)
    assert "methods: STFT" in str(registry) and "wavelet" not in str(registry).lower()


def test_overviews_summarise_their_members():
    record = vaft.omas.available_plots(query="magnetics").find("magnetics_overview")
    assert record.overview_members == ("plasma_current", "pf_coil", "flux_loop", "b_field_probe")
    assert "overview: plasma_current · pf_coil · flux_loop · b_field_probe" in str(
        vaft.omas.available_plots(query="magnetics")
    )


def test_camera_overlays_are_read_from_the_registry():
    frame = vaft.plot.available_plots(query="camera_visible").find("camera_visible_image_frame")
    assert frame.overlays == ("efit_overlay", "field_line")


def test_capability_fields_are_filled_only_where_issue_261_defined_them(catalog):
    # Sources and projection wait for their sub-phases; interaction is stated
    # only by the plots that offer one (the static equilibrium slice summary).
    for record in catalog:
        assert record.sources == {} and record.projection == {}
        if record.name != "equilibrium_overview":
            assert record.interaction == (), record.name
    assert catalog.find("equilibrium_overview").interaction == ("static",)
    assert "sources:" not in str(catalog) and "projection:" not in str(catalog)


def test_every_channel_plot_declares_layouts_and_nothing_else_does():
    for record in vaft.omas.available_plots():
        recipe = RECIPES.get(record.name)
        if isinstance(recipe, LineRecipe):
            assert record.layouts[0] == "overlay"
            assert ("subplots" in record.layouts) == (recipe.index == "channel")
        else:
            assert record.layouts == ()


# ---------------------------------------------------------------------------
# Independent review of the discovery interface
# ---------------------------------------------------------------------------

def test_an_empty_collection_yields_an_empty_catalog():
    for empty in ([], omas.ODC()):
        catalog = vaft.omas.available_plots(empty)
        assert len(catalog) == 0
        assert "(nothing to plot)" in str(catalog)


def test_a_family_query_keeps_the_family_plot():
    names = vaft.plot.available_plots(query="beta").names()
    assert "equilibrium_time_beta" in names
    assert {"equilibrium_time_beta_n", "equilibrium_time_beta_p", "equilibrium_time_beta_t"} <= set(names)


def test_region_counts_are_the_channels_grouped_would_place(shot):
    from vaft.omas._plot_recipes import build_model, normalize_entries
    for name in ("b_field_probe_time_field", "flux_loop_time_flux", "mirnov_time_voltage"):
        record = vaft.omas.available_plots(shot).find(name)
        grouped = build_model(name, normalize_entries(shot, label="key"), layout="grouped")
        placed = {panel.title: len(panel.series) for panel in grouped.models}
        assert record.channels["regions"] == placed, name


def test_a_literal_quantity_names_the_plots_that_carry_it():
    names = vaft.plot.available_plots(query="pressure").names()
    assert names and all(row.quantity == "pressure" for row in vaft.plot.available_plots(query="pressure"))
    assert "barometry_time_pressure" in names and "equilibrium_profile_pressure" in names
    # A subject prefix still wins over a quantity of the same word.
    assert {row.subject for row in vaft.plot.available_plots(query="flux")} == {"flux_loop"}


def test_records_still_compare_equal_to_the_flat_rows():
    rows = vaft.plot.available_plots()
    assert rows[0] == rows[0].row()
    assert rows[0] != rows[1]
    assert rows + () == rows.rows() and () + rows == rows.rows()
    assert len({rows[0], rows[0]}) == 1


def test_an_empty_query_matches_nothing_and_says_so():
    assert "(no plots match)" in str(vaft.plot.available_plots(query=""))
    assert "(no plots match)" in str(vaft.plot.available_plots(query="   "))
