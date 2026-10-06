"""Every canonical plot documents itself under the issue #1505 contract.

The contract is parsed by :mod:`vaft.plot._docstring`, and its structural half
-- a summary sentence, an ``Interpretation`` section, a ``Parameters`` section
that matches the signature when there is one -- decides
:attr:`vaft.plot.PlotDocumentation.conforming`, the same flag the documentation
site reads.  This file adds the half that is policy: which plots must state
their ``Limitations``, that ``Options`` explains an option vocabulary rather
than re-enumerating it, and that ``See Also`` names real plots.

``PENDING`` names the registered plots whose renderers are not yet written to
the contract.  It is exact -- a plot leaves it the moment it conforms, and a
plot that stops conforming fails -- and it may only shrink: a plot registered
after the contract landed must conform from the start, so it never joins the
list: ``PENDING`` must stay a subset of the frozen ``PENDING_AT_LANDING``
snapshot, and ``PENDING_CEILING`` only goes down.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import re

import pytest

import vaft.plot
from vaft.plot import registry
from vaft.plot._docstring import (
    CUSTOM_SECTIONS,
    SECTION_VOCABULARY,
    PlotDocumentation,
    parse_docstring,
    plot_documentation,
)

#: Registered plots whose renderer docstrings are not yet under the contract.
#: Remove a name when its renderer is migrated; never add one.
#: The plots that predate the contract (#1505), frozen when it landed.  Never
#: edit: PENDING may only drop names from it, so no new plot can join PENDING.
PENDING_AT_LANDING: frozenset[str] = frozenset({
    "barometry_time_pressure",
    "camera_visible_animation_frames",
    "camera_visible_image",
    "camera_visible_image_efit_overlay",
    "camera_visible_image_field_line",
    "camera_visible_image_fluctuation",
    "camera_visible_image_frame",
    "camera_visible_image_mhd_power",
    "camera_visible_image_vacuum_field_line",
    "camera_visible_spectrogram",
    "charge_exchange_geometry_poloidal",
    "charge_exchange_profile_fit",
    "charge_exchange_profile_ion_temperature",
    "charge_exchange_profile_velocity_tor",
    "charge_exchange_time_ion_temperature",
    "charge_exchange_time_velocity_tor",
    "chease_overview_profile_validity",
    "chease_overview_refinement_summary",
    "coil_3d_geometry3d",
    "coil_3d_geometry_topview",
    "coil_3d_profile_current",
    "coil_3d_spectrum_current",
    "core_profiles_profile_zeff",
    "core_profiles_time_volume_averaged",
    "current_overview",
    "diagnostics_overview",
    "diagnostics_spectrum_coherence",
    "ec_launchers_time_power",
    "electron_density_profile_gradient",
    "electron_density_time",
    "electron_temperature_profile_gradient",
    "electron_temperature_time",
    "equilibrium_field_psi_vacuum",
    "equilibrium_geometry_boundary",
    "equilibrium_geometry_topview",
    "equilibrium_overview_constraint_coverage",
    "equilibrium_overview_constraint_weights",
    "equilibrium_overview_constraints",
    "equilibrium_overview_convergence",
    "equilibrium_overview_fit_quality",
    "equilibrium_overview_histories",
    "equilibrium_overview_pressure_weight_scan",
    "equilibrium_overview_profiles",
    "equilibrium_overview_residuals",
    "equilibrium_overview_verification",
    "equilibrium_profile_f",
    "equilibrium_profile_ffprime",
    "equilibrium_profile_j_tor",
    "equilibrium_profile_pprime",
    "equilibrium_profile_pressure",
    "equilibrium_text_summary",
    "equilibrium_time_diamagnetic_flux",
    "equilibrium_time_elongation",
    "equilibrium_time_li",
    "equilibrium_time_major_radius",
    "equilibrium_time_minor_radius",
    "equilibrium_time_plasma_current",
    "equilibrium_time_q0",
    "equilibrium_time_qa",
    "equilibrium_time_shape",
    "equilibrium_time_triangularity",
    "equilibrium_time_triangularity_lower",
    "equilibrium_time_triangularity_upper",
    "equilibrium_time_virial",
    "equilibrium_time_w_mag",
    "equilibrium_time_w_mhd",
    "equilibrium_time_w_tot",
    "field_line_topology_field_connection_length",
    "impa_overview",
    "impa_profile_field",
    "impa_time_field",
    "impa_time_voltage",
    "impurity_profile_charge_state_fraction",
    "impurity_profile_composition",
    "interferometer_overview",
    "interferometer_spectrogram",
    "interferometer_spectrum",
    "interferometer_time_n_e_line",
    "ion_temperature_profile",
    "ion_temperature_profile_gradient",
    "kinetic_overview_profiles",
    "limiter_current_time",
    "machine_geometry3d",
    "machine_geometry_topview",
    "magnetics_geometry_poloidal",
    "magnetics_overview",
    "magnetics_overview_plasma_residual",
    "magnetics_overview_vacuum",
    "mhd_linear_field_spectrum",
    "mhd_linear_geometry_island",
    "mhd_linear_overview_eigenfunction",
    "mhd_linear_profile_b_field_perturbed",
    "mhd_linear_profile_chirikov",
    "mhd_linear_profile_displacement",
    "mhd_linear_profile_island_width",
    "mhd_linear_profile_resonant_flux",
    "mhd_linear_spectrum_b_field_perturbed",
    "mhd_linear_time_energy_perturbed",
    "mirnov_spatial_phase",
    "mirnov_spectrum",
    "nbi_profile_current_drive",
    "nbi_profile_electron_heating",
    "nbi_profile_ion_heating",
    "neoclassical_profile_bootstrap_current",
    "ntms_time_delta_prime",
    "passive_structure_field_wall_reduction",
    "passive_structure_geometry_poloidal",
    "passive_structure_geometry_wall_mode",
    "passive_structure_overview_wall_reduction",
    "passive_structure_overview_wall_time",
    "passive_structure_time_current",
    "pf_coil_geometry_poloidal",
    "pf_coil_time_current",
    "pf_coil_time_current_turns",
    "pf_plasma_geometry_poloidal",
    "rogowski_coil_time_current",
    "soft_x_rays_geometry_lines_of_sight",
    "soft_x_rays_overview",
    "soft_x_rays_spectrogram",
    "soft_x_rays_spectrum",
    "soft_x_rays_time_power",
    "spectrometer_uv_time_impurity",
    "spectrometer_uv_time_intensity",
    "startup_proxies_time",
    "summary_time_energy",
    "summary_time_estimated_q95",
    "summary_time_normalized_current",
    "summary_time_power_balance",
    "summary_time_q_star_cylindrical",
    "summary_time_q_star_kink",
    "summary_time_resistive_zeff",
    "summary_time_romero_balance",
    "summary_time_voltage_consumption",
    "tf_coil_time_b_t",
    "tf_coil_time_b_t_vacuum_r",
    "tf_coil_time_current",
    "thermal_pressure_profile",
    "thomson_scattering_geometry_poloidal",
    "thomson_scattering_profile_electron_density",
    "thomson_scattering_profile_electron_temperature",
    "thomson_scattering_profile_fit",
    "thomson_scattering_time_electron_density",
    "thomson_scattering_time_electron_temperature",
    "vacuum_field",
    "vacuum_field_midplane",
    "wall_geometry_poloidal",
})

#: Registered plots whose renderer docstrings are not yet under the contract.
#: Remove a name with ``- {...}`` when its renderer is migrated; never add one.
PENDING: frozenset[str] = PENDING_AT_LANDING - frozenset()

#: ``len(PENDING)`` when the contract landed, lowered with every migration.
PENDING_CEILING = 146

#: Views whose reading is interpretive enough that an adopted plot must say
#: what it cannot show: profiles, maps, spectra, summaries and tables.
LIMITATIONS_VIEWS = frozenset({
    "profile", "spatial", "evolution", "field", "spectrum", "spectrogram", "overview", "table",
})

#: Domains whose plotted values are reconstruction or model outputs, which an
#: adopted plot must bound with ``Limitations`` whatever its view.
LIMITATIONS_DOMAINS = frozenset({"equilibrium"})

SPECS = registry.specs(status=None)
ADOPTED = [spec for spec in SPECS if spec.name not in PENDING]
IDS = [spec.name for spec in ADOPTED]


def _vocabularies(name: str) -> dict[str, tuple[str, ...]]:
    """The option vocabularies the plotting layer defines structurally."""
    from vaft.plot.backend import recipes
    from vaft.plot.display import PROFILE_COORDINATES, PSI_STYLES
    from vaft.plot.style import UNCERTAINTY_MODES, VALIDITY_MODES

    vocabularies = {
        "PROFILE_COORDINATES": PROFILE_COORDINATES,
        "PSI_STYLES": PSI_STYLES,
        "UNCERTAINTY_MODES": UNCERTAINTY_MODES,
        "VALIDITY_MODES": VALIDITY_MODES,
        "SPECTROGRAM_METHODS": recipes.SPECTROGRAM_METHODS,
        "POLOIDAL_OVERLAYS": recipes.POLOIDAL_OVERLAYS,
        "MACHINE_OVERLAYS": recipes.MACHINE_OVERLAYS,
    }
    fields = recipes.field_options_for(name)
    if fields:
        vocabularies["field="] = tuple(fields)
    return vocabularies


# --- the split ------------------------------------------------------------------


def test_pending_names_registered_plots():
    assert PENDING <= {spec.name for spec in SPECS}, sorted(PENDING - {spec.name for spec in SPECS})


def test_pending_is_exactly_the_set_of_non_conforming_plots():
    """A plot may not leave PENDING early, nor linger once it conforms."""
    actual = {spec.name for spec in SPECS if not plot_documentation(spec.name).conforming}
    assert actual == PENDING, {
        "must conform (or, if it predates #1505, stay in PENDING)": sorted(actual - PENDING),
        "conforms: remove from PENDING": sorted(PENDING - actual),
    }


def test_pending_only_shrinks():
    assert PENDING <= PENDING_AT_LANDING, (
        "a plot registered after #1505 must conform; it never joins PENDING",
        sorted(PENDING - PENDING_AT_LANDING),
    )
    assert len(PENDING) <= PENDING_CEILING, "lower PENDING_CEILING as plots are migrated"
    assert len(PENDING_AT_LANDING) == 146, "the landing snapshot is frozen; never edit it"


def test_something_is_under_the_contract():
    assert ADOPTED, "every plot is pending; the contract enforces nothing"


# --- structural, per adopted plot --------------------------------------------


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_conforms(spec):
    documentation = plot_documentation(spec.name)
    assert documentation.conforming, "\n".join(documentation.errors)


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_every_section_is_in_the_vocabulary(spec):
    for title, _ in plot_documentation(spec.name).sections:
        assert title in SECTION_VOCABULARY, title


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_interpretation_is_prose_beyond_the_summary(spec):
    documentation = plot_documentation(spec.name)
    text = documentation.interpretation or ""
    assert len(text) >= 120, text
    assert text.strip() != documentation.summary


# --- policy, per adopted plot --------------------------------------------------


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_interpretive_views_state_their_limitations(spec):
    if spec.view not in LIMITATIONS_VIEWS and spec.domain not in LIMITATIONS_DOMAINS:
        pytest.skip("view and domain need no Limitations")
    assert plot_documentation(spec.name).limitations, "missing Limitations section"


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_options_explain_rather_than_enumerate_a_vocabulary(spec):
    """Options says what a choice means; the option schema lists the choices.

    Spelling out every member of a structural vocabulary makes the docstring a
    second, silently diverging source of truth for it (issue #1505).
    """
    text = plot_documentation(spec.name).options
    if text is None:
        pytest.skip("no Options section")
    for vocabulary, members in _vocabularies(spec.name).items():
        if len(members) < 3:
            continue
        listed = [member for member in members if f"``{member}``" in text or f"'{member}'" in text]
        assert len(listed) < len(members), f"Options re-enumerates {vocabulary}: {members}"


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_see_also_names_registered_plots(spec):
    text = plot_documentation(spec.name).see_also
    if text is None:
        pytest.skip("no See Also section")
    known = {item.name for item in SPECS}
    for line in text.splitlines():
        if not line or line[0].isspace():
            continue
        for target in re.split(r"\s*,\s*", line.partition(" : ")[0].strip()):
            if "." in target:
                continue  # a dotted reference to a function outside the registry
            assert target in known, f"See Also names {target!r}, which is not a registered plot"


@pytest.mark.parametrize("spec", ADOPTED, ids=IDS)
def test_documented_defects_are_tracked_by_issue(spec):
    text = plot_documentation(spec.name).limitations or ""
    if "tracked in" in text.lower():
        assert "#" in text, "a tracked limitation must name its GitHub issue number"


# --- the accessor ----------------------------------------------------------------


def test_the_accessor_reads_the_registered_renderer():
    for spec in SPECS:
        documentation = vaft.plot.documentation(spec.name)
        assert isinstance(documentation, PlotDocumentation)
        assert documentation.name == spec.name
        expected = parse_docstring(inspect.unwrap(spec.renderer).__doc__)
        assert documentation.summary == expected.summary
        assert documentation.sections == expected.sections


def test_the_accessor_refuses_an_unknown_plot():
    with pytest.raises(KeyError, match="no plot named"):
        vaft.plot.documentation("not_a_plot")


def test_the_parsed_form_is_plain_data():
    payload = vaft.plot.documentation("equilibrium_profile_q").as_dict()
    assert json.loads(json.dumps(payload)) == payload
    titles = [section["title"] for section in payload["sections"]]
    assert "Interpretation" in titles and "Limitations" in titles
    assert payload["conforming"] is True


def test_named_sections_are_exposed_for_gui_consumers():
    documentation = vaft.plot.documentation("equilibrium_profile_q")
    assert "rational surfaces" in documentation.interpretation
    assert "not by itself a stability" in " ".join(documentation.limitations.split())
    assert "coordinate=" in documentation.options
    assert documentation.section("Interpretation") == documentation.interpretation


def test_a_one_line_docstring_does_not_conform():
    def renderer(model, *, ax=None, show=False, **style):
        """Some plot."""

    from vaft.plot._docstring import documentation_of

    documentation = documentation_of("x", renderer)
    assert not documentation.conforming
    assert "missing Interpretation section" in documentation.errors


def test_parameters_must_match_the_signature_when_present():
    def renderer(model, *, ax=None, show=False, **style):
        """Some plot.

        Parameters
        ----------
        model : LineSeries
            The model.

        Interpretation
        --------------
        Something worth reading.
        """

    from vaft.plot._docstring import documentation_of

    errors = documentation_of("x", renderer).errors
    assert any("signature has ['model', 'ax', 'show']" in error for error in errors), errors


def test_plot_spec_carries_no_scientific_prose():
    """The registry answers structural questions; the docstring answers scientific ones."""
    fields = {field.name for field in dataclasses.fields(registry.PlotSpec)}
    assert not fields & {
        "interpretation", "limitations", "scientific_use", "references", "options_meaning", "documentation",
    }


def test_vocabulary_lists_every_custom_section():
    assert set(CUSTOM_SECTIONS) <= set(SECTION_VOCABULARY)
