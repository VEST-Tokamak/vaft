"""The computed views declare what they read, and the neutral ones run natively (#439).

Every ``CallableRecipe`` carries ``reads`` (the DD templates its builder and
helpers touch) and ``backend`` (``neutral``: accessor reads only, any data
model; ``omas``: an OMAS ODS is required, ``reason`` names the helper).  The
tests here record what each builder actually reads -- through the accessor
for a neutral builder, through ``omas.ODS`` for an OMAS-bound one -- and fail
on a read the recipe did not declare.  The OMAS recorder is best-effort
(omas internals that read ``omas_data`` directly leave no trace), so an
OMAS-bound under-declaration can slip through; the neutral half is exact.
"""

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pytest

import vaft
import vaft.omas
from vaft.plot.backend import dd
from vaft.plot.backend import recipes as R
from vaft.plot.backend.recipes import build_model, converts_for_builder, missing_required_path
from vaft.omas.entries import normalize_entries

from _read_recorder import accessor_reads, assert_models_equal, ods_reads, undeclared
from _synthetic_inputs import OPTIONS, SYNTHETIC

NEUTRAL = frozenset({
    "nbi_profile_electron_heating", "nbi_profile_ion_heating", "nbi_profile_current_drive",
    "interferometer_spectrogram", "passive_structure_time_current",
    "impa_time_field", "impa_time_voltage", "impa_profile_field",
    "soft_x_rays_geometry_lines_of_sight", "coil_3d_geometry3d", "coil_3d_geometry_topview",
    "pf_coil_geometry_poloidal", "passive_structure_geometry_poloidal", "machine_geometry_poloidal",
    "equilibrium_geometry_topview", "machine_geometry_topview",
    "electron_temperature_field", "electron_density_field",
    "camera_visible_animation_frames", "camera_visible_spectrogram",
    "limiter_current_time", "mirnov_spatial_phase",
    "ntms_time_delta_prime", "mhd_linear_time_energy_perturbed",
    "mhd_linear_profile_displacement", "mhd_linear_profile_b_field_perturbed",
    "mhd_linear_profile_resonant_flux", "mhd_linear_profile_island_width",
    # Built on vaft.omas helpers that read through vaft.ods_access, which
    # dispatches to the registered accessor: native on an IMAS entry too.
    "pf_plasma_geometry_poloidal",
    "equilibrium_overview_verification", "equilibrium_overview_fit_quality",
    "equilibrium_overview_convergence", "equilibrium_overview_constraints",
    "equilibrium_overview_constraint_coverage", "equilibrium_overview_residuals",
    "chease_overview_refinement_summary", "chease_overview_profile_validity",
})
OMAS_BOUND = frozenset({
    "passive_structure_geometry_wall_mode",
    "passive_structure_overview_wall_time", "passive_structure_overview_wall_reduction",
    "passive_structure_field_wall_reduction", "neoclassical_profile_bootstrap_current",
    "equilibrium_field_psi_vacuum", "vacuum_field", "summary_time_power_balance",
    "camera_visible_image", "camera_visible_image_frame", "camera_visible_image_efit_overlay",
    "camera_visible_image_field_line", "camera_visible_image_fluctuation",
    "camera_visible_image_mhd_power", "equilibrium_overview",
    "magnetics_overview_vacuum", "magnetics_overview_plasma_residual",
})

#: Recorded reads that are not the plot's input, per plot, with the reason.
IGNORED_READS: dict[str, dict[str, str]] = {}

#: Neutral views whose synthetic input cannot be written to IMAS and read back.
SYNTHETIC_ROUND_TRIP_UNSUPPORTED: dict[str, str] = {}


def _callables() -> list[str]:
    return [name for name, recipe in R.RECIPES.items() if isinstance(recipe, R.CallableRecipe)]


@pytest.fixture(scope="module")
def sample():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(vaft.data.sample(39915, representation="omas"))


def _input_for(name: str, sample):
    """The sample when it can build ``name``, else the synthetic input, plus its options."""
    if missing_required_path(sample, name) is None and name not in SYNTHETIC:
        return sample, {}
    if name in SYNTHETIC:
        return SYNTHETIC[name](sample), dict(OPTIONS.get(name, {}))
    return sample, {}


def _quiet_build(name, ods, **options):
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        return build_model(name, normalize_entries(ods), **options)


# ---------------------------------------------------------------------------
# the declarations
# ---------------------------------------------------------------------------


def test_every_computed_view_is_classified_and_declares_its_reads():
    names = _callables()
    assert {n for n in names if R.RECIPES[n].backend == R.NEUTRAL} == NEUTRAL
    assert {n for n in names if R.RECIPES[n].backend == R.OMAS_BOUND} == OMAS_BOUND
    assert NEUTRAL | OMAS_BOUND == set(names)
    for name in names:
        recipe = R.RECIPES[name]
        assert recipe.reads, f"{name} declares no reads"
        assert bool(recipe.reason) == (recipe.backend == R.OMAS_BOUND), name


def test_every_read_is_a_well_formed_template():
    for name in _callables():
        recipe = R.RECIPES[name]
        for template in recipe.reads:
            dd.from_template(template)  # raises on a malformed template
            if "." not in template:
                # A bare IDS root says "the whole IDS is deep-copied": only an
                # OMAS-bound builder may say that; a neutral one is leaf-exact.
                assert recipe.backend == R.OMAS_BOUND, f"{name} declares the root {template!r}"


def test_a_misclassified_recipe_is_refused():
    with pytest.raises(ValueError, match="backend"):
        R.CallableRecipe(builder=lambda ods: None, backend="native")
    with pytest.raises(ValueError, match="reason"):
        R.CallableRecipe(builder=lambda ods: None, backend=R.NEUTRAL, reason="x")


def test_every_computed_view_has_an_input_to_record(sample):
    without = [
        name for name in _callables()
        if missing_required_path(sample, name) is not None and name not in SYNTHETIC
    ]
    assert not without, without


# ---------------------------------------------------------------------------
# neutral builders: exact, and native
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(NEUTRAL))
def test_neutral_builders_read_only_what_they_declare(name, sample, monkeypatch):
    ods, options = _input_for(name, sample)
    recipe = R.RECIPES[name]
    with accessor_reads(monkeypatch) as recorded:
        _quiet_build(name, ods, **options)
    assert not undeclared(recorded, recipe.reads, IGNORED_READS.get(name, {})), (
        undeclared(recorded, recipe.reads, IGNORED_READS.get(name, {}))
    )


@pytest.fixture(scope="module")
def sample_entry():
    imas = pytest.importorskip("imas")
    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    yield entry
    entry.close()


@pytest.mark.parametrize("name", sorted(NEUTRAL))
def test_neutral_builders_never_convert_an_imas_entry(name, sample, sample_entry, monkeypatch, tmp_path):
    imas = pytest.importorskip("imas")
    from vaft.imas.access import IDSEntry
    import vaft.imas

    ods, options = _input_for(name, sample)
    with contextlib.ExitStack() as stack:
        if ods is sample:
            entry = sample_entry
        else:
            if name in SYNTHETIC_ROUND_TRIP_UNSUPPORTED:
                pytest.skip(SYNTHETIC_ROUND_TRIP_UNSUPPORTED[name])
            with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vaft.imas.save(ods, tmp_path / "synthetic.nc")
            entry = stack.enter_context(imas.DBEntry(str(tmp_path / "synthetic.nc"), "r", dd_version="3.41.0"))
        bundle = IDSEntry(entry)
        assert not converts_for_builder(bundle, name)
        monkeypatch.setattr(IDSEntry, "as_ods_for", lambda self, names: pytest.fail(f"{name} converted {sorted(names)}"))
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            native = build_model(name, [("39915", bundle)], **options)
    expected = _quiet_build(name, ods, **options)
    assert_models_equal(native, expected, where=name)


# ---------------------------------------------------------------------------
# OMAS-bound builders: declared as far as the recorder sees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(OMAS_BOUND))
def test_omas_bound_builders_read_only_what_they_declare(name, sample, monkeypatch):
    ods, options = _input_for(name, sample)
    recipe = R.RECIPES[name]
    with accessor_reads(monkeypatch) as via_accessor, ods_reads(monkeypatch) as via_ods:
        try:
            _quiet_build(name, ods, **options)
        except Exception:  # noqa: BLE001 - what was read before the raise still counts
            pass
    for path in via_accessor.paths:
        via_ods.add(path)
    missing = undeclared(via_ods, recipe.reads, IGNORED_READS.get(name, {}))
    assert not missing, missing[:40]


def test_the_ignored_reads_are_still_needed(sample, monkeypatch):
    stale = []
    for name, ignored in IGNORED_READS.items():
        ods, options = _input_for(name, sample)
        recipe = R.RECIPES[name]
        with accessor_reads(monkeypatch) as via_accessor, ods_reads(monkeypatch) as via_ods:
            try:
                _quiet_build(name, ods, **options)
            except Exception:  # noqa: BLE001
                pass
        for path in via_accessor.paths:
            via_ods.add(path)
        with_all = undeclared(via_ods, recipe.reads, ignored)
        for template, reason in ignored.items():
            without = {t: r for t, r in ignored.items() if t != template}
            if undeclared(via_ods, recipe.reads, without) == with_all:
                stale.append(f"{name}: {template} no longer needs ignoring ({reason})")
    assert not stale, stale


# ---------------------------------------------------------------------------
# what the facades say
# ---------------------------------------------------------------------------


def test_discovery_reports_the_classification():
    imas = pytest.importorskip("imas")
    import vaft.imas
    from vaft.plot.backend.discovery import describe_one

    entry = imas.DBEntry(str(vaft.data.data_path("samples/39915/imas.nc")), "r", dd_version="3.41.0")
    entries = vaft.imas.normalize_entries(entry)
    neutral = describe_one("machine_geometry_poloidal", entries)
    assert neutral.computation["backend"] == "neutral"
    assert "converted per IDS" not in neutral.reason
    bound = describe_one("equilibrium_overview", entries)
    assert bound.computation["backend"] == "omas"
    assert "converted per IDS" in bound.reason and "update_equilibrium_derived_profiles" in bound.reason
    detail = str(vaft.imas.available_plots(entry, query="equilibrium", detail=True))
    assert "computed: native reads" in detail
    assert "needs an OMAS ODS" in detail


def test_dd_paths_lists_the_reads_with_their_provenance():
    paths = {p.canonical: p for p in dd.dd_paths("mirnov_spatial_phase")}
    recipe = R.RECIPES["mirnov_spatial_phase"]
    for template in recipe.reads:
        path = paths[dd.from_template(template).canonical]
        assert "recipe" in path.attrs["declared_by"], path
        assert path.attrs["backend"] == "neutral", path
        # A read the spec also gates on keeps the spec's role; the rest are inputs.
        assert path.role == "input" or path.attrs["declared_by"] == "spec+recipe", path
    # A read the spec also gates on (the voltage) carries both provenances.
    voltage = paths["magnetics/b_field_pol_probe(:)/voltage/data"]
    assert voltage.attrs["declared_by"] == "spec+recipe" and voltage.role == "required"
    # A view whose builder reads beyond its spec lists those reads as inputs.
    topview = dd.dd_paths("machine_geometry_topview")
    assert any(p.role == "input" and p.attrs["declared_by"] == "recipe" for p in topview)
    assert vaft.omas.dd_mirnov_spatial_phase() == tuple(paths.values())
