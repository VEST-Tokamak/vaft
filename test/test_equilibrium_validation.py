"""The unified equilibrium validation report (issue #72; #253 §13-§16).

What is asserted, and why it is the contract:

* one call covers verification, diagnostic fit, physical validity and
  independent validation, each separately queryable, and a category not
  requested is *absent* rather than ``not_available``;
* ``not_available`` and ``indeterminate`` are never a ``pass``, at the check
  and at every aggregation level;
* a near-singular or boundary-less slice yields ``indeterminate``, not an
  unbounded number and not a false failure;
* every number is the canonical provider's number -- the tests drive
  ``efit_quality`` and the virial wrapper directly and compare;
* the input ODS is never mutated, and no plotting backend is imported;
* the report is JSON-serializable and deterministic;
* what is invariant about a check lives in the registry, not on each result.
"""

from __future__ import annotations

import copy
import inspect
import json
import subprocess
import sys

import numpy as np
import pytest
from omas import ODS

from vaft.omas.sample import sample_ods
from vaft.validation import validate_equilibrium
from vaft.validation.equilibrium import (
    EQUILIBRIUM_CATEGORIES,
    aggregate_status,
    status_summary,
    validate_independent,
    validate_magnetic_fit,
    validate_physical,
    verify_continuity,
    verify_convergence,
    verify_convention,
    verify_structure,
)
from vaft.validation.model import ValidationStatus
from vaft.validation.registry import CHECKS, CheckSpec, checks_in, describe

PASS, WARN, FAIL = str(ValidationStatus.PASS), str(ValidationStatus.WARN), str(ValidationStatus.FAIL)
INDETERMINATE, NOT_AVAILABLE = str(ValidationStatus.INDETERMINATE), str(ValidationStatus.NOT_AVAILABLE)

#: The packaged shot-39915 sample: nine EFIT slices, the last a dead one with
#: no boundary and zero current, magnetics with a diamagnetic loop, and
#: neither core_profiles nor Thomson scattering.
LIVE, DEAD = 0, 8


@pytest.fixture(scope="module")
def sample():
    return sample_ods()


@pytest.fixture(scope="module")
def report(sample):
    return validate_equilibrium(sample)


@pytest.fixture(scope="module")
def efit():
    """The complete EFIT-quality contract: fitted families, a solver error history."""
    from test_efit_fit_quality import _complete_efit_quality_ods

    return _complete_efit_quality_ods()


def _slice(report, category, check, index):
    return next(entry for entry in report[category][check]["slices"] if entry["time_slice"] == index)


# ---------------------------------------------------------------------------
# one call, four questions
# ---------------------------------------------------------------------------

def test_one_call_answers_every_category_separately(report):
    assert tuple(category for category in EQUILIBRIUM_CATEGORIES if category in report) == EQUILIBRIUM_CATEGORIES
    assert set(report["summary"]) == set(EQUILIBRIUM_CATEGORIES)
    assert report["time_slices"] == list(range(9))
    for category in EQUILIBRIUM_CATEGORIES:
        expected = {spec.key.split(".", 1)[1] for spec in checks_in(category)}
        assert set(report[category]) == expected, category
        for entry in report[category].values():
            assert entry["status"] in {PASS, WARN, FAIL, INDETERMINATE, NOT_AVAILABLE}
    assert status_summary(report) == {
        category: ValidationStatus(status) for category, status in report["summary"].items()
    }


def test_a_category_not_requested_is_absent_not_unavailable(sample):
    partial = validate_equilibrium(sample, checks="verification", time_slice=LIVE)
    assert "verification" in partial
    for category in ("diagnostic_fit", "physical_validity", "independent_validation"):
        assert category not in partial
        assert category not in partial["summary"]
    with pytest.raises(ValueError, match="unknown validation categories"):
        validate_equilibrium(sample, checks=("verification", "source_validity"))


def test_time_slice_selection(sample):
    one = validate_equilibrium(sample, checks=("verification",), time_slice=3)
    assert one["time_slices"] == [3] and one["time"] == [pytest.approx(0.319)]
    assert one["verification"]["structure"]["counts"] == {PASS: 1}
    with pytest.raises(IndexError):
        validate_equilibrium(sample, time_slice=9)


def test_an_ods_without_an_equilibrium_is_not_available_everywhere():
    empty = validate_equilibrium(ODS())
    assert empty["status"] == NOT_AVAILABLE
    assert set(empty["summary"].values()) == {NOT_AVAILABLE}
    assert empty["time_slices"] == [] and "reason" in empty
    for category in EQUILIBRIUM_CATEGORIES:
        assert empty[category] == {}


# ---------------------------------------------------------------------------
# the vocabulary is kept honest
# ---------------------------------------------------------------------------

def test_aggregation_never_collapses_the_undecided_into_a_pass():
    S = ValidationStatus
    assert aggregate_status([]) is S.NOT_AVAILABLE
    assert aggregate_status([S.NOT_AVAILABLE, "not_available"]) is S.NOT_AVAILABLE
    assert aggregate_status([S.PASS, S.PASS]) is S.PASS
    assert aggregate_status([S.PASS, S.NOT_AVAILABLE]) is S.INDETERMINATE
    assert aggregate_status([S.PASS, S.INDETERMINATE]) is S.INDETERMINATE
    assert aggregate_status([S.INDETERMINATE, S.WARN]) is S.WARN
    assert aggregate_status([S.WARN, S.FAIL, S.NOT_AVAILABLE]) is S.FAIL


def test_not_available_is_distinct_from_pass_and_from_fail(report):
    # The sample fits nothing (every family prescribed) and carries no kinetic data.
    assert report["summary"]["diagnostic_fit"] == NOT_AVAILABLE
    kinetic = report["independent_validation"]["kinetic_pressure"]
    assert kinetic["status"] == NOT_AVAILABLE
    assert "core_profiles" in kinetic["slices"][0]["reason"]
    assert report["independent_validation"]["thomson_pressure"]["status"] == NOT_AVAILABLE
    # ... while a check with evidence that disagrees says so.
    assert report["physical_validity"]["pressure_consistency"]["status"] == FAIL
    assert report["status"] == FAIL
    assert NOT_AVAILABLE != PASS != FAIL


def test_a_partially_assessable_category_is_indeterminate(report):
    """Eight slices pass the virial closure and the dead one cannot be decided:
    the check is ``indeterminate``, because part of the evidence is missing."""
    virial = report["physical_validity"]["virial"]
    assert virial["counts"] == {INDETERMINATE: 1, PASS: 8}
    assert virial["status"] == INDETERMINATE


# ---------------------------------------------------------------------------
# numerical robustness
# ---------------------------------------------------------------------------

def test_a_slice_without_a_boundary_is_indeterminate_not_a_number(report):
    dead = _slice(report, "physical_validity", "virial", DEAD)
    assert dead["status"] == INDETERMINATE
    assert "non-finite" in dead["reason"]
    for name in ("s_1", "beta_p", "li", "B_pa"):
        assert dead[name] is None  # NaN serializes as null, never as an unbounded value
    live = _slice(report, "physical_validity", "virial", LIVE)
    assert live["status"] == PASS
    assert 0 < live["beta_p"] < 10 and 0 < live["li"] < 3
    # Both closures are named; neither is silently the other.
    assert live["beta_p"] == live["beta_p_lao"] and live["li"] == live["li_lao"]
    assert live["li_bongard"] != live["li_lao"]
    assert _slice(report, "independent_validation", "diamagnetic_energy", DEAD)["status"] == INDETERMINATE


def test_structure_fails_the_dead_slice_and_passes_the_live_one(report):
    live = _slice(report, "verification", "structure", LIVE)
    assert live["status"] == PASS and live["issues"] == []
    assert live["psi_finite"] and live["profiles_1d_psi_monotonic"] and live["magnetic_axis_inside_lcfs"]
    assert live["volume"] > 0 and live["cross_section_area"] > 0
    dead = _slice(report, "verification", "structure", DEAD)
    assert dead["status"] == FAIL
    assert {"degenerate_flux", "missing_lcfs"} <= set(dead["issues"])
    assert dead["volume"] is None


def test_continuity_needs_two_slices_and_flags_the_current_collapse(sample, report):
    assert verify_continuity(sample, time_slice=LIVE)["status"] == NOT_AVAILABLE
    assert verify_continuity(sample, time_slice=[0, 1, 2])["status"] == PASS
    whole = report["verification"]["continuity"]
    assert whole["status"] == WARN
    assert whole["time_strictly_increasing"] and whole["ip_sign_consistent"]
    assert whole["ip_max_relative_step"] > 0.5  # 46 kA to zero at the dead slice
    broken = copy.deepcopy(sample)
    broken["equilibrium.time_slice.3.time"] = float(broken["equilibrium.time_slice.2.time"])
    assert verify_continuity(broken)["status"] == FAIL


def test_an_undeclared_convention_is_indeterminate_not_a_pass(report):
    live = _slice(report, "verification", "convention", LIVE)
    assert live["status"] == INDETERMINATE
    assert live["issues"] == ["cocos_undeclared"]
    assert live["cocos"] is None and live["candidates"] == [1, 2]


# ---------------------------------------------------------------------------
# the providers are composed, not reimplemented
# ---------------------------------------------------------------------------

def test_diagnostic_fit_is_efit_quality_family_by_family(efit):
    from vaft.omas.efit_quality import fit_quality_metrics

    metrics = fit_quality_metrics(efit, time_slice=0)
    fit = validate_magnetic_fit(efit, time_slice=0)
    for family in ("bpol_probe", "flux_loop"):
        assert fit[family]["status"] == PASS
        assert fit[family]["z_rms"] == metrics["families"][family]["z_rms"]
        assert fit[family]["chi_squared_sum"] == metrics["families"][family]["chi_squared_sum"]
    assert fit["pf_current"]["status"] == NOT_AVAILABLE and fit["pf_current"]["fit_role"] == "prescribed"
    # The fixture's Ip constraint is ten sigma off, beyond the registered fail tolerance.
    assert fit["ip"]["z"] == metrics["scalars"]["ip"]["z"] == 10.0
    assert fit["ip"]["status"] == FAIL and describe("diagnostic_fit.ip").tolerance[1] < 10.0
    assert fit["diamagnetic_flux"]["status"] == NOT_AVAILABLE
    assert fit["global"]["chi_squared_reduced"] == metrics["chi_squared_reduced"]
    assert fit["global"]["status"] == PASS


def test_convergence_is_kept_apart_from_fit_and_from_validity(sample, efit):
    # No solver evidence at all on the packaged sample: not a pass, not a fail.
    assert verify_convergence(sample, time_slice=LIVE)["status"] == NOT_AVAILABLE
    # The fixture stopped inside EFIT's acceptance tolerance but short of its exit tolerance.
    result = verify_convergence(efit, time_slice=0)
    assert result["status"] == WARN
    assert result["within_acceptance_tolerance"] and not result["reached_exit_tolerance"]
    assert result["final_error"] == pytest.approx(8.0e-3)
    # A negative run flag is decisive on its own, whatever the iteration error says.
    flagged = copy.deepcopy(efit)
    flagged["equilibrium.code.output_flag"] = -np.ones(9, dtype=int)
    assert verify_convergence(flagged, time_slice=0)["status"] == FAIL
    assert verify_convergence(flagged, time_slice=0)["output_flag"] == -1


def test_virial_quantities_are_the_wrappers_numbers(sample):
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods

    expected = compute_virial_equilibrium_quantities_ods(copy.deepcopy(sample), time_slice=LIVE)[LIVE]
    physical = validate_physical(sample, time_slice=LIVE)
    virial = physical["virial"]
    for name in ("s_1", "s_2", "s_3", "alpha", "B_pa", "beta_p", "li", "W_kin"):
        assert virial[name] == pytest.approx(expected[name])


def test_pressure_consistency_names_both_definitions(report):
    """The sample's pressure profile integrates to a beta_p two orders below
    the virial one -- a real disagreement the report must state, not hide."""
    from vaft.omas.sample import sample_ods
    from vaft.process.equilibrium import as_equilibrium, derive_global_descriptors

    live = _slice(report, "physical_validity", "pressure_consistency", LIVE)
    descriptors = derive_global_descriptors(as_equilibrium(sample_ods(), time_index=LIVE)).values
    assert live["beta_p_pressure_integral"] == pytest.approx(descriptors["beta_p_boundary_average"].value)
    assert live["beta_p_virial"] == _slice(report, "physical_validity", "virial", LIVE)["beta_p"]
    assert live["ratio"] == pytest.approx(live["beta_p_pressure_integral"] / live["beta_p_virial"])
    assert live["status"] == FAIL and abs(live["log_ratio"]) > describe("physical_validity.pressure_consistency").tolerance[1]
    assert _slice(report, "physical_validity", "pressure_consistency", DEAD)["status"] == NOT_AVAILABLE


def test_the_reconstructed_diamagnetic_flux_disagrees_with_the_measurement_in_sign(report):
    """A finding on shot 39915 the report surfaces: the reconstructed flux is
    paramagnetic where the loop measures diamagnetic, while the *measured*
    flux closes the virial energy balance to half a percent."""
    flux = _slice(report, "physical_validity", "diamagnetic_flux", LIVE)
    assert flux["status"] == FAIL
    assert flux["sign_agreement"] is False
    assert flux["measured"] < 0 < flux["computed"]
    energy = _slice(report, "independent_validation", "diamagnetic_energy", LIVE)
    assert energy["status"] == PASS
    assert energy["mui_measured"] < 0
    assert abs(energy["log_ratio"]) < 0.02
    assert energy["W_diamagnetic"] == pytest.approx(energy["W_kin_virial"], rel=0.02)


def test_measurements_can_arrive_on_a_separate_diagnostics_ods(sample):
    equilibrium_only = ODS()
    equilibrium_only["equilibrium"] = copy.deepcopy(sample["equilibrium"])
    alone = validate_physical(equilibrium_only, time_slice=LIVE)
    assert alone["diamagnetic_flux"]["status"] == NOT_AVAILABLE
    assert validate_independent(equilibrium_only, time_slice=LIVE)["diamagnetic_energy"]["status"] == NOT_AVAILABLE
    with_measurement = validate_physical(equilibrium_only, time_slice=LIVE, diagnostics=sample)
    assert with_measurement["diamagnetic_flux"]["status"] == FAIL
    assert "magnetics" not in equilibrium_only  # the graft happened on the working copy


# ---------------------------------------------------------------------------
# independent measurements
# ---------------------------------------------------------------------------

def _core_profiles_from(sample, index, *, scale, electrons_only):
    root = f"equilibrium.time_slice.{index}"
    profiles = ODS()
    profiles["core_profiles.time"] = np.array([float(sample["equilibrium.time"][index])])
    profiles["core_profiles.profiles_1d.0.time"] = float(sample["equilibrium.time"][index])
    profiles["core_profiles.profiles_1d.0.grid.rho_tor_norm"] = np.asarray(sample[f"{root}.profiles_1d.rho_tor_norm"], float)
    pressure = scale * np.asarray(sample[f"{root}.profiles_1d.pressure"], float)
    if electrons_only:
        profiles["core_profiles.profiles_1d.0.electrons.pressure_thermal"] = pressure
    else:
        profiles["core_profiles.profiles_1d.0.pressure_thermal"] = pressure
    return profiles


def test_kinetic_pressure_built_from_the_reconstruction_agrees(sample):
    profiles = _core_profiles_from(sample, LIVE, scale=1.0, electrons_only=False)
    result = validate_independent(sample, time_slice=LIVE, kinetic_profiles=profiles)["kinetic_pressure"]
    assert result["status"] == PASS
    assert result["coverage"] == "thermal_total" and result["coordinate"] == "rho_tor_norm"
    assert result["log_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert result["normalized_rms"] == pytest.approx(0.0, abs=1e-9)
    assert result["time_offset"] == 0.0


def test_kinetic_pressure_far_from_the_reconstruction_fails(sample):
    profiles = _core_profiles_from(sample, LIVE, scale=5.0, electrons_only=False)
    result = validate_independent(sample, time_slice=LIVE, kinetic_profiles=profiles)["kinetic_pressure"]
    assert result["status"] == FAIL
    assert result["sum_ratio"] == pytest.approx(5.0)


def test_electron_only_coverage_is_one_sided(sample):
    """Electrons alone exceeding the reconstructed total is a failure; falling
    short of it is at most a warning, because the ions are unmeasured."""
    above = _core_profiles_from(sample, LIVE, scale=5.0, electrons_only=True)
    below = _core_profiles_from(sample, LIVE, scale=0.2, electrons_only=True)
    high = validate_independent(sample, time_slice=LIVE, kinetic_profiles=above)["kinetic_pressure"]
    low = validate_independent(sample, time_slice=LIVE, kinetic_profiles=below)["kinetic_pressure"]
    assert high["coverage"] == low["coverage"] == "electrons"
    assert high["status"] == FAIL
    assert low["status"] == WARN


def test_a_kinetic_slice_too_far_in_time_is_not_used(sample):
    profiles = _core_profiles_from(sample, LIVE, scale=1.0, electrons_only=False)
    profiles["core_profiles.time"] = np.array([0.5])
    profiles["core_profiles.profiles_1d.0.time"] = 0.5
    result = validate_independent(sample, time_slice=LIVE, kinetic_profiles=profiles)["kinetic_pressure"]
    assert result["status"] == NOT_AVAILABLE
    assert "away" in result["reason"]


def test_thomson_pressure_is_mapped_through_the_2d_psi(sample):
    """Channels on grid nodes along the midplane, carrying exactly the
    reconstructed pressure at their flux surface: the check must pass and the
    mapping must be the equilibrium's own psi(R, Z)."""
    root = f"equilibrium.time_slice.{LIVE}"
    r_grid = np.asarray(sample[f"{root}.profiles_2d.0.grid.dim1"], float)
    z_grid = np.asarray(sample[f"{root}.profiles_2d.0.grid.dim2"], float)
    psi_2d = np.asarray(sample[f"{root}.profiles_2d.0.psi"], float)
    axis, boundary = float(sample[f"{root}.global_quantities.psi_axis"]), float(sample[f"{root}.global_quantities.psi_boundary"])
    z_index = int(np.argmin(np.abs(z_grid - float(sample[f"{root}.global_quantities.magnetic_axis.z"]))))
    psi_1d = np.asarray(sample[f"{root}.profiles_1d.psi"], float)
    pressure_1d = np.asarray(sample[f"{root}.profiles_1d.pressure"], float)
    psi_norm_1d = (psi_1d - axis) / (boundary - axis)

    diagnostics = ODS()
    diagnostics["thomson_scattering.time"] = np.array([float(sample["equilibrium.time"][LIVE])])
    channel = 0
    for r_index in range(0, r_grid.size, 4):
        psi_norm = (psi_2d[r_index, z_index] - axis) / (boundary - axis)
        if not 0.05 <= psi_norm <= 0.95:
            continue
        pressure = float(np.interp(psi_norm, psi_norm_1d, pressure_1d))
        if pressure <= 0:
            continue
        density = 1.0e19
        base = f"thomson_scattering.channel.{channel}"
        diagnostics[f"{base}.position.r"] = float(r_grid[r_index])
        diagnostics[f"{base}.position.z"] = float(z_grid[z_index])
        diagnostics[f"{base}.n_e.data"] = np.array([density])
        diagnostics[f"{base}.t_e.data"] = np.array([pressure / (density * 1.602176634e-19)])
        channel += 1
    assert channel >= 5

    result = validate_independent(sample, time_slice=LIVE, diagnostics=diagnostics)["thomson_pressure"]
    assert result["status"] == PASS
    assert result["channels_inside"] == channel and result["coverage"] == "electrons"
    assert result["log_ratio"] == pytest.approx(0.0, abs=1e-6)
    for base in (f"thomson_scattering.channel.{i}" for i in range(channel)):
        diagnostics[f"{base}.t_e.data"] = np.asarray(diagnostics[f"{base}.t_e.data"]) * 8.0
    assert validate_independent(sample, time_slice=LIVE, diagnostics=diagnostics)["thomson_pressure"]["status"] == FAIL


# ---------------------------------------------------------------------------
# contract: compact results, one registry, no side effects
# ---------------------------------------------------------------------------

def test_every_check_in_the_report_is_registered_and_vice_versa(report):
    seen = set()
    for category in EQUILIBRIUM_CATEGORIES:
        for check in report[category]:
            key = f"{category}.{check}"
            assert describe(key).category == category
            seen.add(key)
    assert seen == set(CHECKS)
    with pytest.raises(KeyError, match="no registered validation check"):
        describe("physical_validity.nonexistent")


def test_registry_entries_are_internally_consistent():
    for spec in CHECKS.values():
        assert spec.key.startswith(spec.category + ".")
        if spec.measure == "rule":
            assert spec.tolerance is None
        else:
            warn, fail = spec.tolerance
            assert 0 <= warn <= fail
    with pytest.raises(ValueError, match="tolerance"):
        CheckSpec("x.y", "x", "1", "p", "m", "relative", None)
    with pytest.raises(ValueError, match="unknown measure"):
        CheckSpec("x.y", "x", "1", "p", "m", "sigma", (1.0, 2.0))


def test_results_carry_numbers_and_a_status_not_invariant_metadata(report):
    """#253 §4-§5: units, tolerances, methods and provenance are stated once."""
    forbidden = {"unit", "units", "tolerance", "provenance", "method", "definition", "reference"}
    for category in EQUILIBRIUM_CATEGORIES:
        for check, entry in report[category].items():
            for result in entry.get("slices", [entry]):
                assert not (forbidden & set(result)), f"{category}.{check}: {forbidden & set(result)}"
    assert "provenance" in report and report["provenance"]["conventions"]["psi_per_radian"] is True


def test_provenance_records_the_wrappers_actual_settings(report):
    from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods

    source = inspect.getsource(compute_virial_equilibrium_quantities_ods)
    methods = report["provenance"]["methods"]
    assert f"n_points={methods['boundary_normalization']['n_points']}" in source
    assert f"samples_per_axis={methods['cell_weighting']['samples_per_axis']}" in source
    assert methods["virial_closure"] == "lao"
    assert report["provenance"]["equilibrium"]["time_slice_count"] == 9


def test_the_report_never_mutates_its_input(sample):
    before = set(sample.flat().keys())
    validate_equilibrium(sample, time_slice=LIVE)
    validate_physical(sample, time_slice=LIVE)
    assert set(sample.flat().keys()) == before
    # The virial wrapper writes the geometric axis; the report ran it on a copy.
    assert f"equilibrium.time_slice.{LIVE}.boundary.geometric_axis.r" not in before


def test_the_report_serializes_deterministically_without_nan(sample, report):
    encoded = json.dumps(report, sort_keys=True)
    assert "NaN" not in encoded and "Infinity" not in encoded
    assert json.loads(encoded) == report
    assert encoded == json.dumps(validate_equilibrium(sample), sort_keys=True)


def test_no_plotting_or_database_layer_is_imported_by_the_report():
    """The report's own path names no plotting or FileDB layer.

    Asserted two ways: the module sources import neither matplotlib nor
    ``vaft.plot``, and running the report leaves no ``vaft.plot`` or
    ``vaft.database.production_qa`` module loaded.  ``matplotlib`` itself is
    *not* asserted absent from ``sys.modules``: ``import vaft.omas`` -- the
    package every provider lives in -- pulls it in at package-init time, which
    is a fact about ``vaft.omas``, not about this report.
    """
    import ast
    from pathlib import Path

    from vaft.validation import equilibrium, registry

    for module in (equilibrium, registry):
        tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
        names = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module]
        names += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        offenders = [name for name in names if name.startswith(("matplotlib", "vaft.plot", "vaft.database"))]
        assert offenders == [], f"{module.__name__} imports {offenders}"

    code = (
        "import sys\n"
        "from vaft.omas.sample import sample_ods\n"
        "from vaft.validation import validate_equilibrium\n"
        "validate_equilibrium(sample_ods(), time_slice=0)\n"
        "print(','.join(sorted(m for m in sys.modules if m.startswith(('vaft.plot', 'vaft.database.production_qa')))))\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"the report pulled in: {result.stdout.strip()}"


def test_the_public_name_lives_on_the_package(sample):
    import vaft.validation
    from vaft.validation import equilibrium

    assert vaft.validation.validate_equilibrium is equilibrium.validate_equilibrium
    assert "validate_equilibrium" in vaft.validation.__all__
