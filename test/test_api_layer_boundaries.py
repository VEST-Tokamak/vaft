import importlib
import importlib.util
import warnings

import numpy as np
import pytest


def test_detect_active_window_selects_peak_containing_region():
    from vaft.process import detect_active_window

    time = np.arange(8, dtype=float)
    values = np.array([0.0, 0.2, 0.3, 0.0, 0.4, 0.9, 0.4, 0.0])

    assert detect_active_window(time, values, threshold=0.1) == (4.0, 6.0)


def test_detect_active_window_uses_full_range_when_threshold_is_not_reached():
    from vaft.process import detect_active_window

    assert detect_active_window([0.0, 1.0], [0.0, 0.0], threshold=0.1) == (0.0, 1.0)


def test_legacy_process_alias_warns_and_preserves_result():
    from vaft.process.signal_processing import vfit_signal_start_end

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vfit_signal_start_end([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])

    assert result == (1.0, 1.0)
    assert any(item.category is DeprecationWarning for item in caught)
    assert "detect_active_window" in str(caught[0].message)


def test_vaft_process_reaches_no_verdicts():
    """The boundary rule #253 §2 states, made executable (issue #337).

    `vaft.process` transforms, infers and checks its own preconditions. Whether
    a datum or a result is *credible* is a verdict, and verdicts live in
    `vaft.validation` -- one namespace, one status vocabulary. A new
    `vaft.process.validate_*` is how that boundary erodes, so it fails here.

    The two names still present are deprecating aliases, kept because they were
    public; both delegate and warn.
    """
    import vaft.process

    aliases = {"validate_equilibrium", "validate_impa"}
    offenders = {
        name
        for name in dir(vaft.process)
        if name.startswith("validate_") and name not in aliases
    }
    assert not offenders, (
        f"{sorted(offenders)} reach a verdict from vaft.process. A precondition is "
        "named check_*_requirements and stays; an assessment belongs in "
        "vaft.validation (issues #253, #337)."
    )


def test_the_renamed_equilibrium_precondition_keeps_its_old_name_working():
    """A public name that moves keeps resolving, warns, and answers identically."""
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process import check_equilibrium_requirements, validate_equilibrium

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = validate_equilibrium(EquilibriumData(), required_for="global")

    assert any(item.category is DeprecationWarning for item in caught)
    assert "check_equilibrium_requirements" in str(caught[0].message)
    assert legacy == check_equilibrium_requirements(
        EquilibriumData(), required_for="global"
    )


def test_the_moved_impa_grading_keeps_its_old_names_working():
    """Both process-side names delegate to `vaft.validation.impa` and warn (#339).

    Their arguments are a whole processed shot, so the delegation is what is
    asserted here; `test_validation_impa.py` covers the answers.
    """
    from vaft.process import grade_impa_quality, validate_impa

    for legacy in (validate_impa, grade_impa_quality):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(TypeError):
                legacy()

        assert any(item.category is DeprecationWarning for item in caught)
        assert "vaft.validation.impa.validate_impa" in str(caught[0].message)


def test_the_moved_cocos_check_keeps_its_old_name_working():
    """`validate_cocos` is a verdict, so it lives in `vaft.validation.cocos` (#337)."""
    from vaft.data.equilibrium import EquilibriumData
    from vaft.process import cocos as process_cocos
    from vaft.validation.cocos import validate_cocos

    equilibrium = EquilibriumData(ip=1.0e5, bt0=0.1, f=[0.04, 0.04], q=[1.0, 2.0],
                                  psi_axis=0.0, psi_boundary=0.1, pprime=[-1.0, -1.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy = process_cocos.validate_cocos(equilibrium, 3, sigma_ip=1)

    deprecations = [item for item in caught if item.category is DeprecationWarning]
    assert deprecations
    assert "vaft.validation.cocos.validate_cocos" in str(deprecations[0].message)
    assert legacy == validate_cocos(equilibrium, 3, sigma_ip=1)
    assert legacy.issues  # COCOS 3 is the wrong index here, so there is something to compare
    assert process_cocos.validate_cocos.__doc__.startswith("Deprecated")


def test_the_validation_layer_has_one_status_vocabulary():
    """#253 §4: findings and verdicts are different shapes, not rival vocabularies.

    `ValidationReport` is a findings list with severities; `ValidationStatus` is
    the verdict vocabulary. They meet at `ValidationReport.status`, which is what
    stops a caller composing findings into a report from inventing a third set of
    strings -- which is how the codebase came to have three.
    """
    from vaft.validation.model import (
        ValidationIssue,
        ValidationReport,
        ValidationStatus,
    )

    assert ValidationReport().status is ValidationStatus.PASS
    assert ValidationReport((ValidationIssue("warning", "c", "f", "m"),)).status is (
        ValidationStatus.WARN
    )
    assert ValidationReport((ValidationIssue("error", "c", "f", "m"),)).status is (
        ValidationStatus.FAIL
    )
    # The gate stays coarser than the verdict: a warning does not stop a caller.
    assert ValidationReport((ValidationIssue("warning", "c", "f", "m"),)).valid


def test_the_findings_model_resolves_from_every_historical_path():
    """It moved to the validation layer; the old import sites still work."""
    from vaft.data import ValidationReport as from_data
    from vaft.data.equilibrium import ValidationReport as from_equilibrium
    from vaft.validation.model import ValidationReport as canonical

    assert from_data is from_equilibrium is canonical
    assert canonical.__module__ == "vaft.validation.model"


def test_importing_the_data_layer_stays_light():
    """`vaft.data` gained a validation import; it must not have gained a layer.

    `vaft.validation.model` is enum-and-dataclass only, so this holds -- but the
    edge is new, and a heavier import landing in the validation package would
    otherwise reach `vaft.data` silently.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, vaft.data; "
            "print(','.join(sorted(m for m in sys.modules "
            "if m.startswith(('vaft.database', 'vaft.plot', 'vaft.omas', 'matplotlib')))))",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", f"import vaft.data pulled: {result.stdout.strip()}"


def test_machine_mapping_public_surface_is_canonical_but_legacy_imports_work():
    from vaft import machine_mapping

    # Canonical IDS entry points are reached through their own submodule, not
    # the package, because the two share a name -- see
    # test_entrypoint_names_never_collide_with_submodules below.
    assert "magnetics" not in machine_mapping.__all__
    assert "magnetics_from_raw_database" in machine_mapping.__all__
    assert not any(name.startswith("vfit_") for name in machine_mapping.__all__)
    assert "VEST_DiamagneticFlux" not in machine_mapping.__all__
    assert "pf_plasma" not in machine_mapping.__all__

    # Package-level lazy exports are cached after first access. Remove a value
    # that an earlier test may have resolved so this test remains order-independent.
    machine_mapping.__dict__.pop("vfit_barometry_static", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        legacy_builder = machine_mapping.vfit_barometry_static

    payload = {}
    legacy_builder(payload)
    assert payload["barometry"]["gauge"][0]["name"] == "PKR-251 Main Gauge"
    assert any(item.category is DeprecationWarning for item in caught)
    assert "diagnostic module" in str(caught[0].message)


def test_unimplemented_pf_plasma_module_is_not_shipped():
    assert importlib.util.find_spec("vaft.machine_mapping.pf_plasma") is None


def test_top_level_namespace_has_no_duplicate_exports():
    import vaft

    assert len(vaft.__all__) == len(set(vaft.__all__))


def test_entrypoint_names_never_collide_with_submodules():
    """No lazily-exported name may share a name with a submodule.

    A module's ``__getattr__`` only runs when normal attribute lookup fails, so
    importing ``vaft.machine_mapping.tf`` binds the *module* onto the package
    and permanently shadows any exported ``tf`` function -- silently, and
    depending only on which import ran first. Exporting both is therefore never
    safe, whichever one happens to win today.
    """
    from pathlib import Path

    from vaft import machine_mapping

    package_dir = Path(machine_mapping.__file__).parent
    submodules = {path.stem for path in package_dir.glob("*.py")} - {"__init__"}
    exported = (
        set(machine_mapping.__all__)
        | set(machine_mapping._EXPORT_MAP)
        | set(machine_mapping._LEGACY_EXPORT_MAP)
    )

    collisions = sorted(exported & submodules)
    assert not collisions, (
        f"these exported names are shadowed by same-named submodules: {collisions}. "
        f"Reach them through their module instead, e.g. "
        f"`from vaft.machine_mapping.{collisions[0]} import {collisions[0]}`, and add "
        f"the name to _ENTRYPOINT_MODULES."
    )


def test_shadowed_entrypoints_explain_themselves():
    from vaft import machine_mapping

    for name in ("magnetics", "tf", "dataset_description"):
        machine_mapping.__dict__.pop(name, None)
        with pytest.raises(AttributeError, match=rf"from vaft\.machine_mapping\.{name} import {name}"):
            getattr(machine_mapping, name)


# --- deprecated aliases carry their removal release (#2168, register L5, L6, P9, P10) ---

_ALIAS_REMOVAL = "0.11.0"


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("vfit_md", "vfit_equilibrium_magnetics"),
        ("vest_md_channel_definitions", "vest_equilibrium_magnetics_channel_definitions"),
    ],
)
def test_magnetics_module_aliases_warn_with_their_removal_release(alias, canonical):
    import vaft.machine_mapping.magnetics as magnetics

    with pytest.warns(DeprecationWarning, match=rf"magnetics\.{alias} is deprecated and will be removed in vaft {_ALIAS_REMOVAL}; use .*{canonical}"):
        value = getattr(magnetics, alias)
    assert value is getattr(magnetics, canonical)
    # The import form warns too, and binds the canonical object.
    namespace = {}
    with pytest.warns(DeprecationWarning, match=alias):
        exec(f"from vaft.machine_mapping.magnetics import {alias}", namespace)  # noqa: S102 - fixed text
    assert namespace[alias] is getattr(magnetics, canonical)
    # The canonical name is not deprecated.
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        getattr(magnetics, canonical)
    with pytest.raises(AttributeError, match="no attribute 'vfit_mdx'"):
        magnetics.vfit_mdx  # noqa: B018


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("vfit_md", "vfit_equilibrium_magnetics"),
        ("vest_md_channel_definitions", "vest_equilibrium_magnetics_channel_definitions"),
    ],
)
def test_machine_mapping_package_aliases_warn_with_their_removal_release(alias, canonical):
    import vaft.machine_mapping as machine_mapping
    import vaft.machine_mapping.magnetics as magnetics

    machine_mapping.__dict__.pop(alias, None)  # an earlier read caches the name
    with pytest.warns(DeprecationWarning, match=rf"{canonical}\(\) instead\. It will be removed in vaft {_ALIAS_REMOVAL}"):
        value = getattr(machine_mapping, alias)
    assert value is getattr(magnetics, canonical)


@pytest.mark.parametrize(
    ("module", "alias", "arguments"),
    [
        ("vaft.process.signal_processing", "VEST_CoilCurrentNoiseReduction", (np.zeros(400),)),
        ("vaft.process.signal_processing", "vfit_signal_startend", ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])),
        ("vaft.process.signal_processing", "vfit_signal_start_end", ([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])),
        ("vaft.process.signal_processing", "signal_onoffset", (np.arange(20.0), np.r_[np.zeros(5), np.ones(10), np.zeros(5)])),
        ("vaft.machine_mapping.utils", "process_signal", (np.arange(10.0), np.ones(10))),
    ],
)
def test_deprecated_wrappers_name_their_removal_release(module, alias, arguments):
    function = getattr(importlib.import_module(module), alias)
    with pytest.warns(DeprecationWarning, match=rf"{alias}\(\) is deprecated and will be removed in vaft {_ALIAS_REMOVAL}"):
        function(*arguments)


def test_the_removal_release_is_the_one_signal_processing_announces():
    from vaft.process import signal_processing

    assert signal_processing._ALIAS_REMOVAL == _ALIAS_REMOVAL
