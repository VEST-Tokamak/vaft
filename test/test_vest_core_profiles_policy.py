"""The VEST kinetic-profile policy lives in vest.yaml and nowhere else (issue #420).

Two halves.  The resolver: the ``core_profiles`` block resolves per shot with
its provenance, every value carries a status, and a malformed block is
refused rather than defaulted.  The layering: the generic processing modules
hold no VEST number and never import the machine-mapping layer, so the only
way a VEST value reaches a fit is through a pipeline that resolved it.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

import pytest

from vaft.machine_mapping.core_profiles import (
    POLICY_STATUSES,
    CoreProfilesPolicy,
    policy_for_ods,
    shot_number,
    vest_core_profiles_policy,
)
from vaft.machine_mapping.utils import VestConfigurationError

ROOT = pathlib.Path(__file__).resolve().parent.parent


# --- the resolver ---------------------------------------------------------------


def test_policy_resolves_for_a_two_diagnostic_shot():
    policy = vest_core_profiles_policy(48224)

    assert isinstance(policy, CoreProfilesPolicy)
    assert policy.shot == 48224
    assert policy.coordinate == "rho_tor_norm"
    assert policy.ti_te_ratio == pytest.approx(0.17)
    assert policy.ti_te_ratio_sigma == pytest.approx(0.08)
    assert policy.ti_te_ratio_status == "inferred"
    assert policy.impurity_species == ("C", "O")
    assert policy.impurity_fractions == {"C": pytest.approx(1e-2), "O": pytest.approx(1e-2)}
    assert policy.impurity_status == {"C": "assumed", "O": "assumed"}
    assert policy.source == "vest.yaml:diagnostics.core_profiles"


def test_policy_carries_the_derivation_provenance_verbatim():
    policy = vest_core_profiles_policy(48224)
    notes = policy.provenance["ti_te_ratio"]

    assert notes["shots"] == [48224, 48226, 48233]
    assert notes["time_window_ms"] == [299, 301]
    assert "pressure-matching" in notes["estimator"]
    assert "0.170" in notes["cross_check"]
    assert "ids_test/fit_ti_te_ratio.py" in notes["derivation_script"]
    assert "revision" in policy.provenance


def test_policy_is_the_same_for_every_shot_until_a_revision_says_otherwise():
    early, late = vest_core_profiles_policy(39915), vest_core_profiles_policy(48224)

    assert early.shot != late.shot
    assert (early.coordinate, early.ti_te_ratio, early.impurity_fractions) == (
        late.coordinate, late.ti_te_ratio, late.impurity_fractions
    )
    assert early.provenance["revision"]["revision_index"] is None


def test_text_records_name_value_status_and_source():
    policy = vest_core_profiles_policy(48224)

    assert policy.ti_te_ratio_text() == (
        "ti_te_ratio=0.17; sigma=0.08; status=inferred; "
        "source=vest.yaml:diagnostics.core_profiles; base"
    )
    unknown = vest_core_profiles_policy(None)
    assert unknown.ti_te_ratio_text().endswith("base; shot=unknown")
    assert policy.impurity_text().startswith("impurity_fractions=C=0.01(assumed),O=0.01(assumed)")


def _yaml_with(tmp_path, **overrides):
    """A minimal vest.yaml holding only the core_profiles block, patched."""
    block = {
        "coordinate": "rho_tor_norm",
        "ti_te_ratio": {"value": 0.17, "sigma": 0.08, "status": "inferred"},
        "impurities": {"species": ["C"], "fractions": {"C": {"value": 0.01, "status": "assumed"}}},
    }
    for dotted, value in overrides.items():
        node = block
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        node[last] = value
    import yaml

    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump({0: {"diagnostics": {"core_profiles": block}}}))
    return str(path)


def test_unknown_status_is_refused(tmp_path):
    with pytest.raises(VestConfigurationError, match="status must be one of"):
        vest_core_profiles_policy(1, info_file=_yaml_with(tmp_path, **{"ti_te_ratio.status": "guessed"}))
    with pytest.raises(VestConfigurationError, match="status must be one of"):
        vest_core_profiles_policy(1, info_file=_yaml_with(tmp_path, **{"impurities.fractions.C.status": "true"}))


def test_unknown_coordinate_is_refused(tmp_path):
    with pytest.raises(VestConfigurationError, match="coordinate must be one of"):
        vest_core_profiles_policy(1, info_file=_yaml_with(tmp_path, coordinate="rho"))


@pytest.mark.parametrize("bad", [-0.1, "nan", "inf", "abc"])
def test_negative_or_non_finite_values_are_refused(tmp_path, bad):
    with pytest.raises(VestConfigurationError):
        vest_core_profiles_policy(1, info_file=_yaml_with(tmp_path, **{"ti_te_ratio.value": bad}))


def test_a_species_without_a_fraction_entry_is_refused(tmp_path):
    with pytest.raises(VestConfigurationError, match="impurities.fractions.O"):
        vest_core_profiles_policy(1, info_file=_yaml_with(tmp_path, **{"impurities.species": ["C", "O"]}))


def test_statuses_are_the_three_the_contract_names():
    assert POLICY_STATUSES == ("assumed", "measured", "inferred")


def test_policy_is_part_of_the_shot_provenance_report():
    from vaft.machine_mapping.provenance import vest_processing_provenance

    report = vest_processing_provenance(48224)
    assert report["core_profiles"]["coordinate"] == "rho_tor_norm"
    assert report["core_profiles"]["ti_te_ratio"]["status"] == "inferred"


# --- the layering ---------------------------------------------------------------


@pytest.mark.parametrize("module", ["vaft/process/profile.py", "vaft/process/atomic.py"])
def test_generic_process_modules_hold_no_vest_policy(module):
    """No VEST number, and no import of the machine layer -- prose may name it."""
    import ast

    # Explicit: the sources carry non-ASCII (a Greek rho in a profile.py
    # comment), and Windows would otherwise decode them as cp1252 and raise.
    source = (ROOT / module).read_text(encoding="utf-8")
    assert "TI_TE_RATIO_VEST = " not in source
    assert "DEFAULT_IMPURITY_FRACTIONS = {" not in source

    imported = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert not [name for name in imported if name.startswith("vaft.machine_mapping")], imported


@pytest.mark.parametrize("module", ["vaft.process.profile", "vaft.process.atomic"])
def test_importing_a_generic_process_module_does_not_load_the_machine_layer(module):
    code = textwrap.dedent(
        f"""
        import sys
        import {module}
        print(' '.join(sorted(m for m in sys.modules if m.startswith('vaft.machine_mapping'))))
        """
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert result.stdout.split() == []


# --- the ODS rule -----------------------------------------------------------------


def test_the_coordinate_list_matches_the_process_layer():
    """The policy validates coordinates without importing the process layer."""
    import vaft.process.profile as profile
    from vaft.machine_mapping.core_profiles import _COORDINATES

    assert _COORDINATES == profile.COORDINATES


def test_reading_the_shot_number_does_not_materialize_an_ids():
    """An OMAS read of a missing leaf creates its parents; the probe must not."""
    from omas import ODS

    ods = ODS()
    assert shot_number(ods) is None
    assert list(ods.keys()) == []

    ods["dataset_description.data_entry.pulse"] = 48224
    assert shot_number(ods) == 48224


def test_policy_for_ods_prefers_the_explicit_shot_then_the_ods_then_the_base():
    from omas import ODS

    ods = ODS()
    ods["dataset_description.data_entry.pulse"] = 48224

    assert policy_for_ods(ods).shot == 48224
    assert policy_for_ods(ods, 39915).shot == 39915
    bare = policy_for_ods(ODS())
    assert bare.shot is None
    assert "shot=unknown" in bare.revision_text


def test_the_retired_atomic_defaults_are_gone():
    """Retirement is the same on both sides: the names do not exist."""
    import vaft.process.atomic as atomic

    for name in ("DEFAULT_IMPURITY_FRACTIONS", "DEFAULT_LINE_RADIATION_SPECIES"):
        assert not hasattr(atomic, name)
        with pytest.raises(AttributeError):
            getattr(atomic, name)
