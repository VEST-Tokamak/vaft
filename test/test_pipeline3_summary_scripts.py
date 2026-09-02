"""The Pipeline 3 summary wrappers against what their consumers actually call.

`notebooks/verification_and_validation.ipynb` imports these scripts and calls
their `generate_*` entry points. Nothing covered this package before, so when
the scripts were rewritten into thin `vaft.database` wrappers the notebook broke
silently (issues #151/#181). These tests pin the contract the notebook relies
on: the entry point exists, its keyword arguments are the ones the notebook
passes, and the sheet schema comes from a `vaft.database` summary preset rather
than a constant duplicated in the script.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pandas as pd
import pytest

from vaft.database import get_summary_preset


WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_3_data_summary"
)

#: script stem -> (entry point, summary preset, output sheet)
SCRIPTS = {
    "gen_volume_averaged_parameter_sheet": (
        "generate_volume_averaged_parameter_sheet",
        "volume_averaged",
        "volume_averaged_parameters.xlsx",
    ),
    "gen_equilibrium_global_history": (
        "generate_equilibrium_global_history_excel",
        "equilibrium_global",
        "equilibrium_global_history.xlsx",
    ),
    "gen_core_profiles_history": (
        "generate_core_profiles_history_excel",
        "core_profiles",
        "core_profiles_history.xlsx",
    ),
}

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="workflow scripts are not part of the distribution"
)


def _load(stem: str):
    spec = importlib.util.spec_from_file_location(stem, WORKFLOW / f"{stem}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("stem", sorted(SCRIPTS))
def test_entry_point_accepts_the_arguments_its_callers_pass(stem):
    entry_point, _, filename = SCRIPTS[stem]
    module = _load(stem)

    assert module.OUTPUT_FILENAME == filename
    function = getattr(module, entry_point, None)
    assert callable(function), f"{stem}.{entry_point} is missing"

    signature = inspect.signature(function)
    parameters = signature.parameters
    # The notebook passes the shot range positionally and the rest by keyword.
    assert list(parameters)[0] == "shot_range"
    for keyword in ("source", "directory", "output_path", "rebuild"):
        assert keyword in parameters, keyword
        assert parameters[keyword].kind is inspect.Parameter.KEYWORD_ONLY
    # `source` is the name now; `directory` stays bindable for existing callers.
    signature.bind(None, source="main", output_path="out.xlsx")
    signature.bind(None, directory="public", output_path="out.xlsx")


@pytest.mark.parametrize("stem", sorted(SCRIPTS))
def test_checked_in_sheet_matches_its_summary_preset(stem):
    _, preset_name, filename = SCRIPTS[stem]
    sheet = WORKFLOW / filename
    if not sheet.exists():
        pytest.skip(f"{filename} has not been materialized in this checkout")

    columns = set(pd.read_excel(sheet).columns)
    missing = [c for c in get_summary_preset(preset_name).columns if c not in columns]
    assert missing == [], f"{filename} is missing preset columns: {missing}"


def test_summary_preset_is_the_only_schema_source():
    """The scripts must not reintroduce a hand-maintained column constant.

    A duplicated `EXPECTED_COLUMNS` is exactly what drifted out of sync with the
    generated sheets in issue #151.
    """
    offenders = []
    for stem in SCRIPTS:
        module = _load(stem)
        for name in ("EXPECTED_COLUMNS", "REQUIRED_COLUMNS_FOR_REPAIR", "SORT_COLUMNS"):
            if hasattr(module, name):
                offenders.append(f"{stem}.{name}")
    assert offenders == [], (
        "these scripts duplicate a schema that vaft.database.get_summary_preset owns: "
        + ", ".join(offenders)
    )
