"""Shot-number extraction from diagnostic .mat filenames.

A layout the extractor does not recognise is skipped by the corrective-pipeline
watcher, so the shot silently never reaches the database. That is exactly what
happened to the whole 48222-48234 campaign, whose files are named
``NeTe_<shot>.mat`` -- a layout neither of the original two patterns matched.
"""

import importlib.util
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_2_corrective_data_update"
    / "update_thomson_scattering_and_core_profile.py"
)

pytestmark = pytest.mark.skipif(
    not WORKFLOW.exists(), reason="corrective-pipeline workflow script not present"
)


@pytest.fixture(scope="module")
def extract():
    pytest.importorskip("h5pyd")
    pytest.importorskip("omas")
    pytest.importorskip("omfit_classes")
    spec = importlib.util.spec_from_file_location("_ts_updater", WORKFLOW)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.extract_shotnumber_of_thomson_scattering


@pytest.mark.parametrize(
    "fname,expected",
    [
        # layout that regressed: the entire 48xxx campaign
        ("NeTe_48223.mat", 48223),
        ("NeTe_48234.mat", 48234),
        ("nete_48222.mat", 48222),          # case-insensitive
        ("NeTe-48226.mat", 48226),          # dash separator
        # long-standing layouts that must keep working
        ("44405_NeTe.mat", 44405),
        ("Shot40330_v10.mat", 40330),
        ("NeTe_Shot48223_v9.mat", 48223),
        ("NeTe_Shot48223_v9_rev.mat", 48223),
        ("NeTe_Shot40330.mat", 40330),
    ],
)
def test_known_layouts(extract, fname, expected):
    assert extract(fname) == expected


def test_shot_tag_wins_over_trailing_number(extract):
    """'NeTe_Shot<shot>_v9' must resolve via the Shot tag, not the version digits."""
    assert extract("NeTe_Shot48223_v9_rev.mat") == 48223


def test_unknown_layout_returns_none(extract):
    # returning None (rather than guessing) lets the watcher log and skip
    assert extract("random_file.mat") is None
    assert extract("NeTe.mat") is None


def test_every_file_in_the_diagnostic_dir_parses(extract):
    """Every real .mat in /srv/vest.diagnostic must map to a shot number.

    Guards against a new naming convention silently dropping a campaign again.
    Skipped off the VEST intranet.
    """
    import re

    watch = Path("/srv/vest.diagnostic")
    if not watch.is_dir():
        pytest.skip("/srv/vest.diagnostic not mounted")
    cx = re.compile(r"^(?:IDS|CES)[_-](\d+)", re.IGNORECASE)

    unparsed = []
    for path in sorted(watch.glob("*.mat")):
        name = path.name
        if cx.match(name):          # ion diagnostics belong to the CES updater
            continue
        shot = extract(name)
        if shot is None:
            unparsed.append(name)
        else:
            assert str(shot) in name, f"{name}: extracted {shot} not present in the name"
    assert not unparsed, f"unrecognised Thomson filename layout(s): {unparsed}"
