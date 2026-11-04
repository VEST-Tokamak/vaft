"""vest.yaml ``machine_history``: the shot boundaries that used to be Python literals (#2267)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from vaft.machine_mapping import machine_history as mh
from vaft.machine_mapping.utils import VestConfigurationError, package_data_path

# PRODUCT IDENTITY PIN. An era name is a static product's `machine_version`,
# and stage products and HSDS sources are keyed by it (#2159 audit gap G14).
# Changing this tuple re-keys stored products: do it only together with the
# vest.yaml edit, in a reviewed change that says so.
PINNED_ERAS = (
    ("vest-pre-43017-pf1906", None, 43016, "1906", "1512", 43016),
    ("vest-43017-45967-pf1906", 43017, 45967, "1906", "2409", 43017),
    ("vest-45968-plus-pf2507", 45968, None, "2507", "2409", 45968),
)


def test_machine_eras_are_pinned():
    observed = tuple(
        (e.name, e.first_shot, e.last_shot, e.pf_geometry, e.wall_geometry, e.reference_shot)
        for e in mh.machine_eras()
    )
    assert observed == PINNED_ERAS


def test_vest_upstream_reexports_the_yaml_eras():
    from vaft.omas import vest_upstream

    assert vest_upstream.VEST_MACHINE_ERAS == mh.machine_eras()
    assert vest_upstream.machine_era_for_shot is mh.machine_era_for_shot
    assert vest_upstream.machine_era("vest-45968-plus-pf2507").first_shot == 45968


# The literals each consumer held before #2267, kept here as the oracle.
def _legacy_trigger_offset(shot: int) -> float:
    if shot < 41446:
        return 0.24
    if shot <= 41451:
        return 0.26
    if shot <= 41659:
        return 0.24
    return 0.26


def _legacy_generation(shot: int) -> int:
    if 29349 < shot <= 42190:
        return 2
    if shot > 42190:
        return 3
    return 1


BOUNDARY_SHOTS = sorted(
    {0, 1, 20000, 29349, 29350, 39915, 41445, 41446, 41451, 41452, 41659, 41660, 42190, 42191,
     43016, 43017, 45530, 45531, 45540, 45541, 45967, 45968, 48224, 60000}
)


@pytest.mark.parametrize("shot", BOUNDARY_SHOTS)
def test_every_boundary_reproduces_the_former_literals(shot):
    from vaft.database import raw as raw_db
    from vaft.machine_mapping.pf_active import pf_geometry_version_for_shot
    from vaft.machine_mapping.pf_passive import wall_geometry_version_for_shot
    from vaft.machine_mapping.spectrometer_uv import legacy_time_shift_s

    assert raw_db._daq_trigger_time_correction(shot) == _legacy_trigger_offset(shot)
    assert legacy_time_shift_s(shot) == _legacy_trigger_offset(shot)
    assert raw_db.vest_check_table(None, shot) == _legacy_generation(shot)
    assert pf_geometry_version_for_shot(shot) == ("2507" if shot >= 45968 else "1906")
    assert wall_geometry_version_for_shot(shot) == ("2409" if shot >= 43017 else "1512")
    assert (shot in mh.sxr_hxr_trigger_fallback_shots()) == (45531 <= shot <= 45540)


def test_compatibility_constants_are_derived():
    from vaft.database import raw as raw_db
    from vaft.machine_mapping import pf_active, pf_passive, soft_x_rays

    assert raw_db.V2_LAST_SHOT == 42190
    assert pf_active.PF_GEOMETRY_2507_FIRST_SHOT == 45968
    assert pf_passive.WALL_GEOMETRY_2409_FIRST_SHOT == 43017
    assert soft_x_rays.HXR_FALLBACK_SHOT_RANGE == range(45531, 45541)


def test_no_shot_keeps_the_historical_geometry():
    from vaft.machine_mapping.pf_active import pf_geometry_version_for_shot
    from vaft.machine_mapping.pf_passive import wall_geometry_version_for_shot

    assert pf_geometry_version_for_shot(None) == "1906"
    assert wall_geometry_version_for_shot(None) == "1512"


# --- validation -------------------------------------------------------------


def _packaged() -> dict:
    return yaml.safe_load(Path(package_data_path("vest.yaml")).read_text(encoding="utf-8"))


def _eras_from(tmp_path: Path, mutate) -> tuple:
    document = _packaged()
    mutate(document["machine_history"])
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return mh._machine_eras(mh._stat_key(str(path)))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda h: h["machine_eras"][1].update(from_shot=43018, reference_shot=43018), "tile every shot"),
        (lambda h: h["machine_eras"][1].update(from_shot=43016), "tile every shot"),
        (lambda h: h["machine_eras"][0].update(from_shot=10), "no from_shot"),
        (lambda h: h["machine_eras"][2].update(to_shot=50000), "no to_shot"),
        (lambda h: h["machine_eras"][2].update(name="vest-pre-43017-pf1906"), "unique"),
        (lambda h: h["machine_eras"][2].update(pf_geometry="2600"), "pf_geometry"),
        (lambda h: h["machine_eras"][0].update(wall_geometry="9999"), "wall_geometry"),
        (lambda h: h["machine_eras"][1].update(reference_shot=50000), "outside the era"),
        (lambda h: h["machine_eras"][1].update(pf_geometyr="1906"), "unknown key"),
        (lambda h: h.update(machine_eras=[]), "non-empty list"),
    ],
)
def test_malformed_machine_eras_fail(tmp_path, mutate, message):
    with pytest.raises(VestConfigurationError, match=message):
        _eras_from(tmp_path, mutate)


def test_overlapping_daq_revisions_fail(tmp_path):
    document = _packaged()
    document["machine_history"]["daq"]["fast_trigger_offset"]["revisions"][1]["from_shot"] = 41450
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="overlap"):
        mh.fast_trigger_offset_s(41700, info_file=str(path))


def test_missing_machine_history_fails_loudly(tmp_path):
    document = copy.deepcopy(_packaged())
    del document["machine_history"]
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="machine_history"):
        mh.fast_trigger_offset_s(41700, info_file=str(path))


def test_negative_shot_is_refused():
    with pytest.raises(VestConfigurationError):
        mh.machine_era_for_shot(-1)


def test_a_gap_between_waveform_generations_fails(tmp_path):
    document = _packaged()
    document["machine_history"]["daq"]["waveform_table_fallback"]["revisions"][0]["to_shot"] = 42180
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    with pytest.raises(VestConfigurationError, match="contiguous"):
        mh.fallback_waveform_generation(42185, info_file=str(path))
    with pytest.raises(VestConfigurationError, match="contiguous"):
        mh.last_shot_of_waveform_generation(2, info_file=str(path))


def test_resolved_daq_values_are_read_only():
    resolved = mh._resolved_daq(mh._stat_key(None), "fast_trigger_offset", 41700)
    with pytest.raises(TypeError):
        resolved["offset_s"] = 1.0  # type: ignore[index]


def test_wall_2409_import_derives_its_reference_shots():
    import importlib.util
    from pathlib import Path as _Path

    script = _Path(__file__).resolve().parents[1] / "workflow" / "em_coupling" / "import_wall_2409.py"
    spec = importlib.util.spec_from_file_location("_import_wall_2409", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PF_REFERENCE_SHOT == {"1906": 43017, "2507": 45968}
