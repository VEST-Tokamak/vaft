"""IMPA is separated from the baseline at the machine-description level (#305).

The absence of IMPA from the static and diagnostics products held by accident
before this: the magnetics geometry asset simply never listed the array's field
codes, and the mapper wrote its fitted geometry per shot instead. These make it
a contract, so a future geometry survey cannot quietly reintroduce IMPA into the
product that defines `main`.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import pytest
import yaml

from vaft.machine_mapping.impa import IMPA_IDENTIFIER_PREFIX, resolve_impa_config
from vaft.machine_mapping.utils import package_data_path
from vaft.omas.vest_upstream import build_static_ods, machine_era_for_shot


HALL_PROBE_TYPE_INDEX = 3
PROBE_NODES = ("magnetics.b_field_pol_probe", "magnetics.b_field_tor_probe")


def _impa_channels(ods) -> list[str]:
    """Every channel this product carries that belongs to the array."""
    found = []
    for node in PROBE_NODES:
        if node not in ods:
            continue
        for index in range(len(ods[node])):
            probe = ods[f"{node}.{index}"]
            identifier = str(probe["identifier"]) if "identifier" in probe else ""
            kind = probe["type"]["index"] if "type" in probe and "index" in probe["type"] else None
            if identifier.startswith(IMPA_IDENTIFIER_PREFIX) or kind == HALL_PROBE_TYPE_INDEX:
                found.append(f"{node}.{index}")
    return found


@pytest.mark.parametrize("shot", [39915, 43017, 45967])
def test_the_static_product_describes_no_internal_probe(shot):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        static, manifest = build_static_ods(machine_era_for_shot(shot).name)
    assert _impa_channels(static) == []
    assert "impa" not in manifest["contents"]


def test_the_machine_description_keeps_impa_out_of_the_baseline_magnetics():
    """One source of truth, and it is not nested inside `magnetics`."""
    source = Path(package_data_path("vest.yaml"))
    content = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert "impa" in content, "the IMPA section is top-level since issue #305"
    for key, block in content.items():
        if not isinstance(block, dict):
            continue
        assert "impa" not in (block.get("magnetics") or {}), key


def test_the_resolved_configuration_still_narrows_by_shot_era():
    """The move changed where the block lives, not how a shot resolves it."""
    from vaft.machine_mapping.impa import impa_expected_fields

    assert impa_expected_fields(39915) == [114, 115, 116, 117, 118, 119, 120, 121]
    # The 2022-04-23 block wires seven channels, without field 121.
    assert impa_expected_fields(35375) == [114, 115, 116, 117, 118, 119, 120]


def test_a_shot_keyed_block_still_overrides_the_top_level_section(tmp_path):
    mapping = {
        "impa": {"label": "base", "channels": {0: {"label": "IMPA 01", "field": 114}}},
        0: {},
        "39915": {"impa": {"label": "overridden"}},
    }
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    assert resolve_impa_config(39915, str(path))["label"] == "overridden"
    assert resolve_impa_config(41524, str(path))["label"] == "base"


def test_a_leftover_nested_block_is_refused_rather_than_silently_losing(tmp_path):
    """Two places to define it is the failure mode the move must not create."""
    mapping = {
        "impa": {"label": "top-level"},
        0: {"magnetics": {"impa": {"label": "stale nested copy"}}},
    }
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="moved to the top-level"):
        resolve_impa_config(39915, str(path))


def test_the_registry_declares_the_source_the_array_is_published_into():
    from vaft.machine_mapping.registry import load_diagnostic_registry

    entry = load_diagnostic_registry()["magnetics.internal_probe"]
    assert entry["publication"] == {"stage": "impa", "source": "impa"}
    assert entry["ids_path"] != "magnetics.b_field_pol_probe"
