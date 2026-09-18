"""A shot-keyed override is found however the mapping file spells its key.

``yaml.safe_load`` reads vest.yaml's own unquoted ``20259:`` as an ``int``;
the resolvers looked the block up by ``str`` alone, so an override written in
the file's own style was silently ignored and the "nested magnetics.impa is
refused" guard never saw an int-keyed block (cold review machine-mapping F1).
"""
from __future__ import annotations

import pytest
import yaml

from vaft.machine_mapping.impa import resolve_impa_config
from vaft.machine_mapping.langmuir_probes import resolve_langmuir_probe_config
from vaft.machine_mapping.utils import (
    _resolve_info_file_path,
    get_diagnostic_info,
    get_static_info,
    load_yaml,
    raw_database_info,
)

KEYS = pytest.mark.parametrize("key", [39915, "39915"], ids=["int", "str"])


def _write(tmp_path, mapping) -> str:
    path = tmp_path / "vest.yaml"
    path.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(path)


@KEYS
def test_an_impa_override_applies_under_either_spelling(tmp_path, key):
    path = _write(tmp_path, {
        "impa": {"label": "base"}, 0: {}, key: {"impa": {"label": "overridden"}},
    })
    assert resolve_impa_config(39915, path)["label"] == "overridden"
    assert resolve_impa_config(41524, path)["label"] == "base"


@KEYS
def test_a_nested_impa_block_in_a_shot_override_is_refused(tmp_path, key):
    path = _write(tmp_path, {
        "impa": {"label": "base"}, 0: {},
        key: {"magnetics": {"impa": {"label": "stale nested copy"}}},
    })
    with pytest.raises(ValueError, match="moved to the top-level"):
        resolve_impa_config(39915, path)


@KEYS
def test_a_langmuir_override_applies_under_either_spelling(tmp_path, key):
    path = _write(tmp_path, {
        0: {"langmuir_probes": {"mid": {"vd3": 1.0, "tip": "a"}}},
        key: {"langmuir_probes": {"mid": {"vd3": 2.0}}},
    })
    assert resolve_langmuir_probe_config("mid", 39915, path) == {"vd3": 2.0, "tip": "a"}
    assert resolve_langmuir_probe_config("mid", 41524, path)["vd3"] == 1.0


@KEYS
def test_the_generic_readers_find_the_block_too(tmp_path, key):
    path = _write(tmp_path, {
        0: {"pf_active": {0: {"label": "PF1", "field": 59, "gain": 1.0}},
            "tf": {"label": "TF"}},
        key: {"pf_active": {0: {"gain": -5.0}}},
    })
    assert raw_database_info(path, 39915, "pf_active")["gains"] == {"0": -5.0}
    assert raw_database_info(path, 41524, "pf_active")["gains"] == {"0": 1.0}
    options = {"info_file": path}
    assert get_diagnostic_info("39915", "pf_active", options) == {0: {"gain": -5.0}}
    assert get_static_info("39915", "pf_active", options) == {0: {"gain": -5.0}}
    # a sparse override answers only for what it holds
    assert get_diagnostic_info("39915", "tf", options) == {"label": "TF"}
    assert get_static_info("39915", "tf", options) == {"label": "TF"}


def test_no_packaged_shot_resolves_differently_now_that_its_block_is_found():
    """The packaged int-keyed blocks carry ``pf_active`` gains only, which no
    resolver below reads, so finding them moves no shipped number."""
    content = load_yaml(_resolve_info_file_path(None))
    keyed = [key for key in content if isinstance(key, int) and key != 0]
    assert keyed, "the packaged mapping no longer carries shot blocks"
    for shot in keyed:
        assert set(content[shot]) == {"pf_active"}, shot
    unkeyed = 39000
    assert unkeyed not in content and str(unkeyed) not in content
    for shot in (*keyed, 39915, 41524, 41672):
        assert resolve_impa_config(shot) == resolve_impa_config(unkeyed), shot
        for assembly in ("mid", "upper"):
            assert resolve_langmuir_probe_config(assembly, shot) == (
                resolve_langmuir_probe_config(assembly, unkeyed)
            ), (shot, assembly)
        for diagnostic in ("tf", "magnetics"):
            assert get_diagnostic_info(str(shot), diagnostic) == (
                get_diagnostic_info(str(unkeyed), diagnostic)
            ), (shot, diagnostic)
