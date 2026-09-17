"""The two live probe tables are joined by list position, so they must agree.

`MD.yaml` supplies each probe's calibration and `VEST_MagneticsGeometry_Full_ver_2302.yaml`
its position, and the mapper walks both in list order: ODS probe *i* takes its
signal and gain from entry *i* of the first and its `(r, z)` from entry *i* of
the second. Each record carries its own `field_code`, so the join is checkable —
and if the two ever drift, every probe past the drift point is published with
another probe's calibration, with nothing to say so.

Issue #843 is what this guards against. A third description of the same probes
lives in `vest.yaml` under `0.magnetics.b_field_pol_probe`; **nothing reads it**,
and it disagrees with the live ordering at 11 of 64 positions, which is exactly
how #843 came to be filed against the wrong file.
"""

from __future__ import annotations

import yaml

from vaft.machine_mapping.magnetics import (
    _load_equilibrium_magnetics_channels,
    _load_static_channels,
)

KIND = "b_field_pol_probe"


def _probes(channels):
    return [c for c in channels if c["kind"] == KIND]


def test_the_two_live_tables_are_the_same_probes_in_the_same_order():
    calibration = _probes(_load_equilibrium_magnetics_channels())
    geometry = _probes(_load_static_channels())
    assert len(calibration) == len(geometry) == 64
    mismatched = [
        (index, int(cal["field_code"]), int(geo["field_code"]))
        for index, (cal, geo) in enumerate(zip(calibration, geometry))
        if int(cal["field_code"]) != int(geo["field_code"])
    ]
    assert mismatched == [], (
        "MD.yaml and the geometry file are joined by list position; these indices "
        f"name different probes, so each would be published with another's "
        f"calibration: {mismatched}"
    )


def test_every_probe_appears_once_in_each_live_table():
    for channels in (_load_equilibrium_magnetics_channels(), _load_static_channels()):
        codes = [int(c["field_code"]) for c in _probes(channels)]
        assert len(set(codes)) == len(codes)


def test_vest_yaml_carries_no_third_probe_description():
    """The duplicate that caused #843 is gone and must not come back.

    It held the same 64 probes in a different order, so it disagreed with the
    live pair at 11 of 64 positions while looking as authoritative as either.
    Nothing read it. Anyone reintroducing probe geometry here would recreate a
    source that can disagree with the one the mapper actually walks.
    """
    import yaml

    from vaft.machine_mapping.utils import package_data_path

    magnetics = yaml.safe_load(
        open(package_data_path("vest.yaml"), encoding="utf-8")
    )[0]["magnetics"]
    assert KIND not in magnetics


def test_the_geometry_files_are_the_only_probe_geometry():
    """Both live tables describe the same 64 probes, by field code."""
    calibration = {int(c["field_code"]) for c in _probes(_load_equilibrium_magnetics_channels())}
    geometry = {int(c["field_code"]) for c in _probes(_load_static_channels())}
    assert calibration == geometry
    assert len(geometry) == 64
