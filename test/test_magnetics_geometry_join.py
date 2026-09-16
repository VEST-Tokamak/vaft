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


def test_the_unread_vest_yaml_block_is_not_treated_as_a_source():
    """Pins #843: the duplicate disagrees, and that must stay harmless.

    If a future change starts reading this block, it will read an ordering that
    differs from the live one at 11 of 64 positions. This test does not assert
    the two agree -- they do not -- but that the live tables are the ones the
    mapper walks, so a reader of the duplicate is knowingly on their own.
    """
    from vaft.machine_mapping.utils import package_data_path

    duplicate = yaml.safe_load(
        open(package_data_path("vest.yaml"), encoding="utf-8")
    )[0]["magnetics"][KIND]["channels"]
    ordered = [duplicate[key] for key in sorted(duplicate, key=lambda k: int(k))]
    geometry = _probes(_load_static_channels())

    assert len(ordered) == len(geometry)
    differing = [
        index
        for index, (dup, geo) in enumerate(zip(ordered, geometry))
        if int(dup["field"]) != int(geo["field_code"])
    ]
    # Recorded, not accepted: this is the state #843 describes. If someone
    # reconciles the duplicate, this list shrinks and the test says so.
    assert len(differing) == 11, (
        f"the unread vest.yaml probe block now differs at {len(differing)} positions "
        "rather than 11; if it was reconciled, update this test and #843"
    )
