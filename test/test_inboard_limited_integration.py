"""Both forward solvers reproduce the inboard-limited 39915 @ 325 ms state (#1469).

EFIT places this LCFS on the inboard (center-stack) limiter at the midplane.
Each test runs the real solver on the packaged sample with the canonical VEST
limiter and skips cleanly when that solver is not installed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from vaft.code.tokamaker import ScanTopology, classify_boundary
from vaft.machine_mapping.wall import wall as canonical_wall
from vaft.omas.sample import sample_ods

pytestmark = pytest.mark.slow

SHOT, TIME = 39915, 0.325
INBOARD_FACE_R = 0.105


@pytest.fixture(scope="module")
def ods():
    data = sample_ods(SHOT)
    canonical_wall(data)
    return data


def _assert_inboard_limited(gfile):
    report = classify_boundary(gfile)
    assert report.topology is ScanTopology.LIMITED, report.reason
    assert report.limiter_contact.side == "inboard"
    assert report.limiter_contact.distance <= report.contact_tolerance
    assert abs(report.limiter_contact.wall_r - INBOARD_FACE_R) < 1e-3
    assert abs(report.limiter_contact.z) < 0.05            # at the midplane


def test_tokamaker_rests_on_the_inboard_limiter(ods, tmp_path):
    from vaft.code import tokamaker

    try:
        tokamaker._oft.import_oft()
    except ImportError as exc:
        pytest.skip(f"OpenFUSIONToolkit is not importable: {exc}")

    config = tokamaker.TokaMakerConfig(
        shot=SHOT, time=TIME, workdir=tmp_path,
        pax=tokamaker.reference_axis_pressure(ods, TIME),
    )
    result = tokamaker.run_tokamaker(tokamaker.prepare_tokamaker_inputs(ods, config), config)

    assert result.ok, result.error
    assert result.scalars["diverted"] is False
    assert result.scalars["lim_point"][0] == pytest.approx(INBOARD_FACE_R, abs=1e-3)
    _assert_inboard_limited(result.gfile)
    # the exported boundary is the LCFS, not the 99 % surface
    r = np.asarray(result.ods["equilibrium.time_slice.0.boundary.outline.r"])
    assert r.min() == pytest.approx(INBOARD_FACE_R, abs=1e-3)


def _rtes() -> Path | None:
    home = os.environ.get("TESHOME")
    for candidate in (
        Path(home) / "bin" / "rtes" if home else None,
        Path(home) / "TES" / "rtes" if home else None,   # TES's own build layout
        Path(os.environ["RTES"]) if os.environ.get("RTES") else None,
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def test_tes_rests_on_the_inboard_limiter(ods, tmp_path):
    from vaft.code import tes

    rtes = _rtes()
    if rtes is None:
        pytest.skip("rtes is not installed ($TESHOME/bin/rtes or $RTES)")

    config = tes.TESConfig(
        executable=str(rtes), workdir=tmp_path, shot=SHOT, time=TIME, bt0=0.15,
        constraint_source="magnetics", betap=0.05, betap_type=1,
    )
    result = tes.run_tes(tes.prepare_tes_inputs(ods, config), config)

    assert result.ok, result.stderr
    _assert_inboard_limited(result.gfile)
