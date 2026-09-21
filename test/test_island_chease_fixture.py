"""The CHEASE fixture behind the #886 island notebook, and the island on it.

``test/data/island_sxr/solovev_vest_chease.geqdsk`` is a CHEASE fixed-boundary
refinement of ``solovev_example(**provenance["solovev_example"])``, made by
``workflow/island_sxr/generate_chease_fixture.py``. These tests pin that the
file is the one its provenance describes, that it is a refinement of that
Solov'ev input rather than some other equilibrium, and that one island spec
means the same thing on both.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from vaft.data.eqdsk import read_geqdsk
from vaft.process.equilibrium import as_equilibrium, solovev_example
from vaft.process.magnetic_island import MagneticIslandSpec, magnetic_island_topology

FIXTURE = Path(__file__).resolve().parent / "data" / "island_sxr"
PROVENANCE = json.loads((FIXTURE / "solovev_vest_chease.json").read_text())

needs_chease = pytest.mark.skipif(
    not os.environ.get("CHEASE"), reason="set CHEASE to a chease executable to regenerate the fixture"
)


@pytest.fixture(scope="module")
def pair():
    solovev = solovev_example(**PROVENANCE["solovev_example"])
    refined = as_equilibrium(read_geqdsk(str(FIXTURE / "solovev_vest_chease.geqdsk")))
    return solovev, refined


def test_the_fixture_is_the_file_its_provenance_describes():
    digest = hashlib.sha256((FIXTURE / "solovev_vest_chease.geqdsk").read_bytes()).hexdigest()
    assert digest == PROVENANCE["geqdsk_sha256"]
    assert PROVENANCE["generator"] == "workflow/island_sxr/generate_chease_fixture.py"
    assert (Path(__file__).resolve().parents[1] / PROVENANCE["generator"]).is_file()


def test_the_fixture_refines_the_recorded_solovev_input(pair):
    solovev, refined = pair
    assert refined.convention.cocos == 2 and solovev.convention.cocos == 11
    np.testing.assert_allclose(refined.magnetic_axis, solovev.magnetic_axis, atol=5e-3)
    assert refined.ip == pytest.approx(solovev.ip, rel=0.05)
    assert np.sign(refined.ip) == np.sign(solovev.ip)
    # Same fixed boundary: the refined LCFS sits on the Solov'ev one.
    from vaft.process.equilibrium import contour_shape_parameters

    a = contour_shape_parameters(solovev.lcfs.r, solovev.lcfs.z)
    b = contour_shape_parameters(refined.lcfs.r, refined.lcfs.z)
    assert b["elongation"] == pytest.approx(a["elongation"], rel=0.02)
    assert b["area"] == pytest.approx(a["area"], rel=0.02)


@pytest.mark.parametrize("m", [2, 3])
def test_one_island_spec_means_the_same_thing_on_both(pair, m):
    """Same m/n, same outboard width, same helicity across COCOS 11 and 2; the
    resonant surface is each equilibrium's own."""
    spec = MagneticIslandSpec(m, 1, 0.03)
    tops = [magnetic_island_topology(eq, spec) for eq in pair]
    for top in tops:
        assert top.q_s == pytest.approx(m, abs=1e-9)
        sfl = top.sfl_map
        r_s = float(sfl.outboard_radius(top.psi_n_s))
        z_axis = sfl.magnetic_axis[1]
        r = np.linspace(r_s - 0.03, r_s + 0.03, 60001)
        x = sfl.outboard_radius(sfl.psi_norm(r, np.full_like(r, z_axis))) - r_s
        inside = r[8 * (x / spec.width) ** 2 - 1.0 <= 1.0]
        assert inside.max() - inside.min() == pytest.approx(spec.width, abs=2e-6)
    solovev_top, refined_top = tops
    assert solovev_top.helicity == refined_top.helicity == -1
    assert 1e-4 < abs(solovev_top.psi_n_s - refined_top.psi_n_s) < 0.05


@needs_chease
def test_regenerating_the_fixture_reproduces_it(tmp_path):
    import runpy

    script = Path(__file__).resolve().parents[1] / PROVENANCE["generator"]
    module = runpy.run_path(str(script))
    assert module["SOLOVEV_PARAMETERS"] == PROVENANCE["solovev_example"]
    assert module["main"](["--output", str(tmp_path)]) == 0
    fresh = as_equilibrium(read_geqdsk(str(tmp_path / "solovev_vest_chease.geqdsk")))
    stored = as_equilibrium(read_geqdsk(str(FIXTURE / "solovev_vest_chease.geqdsk")))
    spec = MagneticIslandSpec(2, 1, 0.03)
    assert magnetic_island_topology(fresh, spec).psi_n_s == pytest.approx(
        magnetic_island_topology(stored, spec).psi_n_s, abs=1e-4)
