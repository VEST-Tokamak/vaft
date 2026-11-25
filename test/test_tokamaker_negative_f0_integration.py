"""A real TokaMaker solve with F0 < 0 keeps F(psi) consistent with its own FF'.

Grad-Shafranov depends on F^2 only, so reversing the vacuum toroidal field
(F0 -> -F0) must flip FPOL and q and change nothing else.  OpenFUSIONToolkit
before 26.9 rebuilt F as a reflection about F0 for F0 < 0 (upstream #321,
fixed in #322): the solve converged, but the written FPOL contradicted FFPRIM
and |q| changed.  This checks it on the packaged VEST sample, whose own F0 is
positive, by solving it both ways.
"""

from __future__ import annotations

import numpy as np
import pytest

from vaft.data.eqdsk import read_geqdsk
from vaft.machine_mapping.wall import wall as canonical_wall
from vaft.omas.sample import sample_ods

pytestmark = pytest.mark.slow

SHOT, TIME = 39915, 0.325


def _solve(ods, workdir, pax, f0):
    from vaft.code import tokamaker

    config = tokamaker.TokaMakerConfig(shot=SHOT, time=TIME, workdir=workdir, pax=pax, f0=f0)
    result = tokamaker.run_tokamaker(tokamaker.prepare_tokamaker_inputs(ods, config), config)
    assert result.ok, result.error
    g = read_geqdsk(result.gfile)
    fpol, ffprim, q = (np.asarray(g[key], dtype=float) for key in ("FPOL", "FFPRIM", "QPSI"))
    psi = np.linspace(g["SIMAG"], g["SIBRY"], fpol.size)
    return {
        "fpol": fpol, "q": q,
        "f2_drop": fpol[0] ** 2 - fpol[-1] ** 2,
        "ffp_integral": -2.0 * np.trapezoid(ffprim, psi),
    }


def test_reversed_toroidal_field_flips_f_and_q_and_nothing_else(tmp_path):
    from vaft.code import tokamaker

    try:
        version = tokamaker._oft.oft_version()
    except ImportError as exc:
        pytest.skip(f"OpenFUSIONToolkit is not importable: {exc}")
    if version is None:
        pytest.skip("OpenFUSIONToolkit reports no version (uninstalled source tree); vaft refuses F0 < 0 on it")
    if version < tokamaker._oft.MIN_OFT_VERSION:
        pytest.skip("OpenFUSIONToolkit %d.%d predates the #322 fix; vaft refuses F0 < 0 on it" % version)

    ods = sample_ods(SHOT)
    canonical_wall(ods)
    pax = tokamaker.reference_axis_pressure(ods, TIME)
    # its own workdir: preparing inputs writes geometry.json there
    base = tokamaker.TokaMakerConfig(shot=SHOT, time=TIME, workdir=tmp_path / "base", pax=pax)
    f0 = abs(tokamaker.prepare_tokamaker_inputs(ods, base).f0)

    forward = _solve(ods, tmp_path / "positive", pax, +f0)
    reversed_ = _solve(ods, tmp_path / "negative", pax, -f0)

    for case in (forward, reversed_):
        assert case["f2_drop"] == pytest.approx(case["ffp_integral"], rel=1e-3)
    np.testing.assert_allclose(reversed_["fpol"], -forward["fpol"], rtol=1e-6, atol=1e-9)
    # q is traced on flux surfaces and repeats to ~1e-5 between solves; the #321
    # error it guards against moves q0 by ~20 %.
    np.testing.assert_allclose(reversed_["q"], -forward["q"], rtol=1e-4, atol=1e-8)
    assert np.all(reversed_["fpol"] < 0)
