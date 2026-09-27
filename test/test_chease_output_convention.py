"""``output_cocos="input"`` returns CHEASE's output in the source's convention.

CHEASE writes COCOS 2. The adapter re-signs that to the source's sign pattern,
which used to be all it did: the ``COCOS=02`` CASE token stayed in the
header. COCOS 1 and 2 (and 11 and 12) share every sign and differ only in the
toroidal direction, so a COCOS 11 source came back
declaring COCOS 2 and read back with its plasma current, toroidal field and
poloidal field all reversed -- with nothing in its signs to say so. Found on
the #886 island fixture.

Every check here is a physical direction in the IMAS frame, read through the
file's declared convention, because that is what a reader of the file gets.
The file keeps psi per radian -- what every g-file reader, ``to_omas``
included, assumes -- so a weber source's index is declared as its per-radian
twin (11 -> 1), which shares its orientation.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
from scipy.interpolate import RectBivariateSpline

from vaft.code.chease import (
    _declare_source_convention,
    _desired_signs_from_info,
    _force_geqdsk_signs,
    _geqdsk_sign_info,
)
from vaft.data.cocos import cocos_spec
from vaft.data.eqdsk import from_equilibrium, read_geqdsk
from vaft.process.equilibrium import (
    as_equilibrium,
    calculate_q_profile_from_psi,
    convert_cocos,
    equilibrium_field_on_grid,
    solovev_example,
)


def _with_q(eq):
    psi_n = (eq.psi_1d - eq.psi_axis) / (eq.psi_boundary - eq.psi_axis)
    q = calculate_q_profile_from_psi(eq.psi, eq.r, eq.z, (eq.psi_1d, eq.f), eq.psi_axis,
                                     eq.psi_boundary, np.clip(psi_n, 0.01, 0.99),
                                     axis_rz=eq.magnetic_axis, cocos=eq.convention.cocos)
    return dataclasses.replace(eq, q=np.asarray(q, dtype=float))


def _physical_directions(eq):
    """Signs of Ip, B_phi and the outboard-midplane B_Z in the IMAS frame."""
    cocos = eq.convention.cocos
    assert cocos is not None, "the record must declare a convention to be read physically"
    frame = cocos_spec(cocos).sigma_rpz  # an even COCOS measures phi clockwise from above
    _, b_z, b_phi = equilibrium_field_on_grid(eq.r, eq.z, eq.psi, eq.psi_1d, eq.f, cocos=cocos)
    r_axis, z_axis = eq.magnetic_axis
    probe = (r_axis + 0.1, z_axis)
    return {
        "ip": int(np.sign(eq.ip)) * frame,
        "b_phi": int(np.sign(RectBivariateSpline(eq.r, eq.z, b_phi).ev(*probe))) * frame,
        "b_z": int(np.sign(RectBivariateSpline(eq.r, eq.z, b_z).ev(*probe))),
    }


@pytest.fixture(scope="module")
def source():
    """A COCOS 11 source, as the #886 fixture generator hands CHEASE."""
    return _with_q(solovev_example(toroidal_field=0.13, a_parameter=0.5))


def _chease_like_output(source_eq):
    """The source's physics written the way CHEASE writes it: COCOS 2, per radian."""
    geqdsk = from_equilibrium(convert_cocos(source_eq, 2))
    geqdsk.mapping["CASE"] = "  FROM CHEASE BUT COCOS=02 ,SI UNITS20260921"
    return geqdsk


def _export_as_input(raw, source_geqdsk):
    signed, *_ = _force_geqdsk_signs(raw, **_desired_signs_from_info(_geqdsk_sign_info(source_geqdsk)))
    return _declare_source_convention(signed, source_geqdsk)


def test_the_chease_like_output_is_the_same_physics_in_cocos_2(source):
    """Sanity of the stand-in: read as declared, it is the source's plasma."""
    raw = as_equilibrium(_chease_like_output(source))
    assert raw.convention.cocos == 2
    assert _physical_directions(raw) == _physical_directions(source)


def test_input_export_returns_the_source_orientation_and_physics(source):
    source_geqdsk = from_equilibrium(source)
    exported, info = _export_as_input(_chease_like_output(source), source_geqdsk)
    assert info["declared_cocos"] == 1
    assert "COCOS=01" in exported.mapping["CASE"] and "COCOS=02" not in exported.mapping["CASE"]
    back = as_equilibrium(exported)
    assert back.convention.cocos == 1
    assert not back.convention.contradicted
    assert _physical_directions(back) == _physical_directions(source)
    # Per radian, as CHEASE wrote it and as a g-file stores it.
    np.testing.assert_allclose(back.psi * 2 * np.pi, source.psi, rtol=1e-6, atol=1e-9 * np.ptp(source.psi))


def test_the_ods_of_the_export_carries_the_source_flux_in_weber(source):
    """``to_omas`` multiplies a g-file's psi by 2*pi; the export must not have."""
    from vaft.data.eqdsk import ods_psi_to_wb_per_radian_factor

    exported, _ = _export_as_input(_chease_like_output(source), from_equilibrium(source))
    ods = exported.to_omas()
    quantities = ods["equilibrium.time_slice.0.global_quantities"]
    span = float(quantities["psi_boundary"]) - float(quantities["psi_axis"])
    assert span == pytest.approx(source.psi_boundary - source.psi_axis, rel=1e-6)
    assert ods_psi_to_wb_per_radian_factor(ods) == pytest.approx(1.0 / (2 * np.pi))


def test_a_contradicted_source_header_is_not_trusted(source):
    """Per-radian COCOS 1 data under a stale ``COCOS=11`` header: the header is
    contradicted, the signs allow 1 or 2, so nothing is declared and psi is
    left exactly as CHEASE wrote it."""
    per_radian = convert_cocos(source, 1)
    source_geqdsk = from_equilibrium(per_radian)
    source_geqdsk.mapping["CASE"] = "stale header COCOS=11"
    stale = as_equilibrium(source_geqdsk).convention
    assert stale.cocos == 11 and stale.contradicted
    raw = _chease_like_output(source)
    exported, info = _export_as_input(raw, source_geqdsk)
    assert info["declared_cocos"] is None
    assert "COCOS" not in exported.mapping["CASE"]
    np.testing.assert_allclose(np.abs(exported["PSIRZ"]), np.abs(raw["PSIRZ"]))


def test_re_signing_alone_left_a_header_that_reverses_the_plasma(source):
    """What the adapter used to write: the right signs under COCOS 2's label."""
    source_geqdsk = from_equilibrium(source)
    signed, *_ = _force_geqdsk_signs(_chease_like_output(source),
                                     **_desired_signs_from_info(_geqdsk_sign_info(source_geqdsk)))
    stale = as_equilibrium(signed)
    assert stale.convention.cocos == 2 and not stale.convention.contradicted
    reversed_ = {key: -value for key, value in _physical_directions(source).items()}
    assert _physical_directions(stale) == reversed_


def test_an_unpinned_source_gets_no_declaration(source):
    """A weber-per-radian source whose toroidal direction its signs cannot tell
    (COCOS 1 or 2) is exported without a COCOS token rather than with a guess."""
    source_geqdsk = from_equilibrium(convert_cocos(source, 1))
    source_geqdsk.mapping["CASE"] = "EFIT-like, no convention declared"
    assert as_equilibrium(source_geqdsk).convention.identified == (1, 2)
    exported, info = _export_as_input(_chease_like_output(source), source_geqdsk)
    assert info["declared_cocos"] is None
    assert "COCOS" not in exported.mapping["CASE"]
    assert as_equilibrium(exported).convention.identified == (1, 2)


@pytest.mark.skipif(not os.environ.get("CHEASE"), reason="set CHEASE to a chease executable")
def test_a_real_chease_refinement_keeps_the_source_physics(source, tmp_path):
    from vaft.code.chease import CHEASEConfig, refine_equilibrium

    result = refine_equilibrium(from_equilibrium(source),
                                CHEASEConfig(workdir=tmp_path, nw=129, create_plot=False))
    assert result.ok
    refined = as_equilibrium(read_geqdsk(str(result.refined_geqdsk)))
    assert refined.convention.cocos == 1 and not refined.convention.contradicted
    assert _physical_directions(refined) == _physical_directions(source)
    source_span = source.psi_boundary - source.psi_axis
    assert (refined.psi_boundary - refined.psi_axis) * 2 * np.pi == pytest.approx(source_span, rel=0.05)
    # And the ODS the adapter builds from it is in weber, not 2*pi too large.
    quantities = result.refined_ods["equilibrium.time_slice.0.global_quantities"]
    ods_span = float(quantities["psi_boundary"]) - float(quantities["psi_axis"])
    assert ods_span == pytest.approx(source_span, rel=0.05)
