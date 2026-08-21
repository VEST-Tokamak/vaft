"""Binary-free unit tests for kinetic constraints in ``vaft.code.efit``.

These tests exercise the pure-Python namelist / math logic only; none of them
run the EFIT binary.  A tiny dict-backed fake ODS mimics the OMAS dotted-path
access used by ``kinetic_pressure_points`` so we can assert the pressure math
without constructing a real omas.ODS.

Run with: ``pytest test/test_efit_kinetic.py -q``.
"""

import re

import numpy as np
import pytest

# --- import the module under test from the installed package ----------------
from vaft.code import efit as km


# --------------------------------------------------------------------------- #
# Minimal fake ODS supporting dotted-path + list-index access
# --------------------------------------------------------------------------- #

class FakeNode(dict):
    """dict that resolves dotted keys and numeric list indices like omas."""

    def __getitem__(self, key):
        if not isinstance(key, str):
            return super().__getitem__(key)
        cur = self
        for part in key.split("."):
            if isinstance(cur, (list, tuple)):
                cur = cur[int(part)]
            else:
                cur = dict.__getitem__(cur, part)
        return cur

    def __contains__(self, key):
        try:
            self[key]
            return True
        except Exception:
            return False


def _make_raw_ods():
    """ODS with thomson_scattering + charge_exchange at t = 0.300 s."""
    t = [0.299, 0.300, 0.301]

    def ts_channel(r, ne, sne, te, ste):
        ch = FakeNode()
        ch["position"] = FakeNode(r=r)
        ch["n_e"] = FakeNode(data=[0, ne, 0], data_error_upper=[0, sne, 0])
        ch["t_e"] = FakeNode(data=[0, te, 0], data_error_upper=[0, ste, 0])
        return ch

    ts = FakeNode()
    ts["time"] = t
    ts["channel"] = [
        ts_channel(0.40, 1.0e19, 1.0e18, 100.0, 5.0),
        ts_channel(0.55, 1.0e19, 1.0e18, 100.0, 5.0),
    ]

    def cx_channel(r, ti, sti):
        ch = FakeNode()
        ch["position"] = FakeNode(r=FakeNode(time=t, data=[r, r, r]))
        ion = FakeNode()
        ion["t_i"] = FakeNode(data=[ti, ti, ti], data_error_upper=[sti, sti, sti])
        ch["ion"] = [ion]
        return ch

    cx = FakeNode()
    cx["time"] = t
    # four CX channels spanning the TS radii; deterministic Ti field
    cx["channel"] = [
        cx_channel(0.35, 30.0, 3.0),
        cx_channel(0.45, 30.0, 3.0),
        cx_channel(0.55, 30.0, 3.0),
        cx_channel(0.65, 30.0, 3.0),
    ]

    ods = FakeNode()
    ods["thomson_scattering"] = ts
    ods["charge_exchange"] = cx
    return ods


def _make_spline_ods():
    """ODS with a core_profiles.profiles_1d slice carrying pressure_thermal."""
    npts = 11
    psi_n = np.linspace(0.0, 1.0, npts)
    p = 500.0 * (1.0 - psi_n ** 2)          # peaked, zero at edge
    prof = FakeNode()
    prof["time"] = 0.300
    prof["grid"] = FakeNode(rho_pol_norm=np.sqrt(psi_n))
    prof["pressure_thermal"] = p
    cp = FakeNode()
    cp["profiles_1d"] = [prof]
    ods = FakeNode()
    ods["core_profiles"] = cp
    return ods


# --------------------------------------------------------------------------- #
# kinetic_pressure_points: math
# --------------------------------------------------------------------------- #

def test_raw_pressure_and_sigpre_formula():
    ods = _make_raw_ods()
    # Recover the exact ne/Te/Ti/sigmas the internal extractor produced, then
    # assert kinetic_pressure_points wires the documented formulas onto them.
    R, ne, Te, sne, sTe, Ti, sTi = km._raw_ne_te_ti(ods, 300.0, None)

    p_exp = km.EQE * ne * (Te + Ti)
    sig_exp = km.EQE * np.sqrt(
        ((Te + Ti) * sne) ** 2 + (ne * sTe) ** 2 + (ne * sTi) ** 2
    )
    sig_exp = np.maximum(sig_exp, 0.05 * p_exp)

    pts = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw5")

    assert len(pts) == len(R)
    assert pts.zpress is not None and all(z == 0.0 for z in pts.zpress)
    assert np.allclose(pts.pressr, p_exp)
    assert np.allclose(pts.sigpre, sig_exp)
    # p = e * ne * (Te + Ti) exactly
    assert np.allclose(np.array(pts.pressr), km.EQE * ne * (Te + Ti))


def test_raw6_adds_zero_pressure_edge_anchor():
    ods = _make_raw_ods()
    p5 = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw5")
    p6 = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw6")

    assert len(p6) == len(p5) + 1
    # 6th anchor: RPRESS=-1.0 (psi_N=1), PRESSR=0, SIGPRE=0.05*max(p), ZPRESS=0
    assert p6.rpress[-1] == -1.0
    assert p6.pressr[-1] == 0.0
    assert p6.zpress[-1] == 0.0
    assert np.isclose(p6.sigpre[-1], 0.05 * max(p5.pressr))
    # the leading 5 points are unchanged
    assert np.allclose(p6.pressr[:-1], p5.pressr)


def test_spline_has_129_points_and_negative_rpress():
    ods = _make_spline_ods()
    pts = km.kinetic_pressure_points(ods, 300.0, encoding="spline")

    assert len(pts) == 129
    assert len(pts.rpress) == len(pts.pressr) == len(pts.sigpre) == len(pts.fwtpre) == 129
    # flux-space: no ZPRESS line, RPRESS = -psi_N in [-1, 0]
    assert pts.zpress is None
    assert all(r <= 0.0 for r in pts.rpress)
    assert np.isclose(pts.rpress[0], 0.0) and np.isclose(pts.rpress[-1], -1.0)
    # constant SIGPRE = SIG_FRAC * p(axis)
    assert np.isclose(pts.sigpre[0], km.SPLINE_SIG_FRAC * pts.pressr[0])
    assert all(s == pts.sigpre[0] for s in pts.sigpre)
    # FWTPRE taper in (0, 1]
    assert all(0.0 < w <= 1.0 for w in pts.fwtpre)


# --------------------------------------------------------------------------- #
# inject_pressure_constraint
# --------------------------------------------------------------------------- #

_BASE_KFILE = """ &IN1
 ISHOT= 48233
 PLASMA= 12345.6
 RCENTR= 0.4
 /
"""


def test_inject_pressure_constraint_raw_block():
    ods = _make_raw_ods()
    pts = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw6")
    out = km.inject_pressure_constraint(_BASE_KFILE, pts, encoding="raw6")

    lines = out.splitlines()
    slash = next(i for i, ln in enumerate(lines) if ln.strip() == "/")
    header = "\n".join(lines[:slash])

    # KPRFIT block present and inserted BEFORE the IN1 terminator
    assert "KPRFIT= 1" in header
    assert f"NPRESS= {len(pts)}" in header
    assert "KPRESSB= 0" in header
    for key in ("RPRESS=", "ZPRESS=", "PRESSR=", "SIGPRE=", "FWTPRE="):
        assert key in header, key
    # everything sits above the closing slash
    assert "KPRFIT= 1" not in "\n".join(lines[slash:])


def test_inject_pressure_constraint_spline_omits_zpress():
    ods = _make_spline_ods()
    pts = km.kinetic_pressure_points(ods, 300.0, encoding="spline")
    out = km.inject_pressure_constraint(_BASE_KFILE, pts, encoding="spline")

    assert "KPRFIT= 1" in out
    assert "RPRESS=" in out
    # flux-space encoding must NOT emit a ZPRESS array
    assert "ZPRESS=" not in out


def test_inject_requires_terminator():
    ods = _make_raw_ods()
    pts = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw5")
    with pytest.raises(RuntimeError):
        km.inject_pressure_constraint(" &IN1\n PLASMA= 1.0\n", pts)


# --------------------------------------------------------------------------- #
# scale_plasma
# --------------------------------------------------------------------------- #

def test_scale_plasma_rescales_value():
    out = km.scale_plasma(_BASE_KFILE, 0.5)
    m = re.search(r"(?m)^\s*PLASMA\s*=\s*([-+0-9.eE]+)", out)
    assert m is not None
    assert np.isclose(float(m.group(1)), 12345.6 * 0.5)
    # untouched fields survive
    assert "ISHOT= 48233" in out
    assert "RCENTR= 0.4" in out


def test_scale_plasma_missing_raises():
    with pytest.raises(RuntimeError):
        km.scale_plasma(" &IN1\n RCENTR= 0.4\n /\n", 0.5)


# --------------------------------------------------------------------------- #
# run_kinetic_efit: skipped when no executable
# --------------------------------------------------------------------------- #

def test_run_kinetic_efit_skipped_without_executable(tmp_path, monkeypatch):
    monkeypatch.delenv("EFITHOME", raising=False)
    monkeypatch.delenv("EFIT", raising=False)
    ods = _make_raw_ods()
    pts = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw6")
    config = km.KineticEFITConfig(
        executable=None, workdir=str(tmp_path), shot=48233, time_ms=300.0
    )
    inputs = km.KineticEFITInputs(
        workdir=tmp_path,
        ods=ods,
        base_kfile_text=_BASE_KFILE,
        points=pts,
        kfiles=(),
    )
    result = km.run_kinetic_efit(inputs, config)

    assert result.status == "skipped"
    assert result.converged is False
    assert result.ok is False
    assert "EFIT" in result.reason


def test_run_kinetic_efit_skipped_bad_executable_path(tmp_path, monkeypatch):
    monkeypatch.delenv("EFITHOME", raising=False)
    monkeypatch.delenv("EFIT", raising=False)
    ods = _make_raw_ods()
    pts = km.kinetic_pressure_points(ods, 300.0, None, encoding="raw6")
    config = km.KineticEFITConfig(
        executable=str(tmp_path / "does_not_exist_efit"),
        workdir=str(tmp_path),
        shot=48233,
        time_ms=300.0,
    )
    inputs = km.KineticEFITInputs(
        workdir=tmp_path, base_kfile_text=_BASE_KFILE, points=pts
    )
    result = km.run_kinetic_efit(inputs, config)
    assert result.status == "skipped"
