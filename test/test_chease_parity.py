"""jsk95 parity tests for vaft.code.chease.

Pure text/logic assertions -- no CHEASE binary required. Run from the
feature/kinetic_profile_eq worktree (where vaft/code/chease.py exists) after
applying the edits in chease_jsk95.md:

    pytest -q scratchpad/port/test_chease_parity.py

These tests exercise:
  * namelist writer emits EPSLON=1.0E-10 and NIDEAL=11 (jsk95 defaults),
  * CSSPEC and QSPEC are driven by the q95 (double-sqrt) constraint,
  * _resolve_executable honors the $CHEASE environment variable.
"""

import os
from pathlib import Path
import stat

import numpy as np
import pytest

from vaft.code import chease as ch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_geqdsk(n: int = 129):
    """Minimal dict-backed GEQDSK sufficient for _write_expeq with target_psin=0.

    A plain dict supports both ``key in geqdsk`` and ``geqdsk[key]`` used by
    _write_expeq, and RBBBS/ZBBBS let _target_boundary short-circuit (no skimage).
    """
    x = np.linspace(0.0, 1.0, n)
    # A monotone, cubically-interpolable q profile: q0=1.86, rising to edge.
    q = 1.86 + 3.0 * x**2

    # Simple circular boundary, R0=1.0 m, a=0.3 m.
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    rbbbs = 1.0 + 0.3 * np.cos(theta)
    zbbbs = 0.0 + 0.3 * np.sin(theta)

    return {
        "NW": n,
        "NH": n,
        "PSIRZ": np.zeros((n, n)),
        "PPRIME": -np.linspace(1.0, 0.0, n),  # negative-definite interior
        "FFPRIM": -np.linspace(2.0, 0.5, n),
        "PRES": np.linspace(5.0e3, 0.0, n),
        "FPOL": np.linspace(0.5, 0.4, n),
        "QPSI": q,
        "RCENTR": 1.0,
        "BCENTR": 0.5,     # sign(+)
        "CURRENT": -1.0e5,  # sign(-)  => sign_q = -1
        "RBBBS": rbbbs,
        "ZBBBS": zbbbs,
    }


def _expected_q95(q, qloc):
    from scipy.interpolate import interp1d

    x = np.linspace(0.0, 1.0, len(q))
    return float(interp1d(x, q, kind="cubic", fill_value="extrapolate")(qloc))


# ---------------------------------------------------------------------------
# (d) EPSLON / NIDEAL defaults
# ---------------------------------------------------------------------------

def test_namelist_epslon_and_nideal_defaults_match_jsk95():
    cfg = ch.CHEASEConfig()
    assert cfg.nideal == 11
    assert cfg.epslon_exponent == 10
    params = {
        "ASPCT": 0.3, "R0EXP": 1.0, "B0EXP": 0.5, "CURRT": 0.1,
        "QSPEC": 1.9, "CSSPEC": 0.9872864, "QLOC": np.sqrt(0.95),
        "SIGNB0XP": 1.0, "SIGNIPXP": 1.0,
    }
    text = "".join(ch._namelist_lines(cfg, params))
    assert "EPSLON=1.0E-10," in text
    assert "NIDEAL=11," in text
    assert "NCSCAL=1," in text


def test_namelist_epslon_exponent_is_configurable():
    cfg = ch.CHEASEConfig(epslon_exponent=9)
    params = {
        "ASPCT": 0.3, "R0EXP": 1.0, "B0EXP": 0.5, "CURRT": 0.1,
        "QSPEC": 1.9, "CSSPEC": 0.0, "QLOC": 0.0,
        "SIGNB0XP": 1.0, "SIGNIPXP": 1.0,
    }
    text = "".join(ch._namelist_lines(cfg, params))
    assert "EPSLON=1.0E-9," in text


# ---------------------------------------------------------------------------
# (a) q95 constraint -> QSPEC / CSSPEC, and namelist CSSPEC wiring
# ---------------------------------------------------------------------------

def test_csspec_double_sqrt_and_qspec_from_q95(tmp_path):
    cfg = ch.CHEASEConfig(target_psin=0.0)  # use RBBBS boundary, skip skimage
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)

    qloc = float(np.sqrt(0.95))
    sign_q = np.sign(geq["CURRENT"]) * np.sign(geq["BCENTR"])  # -1
    expected_qspec = _expected_q95(geq["QPSI"], qloc) * sign_q

    # CSSPEC is the double sqrt: sqrt(sqrt(0.95)) == 0.95 ** 0.25.
    assert params["CSSPEC"] == pytest.approx(0.95**0.25, rel=1e-12)
    assert params["QLOC"] == pytest.approx(qloc, rel=1e-12)
    # QSPEC is q sampled at psi_N = sqrt(0.95), NOT at 0.95, times the signs.
    assert params["QSPEC"] == pytest.approx(expected_qspec, rel=1e-9)
    # Sanity: QSPEC is negative here (sign_q = -1) and q(sqrt0.95) != q(0.95).
    assert params["QSPEC"] < 0.0
    assert _expected_q95(geq["QPSI"], qloc) != pytest.approx(
        _expected_q95(geq["QPSI"], 0.95), rel=1e-6
    )


def test_namelist_emits_q95_csspec(tmp_path):
    cfg = ch.CHEASEConfig(target_psin=0.0)
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)
    text = "".join(ch._namelist_lines(cfg, params))
    assert f"CSSPEC={0.95**0.25:.6f}," in text
    # QSPEC appears on the CURRT line; check the numeric value is present.
    assert "CSSPEC=0.000000," not in text


def test_legacy_axis_q_when_constraint_disabled(tmp_path):
    cfg = ch.CHEASEConfig(target_psin=0.0, q95_constraint=False)
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)
    sign_q = np.sign(geq["CURRENT"]) * np.sign(geq["BCENTR"])
    assert params["CSSPEC"] == 0.0
    assert params["QSPEC"] == pytest.approx(geq["QPSI"][0] * sign_q, rel=1e-12)
    text = "".join(ch._namelist_lines(cfg, params))
    assert "CSSPEC=0.000000," in text  # byte-identical to pre-jsk95 output


# ---------------------------------------------------------------------------
# (b) edge-zeroing helper
# ---------------------------------------------------------------------------

def test_edge_zero_zeros_positive_edge_and_flattens_ffprim():
    psin = np.linspace(0.0, 1.0, 11)
    pprime = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5, 0.7, 0.9])
    ffprim = np.arange(11, dtype=float)
    pp, ff, qloc = ch._edge_zero_profiles(psin, pprime, ffprim, float(np.sqrt(0.95)))
    # Positive edge points (indices 8,9) zeroed; index 10 (edge) untouched by loop.
    assert pp[8] == 0.0 and pp[9] == 0.0
    assert pp[7] == -1.0
    # FF' flattened inward across the zeroed band (held at just-outside value).
    assert ff[9] == pytest.approx(10.0)  # ff[9] <- ff[10]
    assert ff[8] == pytest.approx(10.0)  # ff[8] <- ff[9] (already updated)
    # qloc = sqrt(0.95) > 0.3, so the eqdsk nudge branch never fires.
    assert qloc == pytest.approx(np.sqrt(0.95))


# ---------------------------------------------------------------------------
# (c) FFT boundary smoothing
# ---------------------------------------------------------------------------

def test_fft_boundary_returns_nf_points_on_circle():
    theta = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    rz = np.column_stack([1.0 + 0.3 * np.cos(theta), 0.3 * np.sin(theta)])
    out = ch._smooth_boundary_fft(rz, nf=128)
    assert out.shape == (128, 2)
    # A circle of minor radius 0.3 about (1.0, 0.0) is reproduced.
    rad = np.sqrt((out[:, 0] - 1.0) ** 2 + out[:, 1] ** 2)
    assert np.allclose(rad, 0.3, atol=1e-3)


# ---------------------------------------------------------------------------
# (e) $CHEASE executable resolution
# ---------------------------------------------------------------------------

def test_resolve_executable_honors_CHEASE_env(tmp_path, monkeypatch):
    exe = tmp_path / "chease"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.delenv("CHEASEHOME", raising=False)
    monkeypatch.delenv("CHEASE_EXEC_DIR", raising=False)
    monkeypatch.setenv("CHEASE", str(exe))
    resolved = ch._resolve_executable(ch.CHEASEConfig())
    assert resolved == exe


def test_resolve_executable_CHEASE_dir(tmp_path, monkeypatch):
    exe = tmp_path / "chease"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.delenv("CHEASEHOME", raising=False)
    monkeypatch.delenv("CHEASE_EXEC_DIR", raising=False)
    monkeypatch.setenv("CHEASE", str(tmp_path))  # directory containing 'chease'
    resolved = ch._resolve_executable(ch.CHEASEConfig())
    assert resolved == exe


def test_resolve_executable_config_env_precedence(tmp_path, monkeypatch):
    exe = tmp_path / "chease"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.delenv("CHEASEHOME", raising=False)
    monkeypatch.delenv("CHEASE", raising=False)
    monkeypatch.delenv("CHEASE_EXEC_DIR", raising=False)
    resolved = ch._resolve_executable(ch.CHEASEConfig(env={"CHEASE": str(exe)}))
    assert resolved == exe


# ---------------------------------------------------------------------------
# (i) Comparison figure: migrated out of the pyplot shim into vaft.plot (#139)
# ---------------------------------------------------------------------------
def _plottable_geqdsk(n: int = 33, *, scale: float = 1.0):
    geq = _fake_geqdsk(n)
    geq["PSIRZ"] = np.outer(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, n))
    geq["SIMAG"] = 0.0
    geq["SIBRY"] = 1.0
    geq["RLEFT"] = 0.6
    geq["RDIM"] = 0.8
    geq["ZMID"] = 0.0
    geq["ZDIM"] = 1.2
    geq["RLIM"] = geq["RBBBS"] * 1.15
    geq["ZLIM"] = geq["ZBBBS"] * 1.15
    for key in ("QPSI", "PRES", "PPRIME", "FFPRIM"):
        geq[key] = np.asarray(geq[key], dtype=float) * scale
    return geq


def test_comparison_model_keeps_the_four_profile_comparisons_and_the_geometry():
    from vaft.plot.models import Field2D, GeometryLayers, Profile1D

    original = _plottable_geqdsk()
    refined = _plottable_geqdsk(scale=1.1)
    model = ch._comparison_model(original, refined)

    profiles = [panel for panel in model.models if isinstance(panel, Profile1D)]
    assert [panel.title for panel in profiles] == [
        "Safety factor",
        "Pressure",
        "Pressure derivative",
        "FF prime",
    ]
    for panel in profiles:
        # Both equilibria are compared in every profile panel.
        assert [series.label for series in panel.series] == ["input", "CHEASE"]
    assert np.allclose(profiles[0].series[1].y, np.asarray(refined["QPSI"], dtype=float))

    fields = [panel for panel in model.models if isinstance(panel, Field2D)]
    assert len(fields) == 1
    # The refined flux map carries the input boundary as an overlay.
    assert [layer.label for layer in fields[0].overlays] == ["input boundary"]

    geometry = [panel for panel in model.models if isinstance(panel, GeometryLayers)]
    assert len(geometry) == 1
    labels = [layer.label for layer in geometry[0].layers if layer.label]
    assert labels == ["input boundary", "CHEASE boundary"]
    # Limiters are drawn for both, unlabeled.
    assert len(geometry[0].layers) == 4


def test_comparison_metrics_is_public_and_reports_current_diff():
    # Issue #172: promoted out of the private surface so the chease validation
    # stage can reuse it, extended with an Ip/CURRENT comparison.
    original = _plottable_geqdsk()
    refined = _plottable_geqdsk(scale=1.1)
    refined["CURRENT"] = original["CURRENT"] * 1.05

    metrics = ch.comparison_metrics(original, refined)

    assert metrics["current_abs_diff"] == pytest.approx(abs(original["CURRENT"] * 0.05))
    assert metrics["current_rel_diff"] == pytest.approx(0.05)
    # The four profile RMS-relative terms moved because `refined` is a scaled
    # copy of `original`.
    for key in ("q_rms_rel", "pressure_rms_rel", "pprime_rms_rel", "ffprim_rms_rel"):
        assert metrics[key] > 0


def test_comparison_metrics_current_rel_diff_is_nan_for_zero_input_current():
    original = _plottable_geqdsk()
    original["CURRENT"] = 0.0
    refined = _plottable_geqdsk(scale=1.1)

    metrics = ch.comparison_metrics(original, refined)

    assert metrics["current_abs_diff"] == pytest.approx(abs(refined["CURRENT"]))
    assert np.isnan(metrics["current_rel_diff"])


def test_comparison_plot_is_rendered_by_vaft_plot_and_saved(tmp_path):
    import matplotlib

    matplotlib.use("Agg")

    target = tmp_path / "chease_comparison.png"
    result = ch._create_comparison_plot(
        _plottable_geqdsk(), _plottable_geqdsk(scale=1.1), target
    )
    assert Path(result) == target
    assert target.stat().st_size > 10_000
