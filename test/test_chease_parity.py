"""Input-fidelity tests for vaft.code.chease.

Pure text/logic assertions -- no CHEASE binary required.

Most of what this file pins comes from the legacy VEST CHEASE workflow the
adapter was ported from, and is kept deliberately so a refactor cannot quietly
change the inputs VAFT writes and make its refined equilibria incomparable with
the ones that workflow already produced: edge-zeroing, FFT boundary smoothing,
``EPSLON``, and ``$CHEASE`` resolution.

Two things the donor did are deliberately *not* reproduced, because they were
conventions rather than physics:

  * ``NIDEAL=11``, which upstream CHEASE rejects outright (#717). The adapter
    now selects NIDEAL from its GEQDSK output contract (6, #516); the writer
    still emits 11 through the deprecated raw override.
  * sampling q at ``sqrt(0.95)``, which constrains the surface at
    ``psi_norm = 0.9747`` rather than the conventional q95 surface at 0.95.
    ``q_constraint_psi_norm`` now means what it says.
"""

import os
from pathlib import Path
import stat

import numpy as np
import pytest

from vaft.code import chease as ch

from external_code_stubs import write_launchable_stub


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


def _q_at(q, psi_norm):
    """q interpolated at a normalized-poloidal-flux location."""
    from scipy.interpolate import interp1d

    x = np.linspace(0.0, 1.0, len(q))
    return float(interp1d(x, q, kind="cubic", fill_value="extrapolate")(psi_norm))


# ---------------------------------------------------------------------------
# (d) EPSLON / NIDEAL defaults
# ---------------------------------------------------------------------------

#: Minimal params for the namelist writer. Only the EPSLON and NIDEAL lines
#: are under test here, so the constraint fields carry placeholder values --
#: the surface itself is asserted in section (a).
_WRITER_PARAMS = {
    "ASPCT": 0.3, "R0EXP": 1.0, "B0EXP": 0.5, "CURRT": 0.1,
    "QSPEC": 1.9, "CSSPEC": float(np.sqrt(0.95)),
    "SIGNB0XP": 1.0, "SIGNIPXP": 1.0,
}


def test_the_default_nideal_is_one_upstream_chease_accepts():
    """The default has to be runnable, not merely faithful to jsk95.

    Upstream CHEASE validates the range in ``cotrol.f90`` (0 to 10) and quits
    before doing any equilibrium work on anything outside it, so the previous
    default of 11 meant a bare ``CHEASEConfig()`` could not run at all against
    a CHEASE built from the public repository. See #717. Since #516 a caller
    names the output, not the number.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the default path must not warn
        cfg = ch.CHEASEConfig()
    assert cfg.output == "geqdsk"
    assert cfg.nideal is None
    assert cfg.resolved_nideal == ch.CHEASE_OUTPUT_NIDEAL["geqdsk"] == 6
    text = "".join(ch._namelist_lines(cfg, _WRITER_PARAMS))
    assert "NIDEAL=6," in text
    assert "NEQDSK=0" in text  # EXPEQ input, written by the adapter itself


def test_an_output_the_adapter_cannot_read_is_refused():
    """NIDEAL=9 and friends write files nothing here reads (#516)."""
    with pytest.raises(ValueError, match="not supported"):
        ch.CHEASEConfig(output="gyrokinetic")


def test_namelist_epslon_default_and_jsk95_nideal_on_request():
    """jsk95 parity is a property of the writer, not of the default.

    ``NIDEAL=11`` is what the VEST jsk95 workflow runs, against the CHEASE
    revision that group uses. It stopped being the default in #717 because
    upstream rejects it, but asking for it must still emit it -- that is the
    parity claim this file exists to defend. Asking is deprecated (#516), so
    it warns.
    """
    with pytest.warns(FutureWarning, match="deprecated"):
        cfg = ch.CHEASEConfig(nideal=11)
    assert cfg.epslon_exponent == 10
    assert cfg.resolved_nideal == 11
    text = "".join(ch._namelist_lines(cfg, _WRITER_PARAMS))
    assert "EPSLON=1.0E-10," in text
    assert "NIDEAL=11," in text
    assert "NCSCAL=1," in text


def test_namelist_epslon_exponent_is_configurable():
    cfg = ch.CHEASEConfig(epslon_exponent=9)
    params = {
        "ASPCT": 0.3, "R0EXP": 1.0, "B0EXP": 0.5, "CURRT": 0.1,
        "QSPEC": 1.9, "CSSPEC": 0.0,
        "SIGNB0XP": 1.0, "SIGNIPXP": 1.0,
    }
    text = "".join(ch._namelist_lines(cfg, params))
    assert "EPSLON=1.0E-9," in text


# ---------------------------------------------------------------------------
# (a) q constraint -> QSPEC / CSSPEC, and namelist CSSPEC wiring
# ---------------------------------------------------------------------------

def test_q_is_constrained_on_the_surface_the_config_names(tmp_path):
    """QSPEC is q at psi_norm, and CSSPEC is that same surface on CHEASE's mesh.

    CHEASE reads CSSPEC on ``s = sqrt(psi_norm)`` and squares it back
    (``norept.f90``), so a constraint at 0.95 has to be written as
    ``sqrt(0.95)`` and no more. The donor applied the root twice and so
    constrained 0.9747 instead; that is the deviation this asserts is gone.
    """
    cfg = ch.CHEASEConfig(target_psin=0.0)  # use RBBBS boundary, skip skimage
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)

    sign_q = np.sign(geq["CURRENT"]) * np.sign(geq["BCENTR"])  # -1

    assert params["CONSTRAINT_PSI_NORM"] == pytest.approx(0.95, rel=1e-12)
    assert params["CSSPEC"] == pytest.approx(np.sqrt(0.95), rel=1e-12)
    # What CHEASE will read back out of CSSPEC is the surface we asked for.
    assert params["CSSPEC"] ** 2 == pytest.approx(0.95, rel=1e-12)
    # QSPEC is q sampled at that same surface, times the signs.
    assert params["QSPEC"] == pytest.approx(_q_at(geq["QPSI"], 0.95) * sign_q, rel=1e-9)
    assert params["QSPEC"] < 0.0  # sign_q = -1 for this fixture


def test_the_old_double_root_surface_is_not_what_gets_constrained(tmp_path):
    """Guards the correction itself, since both forms look plausible.

    q at 0.95 and q at sqrt(0.95) differ on any real profile, so asserting the
    new value alone would not catch a revert to the old one.
    """
    cfg = ch.CHEASEConfig(target_psin=0.0)
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)

    donor_surface = float(np.sqrt(0.95))
    assert _q_at(geq["QPSI"], donor_surface) != pytest.approx(
        _q_at(geq["QPSI"], 0.95), rel=1e-6
    )
    assert params["CSSPEC"] != pytest.approx(0.95**0.25, rel=1e-6)
    assert params["CONSTRAINT_PSI_NORM"] != pytest.approx(donor_surface, rel=1e-6)


def test_namelist_emits_the_constraint_surface(tmp_path):
    cfg = ch.CHEASEConfig(target_psin=0.0)
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)
    text = "".join(ch._namelist_lines(cfg, params))
    assert f"CSSPEC={np.sqrt(0.95):.6f}," in text
    # QSPEC appears on the CURRT line; check the numeric value is present.
    assert "CSSPEC=0.000000," not in text


def test_a_configurable_constraint_surface_is_honoured(tmp_path):
    """The surface is a knob, not a constant folded into the writer."""
    cfg = ch.CHEASEConfig(target_psin=0.0, q_constraint_psi_norm=0.8)
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)
    sign_q = np.sign(geq["CURRENT"]) * np.sign(geq["BCENTR"])
    assert params["CSSPEC"] == pytest.approx(np.sqrt(0.8), rel=1e-12)
    assert params["QSPEC"] == pytest.approx(_q_at(geq["QPSI"], 0.8) * sign_q, rel=1e-9)


def test_axis_q_when_no_constraint_surface_is_given(tmp_path):
    cfg = ch.CHEASEConfig(target_psin=0.0, q_constraint_psi_norm=None)
    geq = _fake_geqdsk()
    params = ch._write_expeq(geq, tmp_path / "EXPEQ", cfg)
    sign_q = np.sign(geq["CURRENT"]) * np.sign(geq["BCENTR"])
    assert params["CSSPEC"] == 0.0
    assert params["QSPEC"] == pytest.approx(geq["QPSI"][0] * sign_q, rel=1e-12)
    text = "".join(ch._namelist_lines(cfg, params))
    assert "CSSPEC=0.000000," in text  # CHEASE's own "constrain on axis"


# ---------------------------------------------------------------------------
# (b) edge-zeroing helper
# ---------------------------------------------------------------------------

def test_edge_zero_zeros_positive_edge_and_flattens_ffprim():
    psin = np.linspace(0.0, 1.0, 11)
    pprime = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.5, 0.7, 0.9])
    ffprim = np.arange(11, dtype=float)
    pp, ff, surface = ch._edge_zero_profiles(psin, pprime, ffprim, 0.95)
    # Positive edge points (indices 8,9) zeroed; index 10 (edge) untouched by loop.
    assert pp[8] == 0.0 and pp[9] == 0.0
    assert pp[7] == -1.0
    # FF' flattened inward across the zeroed band (held at just-outside value).
    assert ff[9] == pytest.approx(10.0)  # ff[9] <- ff[10]
    assert ff[8] == pytest.approx(10.0)  # ff[8] <- ff[9] (already updated)
    # 0.95 > 0.3, so the donor's inward-nudge branch never fires for an edge
    # constraint; it exists for a near-axis one.
    assert surface == pytest.approx(0.95)


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
    exe = write_launchable_stub(tmp_path / "chease")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.delenv("CHEASEHOME", raising=False)
    monkeypatch.delenv("CHEASE_EXEC_DIR", raising=False)
    monkeypatch.setenv("CHEASE", str(exe))
    resolved = ch._resolve_executable(ch.CHEASEConfig())
    assert resolved == exe


def test_resolve_executable_CHEASE_dir(tmp_path, monkeypatch):
    exe = write_launchable_stub(tmp_path / "chease")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.delenv("CHEASEHOME", raising=False)
    monkeypatch.delenv("CHEASE_EXEC_DIR", raising=False)
    monkeypatch.setenv("CHEASE", str(tmp_path))  # directory containing 'chease'
    resolved = ch._resolve_executable(ch.CHEASEConfig())
    assert resolved == exe


def test_resolve_executable_config_env_precedence(tmp_path, monkeypatch):
    exe = write_launchable_stub(tmp_path / "chease")
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


# ---------------------------------------------------------------------------
# (e) What CHEASE receives is visible to the caller (#885)
# ---------------------------------------------------------------------------

def test_the_materialized_input_is_the_expeq_content_in_source_units(tmp_path):
    """The record must say what EXPEQ says, and be comparable with the source.

    Undo EXPEQ's normalization and sign pattern and the record's p'/FF' must
    come back; undo the resampling and the source's must come back.
    """
    import json

    from vaft.data.eqdsk import read_geqdsk
    from vaft.data.resources import data_path

    source = data_path("efit/g039915.00319")
    inputs = ch.prepare_chease_inputs(
        source, ch.CHEASEConfig(workdir=tmp_path, create_plot=False)
    )
    m = inputs.materialized
    geq = read_geqdsk(source)

    # Resampling only, in the source's own sign.
    x = np.linspace(0.0, 1.0, len(geq["PPRIME"]))
    np.testing.assert_allclose(m.pprime_resampled, np.interp(m.psi_norm, x, geq["PPRIME"]))
    np.testing.assert_allclose(m.ffprime_resampled, np.interp(m.psi_norm, x, geq["FFPRIM"]))

    # The EXPEQ file, parsed back, is the record re-normalized.
    lines = inputs.expeq.read_text(encoding="utf-8").splitlines()
    count = int(lines[3])
    boundary = np.array([[float(v) for v in line.split()] for line in lines[4 : 4 + count]])
    r0 = m.parameters["R0EXP"]
    np.testing.assert_allclose(boundary * r0, m.boundary, rtol=1e-10, atol=1e-12)
    n = int(lines[4 + count])
    offset = 4 + count + 2 + n  # skip N, "1 0", then the sqrt(psi_N) column
    pp = np.array([float(v) for v in lines[offset : offset + n]])
    ff = np.array([float(v) for v in lines[offset + n : offset + 2 * n]])
    b0 = m.parameters["B0EXP"]
    psi_sign = m.sign_transform["psi"]
    assert psi_sign == -1  # 39915 is flipped into CHEASE's pattern, so the sign is exercised
    # The written profiles, not only the resampled ones, keep the source's sign.
    assert np.sign(np.median(m.pprime)) == np.sign(np.median(geq["PPRIME"]))
    assert np.sign(np.median(m.ffprime)) == np.sign(np.median(geq["FFPRIM"]))
    np.testing.assert_allclose(pp, -np.abs(m.pprime) * ch.MU0 * r0**2 / b0, rtol=1e-10)
    np.testing.assert_allclose(ff, -(psi_sign * m.ffprime) / b0, rtol=1e-10)

    record = json.loads((tmp_path / "chease_input_manifest.json").read_text())
    assert record["nideal"] == 6 and record["neqdsk"] == 0
    assert record["expeq_boundary_points"] == m.boundary.shape[0]
    assert record["q_constraint_psi_norm_used"] == pytest.approx(0.95)


def test_edge_conditioning_is_reported_where_it_changed_the_profile(tmp_path):
    """A reversed edge p' sample is zeroed, and the record says which ones."""
    geq = _fake_geqdsk()
    geq["PPRIME"] = geq["PPRIME"].copy()
    geq["PPRIME"][-3:] = 0.2  # opposite to the negative bulk
    payload = ch._expeq_payload(geq, ch.CHEASEConfig(target_psin=0.0))
    m = ch._materialized_input(payload, ch.CHEASEConfig(target_psin=0.0), {"psi": 1})
    changed = np.flatnonzero(m.edge_modified)
    assert changed.size > 0
    assert m.psi_norm[changed].min() > 0.97  # the edge, and only the edge
    assert np.all(m.pprime[changed] == 0.0)
    assert np.all(m.pprime[~m.edge_modified] == m.pprime_resampled[~m.edge_modified])
    # The donor loop starts one sample in from the separatrix, so the last
    # sample is never examined and a reversal there survives into EXPEQ as
    # -|p'|. Recorded, not endorsed: the record has to show it.
    assert not m.edge_modified[-1] and m.pprime[-1] > 0.0


def test_the_record_undoes_the_psi_flip_on_the_written_profiles():
    """p' and FF' flip with psi on the way into CHEASE, and back in the record."""
    geq = _fake_geqdsk()
    cfg = ch.CHEASEConfig(target_psin=0.0)
    payload = ch._expeq_payload(geq, cfg)
    kept = ch._materialized_input(payload, cfg, {"psi": 1})
    flipped = ch._materialized_input(payload, cfg, {"psi": -1})
    np.testing.assert_array_equal(flipped.pprime, -kept.pprime)
    np.testing.assert_array_equal(flipped.ffprime, -kept.ffprime)
    np.testing.assert_array_equal(flipped.pressure, kept.pressure)  # not a d/dpsi quantity


def test_a_q_constraint_on_axis_is_not_reported_as_a_surface(tmp_path):
    cfg = ch.CHEASEConfig(target_psin=0.0, q_constraint_psi_norm=None)
    m = ch._materialized_input(ch._expeq_payload(_fake_geqdsk(), cfg), cfg, {"psi": 1})
    assert m.q_constraint_psi_norm is None
    assert m.csspec == 0.0


def test_the_solver_mesh_defaults_and_overrides_reach_the_namelist():
    """``nw`` is the output box; the solve's own mesh is NS x NT (#885)."""
    text = "".join(ch._namelist_lines(ch.CHEASEConfig(), _WRITER_PARAMS))
    assert "NS=150, NT=150," in text
    assert "NPSI=200, NCHI=100," in text
    assert "NRBOX=513," in text

    cfg = ch.CHEASEConfig(ns=60, nt=40, npsi=120, nchi=64)
    text = "".join(ch._namelist_lines(cfg, _WRITER_PARAMS))
    assert "NS=60, NT=40," in text
    assert "NPSI=120, NCHI=64," in text

    with pytest.raises(ValueError, match="at least 2"):
        ch.CHEASEConfig(ns=1)
