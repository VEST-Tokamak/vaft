"""A shot EFIT does not apply to is a result, not a failure (#205).

A vacuum shot reached the EFIT constraint step and failed there ("every one of
the 65 selected instants carries less than 15000.0 A", vestserver 48927,
2026-10-06), so the new-shot worker retried it until it gave up.  The
constraint step now records the verdict, and the k-file, EFIT and EFIT-ODS
steps pass it on the way they pass on ``efit.run=false``: the EFIT stage ends
``no_output`` with ``skipped: not applicable: ...``, which replication accepts.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from omas import load_omas_json, save_omas_json

from _plasma_timing_fixtures import current, grid, pickup_only, synthetic_ods
from vaft.code.efit.applicability import (
    constraints_not_applicable_reason,
    kfile_manifest_not_applicable_reason,
    kfile_manifest_paths,
    kfile_manifest_text,
    not_applicable_constraints,
)
from vaft.database.replication import ProductNotEligibleError, _nothing_to_replicate

WORKFLOW = Path(__file__).parents[1] / "workflow/automatic_pipeline_1_routine_data_processing"


def _module(name):
    spec = importlib.util.spec_from_file_location(f"{name}_not_applicable", WORKFLOW / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONSTRAINTS = _module("generate_constraints_ods")


def _vacuum():
    t = grid()
    ods = synthetic_ods(ip=pickup_only(t), t=t)
    ods["magnetics.ip.0.time"] = t  # the vacuum cut reads the node's own clock
    return ods


def _run_constraints(monkeypatch, tmp_path, ods):
    monkeypatch.setattr(CONSTRAINTS, "compose_stage_products", lambda **_: (ods, {}))
    output = tmp_path / "constraints" / "48927_constraints.json"
    monkeypatch.setattr(sys, "argv", [
        "generate_constraints_ods.py", "--shot", "48927", "--eddy-ods", "e", "--diagnostics-ods", "d",
        "--output", str(output),
    ])
    return CONSTRAINTS.main(), output


def test_a_vacuum_shot_records_efit_as_not_applicable(monkeypatch, tmp_path):
    code, output = _run_constraints(monkeypatch, tmp_path, _vacuum())

    assert code == 0
    reason = constraints_not_applicable_reason(load_omas_json(str(output), consistency_check=False))
    assert reason.startswith("peak |Ip| ") and "below CUTIP 15000 A" in reason
    assert "shot_class Vacuum" in reason


def test_a_pickup_classified_plasma_is_still_not_applicable():
    """48927: a 3.3 kA pickup that shot_class reads as a pulse is still no plasma for EFIT."""
    t = grid()
    ods = synthetic_ods(ip=current(t, peak=3.3e3), t=t)
    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=65, threshold=15000.0)

    reason = CONSTRAINTS._efit_not_applicable(ods, error)
    assert reason.startswith("peak |Ip| 3.") and "all 65 constraint instants" in reason


def test_a_window_that_missed_the_discharge_still_fails():
    """The current reaches CUTIP outside the selected instants: the window is wrong, a fault."""
    t = grid()
    ods = synthetic_ods(ip=current(t, peak=60e3), t=t)
    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=3, threshold=15000.0)

    assert CONSTRAINTS._efit_not_applicable(ods, error) is None


def test_light_without_current_is_not_a_vacuum():
    """H-alpha saw a discharge the current never shows: maybe a dead Ip channel, so fail."""
    from _plasma_timing_fixtures import light

    t = grid()
    ods = synthetic_ods(slow=light(t), fast=light(t), ip=pickup_only(t), t=t)
    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=3, threshold=15000.0)

    assert CONSTRAINTS._efit_not_applicable(ods, error) is None


def test_an_eddy_stage_without_output_is_a_verdict_not_a_failure(monkeypatch, tmp_path):
    """#1568 unscoped: the eddy manifest says `no_output` (a PF circuit was never
    recorded). Composition refuses it by name; the constraint step used to let
    that escape, so the worker retried the shot to `gave_up` for a product that
    cannot exist (cold review 0.8.0 delta-absorb-16 F1). It is now the #205
    carrier, which the k-file and EFIT steps already pass on."""
    eddy_manifest = tmp_path / "eddy-manifest.json"
    eddy_manifest.write_text(json.dumps({
        "status": "no_output",
        "eddy_status": "skipped: required input unavailable: pf_active circuits PF5, PF6 were never recorded",
    }), encoding="utf-8")
    output = tmp_path / "constraints" / "48500_constraints.json"
    monkeypatch.setattr(sys, "argv", [
        "generate_constraints_ods.py", "--shot", "48500", "--eddy-ods", str(tmp_path / "absent-eddy.json"),
        "--diagnostics-ods", str(tmp_path / "absent-diagnostics.json"),
        "--eddy-manifest", str(eddy_manifest), "--output", str(output),
    ])

    assert CONSTRAINTS.main() == 0
    reason = constraints_not_applicable_reason(load_omas_json(str(output), consistency_check=False))
    assert reason.startswith("eddy produced no output: skipped: required input unavailable")
    assert "PF5, PF6" in reason


def test_a_refused_verdict_still_fails_the_step(monkeypatch, tmp_path):
    """main() re-raises when the cut is a fault: no product, a non-zero exit."""
    monkeypatch.setattr(CONSTRAINTS, "_efit_not_applicable", lambda ods, error, **_: None)
    with pytest.raises(CONSTRAINTS.NoPlasmaCurrentError):
        _run_constraints(monkeypatch, tmp_path, _vacuum())
    assert not (tmp_path / "constraints").exists() or not list((tmp_path / "constraints").iterdir())


def test_a_condemned_rogowski_is_a_fault_not_a_verdict():
    """cold review 0.8.0 delta-absorb-16 F3: a dead or drifting plasma-current
    Rogowski reads below CUTIP on every shot. The diagnostics stage condemns it
    in `magnetics.rogowski_coil.0.current.validity` (#1373) -- the mapper writes
    no `magnetics.ip.0.validity` -- and that is the channel, not the shot."""
    t = grid()
    ods = synthetic_ods(ip=pickup_only(t), t=t)
    ods["magnetics.rogowski_coil.0.current.validity"] = -2
    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=3, threshold=15000.0)

    assert CONSTRAINTS._efit_not_applicable(ods, error) is None
    ods["magnetics.rogowski_coil.0.current.validity"] = 0
    assert CONSTRAINTS._efit_not_applicable(ods, error) is not None


def test_the_verdict_names_the_window_it_was_judged_over():
    """`magnetics.ip.0` is stored on the analysis grid; the verdict must not
    claim the whole record (cold review 0.8.0 delta-absorb-16 F3)."""
    t = grid()
    ods = synthetic_ods(ip=pickup_only(t), t=t)
    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=3, threshold=15000.0)

    reason = CONSTRAINTS._efit_not_applicable(ods, error)
    assert "over the diagnostics window" in reason
    assert "whole record" not in reason


def test_no_plasma_current_at_all_is_not_a_verdict():
    from omas import ODS

    error = CONSTRAINTS.NoPlasmaCurrentError("all below", count=3, threshold=15000.0)
    assert CONSTRAINTS._efit_not_applicable(ODS(consistency_check=False), error) is None


def test_the_cut_still_raises_a_value_error():
    """Existing callers catching ValueError keep working."""
    assert issubclass(CONSTRAINTS.NoPlasmaCurrentError, ValueError)


def test_a_constraints_product_with_slices_is_reconstructed_whatever_its_comment():
    ods = not_applicable_constraints("Vacuum shot")
    assert constraints_not_applicable_reason(ods) == "Vacuum shot"
    ods["equilibrium.time"] = np.array([0.31])
    assert constraints_not_applicable_reason(ods) is None


def test_kfile_manifest_marker_is_not_a_path():
    text = kfile_manifest_text("Vacuum shot (no pulse)\n over two lines")
    assert kfile_manifest_paths(text) == []
    assert kfile_manifest_not_applicable_reason(text) == "Vacuum shot (no pulse) over two lines"
    assert kfile_manifest_not_applicable_reason("/a/k048927.00310\n") is None


def _script(name, *args):
    result = subprocess.run([sys.executable, str(WORKFLOW / f"{name}.py"), *map(str, args)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-3000:]


def test_the_verdict_runs_through_kfile_efit_and_replication(tmp_path):
    """constraints -> k-file -> EFIT (switched on, even without a binary) -> EFIT ODS -> replication."""
    shot = 48927
    root = tmp_path / "efit" / "magnetic" / str(shot)
    constraints = root / "constraints" / "constraints.json"
    constraints.parent.mkdir(parents=True)
    save_omas_json(not_applicable_constraints("Vacuum shot (no light); all 65 below CUTIP"), str(constraints))
    kfiles = root / "kfile" / "manifest.txt"
    _script("generate_kfile", "--shot", shot, "--constraints-ods", constraints, "--output", kfiles)
    assert not list(kfiles.parent.glob("k0*"))

    gfiles, status, artifacts = root / "gfile" / "manifest.txt", root / "status.txt", root / "artifacts.json"
    _script("run_efit_reconstruction", "--shot", shot, "--kfile-manifest", kfiles,
            "--gfile-manifest", gfiles, "--status", status, "--artifact-manifest", artifacts,
            "--run", "true", "--executable", tmp_path / "no-efit")
    assert status.read_text().startswith("skipped: not applicable: Vacuum shot (no light)")
    assert gfiles.read_text() == ""

    product, manifest_path = root / "output" / "efit.json", root / "metadata" / "manifest.json"
    _script("generate_efit_ods", "--shot", shot, "--gfile-manifest", gfiles, "--status", status,
            "--constraints-ods", constraints, "--kfile-manifest", kfiles,
            "--artifact-manifest", artifacts, "--output", product, "--metadata", manifest_path)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "no_output" and manifest["slice_statuses"] == []
    assert "not applicable" in _nothing_to_replicate(manifest, "efit")


@pytest.mark.parametrize("reason", [
    "skipped: EFIT executable unavailable: /x",
    "skipped: not applicable",  # only the verdict's own form, with its colon
])
def test_replication_still_refuses_a_skip_that_is_a_fault(reason):
    manifest = {"status": "no_output", "efit_status": reason}
    with pytest.raises(ProductNotEligibleError):
        _nothing_to_replicate(manifest, "efit")


# --- #1731 class shots: what "the current reached CUTIP" means ------------------------------

def _class_shot(ip, *, lit=False):
    from _plasma_timing_fixtures import light

    t = grid()
    kwargs = {"slow": light(t), "fast": light(t)} if lit else {}
    return synthetic_ods(ip=ip(t), t=t, **kwargs)


def _error():
    return CONSTRAINTS.NoPlasmaCurrentError("all below", count=14, threshold=15000.0)


def test_a_wrong_sign_pickup_swing_is_not_a_missed_discharge():
    """48932-48937: -12 to -22 kA of PF pickup ahead of a +5 kA pulse.  The
    swing is current in the wrong direction, not a discharge the window missed."""

    def ip(t):
        y = current(t, onset=0.3104, offset=0.3267, peak=5e3)
        swing = (t >= 0.3010) & (t < 0.3060)
        y[swing] -= 20e3 * np.sin(np.pi * (t[swing] - 0.3010) / 0.005)
        return y

    ods = _class_shot(ip)
    reason = CONSTRAINTS._efit_not_applicable(ods, _error())
    assert reason is not None and "in the discharge direction" in reason
    assert reason.startswith("peak |Ip| 5.")


def test_single_sample_ringing_is_judged_on_the_box_averaged_current():
    """48886: spikes read 41 kA, every millisecond averages below 9 kA."""

    def ip(t):
        y = current(t, onset=0.3057, offset=0.3144, peak=6e3)
        y[::50] += 40e3 * np.where(np.arange(y[::50].size) % 2, 1.0, -1.0)
        return y

    ods = _class_shot(ip)
    assert np.max(np.abs(ods["magnetics.ip.0.data"])) > 30e3
    assert CONSTRAINTS._efit_not_applicable(ods, _error()) is not None
    # the same spikes averaged over a single sample are a missed discharge again
    assert CONSTRAINTS._efit_not_applicable(ods, _error(), average_window=0.0) is None


def test_a_lit_pulse_below_cutip_is_a_small_discharge_not_a_dead_channel():
    """48913/48924: 2.4-2.8 kA under a lit window -- the current shows the pulse,
    so the channel is alive and the discharge is too small for EFIT."""
    ods = _class_shot(lambda t: current(t, peak=2.6e3, noise=30.0), lit=True)
    reason = CONSTRAINTS._efit_not_applicable(ods, _error())
    assert reason is not None and "shot_class Plasma" in reason


def test_a_reverse_current_with_no_accepted_pulse_is_judged_on_its_magnitude():
    """The timing accepts no pulse from a negative-only record, so |Ip| decides."""
    ods = _class_shot(lambda t: -current(t, peak=60e3))
    assert CONSTRAINTS._efit_not_applicable(ods, _error()) is None


def _small_pulse_then_large_reverse(t):
    y = current(t, onset=0.285, offset=0.300, peak=8e3, rise=3e-3)
    return y - current(t, onset=0.320, offset=0.345, peak=40e3, noise=0.0)


def test_a_large_reverse_current_under_light_is_not_dismissed_as_pickup():
    """cold review: a +8 kA accepted pulse sets the direction; a lit -40 kA
    current after it may be the discharge, so the verdict stays a fault."""
    ods = _class_shot(_small_pulse_then_large_reverse, lit=True)
    assert CONSTRAINTS._efit_not_applicable(ods, _error()) is None


def test_the_same_reverse_current_in_the_dark_is_pickup():
    ods = _class_shot(_small_pulse_then_large_reverse)
    reason = CONSTRAINTS._efit_not_applicable(ods, _error())
    assert reason is not None and "in the discharge direction" in reason
