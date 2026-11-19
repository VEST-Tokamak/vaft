"""Negative controls for the plasma-free vacuum benchmark.

The vacuum benchmark (:mod:`vaft.validation.vacuum_benchmark`) asks
whether the active-coil / passive-wall model reproduces measured plasma-free
magnetics.  That is only evidence if the benchmark *responds* when a sensor's
identity, sign or position is wrong.  Each control below introduces one such
error into a deep copy of the packaged 39915 sample and checks the response
on the per-sensor residual, not on a global score.

The expected observable (direction) and a detectability threshold are
declared in ``CONTROLS`` before any benchmark ran.  The *direction* follows
from physics; the *thresholds* (2x, 1.2x) are round "clearly detectable"
margins chosen with the direction, not derived quantitatively -- how much a
residual grows depends on how good the unperturbed model already is.  The
sensors were chosen by geometry only, before looking at their residuals: the
probe nearest the midplane in the inboard H2 column and an outboard C-column
probe below the midplane away from C4-04 (the probe #190 already flags).

Observed on 39915 at authoring (develop 7b1fcf19), for the record -- not used
to set the thresholds: swap H2-04 x5.9 / C3-04 x2.41; sign C1-04 x9.1, flux
loop #4 x82; move C3-04 x1.50.  The C3-04 swap and the move margins are thin.
Note that correlation alone would *not* detect the swap or the move (C3-04's
correlation rises to 0.99 under both): the amplitude-normalized residual is
the observable that does.

* **swap** -- H2-04 (inboard, R 0.089 m, Z +0.04 m, beside the central
  solenoid) and C3-04 (outboard, R 0.796 m, Z -0.14 m).  Inboard midplane Bz
  is dominated by the solenoid's own field, outboard Bz by its much weaker
  return flux, so each record lands on a sensor whose model differs in
  amplitude.  Expected: each channel's ``normalized_residual`` grows at
  least 2x.
* **sign** -- C1-04 (outboard probe) and flux loop #4.  Pearson correlation
  is odd in the measured series, so negating it negates the correlation
  *exactly*; the residual against a positively correlated model must grow.
  Expected: correlation sign-flipped to 1e-9, ``normalized_residual`` >= 2x.
* **move** -- C3-04 moved 0.20 m radially inward (R 0.796 -> 0.596 m), toward
  the PF coils' near field.  Expected: ``normalized_residual`` grows >= 1.2x.
  The smallest threshold of the set, because nothing here guarantees the
  model's field gradient at that probe is large; it is the declared
  sensitivity, not a fitted one.

And for every control: every *other* channel's per-sensor row is unchanged
-- the perturbation touches one sensor's evidence and nothing else's -- and
neither the source ODS nor the packaged file is modified.

Evidence type: ``experimental_physical`` for the unperturbed residuals (real
39915 plasma-free data against an independently specified machine model);
the controls themselves are a *self-consistency* check of the benchmark's
sensitivity, not independent evidence that the model is right.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from _negative_controls import flip_measured_sign, move_sensor, swap_measured_signals
from _sample_fixtures import sample_ods
from vaft.validation.vacuum_benchmark import run_benchmark_case

PROBE = "b_field_pol_probe"
LOOP = "flux_loop"

# Declared before the run (see module docstring).  index -> expected name is
# asserted, so a re-ordered sample cannot silently retarget a control.
CONTROLS = {
    "swap": {
        "perturb": lambda ods: swap_measured_signals(ods, PROBE, 12, 40),
        "affected": ["MagneticFieldProbe_H2-04_Bz", "MagneticFieldProbe_C3-04_Bz"],
        "min_residual_ratio": 2.0,
    },
    "sign_probe": {
        "perturb": lambda ods: flip_measured_sign(ods, PROBE, 29),
        "affected": ["MagneticFieldProbe_C1-04_Bz"],
        "min_residual_ratio": 2.0,
        "correlation_negated": True,
    },
    "sign_loop": {
        "perturb": lambda ods: flip_measured_sign(ods, LOOP, 1),
        "affected": ["Flux Loop - #4"],
        "min_residual_ratio": 2.0,
        "correlation_negated": True,
    },
    "move": {
        "perturb": lambda ods: move_sensor(ods, PROBE, 40, dr=-0.20),
        "affected": ["MagneticFieldProbe_C3-04_Bz"],
        "min_residual_ratio": 1.2,
    },
}


#: Relative tolerance for an "unchanged" float metric of an unperturbed sensor.  Moving one probe
#: changes the last bits of a few other probes' metrics on some platforms (vestserver py3.14 /
#: numpy 2.5.3: L-03/L-04 differ by <= 2.4e-13 relative; bit-identical on the py3.12 Mac env).
#: That is floating-point round-off, not attribution: the smallest declared control effect is a
#: 1.2x change, twelve orders of magnitude larger.  Status, reasons and positions stay exact.
UNCHANGED_RTOL = 1e-12


def _same(left, right) -> bool:
    if isinstance(left, float) and isinstance(right, float):
        if np.isnan(left) and np.isnan(right):
            return True
        return bool(np.isclose(left, right, rtol=UNCHANGED_RTOL, atol=0.0))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return left == right


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(case) -> dict[str, dict]:
    return {row["name"]: row for row in case["metrics"]["channels"]}


@pytest.fixture(scope="module")
def packaged_file() -> Path:
    import vaft

    return Path(vaft.data.sample(39915, representation="omas"))


@pytest.fixture(scope="module")
def stored_sha256(packaged_file) -> str:
    return _sha256(packaged_file)


@pytest.fixture(scope="module")
def nominal():
    ods = sample_ods(39915)
    return ods, run_benchmark_case(ods, shot=39915, machine_era="packaged")


@pytest.fixture(scope="module", params=sorted(CONTROLS))
def control(request, nominal):
    spec = CONTROLS[request.param]
    ods, case = nominal
    before = {
        name: np.array(ods[f"magnetics.{kind}.{i}.{q}.data"], copy=True)
        for name, kind, i, q in (
            ("p12", PROBE, 12, "field"), ("p29", PROBE, 29, "field"),
            ("p40", PROBE, 40, "field"), ("f1", LOOP, 1, "flux"),
        )
    }
    position = (float(ods[f"magnetics.{PROBE}.40.position.r"]), float(ods[f"magnetics.{PROBE}.40.position.z"]))
    perturbed, record = spec["perturb"](ods)
    result = run_benchmark_case(perturbed, shot=39915, machine_era="negative-control")
    return request.param, spec, record, case, result, ods, before, position


def test_each_control_targets_the_declared_sensors(control):
    _, spec, record, *_ = control
    assert record["affected"] == spec["affected"]


def test_the_affected_sensors_respond_in_the_declared_direction(control):
    name, spec, record, case, result, *_ = control
    base, pert = _rows(case), _rows(result)
    for sensor in spec["affected"]:
        assert base[sensor]["status"] == "evaluated", f"{sensor} must be scored in the control run"
        assert pert[sensor]["status"] == "evaluated"
        ratio = pert[sensor]["normalized_residual"] / base[sensor]["normalized_residual"]
        assert ratio >= spec["min_residual_ratio"], (
            f"{name}: {sensor} normalized residual changed by x{ratio:.3g}, declared >= "
            f"x{spec['min_residual_ratio']}"
        )
        if spec.get("correlation_negated"):
            # Pearson correlation is odd in one argument: exact up to round-off.
            assert pert[sensor]["correlation"] == pytest.approx(-base[sensor]["correlation"], abs=1e-9)


def test_no_other_sensor_moves(control):
    """Per-sensor attribution: the error shows where it was put and nowhere else."""
    _, spec, _, case, result, *_ = control
    base, pert = _rows(case), _rows(result)
    assert set(base) == set(pert)
    for sensor in sorted(set(base) - set(spec["affected"])):
        # The whole row: status, reason, every metric, position and window.
        assert set(base[sensor]) == set(pert[sensor]), sensor
        for key in base[sensor]:
            assert _same(base[sensor][key], pert[sensor][key]), (
                f"{sensor}.{key} changed under a perturbation of {spec['affected']}"
            )
    # Array-contradiction attribution is case-level: no *other* probe may be
    # newly flagged or unflagged by the perturbation.
    def others(flagged):
        return sorted(entry["channel"] for entry in flagged if entry["channel"] not in spec["affected"])

    assert others(result["channels"]["flagged"]) == others(case["channels"]["flagged"])


def test_a_control_never_touches_the_source_or_the_stored_product(control, packaged_file, stored_sha256):
    *_, ods, before, position = control
    for name, kind, i, q in (
        ("p12", PROBE, 12, "field"), ("p29", PROBE, 29, "field"),
        ("p40", PROBE, 40, "field"), ("f1", LOOP, 1, "flux"),
    ):
        assert np.array_equal(np.asarray(ods[f"magnetics.{kind}.{i}.{q}.data"]), before[name])
    assert (float(ods[f"magnetics.{PROBE}.40.position.r"]), float(ods[f"magnetics.{PROBE}.40.position.z"])) == position
    assert _sha256(packaged_file) == stored_sha256


def test_the_packaged_file_still_matches_its_manifest(packaged_file):
    """The stored product's identity, from its own manifest, not from this run."""
    import yaml

    manifest = yaml.safe_load((packaged_file.parent / "manifest.yaml").read_text(encoding="utf-8"))
    assert _sha256(packaged_file) == manifest["representations"]["omas"]["sha256"]
