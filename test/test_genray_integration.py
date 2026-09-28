"""GENRAY with the real executable (#264). Skips cleanly without ``$GENRAYHOME``.

Two runs: upstream's EC ITER regression case against its gold ``genray.nc``
(needs ``$GENRAY_SOURCE_DIR`` for the case files), and a VEST 48224 case built
entirely by ``vaft.code.genray`` from the packaged sample plus a launcher at the
6 kW ECH port.
"""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

pytest.importorskip("netCDF4")

GENRAYHOME = os.environ.get("GENRAYHOME")
pytestmark = pytest.mark.skipif(
    not GENRAYHOME or not (Path(GENRAYHOME) / "bin" / "xgenray").is_file(),
    reason="GENRAY is not installed ($GENRAYHOME/bin/xgenray); see install/install_genray.sh",
)

REPOSITORY = Path(__file__).resolve().parents[1]


def _checker():
    spec = importlib.util.spec_from_file_location("check_genray", REPOSITORY / "install" / "check_genray.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_ec_iter_regression_case_reproduces_upstream_gold(tmp_path):
    source = os.environ.get("GENRAY_SOURCE_DIR")
    if not source:
        pytest.skip("GENRAY_SOURCE_DIR is not set, so upstream's reference case is not available")
    checker = _checker()
    case = Path(source) / checker.REFERENCE_CASE
    for name in ("genray.dat", "equilib.dat"):
        shutil.copy2(case / name, tmp_path / name)

    subprocess.run([str(Path(GENRAYHOME) / "bin" / "xgenray")], cwd=tmp_path, capture_output=True, timeout=600, check=True)

    assert checker.compare_with_gold(tmp_path / "genray.nc", case / "gold-genray.nc") == []


@pytest.fixture(scope="module")
def vest_48224_with_launcher():
    import vaft.data
    import vaft.omas

    ods = vaft.omas.load(vaft.data.sample(48224))
    beam = "ec_launchers.beam.0"
    ods["ec_launchers.ids_properties.homogeneous_time"] = 0
    ods[f"{beam}.name"] = "ECH 6 kW 2.45 GHz"
    ods[f"{beam}.time"] = np.array([0.3])
    for path, value in {
        "launching_position.r": 0.8021,
        "launching_position.z": -0.36,
        "launching_position.phi": math.radians(210.0),
        "steering_angle_pol": 0.0,
        "steering_angle_tor": 0.0,
    }.items():
        ods[f"{beam}.{path}"] = np.array([value])
    ods[f"{beam}.frequency.time"] = np.array([0.3])
    ods[f"{beam}.frequency.data"] = np.array([2.45e9])
    return ods


def test_a_vest_case_runs_end_to_end_into_waves(vest_48224_with_launcher, tmp_path):
    from vaft.code.genray import GENRAYConfig, run

    ods = vest_48224_with_launcher.copy()
    config = GENRAYConfig(mode="X", time=0.3, power_w=3000.0, zeff=2.0, minimum_temperature_ev=1.0, timeout=600)

    result = run(ods, config, workdir=tmp_path)

    assert result.ok, result.stdout[-2000:]
    beam = ods["waves.coherent_wave.0.beam_tracing.0.beam.0"]
    r, z, phi = (np.asarray(beam[f"position.{c}"]) for c in ("r", "z", "phi"))
    # GENRAY starts at the plasma boundary on the launch line: inside the
    # launcher radius, at the launcher's height and toroidal angle, heading in.
    assert r[0] < 0.8021
    assert z[0] == pytest.approx(-0.36, abs=1e-3)
    # GENRAY reports -150 deg for 210 deg: the same angle, wrapped. A reversed
    # toroidal sense would give +150 deg, which this rejects.
    assert math.remainder(phi[0] - math.radians(210.0), 2 * math.pi) == pytest.approx(0.0, abs=1e-3)
    assert np.asarray(beam["wave_vector.k_r"])[0] < 0.0
    assert beam["power_initial"] == pytest.approx(3000.0, rel=1e-6)
    absorbed = np.asarray(beam["electrons.power"])
    assert np.all(np.diff(absorbed) >= -1e-9) and absorbed[-1] <= 3000.0 + 1e-6
    assert ods["waves.coherent_wave.0.global_quantities.0.frequency"] == pytest.approx(2.45e9)


def test_an_oblique_launch_keeps_the_imas_toroidal_sense(vest_48224_with_launcher, tmp_path):
    """Aim 20 deg toward +phi: GENRAY's first traced N_phi must be positive.

    R N_phi is conserved in an axisymmetric plasma, so a reversed alfast sign or
    frame handedness would start the ray with N_phi < 0.
    """
    from vaft.code.genray import GENRAYConfig, run

    ods = vest_48224_with_launcher.copy()
    ods["ec_launchers.beam.0.steering_angle_tor"] = np.array([math.radians(20.0)])
    config = GENRAYConfig(mode="X", time=0.3, power_w=3000.0, zeff=2.0, minimum_temperature_ev=1.0, timeout=600)

    result = run(ods, config, workdir=tmp_path)

    if not result.ok:
        pytest.skip(f"the oblique ray does not enter the 48224 plasma: {result.parsed and result.parsed.get('stop_reasons')}")
    k_tor = np.asarray(ods["waves.coherent_wave.0.beam_tracing.0.beam.0.wave_vector.k_tor"])
    assert k_tor[0] > 0.0
