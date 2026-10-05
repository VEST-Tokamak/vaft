"""The MITIM adapter's availability, configuration and launch contract (#1588 stage A1).

No MITIM is installed in CI. A stub ``mitim_tools`` package on ``PYTHONPATH`` plays
the isolated interpreter (this test's own Python), and a stub GACODE tree supplies
the launchers. The stub ``NEOtools`` copies the recorded 48224 NEO run, so result
discovery is exercised against real NEO output files.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from vaft.code import mitim
from vaft.code.gacode._types import GACODEConfig

ROOT = Path(__file__).resolve().parents[1]
NEO_RUN = ROOT / "test" / "data" / "gacode" / "neo_vest_48224_profile"
SAMPLE = ROOT / "vaft" / "data" / "kineticEfit" / "ods_48224_300ms.json"

NEOTOOLS = '''
import json, os, shutil, time
from pathlib import Path

class NEO:
    def __init__(self, rhos):
        self.rhos = list(rhos)
    def prep(self, input_gacode, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        assert Path(input_gacode).is_file()
    def run(self, subfolder, cold_start=True, code_settings=None):
        mode = os.environ.get("STUB_MODE", "ok")
        if mode == "sleep":
            time.sleep(30)
        if mode == "raise":
            raise RuntimeError("stub NEO failure")
        target = self.folder / subfolder / "rho_0"
        target.mkdir(parents=True, exist_ok=True)
        if mode != "empty":
            for path in Path(os.environ["STUB_NEO_RUN"]).iterdir():
                shutil.copy(path, target / path.name)
        (self.folder / "seen_env.json").write_text(json.dumps(
            {"MITIM_CONFIG": os.environ.get("MITIM_CONFIG"), "PATH": os.environ.get("PATH")}))
    def read(self, label):
        pass
'''


def _stub_mitim(root: Path, *, version="5.3.0", portals=True) -> Path:
    package = root / "site"
    (package / "mitim_tools" / "gacode_tools").mkdir(parents=True)
    (package / "mitim_tools" / "__init__.py").write_text(
        f"from pathlib import Path\n__version__ = {version!r}\n__mitimroot__ = Path(__file__).parents[1]\n")
    (package / "templates").mkdir()
    (package / "templates" / "input.neo.controls").write_text("")
    (package / "mitim_tools" / "gacode_tools" / "__init__.py").write_text("")
    (package / "mitim_tools" / "gacode_tools" / "NEOtools.py").write_text(NEOTOOLS)
    if portals:
        (package / "mitim_modules" / "portals").mkdir(parents=True)
        for init in ("mitim_modules/__init__.py", "mitim_modules/portals/__init__.py"):
            (package / init).write_text("")
        (package / "mitim_modules" / "portals" / "PORTALSmain.py").write_text("")
    return package


def _stub_gacode(root: Path) -> Path:
    home = root / "gacode"
    for code in ("neo", "tglf"):
        launcher = home / code / "bin" / code
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o755)
    (home / "shared" / "bin").mkdir(parents=True)
    (home / "platform" / "build").mkdir(parents=True)
    (home / "platform" / "build" / "make.inc.STUB").write_text("")
    return home


@pytest.fixture
def ready(tmp_path, monkeypatch):
    site = _stub_mitim(tmp_path)
    home = _stub_gacode(tmp_path)
    monkeypatch.setenv("PYTHONPATH", str(site))
    monkeypatch.setenv("STUB_NEO_RUN", str(NEO_RUN))
    monkeypatch.setenv("STUB_MODE", "ok")
    config = mitim.MITIMConfig(python=sys.executable,
                               gacode=GACODEConfig(home=str(home), platform="STUB"),
                               env={"PYTHONPATH": str(site)})
    return config


# --------------------------------------------------------------------------- availability


def test_importing_the_adapter_never_imports_mitim():
    import subprocess

    code = "import sys, vaft.code.mitim; print(any(m.startswith(('mitim_tools','mitim_modules')) for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT)}, check=True)
    assert out.stdout.strip() == "False"


def test_no_interpreter_is_configuration_missing(monkeypatch):
    monkeypatch.delenv(mitim.MITIM_PYTHON_ENV, raising=False)
    assert mitim.mitim_availability().status == "configuration_missing"
    missing = mitim.mitim_availability(mitim.MITIMConfig(python="/no/such/python"))
    assert missing.status == "not_installed"


def test_an_interpreter_without_mitim_is_not_installed(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    found = mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable))
    assert found.status == "not_installed" and "mitim_tools" in found.detail


def test_an_unlisted_version_is_unsupported_not_assumed(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(_stub_mitim(tmp_path, version="4.0.0")))
    found = mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable))
    assert found.status == "unsupported_version" and found.version == "4.0.0"


def test_missing_portals_and_missing_gacode_are_named(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(_stub_mitim(tmp_path / "a", portals=False)))
    assert mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable)).status == "portals_unavailable"
    monkeypatch.setenv("PYTHONPATH", str(_stub_mitim(tmp_path / "b")))
    for variable in ("GACODEHOME", "GACODE_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    assert mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable)).status == "gacode_unavailable"


def test_ready_reports_versions_and_the_gacode_build(ready):
    found = mitim.mitim_availability(ready)
    assert found.ready, found.detail
    assert found.version == "5.3.0" and found.portals
    assert found.gacode_platform == "STUB" and found.gacode_root.endswith("gacode")
    assert found.python_version


# --------------------------------------------------------------------------- configuration


def test_the_per_run_config_keeps_mitim_inside_vafts_allocation(tmp_path):
    document = mitim.mitim_user_config(mitim.MITIMConfig(cores=4), tmp_path, username="u")
    assert set(document["preferences"]) - {"verbose_level", "dpi_notebook"} == set(mitim.config.MITIM_CODES)
    assert {v for k, v in document["preferences"].items() if k in mitim.config.MITIM_CODES} == {
        mitim.VAFT_MACHINE}
    machine = document[mitim.VAFT_MACHINE]
    assert machine["machine"] == "local" and "slurm" not in machine
    assert machine["cores_per_node"] == 4
    assert machine["scratch"].startswith(str(tmp_path.resolve()))


# --------------------------------------------------------------------------- launch


@pytest.fixture
def profile():
    if not SAMPLE.exists():
        pytest.skip("the packaged 48224 sample is a repository asset")
    from omas import load_omas_json

    from vaft.process.transport_state import TransportStateKey, resolve_transport_state

    ods = load_omas_json(str(SAMPLE), consistency_check=False)
    state = resolve_transport_state(ods, TransportStateKey(48224, 0.3, "magnetics"), efit_quality="good")
    assert state.resolved, state.reasons
    return state.profile


def test_the_smoke_run_writes_vafts_input_and_reads_neo_back(ready, profile, tmp_path):
    result, outputs = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", ready)
    assert result.ok and result.runtime_status == "completed", (result.stderr, result.result)
    assert len(outputs) == 1 and outputs[0].transport is not None
    record = json.loads((tmp_path / "run" / "record.json").read_text())
    assert record["mitim"]["version"] == "5.3.0"
    assert record["arguments"]["rhos"] == [0.5]
    assert len(record["arguments"]["input_gacode_sha256"]) == 64
    assert len(record["mitim_config_sha256"]) == 64
    seen = json.loads((tmp_path / "run" / "neo" / "seen_env.json").read_text())
    assert seen["MITIM_CONFIG"] == str(tmp_path / "run" / "mitim_config.json")
    assert seen["PATH"].split(os.pathsep)[0].endswith(os.path.join("neo", "bin"))


def test_a_failing_driver_is_a_failed_result_not_an_exception(ready, profile, tmp_path, monkeypatch):
    monkeypatch.setenv("STUB_MODE", "raise")
    result, outputs = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", ready)
    assert not result.ok and result.status == "failed" and outputs == []
    assert "stub NEO failure" in result.result["error"]
    monkeypatch.setenv("STUB_MODE", "empty")
    result, _ = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", ready)
    assert not result.ok and "no out.neo.transport" in result.result["error"]


def test_a_timeout_is_a_result(ready, profile, tmp_path, monkeypatch):
    from dataclasses import replace

    monkeypatch.setenv("STUB_MODE", "sleep")
    result, _ = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", replace(ready, timeout=3))
    assert result.runtime_status == "timeout" and result.timed_out and not result.ok
    assert json.loads((tmp_path / "run" / "record.json").read_text())["runtime_status"] == "timeout"


def test_a_stale_result_is_never_reported(ready, profile, tmp_path, monkeypatch):
    first, _ = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", ready)
    assert first.ok
    monkeypatch.setenv("STUB_MODE", "raise")
    second, outputs = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", ready)
    assert not second.ok and outputs == []


def test_a_run_is_refused_unless_ready(tmp_path, monkeypatch):
    monkeypatch.delenv(mitim.MITIM_PYTHON_ENV, raising=False)
    with pytest.raises(FileNotFoundError, match="configuration_missing"):
        mitim.run_mitim_driver("neo_smoke", {}, tmp_path, mitim.MITIMConfig())


def test_drivers_import_only_mitim_and_the_standard_library():
    drivers = ROOT / "vaft" / "code" / "mitim" / "drivers"
    for path in drivers.glob("*.py"):
        assert "vaft" not in "".join(
            line for line in path.read_text().splitlines(True) if line.lstrip().startswith(("import", "from"))
        ), path.name


def test_a_wheel_install_without_templates_is_not_installed(tmp_path, monkeypatch):
    site = _stub_mitim(tmp_path)
    (site / "templates" / "input.neo.controls").unlink()
    monkeypatch.setenv("PYTHONPATH", str(site))
    found = mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable))
    assert found.status == "not_installed" and "templates" in found.detail
