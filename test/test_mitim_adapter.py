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
TGLF_RUN = ROOT / "test" / "data" / "gacode" / "tglf_reg05"
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
    def run(self, subfolder, cold_start=True, code_settings=None, extraOptions=None):
        mode = os.environ.get("STUB_MODE", "ok")
        if mode == "sleep":
            time.sleep(30)
        if mode == "raise":
            raise RuntimeError("stub NEO failure")
        if mode == "die":
            os._exit(3)  # killed before the driver can write result.json
        # MITIM 5.3.0's layout: every radius in one folder, each file suffixed by it.
        target = self.folder / subfolder
        target.mkdir(parents=True, exist_ok=True)
        if mode != "empty":
            roa_for = json.loads(os.environ.get("STUB_ROA_FOR", "{}"))   # MITIM's own map
            for rho in self.rhos:
                for path in Path(os.environ["STUB_NEO_RUN"]).iterdir():
                    shutil.copy(path, target / f"{path.name}_{rho:.4f}")
                if f"{rho:.4f}" in roa_for:
                    (target / f"input.neo_{rho:.4f}").write_text(
                        f"RMIN_OVER_A = {roa_for[f'{rho:.4f}']}\\nDENS_1 = 0.8\\nTEMP_1 = 1.0\\n")
        (self.folder / "seen_env.json").write_text(json.dumps(
            {"MITIM_CONFIG": os.environ.get("MITIM_CONFIG"), "PATH": os.environ.get("PATH")}))
    def read(self, label):
        pass
'''


TGLFTOOLS = '''
import os, shutil
from pathlib import Path

class TGLFinput:
    @classmethod
    def initialize_in_memory(cls, parameters):
        self = cls(); self.parameters = dict(parameters); self.file = None; return self
    def write_state(self):
        Path(self.file).write_text("".join(f"{k} = {v}\\n" for k, v in self.parameters.items()))

class TGLF:
    def __init__(self, rhos):
        self.rhos = list(rhos)
    def prep(self, input_gacode, folder, cold_start=True, forceIfcold_start=True):
        self.folder = Path(folder); self.folder.mkdir(parents=True, exist_ok=True)
        assert Path(input_gacode).is_file()
    def run(self, subfolder, code_settings=None, extraOptions=None, cold_start=True, forceIfcold_start=True):
        target = self.folder / subfolder; target.mkdir(parents=True, exist_ok=True)
        (self.folder / "options.txt").write_text(repr((code_settings, extraOptions)))
        import json
        roa_for = json.loads(os.environ.get("STUB_ROA_FOR", "{}"))   # MITIM's own rho -> r/a
        for rho in self.rhos:
            label = f"{rho:.4f}"
            for path in Path(os.environ["STUB_TGLF_RUN"]).iterdir():
                if path.name != "input.tglf":
                    shutil.copy(path, target / f"{path.name}_{label}")
            roa = roa_for.get(label, rho)
            (target / f"input.tglf_{label}").write_text(f"SAT_RULE = 3\\nRMIN_LOC = {roa}\\n")
'''

PROFILESTOOLS = '''
class gacode_state:
    def __init__(self, path):
        self.derived = {"roa": [0.0, 0.5, 1.0]}
        self.profiles = {"rho(-)": [0.0, 0.4, 1.0]}
    def derive_quantities(self, mi_ref=None):
        pass
    def to_tglf(self, r, code_settings="SAT0", r_is_rho=True):
        assert r_is_rho is False
        return {roa: {"SAT_RULE": 3, "RMIN_LOC": roa, "NKY": 19} for roa in r}
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
    (package / "mitim_tools" / "gacode_tools" / "TGLFtools.py").write_text(TGLFTOOLS)
    (package / "mitim_tools" / "gacode_tools" / "PROFILEStools.py").write_text(PROFILESTOOLS)
    (package / "mitim_tools" / "misc_tools").mkdir()
    (package / "mitim_tools" / "misc_tools" / "__init__.py").write_text("")
    (package / "mitim_tools" / "misc_tools" / "PLASMAtools.py").write_text("md_u = 2.0\n")
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
    monkeypatch.setenv("STUB_TGLF_RUN", str(TGLF_RUN))
    monkeypatch.setenv("STUB_MODE", "ok")
    config = mitim.MITIMConfig(python=sys.executable,
                               gacode=GACODEConfig(home=str(home), platform="STUB"),
                               env={"PYTHONPATH": str(site)})
    return config


# --------------------------------------------------------------------------- availability


def test_importing_the_adapter_never_imports_mitim():
    import subprocess

    # An editable install elsewhere can win over PYTHONPATH, so the child names the
    # tree it imported and the test skips rather than test another checkout.
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import vaft, vaft.code.mitim; "
            "print(vaft.__file__); "
            "print(any(m.startswith(('mitim_tools','mitim_modules')) for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code, str(ROOT)], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": str(ROOT)}, check=False)
    lines = out.stdout.strip().splitlines()
    if out.returncode != 0 or not lines or not lines[0].startswith(str(ROOT)):
        pytest.skip(f"the child imported another vaft checkout: {lines[:1] or out.stderr[-200:]}")
    assert lines[-1] == "False"


def test_no_interpreter_is_configuration_missing(monkeypatch):
    monkeypatch.delenv(mitim.MITIM_PYTHON_ENV, raising=False)
    assert mitim.mitim_availability().status == "configuration_missing"
    missing = mitim.mitim_availability(mitim.MITIMConfig(python="/no/such/python"))
    assert missing.status == "not_installed"


def test_an_interpreter_without_mitim_is_not_installed(tmp_path, monkeypatch):
    found = mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable,
                                                        env={"PYTHONPATH": str(tmp_path)}))
    assert found.status == "not_installed" and "mitim_tools" in found.detail


def test_an_unlisted_version_is_unsupported_not_assumed(tmp_path, monkeypatch):
    site = str(_stub_mitim(tmp_path, version="4.0.0"))
    found = mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable, env={"PYTHONPATH": site}))
    assert found.status == "unsupported_version" and found.version == "4.0.0"


def test_missing_portals_and_missing_gacode_are_named(tmp_path, monkeypatch):
    site = str(_stub_mitim(tmp_path / "a", portals=False))
    config = mitim.MITIMConfig(python=sys.executable, env={"PYTHONPATH": site})
    assert mitim.mitim_availability(config).status == "portals_unavailable"
    site = str(_stub_mitim(tmp_path / "b"))
    for variable in ("GACODEHOME", "GACODE_ROOT"):
        monkeypatch.delenv(variable, raising=False)
    config = mitim.MITIMConfig(python=sys.executable, env={"PYTHONPATH": site})
    assert mitim.mitim_availability(config).status == "gacode_unavailable"


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
    result, outputs = mitim.run_neo_smoke(profile, [0.5, 0.7], tmp_path / "run", ready)
    assert result.ok and result.runtime_status == "completed", (result.stderr, result.result)
    assert [Path(d).name for d in result.result["run_directories"]] == ["rho_0.5000", "rho_0.7000"]
    assert len(outputs) == 2 and all(out.transport is not None for out in outputs)
    record = json.loads((tmp_path / "run" / "record.json").read_text())
    assert record["mitim"]["version"] == "5.3.0"
    assert record["arguments"]["rhos"] == [0.5, 0.7]
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
    monkeypatch.setenv("STUB_MODE", "die")
    second, outputs = mitim.run_neo_smoke(profile, [0.5], tmp_path / "run", ready)
    assert second.returncode == 3 and second.result is None
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
    found = mitim.mitim_availability(mitim.MITIMConfig(python=sys.executable,
                                                        env={"PYTHONPATH": str(site)}))
    assert found.status == "not_installed" and "templates" in found.detail


def test_the_callers_pythonpath_never_reaches_the_mitim_interpreter(ready, profile, tmp_path, monkeypatch):
    from dataclasses import replace

    from vaft.code.mitim.runner import _command, _environment

    monkeypatch.setenv("PYTHONPATH", "/caller/worktree:/caller/py314/site-packages")
    monkeypatch.setenv("PYTHONHOME", "/caller/home")
    bare = replace(ready, env={})
    environment = _environment(bare, tmp_path / "c.json")
    assert "PYTHONPATH" not in environment and "PYTHONHOME" not in environment
    command = _command(bare, "/mitim/python", "driver.py")
    assert command[:5] == ("env", "-u", "PYTHONPATH", "-u", "PYTHONHOME")
    (path,) = [part for part in command if part.startswith("PYTHONPATH=")]
    home = ready.gacode.home
    assert path == "PYTHONPATH=" + os.pathsep.join([f"{home}/f2py", f"{home}/f2py/pygacode"])
    assert "/caller/" not in path   # GACODE's f2py only, never the caller's entries
    assert "PYTHONPATH=" + ready.env["PYTHONPATH"] in _command(ready, "/mitim/python", "driver.py")
    # The probe drops it too: with the stub only on the caller's PYTHONPATH, MITIM is absent.
    monkeypatch.setenv("PYTHONPATH", ready.env["PYTHONPATH"])
    assert mitim.mitim_availability(bare).status == "not_installed"
    assert mitim.mitim_availability(ready).ready


def test_a_slow_probe_is_a_timeout_not_a_broken_install(tmp_path):
    slow = tmp_path / "python"
    slow.write_text("#!/bin/sh\nsleep 5\n")
    slow.chmod(0o755)
    found = mitim.mitim_availability(mitim.MITIMConfig(python=str(slow)), timeout=1)
    assert found.status == "probe_timeout"


def _mitim_maps_back(monkeypatch, profile, surfaces, shift=None):
    import json

    rho = mitim.rho_tor_norm_at(profile, surfaces)
    roa = {f"{x:.4f}": r + (shift or {}).get(r, 0.0) for x, r in zip(rho, surfaces)}
    monkeypatch.setenv("STUB_ROA_FOR", json.dumps(roa))


def test_mitim_tglf_runs_at_the_bridged_rho_and_is_keyed_back_by_r_over_a(ready, profile, tmp_path, monkeypatch):
    surfaces = [0.4, 0.7]
    _mitim_maps_back(monkeypatch, profile, surfaces)
    result, outputs = mitim.run_mitim_tglf(profile, surfaces, tmp_path / "run", ready,
                                           extra_options={"NKY": 12})
    assert result.ok, (result.stderr, result.result)
    rho = mitim.rho_tor_norm_at(profile, surfaces)
    assert result.record["arguments"]["rho_tor_norm"] == pytest.approx(list(rho))
    assert result.record["arguments"]["r_over_a"] == surfaces
    assert result.result["status"] == "ok"
    assert sorted(outputs) == surfaces and all(o.gbflux is not None for o in outputs.values())
    assert "{'NKY': 12}" in (tmp_path / "run" / "tglf" / "options.txt").read_text()


def test_mitim_local_inputs_are_written_at_exact_r_over_a(ready, profile, tmp_path):
    result, parsed = mitim.mitim_tglf_local_inputs(profile, [0.3, 0.6], tmp_path / "run", ready)
    assert result.ok, (result.stderr, result.result)
    assert sorted(parsed) == [0.3, 0.6]
    assert parsed[0.6]["RMIN_LOC"] == 0.6 and parsed[0.6]["NKY"] == 19
    assert result.result["rho_tor_norm"] == pytest.approx([0.24, 0.52])   # MITIM's own map


def test_colliding_rho_labels_are_refused_before_running(ready, profile, tmp_path):
    with pytest.raises(ValueError, match="collide"):
        mitim.run_mitim_tglf(profile, [0.5, 0.50001], tmp_path / "run", ready)


def test_a_surface_mitim_ran_elsewhere_is_reported_missing(ready, profile, tmp_path, monkeypatch):
    # MITIM maps r/a 0.4 back to 0.41: that surface is not the one asked for.
    surfaces = [0.4, 0.7]
    _mitim_maps_back(monkeypatch, profile, surfaces, shift={0.4: 0.01})
    result, outputs = mitim.run_mitim_tglf(profile, surfaces, tmp_path / "run", ready)
    assert sorted(outputs) == [0.7]
    assert not result.ok and result.result["status"] == "partial"
    assert result.result["missing_r_over_a"] == [0.4]
    assert result.result["shifted_r_over_a"] == {0.4: pytest.approx(0.41)}
    # The run directory says the same as the returned result (cold review 0.8.0
    # delta-absorb-17 transport F6): record.json, result.json and result.record.
    assert result.record["result_status"] == "partial"
    assert result.record["missing_r_over_a"] == [0.4]
    on_disk = json.loads((tmp_path / "run" / "record.json").read_text(encoding="utf-8"))
    assert on_disk["result_status"] == "partial" and on_disk["missing_r_over_a"] == [0.4]
    assert on_disk["shifted_r_over_a"] == {"0.4": pytest.approx(0.41)}
    written = json.loads((tmp_path / "run" / "result.json").read_text(encoding="utf-8"))
    assert written["status"] == "partial" and written["missing_r_over_a"] == [0.4]


def test_mitim_neo_checks_the_radius_it_ran_and_keeps_its_inputs(ready, profile, tmp_path, monkeypatch):
    surfaces = [0.4, 0.7]
    _mitim_maps_back(monkeypatch, profile, surfaces, shift={0.7: 0.02})
    result, outputs, inputs = mitim.run_mitim_neo(profile, surfaces, tmp_path / "run", ready,
                                                  extra_options={"ROTATION_MODEL": 1})
    assert sorted(outputs) == [0.4] and inputs[0.4]["DENS_1"] == 0.8
    assert result.result["status"] == "partial" and result.result["missing_r_over_a"] == [0.7]
    assert result.record["arguments"]["extra_options"] == {"ROTATION_MODEL": 1}
    assert result.record["result_status"] == "partial"
    on_disk = json.loads((tmp_path / "run" / "record.json").read_text(encoding="utf-8"))
    assert on_disk["result_status"] == "partial" and on_disk["missing_r_over_a"] == [0.7]
