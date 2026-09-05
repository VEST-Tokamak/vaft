"""External-code home conventions and compatibility lookup tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from vaft.code import chease, efit, gpec, nubeam
from vaft.code.tes import runner as tes_runner
from vaft.code.tes.config import TESConfig


_EXTERNAL_ENVIRONMENT = (
    "GPECHOME",
    "CHEASEHOME",
    "CHEASE",
    "CHEASE_EXEC_DIR",
    "EFITHOME",
    "EFIT",
    "TESHOME",
    "RTES",
    "NUBEAMHOME",
    # TokaMaker (Open FUSION Toolkit) is imported in-process rather than run
    # as a $XHOME/bin binary; these steer library discovery and sys.path.
    "OFT_ROOTPATH",
    "OFT_LIBRARY_DIR",
    "OFT_INSTALL_DIR",
)


@pytest.fixture(autouse=True)
def clear_external_code_environment(monkeypatch):
    for name in _EXTERNAL_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)


def _executable(root: Path, relative_path: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_canonical_home_layouts_resolve_expected_executables(monkeypatch, tmp_path):
    gpec_executable = _executable(tmp_path / "gpec", "bin/dcon")
    chease_executable = _executable(tmp_path / "chease", "bin/chease")
    efit_executable = _executable(tmp_path / "efit", "bin/efit")
    tes_executable = _executable(tmp_path / "tes", "bin/rtes")
    nubeam_executable = _executable(tmp_path / "nubeam", "bin/nubeam_comp_exec")
    monkeypatch.setenv("GPECHOME", str(tmp_path / "gpec"))
    monkeypatch.setenv("CHEASEHOME", str(tmp_path / "chease"))
    monkeypatch.setenv("EFITHOME", str(tmp_path / "efit"))
    monkeypatch.setenv("TESHOME", str(tmp_path / "tes"))
    monkeypatch.setenv("NUBEAMHOME", str(tmp_path / "nubeam"))

    assert gpec._executable(gpec.GPECSuiteConfig(), "dcon") == gpec_executable
    assert chease.find_chease_executable() == chease_executable
    assert efit.find_efit_executable() == efit_executable
    assert tes_runner._resolve_executable(TESConfig()) == str(tes_executable)
    assert nubeam.find_nubeam_executable() == nubeam_executable


def test_invalid_home_is_not_masked_by_legacy_executable(monkeypatch, tmp_path):
    legacy_efit = _executable(tmp_path / "legacy", "efit")
    monkeypatch.setenv("EFITHOME", str(tmp_path / "uncompiled-efit"))
    monkeypatch.setenv("EFIT", str(legacy_efit))

    with pytest.raises(FileNotFoundError) as error:
        efit.find_efit_executable()

    message = str(error.value)
    assert "EFITHOME" in message
    assert str(tmp_path / "uncompiled-efit/bin/efit") in message
    assert "Compile or install EFIT" in message


@pytest.mark.parametrize(
    ("home_variable", "relative_path", "resolve"),
    [
        (
            "GPECHOME",
            "bin/dcon",
            lambda: gpec._executable(gpec.GPECSuiteConfig(), "dcon"),
        ),
        ("CHEASEHOME", "bin/chease", chease.find_chease_executable),
        ("EFITHOME", "bin/efit", efit.find_efit_executable),
        (
            "TESHOME",
            "bin/rtes",
            lambda: tes_runner._resolve_executable(TESConfig()),
        ),
        ("NUBEAMHOME", "bin/nubeam_comp_exec", nubeam.find_nubeam_executable),
    ],
)
def test_each_adapter_reports_missing_home_executable(
    monkeypatch, tmp_path, home_variable, relative_path, resolve
):
    root = tmp_path / home_variable.lower()
    monkeypatch.setenv(home_variable, str(root))

    with pytest.raises(FileNotFoundError) as error:
        resolve()

    message = str(error.value)
    assert home_variable in message
    assert str(root / relative_path) in message
    assert "Compile or install" in message


@pytest.mark.skipif(os.name == "nt", reason="Windows does not expose POSIX execute bits")
def test_non_executable_home_binary_has_actionable_error(monkeypatch, tmp_path):
    binary = tmp_path / "chease/bin/chease"
    binary.parent.mkdir(parents=True)
    binary.write_text("not executable", encoding="utf-8")
    binary.chmod(0o644)
    monkeypatch.setenv("CHEASEHOME", str(tmp_path / "chease"))

    with pytest.raises(PermissionError) as error:
        chease.find_chease_executable()

    message = str(error.value)
    assert "CHEASEHOME" in message
    assert str(binary) in message
    assert "not executable" in message


def test_legacy_variables_work_when_home_is_unset(monkeypatch, tmp_path):
    legacy_efit = _executable(tmp_path, "legacy/efit")
    legacy_chease = _executable(tmp_path, "legacy/chease")
    legacy_rtes = _executable(tmp_path, "legacy/rtes")
    monkeypatch.setenv("EFIT", str(legacy_efit))
    monkeypatch.setenv("CHEASE_EXEC_DIR", str(legacy_chease.parent))
    monkeypatch.setenv("RTES", str(legacy_rtes))

    assert efit.find_efit_executable() == legacy_efit
    assert chease.find_chease_executable() == legacy_chease
    assert tes_runner._resolve_executable(TESConfig()) == str(legacy_rtes)


def test_unconfigured_message_names_home_and_expected_layout():
    message = efit._efit_unconfigured_reason()

    assert "EFITHOME" in message
    assert "bin/efit" in message
    assert "$EFIT" in message


def test_initialization_notebook_documents_supported_names():
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks/initialize_external_fusion_codes.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    reference = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    for name in _EXTERNAL_ENVIRONMENT + ("VAFT_FILEDB_DIR",):
        assert f"`{name}`" in reference
