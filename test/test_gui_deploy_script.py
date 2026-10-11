"""``vaft/deploy/gui/test-instance.sh`` (#1862): dry runs only, nothing installed."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "vaft" / "deploy" / "gui" / "test-instance.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _run(*args, env=None, tmp_path=None):
    environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin"}
    if tmp_path is not None:
        environment.update({
            "VAFT_GUI_TEST_SITE": str(tmp_path / "site.conf"), "VAFT_GUI_TEST_ENV": str(tmp_path / "missing.env"),
            "VAFT_GUI_TEST_UNIT": str(tmp_path / "unit"), "VAFT_GUI_TEST_VENV": str(tmp_path / "venv"),
            "VAFT_GUI_TEST_STATE": str(tmp_path / "state"), "VAFT_GUI_TEST_SNIPPETS": str(tmp_path / "snippets"),
            "VAFT_GUI_TEST_CONFD": str(tmp_path / "confd.conf"),
        })
    environment.update(env or {})
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=environment, timeout=60)


def test_install_never_starts_without_a_filled_env_file(tmp_path):
    result = _run("install", "abc123", "--dry-run", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert "vaft@abc123" in result.stdout and "NOT STARTED" in result.stdout
    assert "enable --now" not in result.stdout.replace("then: systemctl enable --now vaft-gui-test", "")


def test_install_starts_once_the_account_is_filled_in(tmp_path):
    env_file = tmp_path / "missing.env"
    env_file.write_text("HS_ENDPOINT=http://127.0.0.1:5101\nHS_USERNAME=gui-test-reader\nHS_PASSWORD=filled\n")
    result = _run("install", "abc123", "--dry-run", tmp_path=tmp_path)
    assert "+ systemctl enable --now vaft-gui-test" in result.stdout


def test_publishing_needs_the_approved_ranges(tmp_path):
    result = _run("nginx", "--dry-run", tmp_path=tmp_path)
    assert result.returncode != 0 and "approved internal ranges" in result.stderr


def test_without_root_and_without_dry_run_nothing_runs(tmp_path):
    import os

    if os.geteuid() == 0:
        pytest.skip("running as root")
    result = _run("install", "abc123", tmp_path=tmp_path)
    assert result.returncode == 1 and "run as root" in result.stderr and "+ " not in result.stdout


def test_remove_leaves_the_production_service_alone():
    # The real default paths, printed only (--dry-run).
    result = _run("remove", "--dry-run")
    assert result.returncode == 0
    for line in result.stdout.splitlines():
        if line.startswith("+ "):
            assert "vaft-gui-test" in line or "nginx" in line or "daemon-reload" in line, line
            assert "vaft-gui.service" not in line and "/etc/vaft-gui.env" not in line and "/opt/vaft-gui " not in line
