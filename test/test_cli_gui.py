"""``vaft gui``: parsing, the missing-Panel message and argument pass-through (#1086)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from vaft.cli import gui as gui_cli
from vaft.cli._main import main as cli_main

ROOT = Path(__file__).resolve().parents[1]


def test_help_parses_without_importing_panel():
    code = (
        "import sys\n"
        "from vaft.cli._main import main\n"
        "try:\n"
        "    main(['gui', '--help'])\n"
        "except SystemExit as exit:\n"
        "    assert exit.code == 0, exit.code\n"
        "assert 'panel' not in sys.modules, 'help imported panel'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ssh -L 5006:localhost:5006" in result.stdout


def test_missing_panel_exits_1_naming_the_extra(monkeypatch, capsys):
    def missing():
        raise ImportError("The VAFT GUI needs the panel package; install it with `pip install 'vaft[gui]'`.")

    monkeypatch.setattr("vaft.gui.require_panel", missing)
    assert cli_main(["gui", "--no-show"]) == 1
    assert "vaft[gui]" in capsys.readouterr().err


def test_arguments_reach_serve(monkeypatch):
    pytest.importorskip("panel")
    calls = []
    monkeypatch.setattr("vaft.gui.app.serve", lambda **kwargs: calls.append(kwargs))
    assert gui_cli.main([
        "--shot", "41524", "41672", "--source", "main", "--plot", "plasma_current_time",
        "--port", "5010", "--no-show", "--allow-websocket-origin", "vest:5010",
    ]) == 0
    assert calls == [{
        "address": "127.0.0.1", "port": 5010, "show": False,
        "websocket_origin": ["vest:5010"], "auth": "auto", "hosted": False, "prefix": None,
        "sample": None, "file": None,
        "shot": [41524, 41672], "namespace": "main", "plot": "plasma_current_time",
    }]


def test_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        gui_cli.main(["--sample", "39915", "--shot", "39915"])


def test_hosted_passes_through_and_refuses_files(monkeypatch, capsys):
    pytest.importorskip("panel")
    calls = []
    monkeypatch.setattr("vaft.gui.app.serve", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setenv("VAFT_GUI_PASSWORD", "team")
    assert gui_cli.main(["--hosted", "--prefix", "/gui", "--shot", "39915", "--no-show"]) == 0
    assert calls[-1]["hosted"] is True and calls[-1]["prefix"] == "/gui"
    with pytest.raises(SystemExit):
        gui_cli.main(["--hosted", "--file", "eq.json"])
    assert "opens no files" in capsys.readouterr().err


def test_a_hosted_server_without_its_password_does_not_start(monkeypatch, capsys):
    pytest.importorskip("panel")
    served = []
    monkeypatch.setattr("panel.serve", lambda *args, **kwargs: served.append(kwargs))
    monkeypatch.delenv("VAFT_GUI_PASSWORD", raising=False)
    assert gui_cli.main(["--hosted", "--no-show"]) == 1
    assert "VAFT_GUI_PASSWORD" in capsys.readouterr().err and not served
