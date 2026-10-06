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
        "websocket_origin": ["vest:5010"], "auth": "auto", "sample": None, "file": None,
        "shot": [41524, 41672], "namespace": "main", "plot": "plasma_current_time", "workspace": None,
    }]


def test_the_first_workspace_is_checked_against_the_registry(monkeypatch, capsys):
    pytest.importorskip("panel")
    calls = []
    monkeypatch.setattr("vaft.gui.app.serve", lambda **kwargs: calls.append(kwargs))
    assert gui_cli.main(["--workspace", "database", "--no-show"]) == 0
    assert calls[-1]["workspace"] == "database"
    assert gui_cli.main(["--workspace", "nope", "--no-show"]) == 2
    assert "plots, database" in capsys.readouterr().err and len(calls) == 1


def test_sources_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        gui_cli.main(["--sample", "39915", "--shot", "39915"])
