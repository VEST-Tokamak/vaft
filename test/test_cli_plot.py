"""``vaft plot``: the canonical plots from the command line (issue #477)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import matplotlib

matplotlib.use("Agg")

import pytest

import vaft
import vaft.omas
from vaft.cli import plot as plot_cli
from vaft.cli._main import main as cli_main


@pytest.fixture(scope="module")
def sample_ods():
    import contextlib
    import io
    import warnings

    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def _fake_module(name, **attrs):
    module = ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _no_hsds(sample_ods):
    open_ods = Mock(return_value=sample_ods)
    return open_ods, patch.dict(
        "sys.modules",
        {"vaft.database.lazy_ods": _fake_module("vaft.database.lazy_ods", open_ods=open_ods, h5pyd=None)},
    )


def test_options_are_typed_python_literals_or_strings():
    parser = plot_cli._parser()
    assert plot_cli._parse_option("nperseg=256", parser) == ("nperseg", 256)
    assert plot_cli._parse_option("style=normalized", parser) == ("style", "normalized")
    assert plot_cli._parse_option("selection=[1, 2]", parser) == ("selection", [1, 2])
    assert plot_cli._parse_option("time=0.32", parser) == ("time", 0.32)
    with pytest.raises(SystemExit) as raised:
        plot_cli._parse_option("nonsense", parser)
    assert raised.value.code == 2


def test_out_writes_the_figure_without_a_display(tmp_path, sample_ods, capsys):
    open_ods, substitution = _no_hsds(sample_ods)
    target = tmp_path / "ip.png"
    with substitution:
        code = cli_main(["plot", "plasma_current_time", "--shot", "39915", "--out", str(target)])
    assert code == 0 and target.exists() and target.stat().st_size > 1000
    assert capsys.readouterr().out.strip() == str(target)
    args, kwargs = open_ods.call_args
    assert args[0] == 39915 and kwargs["source"] == "main"
    assert kwargs["ids"] == ["dataset_description", "magnetics"]


def test_without_out_the_figure_is_shown(monkeypatch, sample_ods):
    from vaft.database import plotting

    seen = {}

    def fake_render(name, shot, source=None, **kwargs):
        seen.update(name=name, shot=shot, source=source, **kwargs)
        return None, None

    monkeypatch.setattr(plotting, "render", fake_render)
    code = plot_cli.main(["plasma_current_time", "--shot", "39915", "--shot", "41524", "--source", "main",
                          "--option", "selection=all", "--no-lazy"])
    assert code == 0
    assert seen["name"] == "plasma_current_time" and seen["shot"] == [39915, 41524]
    assert seen["show"] is True and seen["lazy"] is False and seen["selection"] == "all"


def test_list_prints_the_catalogue(monkeypatch, capsys):
    from vaft.database import plotting

    monkeypatch.setattr(plotting, "available_plots", lambda *a, **k: f"Available plots -- {a} {k}")
    assert plot_cli.main(["--list", "--query", "equilibrium", "--detail"]) == 0
    out = capsys.readouterr().out
    assert "Available plots" in out and "'query': 'equilibrium'" in out and "'detail': True" in out


def test_errors_are_reported_with_exit_code_one(sample_ods, capsys):
    open_ods, substitution = _no_hsds(sample_ods)
    with substitution:
        code = plot_cli.main(["plasma_current_time", "--shot", "39915", "--source", "nope"])
    assert code == 1 and "nope" in capsys.readouterr().err and not open_ods.called
    with substitution:
        code = plot_cli.main(["no_such_plot", "--shot", "39915"])
    assert code == 1 and "no_such_plot" in capsys.readouterr().err


def test_usage_errors_exit_two():
    for argv in (["plasma_current_time"], ["--shot", "39915"]):
        with pytest.raises(SystemExit) as raised:
            plot_cli.main(argv)
        assert raised.value.code == 2


def test_the_plot_command_imports_nothing_heavy_before_parsing():
    code = (
        "import sys, vaft.cli.plot; "
        "assert 'matplotlib.pyplot' not in sys.modules and 'vaft.omas' not in sys.modules, "
        "sorted(m for m in sys.modules if m.startswith(('matplotlib', 'vaft.omas')))"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


def test_the_console_script_is_declared():
    import tomllib

    project = tomllib.loads(Path(vaft.__file__).resolve().parents[1].joinpath("pyproject.toml").read_text())["project"]
    assert project["scripts"]["vaft"] == "vaft.cli._main:main"


def test_render_to_file_is_pyplot_free_at_import():
    code = "import sys, vaft.database.plotting; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)
