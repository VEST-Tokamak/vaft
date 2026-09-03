"""Where a figure is shown decides which interactive control can work there."""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

import vaft
import vaft.omas
from vaft.plot import environment
from vaft.plot.environment import Environment, default_interaction_backend, detect_environment
from vaft.plot.renderers.interactive import BACKENDS, resolve_backend


@pytest.fixture(scope="module")
def shot():
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vaft.omas.load(str(vaft.data.data_path("samples/39915/omas.json.gz")))


def test_a_plain_process_under_agg_gets_no_live_control():
    env = detect_environment()
    assert env.kind in ("terminal", "ipython") and env.backend == "agg" and not env.live_figures
    assert default_interaction_backend(env) == "none"


@pytest.mark.parametrize("env, expected", [
    (Environment("terminal", "macosx", True, False), "matplotlib"),
    (Environment("jupyter", "module://ipympl.backend_nbagg", True, True), "matplotlib"),
    (Environment("jupyter", "module://matplotlib_inline.backend_inline", False, True), "ipywidgets"),
    (Environment("vscode", "module://matplotlib_inline.backend_inline", False, True), "ipywidgets"),
    (Environment("jupyter", "module://matplotlib_inline.backend_inline", False, False), "none"),
    (Environment("ipython", "agg", False, False), "none"),
])
def test_the_auto_backend_follows_the_environment(env, expected):
    assert default_interaction_backend(env) == expected


def test_kernels_are_told_apart_by_their_shell_and_host(monkeypatch):
    monkeypatch.setattr(environment, "_shell_class_name", lambda: "ZMQInteractiveShell")
    monkeypatch.delenv("VSCODE_PID", raising=False)
    assert detect_environment().kind == "jupyter"
    monkeypatch.setenv("VSCODE_PID", "1")
    assert detect_environment().kind == "vscode"
    monkeypatch.setattr(environment, "_shell_class_name", lambda: "TerminalInteractiveShell")
    assert detect_environment().kind == "ipython"
    monkeypatch.setattr(environment, "_shell_class_name", lambda: "")
    assert detect_environment().kind == "terminal"


def test_the_public_backends_and_their_resolution():
    assert BACKENDS == ("auto", "matplotlib", "ipywidgets", "none")
    assert resolve_backend("auto") == "none"
    with pytest.warns(UserWarning, match="inert"):
        assert resolve_backend("matplotlib") == "matplotlib"
    with pytest.raises(ValueError, match="backend must be one of"):
        resolve_backend("qt")


def test_the_widget_backend_redraws_a_figure_pyplot_never_sees(shot):
    pytest.importorskip("ipywidgets")
    before = set(plt.get_fignums())
    result = vaft.omas.plot_equilibrium_interactive(shot, backend="ipywidgets")
    assert set(plt.get_fignums()) == before  # not a pyplot figure: never auto-displayed
    slider = result.widget
    assert type(slider).__name__ == "IntSlider" and slider.vaft_output is not None
    slider.value = 1
    assert result.navigator.position == 1
    assert "slice 2 of 9, selected" in result.figure._suptitle.get_text()
    result.navigator.select(0.3245)
    assert slider.value == result.navigator.position


@pytest.mark.skipif(shutil.which("ipython") is None, reason="IPython terminal not installed")
def test_an_ipython_terminal_is_detected_as_such():
    code = ("from vaft.plot.environment import detect_environment; "
            "print(detect_environment().kind)")
    out = subprocess.run(["ipython", "--no-banner", "-c", code], capture_output=True, text=True,
                         env={**__import__('os').environ, "MPLBACKEND": "agg"}, timeout=300)
    assert out.stdout.strip().splitlines()[-1] == "ipython", out.stderr[-500:]


@pytest.mark.skipif(shutil.which("jupyter") is None, reason="Jupyter not installed")
def test_a_jupyter_kernel_gets_a_slider_that_redraws_the_figure(tmp_path):
    import os

    pytest.importorskip("nbformat")
    import nbformat

    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(
        "import IPython.display as ipd\n"
        "shown = []\n"
        "_orig = ipd.display\n"
        "ipd.display = lambda *objs, **kw: (shown.extend(type(o).__name__ for o in objs), _orig(*objs, **kw))\n"
        "import vaft, vaft.omas, vaft.data\n"
        "from vaft.plot.environment import detect_environment, default_interaction_backend\n"
        "ods = vaft.omas.load(str(vaft.data.data_path('samples/39915/omas.json.gz')))\n"
        "res = vaft.omas.plot_equilibrium_interactive(ods)\n"
        "res.widget.value = 2\n"
        "import json, matplotlib.pyplot as plt\n"
        "print(json.dumps({'kind': detect_environment().kind, 'backend': default_interaction_backend(),"
        " 'shown': shown, 'selected': res.navigator.selected, 'pyplot': plt.get_fignums()}))\n"
    )]
    path = tmp_path / "kernel.ipynb"
    nbformat.write(nb, path)
    subprocess.run(
        ["jupyter", "nbconvert", "--to", "notebook", "--execute", "--allow-errors", str(path),
         "--output", "out.ipynb", "--output-dir", str(tmp_path), "--ExecutePreprocessor.timeout=600"],
        capture_output=True, text=True, timeout=900,
        env={**os.environ, "PYTHONPATH": str(Path(vaft.__file__).resolve().parents[1])},
    )
    executed = nbformat.read(tmp_path / "out.ipynb", as_version=4)
    streams = [o["text"] for o in executed.cells[0]["outputs"] if o.get("output_type") == "stream"]
    errors = [o for o in executed.cells[0]["outputs"] if o.get("output_type") == "error"]
    assert not errors, errors[0]["evalue"] if errors else ""
    report = json.loads([line for text in streams for line in text.splitlines() if line.startswith("{")][-1])
    assert report["kind"] == "jupyter" and report["backend"] == "ipywidgets"
    assert report["shown"][:2] == ["Image", "VBox"] and report["shown"].count("Image") == 2
    assert report["selected"] == 2 and report["pyplot"] == []
