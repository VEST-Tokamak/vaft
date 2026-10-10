"""``vaft plot``: the canonical plots from the command line (issue #477)."""

from __future__ import annotations

import contextlib
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


class _Substituted(contextlib.ExitStack):
    """The fake lazy store plus the shot's IDS listing; re-enterable, as the tests reuse it."""

    def __init__(self, open_ods, stored):
        super().__init__()
        self._open_ods, self._stored = open_ods, stored

    def __enter__(self):
        stack = super().__enter__()
        module = _fake_module("vaft.database.lazy_ods", open_ods=self._open_ods, h5pyd=None)
        stack.enter_context(patch.dict("sys.modules", {"vaft.database.lazy_ods": module}))
        stack.enter_context(patch("vaft.database.plotting.stored_ids", lambda shot, source=None: self._stored))
        return stack


def _no_hsds(sample_ods):
    # The adapters list the shot's IDS before opening; the sample's are what it stores.
    open_ods = Mock(return_value=sample_ods)
    return open_ods, _Substituted(open_ods, tuple(sample_ods.keys()))


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

    project = tomllib.loads(Path(vaft.__file__).resolve().parents[1].joinpath("pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["scripts"]["vaft"] == "vaft.cli._main:main"


def test_render_to_file_is_pyplot_free_at_import():
    code = "import sys, vaft.database.plotting; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=300)


def test_an_unwritable_output_path_and_an_interrupt_are_reported_not_dumped(tmp_path, sample_ods, capsys, monkeypatch):
    open_ods, substitution = _no_hsds(sample_ods)
    missing = tmp_path / "no_such_dir" / "ip.png"
    with substitution:
        code = plot_cli.main(["plasma_current_time", "--shot", "39915", "--out", str(missing)])
    assert code == 1 and "vaft plot: error:" in capsys.readouterr().err and not missing.exists()
    from vaft.database import plotting

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(plotting, "render", interrupted)
    assert plot_cli.main(["plasma_current_time", "--shot", "39915"]) == 130
    assert "interrupted" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --out on the database path takes every renderer's return shape (cold review 0.7.0 plot G12)
# ---------------------------------------------------------------------------

class _FakeAnimation:
    """What ``plot_*(..., animation=True)`` returns: saved, not unpacked."""

    def __init__(self):
        self.saved = None

    def save(self, path):
        Path(path).write_bytes(b"movie")
        self.saved = path
        return path


_FIGURE = object()  # what the fake renderers draw; the save must receive exactly this


def _render_returning(result_for):
    def fake_render(name, shot, source=None, **kwargs):
        return result_for(name, _FIGURE, "axes", kwargs)

    return fake_render


def _saving_the_figure(monkeypatch):
    """``vaft.plot.save_figure`` replaced by one that checks what it is handed and writes the file."""
    def fake_save(figure, path, *, figure_options=None, **kwargs):
        assert figure is _FIGURE, figure
        Path(path).write_bytes(b"still")
        return path

    # Through sys.modules, not the dotted string: the substitution above
    # drops the modules a block imported, so ``vaft.plot`` may be a fresh
    # module while the ``vaft`` package still holds the old one.  Patched
    # where save_rendered looks it up (its own module) and on the package.
    import vaft.plot as plot_package
    import vaft.plot.style as style_module

    monkeypatch.setattr(style_module, "save_figure", fake_save)
    monkeypatch.setattr(plot_package, "save_figure", fake_save)


def test_out_saves_an_image_sequence_views_three_tuple(monkeypatch, tmp_path, capsys):
    from vaft.database import plotting

    _saving_the_figure(monkeypatch)
    monkeypatch.setattr(plotting, "render", _render_returning(lambda n, f, a, k: (f, a, object())))
    target = tmp_path / "frames.png"
    code = plot_cli.main(["camera_visible_animation_frames", "--shot", "1", "--out", str(target)])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert target.read_bytes() == b"still" and out.strip() == str(target)


def test_out_saves_an_animation_through_its_own_save(monkeypatch, tmp_path, capsys):
    from vaft.database import plotting

    animation = _FakeAnimation()
    monkeypatch.setattr(plotting, "render", _render_returning(lambda n, f, a, k: animation))
    target = tmp_path / "movie.gif"
    code = plot_cli.main(["camera_visible_image", "--shot", "1", "--option", "animation=True", "--out", str(target)])
    assert code == 0, capsys.readouterr().err
    assert animation.saved == str(target) and target.read_bytes() == b"movie"


def test_out_refuses_an_interactive_figure_with_one_line(monkeypatch, tmp_path, capsys):
    from vaft.database import plotting
    from vaft.plot.renderers.interactive import Interactive

    _saving_the_figure(monkeypatch)
    monkeypatch.setattr(plotting, "render", _render_returning(lambda n, f, a, k: Interactive(f, a, object(), ())))
    target = tmp_path / "live.png"
    code = plot_cli.main(["plasma_current_time", "--shot", "1", "--option", "interactive=True", "--out", str(target)])
    err = capsys.readouterr().err
    assert code == 1 and "interactive=" in err and "Traceback" not in err and not target.exists()


def test_list_takes_one_shot(monkeypatch):
    from vaft.database import plotting

    monkeypatch.setattr(plotting, "available_plots", lambda *a, **k: "never reached")
    with pytest.raises(SystemExit) as raised:
        plot_cli.main(["--list", "--shot", "1", "--shot", "2"])
    assert raised.value.code == 2


@pytest.mark.parametrize("key", ["shot", "source", "lazy", "show", "name"])
def test_an_option_the_command_itself_sets_is_refused_by_name(key, capsys):
    """``--option shot=1`` used to surface as Python's "multiple values" error (cold review 0.7.0 plot F8)."""
    with pytest.raises(SystemExit) as raised:
        plot_cli.main(["plasma_current_time", "--shot", "39915", "--option", f"{key}=1"])
    assert raised.value.code == 2
    assert f"--option {key} is reserved" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --out on the --sample/--file/--request path takes the same shapes (review of #1924, R3)
# ---------------------------------------------------------------------------

def _sample_request_rendering(monkeypatch, result):
    from vaft.plot.request import PlotRequest

    monkeypatch.setattr(PlotRequest, "render", lambda self, **kwargs: result)


def test_sample_out_saves_an_image_sequence_views_three_tuple(monkeypatch, tmp_path, capsys):
    _saving_the_figure(monkeypatch)
    _sample_request_rendering(monkeypatch, (_FIGURE, "axes", object()))
    target = tmp_path / "frames.png"
    code = plot_cli.main(["camera_visible_animation_frames", "--sample", "39915", "--out", str(target)])
    out, err = capsys.readouterr()
    assert code == 0, err
    assert target.read_bytes() == b"still" and out.strip() == str(target)


def test_sample_out_saves_an_animation_through_its_own_save(monkeypatch, tmp_path, capsys):
    animation = _FakeAnimation()
    _sample_request_rendering(monkeypatch, animation)
    target = tmp_path / "movie.gif"
    code = plot_cli.main(["camera_visible_image", "--sample", "39915", "--option", "animation=True", "--out", str(target)])
    assert code == 0, capsys.readouterr().err
    assert animation.saved == str(target) and target.read_bytes() == b"movie"


def test_sample_out_refuses_an_interactive_figure_with_one_line(monkeypatch, tmp_path, capsys):
    from vaft.plot.renderers.interactive import Interactive

    _sample_request_rendering(monkeypatch, Interactive(_FIGURE, "axes", object(), ()))
    target = tmp_path / "live.png"
    code = plot_cli.main(["plasma_current_time", "--sample", "39915", "--option", "interactive=True", "--out", str(target)])
    err = capsys.readouterr().err
    assert code == 1 and "interactive=" in err and "Traceback" not in err and not target.exists()
