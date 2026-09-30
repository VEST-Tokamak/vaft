"""``vaft.help()`` and ``vaft help``: lazy, read-only, secret-free (#1203)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vaft
from vaft._help import _providers, topics
from vaft._help import help as vaft_help
from vaft._help._model import HelpPage
from vaft.cli._main import main as cli_main

SUBSYSTEMS = (
    "vaft.plot",
    "vaft.omas",
    "vaft.imas",
    "vaft.formula",
    "vaft.process",
    "vaft.database",
    "vaft.code",
    "vaft.data",
    "vaft.validation",
    "matplotlib",
    "omas",
)
HS_VARIABLES = ("HS_ENDPOINT", "HS_USERNAME", "HS_PASSWORD", "HS_API_KEY")
CODE_HOMES = tuple(variable for _name, variable, _layout in _providers.CODES)
#: Legacy variables some adapters also read; cleared so the probe is isolated.
CODE_LEGACY = ("EFIT", "CHEASE", "CHEASE_EXEC_DIR", "RTES")


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600)


def _loaded(prefixes) -> str:
    return (
        "sorted(m for m in sys.modules if any(m == p or m.startswith(p + '.') for p in "
        f"{tuple(prefixes)!r}))"
    )


# -- lazy imports ------------------------------------------------------------
def test_import_vaft_and_the_help_attribute_import_no_subsystem():
    _run(
        "import sys, vaft; vaft.help; vaft.help(); "
        f"loaded = {_loaded(SUBSYSTEMS)}; assert not loaded, loaded"
    )


def test_a_topic_imports_only_its_own_subsystem():
    _run(
        "import sys, vaft; vaft.help('validation'); "
        f"loaded = {_loaded(('vaft.plot', 'vaft.omas', 'matplotlib', 'vaft.formula'))}; "
        "assert not loaded, loaded; "
        "vaft.help('plot'); assert 'vaft.plot' in sys.modules"
    )


# A cold process -- fresh HOME, cache and temp directories, nothing imported
# beforehand -- is where an import-time side effect would show; these topics
# must not import the libraries that have them.
_COLD = """
import os, sys, pathlib
root = pathlib.Path(sys.argv[1])
def files():
    return sorted(str(p) for p in root.rglob('*'))
import vaft
before_env, before_files = dict(os.environ), files()
for name, probe in (('database', False), ('code', True), ('validation', False), ('data', False), ('cli', False)):
    page = vaft.help(name, probe=probe)
    assert not page.warnings, (name, page.warnings)
    str(page), page._repr_markdown_(), page.as_dict()
heavy = sorted(m for m in ('omas', 'matplotlib', 'vaft.plot', 'vaft.omas', 'vaft.imas', 'sklearn', 'h5pyd')
               if m in sys.modules)
assert not heavy, heavy
changed = {k for k in set(os.environ) | set(before_env) if os.environ.get(k) != before_env.get(k)}
assert not changed, changed
assert files() == before_files, sorted(set(files()) - set(before_files))
"""


def test_light_topics_have_no_import_side_effects_in_a_cold_process(tmp_path):
    root = tmp_path / "cold"
    env = dict(os.environ)
    for name in ("home", "cache", "tmp", "config"):
        (root / name).mkdir(parents=True)
    env.update(
        HOME=str(root / "home"),
        USERPROFILE=str(root / "home"),
        XDG_CACHE_HOME=str(root / "cache"),
        XDG_CONFIG_HOME=str(root / "config"),
        MPLCONFIGDIR=str(root / "config"),
        TMPDIR=str(root / "tmp"),
        TEMP=str(root / "tmp"),
        TMP=str(root / "tmp"),
    )
    for variable in CODE_HOMES + CODE_LEGACY + HS_VARIABLES:
        env.pop(variable, None)
    subprocess.run([sys.executable, "-c", _COLD, str(root)], check=True, timeout=600, env=env)


def test_the_help_command_imports_no_subsystem_before_parsing():
    _run(
        "import sys, vaft.cli.help; "
        f"loaded = {_loaded(SUBSYSTEMS)}; assert not loaded, loaded"
    )


def test_help_stays_out_of_star_imports_but_is_listed():
    assert "help" not in vaft.__all__
    assert "help" in dir(vaft)
    assert vaft.help is vaft_help


# -- read only ---------------------------------------------------------------
def test_help_plot_does_not_make_matplotlib_choose_a_backend():
    # A fresh process: the backend is still Matplotlib's automatic sentinel,
    # which reading ``matplotlib.get_backend()`` would resolve.
    _run(
        "import os, matplotlib, vaft; r = matplotlib.rcParams; "
        "snap = lambda: ({k: repr(v) for k, v in dict.items(r)}, r._get_backend_or_none(), "
        "os.environ.get('MPLBACKEND'), os.environ.get('MPLCONFIGDIR')); "
        "import vaft.plot; before = snap(); assert before[1] is None, before[1]; "
        "page = vaft.help('plot'); str(page); page._repr_markdown_(); "
        "assert snap() == before; assert not page.warnings, page.warnings"
    )


def _state():
    import matplotlib

    params = matplotlib.rcParams
    return (
        {key: repr(value) for key, value in dict.items(params)},
        params._get_backend_or_none(),
        dict(os.environ),
    )


@pytest.mark.parametrize("probe", [False, True])
def test_help_changes_no_runtime_state(probe, monkeypatch):
    for variable in CODE_HOMES + CODE_LEGACY:
        monkeypatch.delenv(variable, raising=False)
    # Importing a subsystem can carry third-party import side effects (an
    # environment variable set by a library at import); help's own contract
    # is that describing an imported subsystem writes nothing.
    for name in topics():
        vaft_help(name, probe=probe)
    for name in topics():
        before = _state()
        page = vaft_help(name, probe=probe)
        str(page), page._repr_markdown_(), json.dumps(page.as_dict())
        assert _state() == before, name


# -- secrets -----------------------------------------------------------------
SECRETS = ("SEKRIT-PW-123", "SEKRIT-KEY-456", "SEKRIT-ENV-789")


@pytest.fixture
def hscfg_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    for name in HS_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    return home


def test_database_help_reports_configuration_without_secrets(hscfg_home, monkeypatch, capsys):
    (hscfg_home / ".hscfg").write_text(
        "hs_endpoint = https://hsds.example.org\n"
        "hs_username = student\n"
        f"hs_password = {SECRETS[0]}\n"
        f"hs_api_key = {SECRETS[1]}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HS_PASSWORD", SECRETS[2])
    monkeypatch.setenv("HS_API_KEY", SECRETS[2])

    page = vaft_help("database")
    status = dict(next(s for s in page.sections if s.title == "HSDS configuration").rows)
    assert status["endpoint"].startswith("set")
    assert status["username"].startswith("set")
    outputs = [str(page), repr(page), page._repr_markdown_(), json.dumps(page.as_dict())]
    for fmt in ("text", "markdown", "json"):
        assert cli_main(["help", "database", "--format", fmt]) == 0
        captured = capsys.readouterr()
        outputs += [captured.out, captured.err]
    for text in outputs:
        for forbidden in (*SECRETS, "HS_PASSWORD", "HS_API_KEY", "hs_password", "hs_api_key"):
            assert forbidden not in text


def test_database_help_without_configuration_points_to_hsds_configure(hscfg_home):
    page = vaft_help("database")
    section = next(s for s in page.sections if s.title == "HSDS configuration")
    assert dict(section.rows) == {"endpoint": "not set", "username": "not set"}
    assert "vaft hsds configure" in section.note
    assert any(d.name == "source" and d.value == "main" for d in page.defaults)


# -- every topic, offline, no solvers ----------------------------------------
@pytest.mark.parametrize("name", topics())
@pytest.mark.parametrize("probe", [False, True])
def test_every_topic_renders_offline_without_solvers(name, probe, hscfg_home, monkeypatch):
    for variable in CODE_HOMES + CODE_LEGACY:
        monkeypatch.delenv(variable, raising=False)
    page = vaft_help(name, probe=probe)
    assert isinstance(page, HelpPage)
    assert page.topic == name and page.summary
    assert str(page) and page._repr_markdown_()
    json.dumps(page.as_dict())
    assert not page.warnings, page.warnings
    if name == "code" and probe:
        rows = dict(page.sections[0].rows)
        assert all(
            value.startswith("unavailable") for value in rows.values()
        ), rows


def test_code_layouts_match_the_adapters():
    from vaft.code.chease import CHEASE_HOME_EXECUTABLE
    from vaft.code.efit.magnetic import EFIT_HOME_EXECUTABLE
    from vaft.code.gacode._runtime import launcher_relative_path
    from vaft.code.genray.config import GENRAY_HOME_EXECUTABLE
    from vaft.code.nubeam.config import NUBEAM_HOME_EXECUTABLE
    from vaft.code.tes.runner import TES_HOME_EXECUTABLE

    layouts = {name: Path(layout) for name, _variable, layout in _providers.CODES}
    assert layouts["EFIT"] == EFIT_HOME_EXECUTABLE
    assert layouts["CHEASE"] == CHEASE_HOME_EXECUTABLE
    assert layouts["GACODE"] == launcher_relative_path("neo")
    assert layouts["GENRAY"] == GENRAY_HOME_EXECUTABLE
    assert layouts["NUBEAM"] == NUBEAM_HOME_EXECUTABLE
    assert layouts["TES"] == TES_HOME_EXECUTABLE
    assert layouts["GPEC"] == Path("bin") / "dcon"
    assert layouts["FLARE"] == Path("bin") / "flare"


def test_code_probe_finds_an_installed_executable(tmp_path, monkeypatch):
    home = tmp_path / "efit"
    executable = home / "bin" / ("efit.exe" if os.name == "nt" else "efit")
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"MZ" if os.name == "nt" else b"#!/bin/sh\n")
    executable.chmod(0o755)
    monkeypatch.setenv("EFITHOME", str(home))
    monkeypatch.setenv("CHEASEHOME", str(tmp_path / "missing"))
    rows = dict(vaft_help("code", probe=True).sections[0].rows)
    assert rows["EFIT"].startswith("available")
    assert rows["CHEASE"].startswith("broken install")


def test_markdown_keeps_placeholders_visible():
    markdown = vaft_help("cli")._repr_markdown_()
    assert "vaft &lt;command&gt; --help" in markdown
    overview = vaft_help()._repr_markdown_()
    assert "`vaft help <topic>`" in overview  # untouched inside a code span


def test_a_failing_provider_becomes_a_warning(monkeypatch):
    def broken(topic, *, probe=False):
        raise ImportError("no module named 'omas'")

    monkeypatch.setattr(_providers, "omas", broken)
    page = vaft_help("omas")
    assert page.entry_points and page.warnings
    assert "ImportError" in page.warnings[0]
    assert "Warnings" in str(page)


def test_defaults_are_kept_in_their_four_kinds():
    kinds = {d.kind for d in vaft_help("database").defaults}
    assert {"data-access", "scientific"} <= kinds
    plot = vaft_help("plot")
    assert {"runtime", "presentation"} <= {d.kind for d in plot.defaults}
    assert "Scientific choices" in str(vaft_help("database"))


# -- items -------------------------------------------------------------------
def test_items_are_answered_by_the_subsystem_catalogs():
    from vaft.formula import catalog as formulas
    from vaft.validation import registry

    formula = formulas.list_formulas("equilibrium")[0]
    assert vaft_help("formula", formula.name) == formulas.show(formula.name)
    key = next(iter(registry.CHECKS))
    assert vaft_help("validation", key) is registry.describe(key)
    assert vaft_help("database", "main").name == "main"
    assert vaft_help("data", "39915")["shot"] == 39915
    assert "vaft plot --help" in vaft_help("cli", "plot")


def test_unknown_topics_and_items_are_errors():
    with pytest.raises(KeyError, match="choose from"):
        vaft_help("nope")
    with pytest.raises(KeyError):
        vaft_help("validation", "no.such.check")
    with pytest.raises(ValueError, match="no per-item view"):
        vaft_help("code", "efit")


# -- CLI ---------------------------------------------------------------------
def test_help_command(capsys):
    assert cli_main(["help"]) == 0
    assert "Topics" in capsys.readouterr().out
    assert cli_main(["help", "cli", "--format", "json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["topic"] == "cli"
    assert any(row[0] == "vaft help" for row in data["sections"][0]["rows"])
    assert cli_main(["help", "validation", "no.such.check"]) == 2
    assert "no.such.check" in capsys.readouterr().err
    assert cli_main(["help", "Validation"]) == 0
    capsys.readouterr()
    assert cli_main(["help", "data", "abc"]) == 2
    assert "must be an integer" in capsys.readouterr().err
    assert cli_main(["help", "database", "main", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["name"] == "main"


def test_an_item_whose_subsystem_is_missing_exits_cleanly(monkeypatch, capsys):
    def missing(query):
        raise ModuleNotFoundError("No module named 'omas'")

    monkeypatch.setattr(_providers, "omas_item", missing)
    assert cli_main(["help", "omas", "ip"]) == 1
    assert "unavailable here" in capsys.readouterr().err


def test_argparse_help_is_unchanged_and_lists_help(capsys):
    assert cli_main(["--help"]) == 0
    out = capsys.readouterr().out
    assert "usage:" in out and "help" in out
    with pytest.raises(SystemExit) as raised:
        cli_main(["help", "--help"])
    assert raised.value.code == 0
    assert "topics: overview" in capsys.readouterr().out
