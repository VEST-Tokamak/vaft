"""``vaft.setup()`` and ``vaft setup``: explicit, non-scientific, idempotent (#1203)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import vaft
from vaft import _setup
from vaft._setup import VAFTSetup, setup
from vaft.cli._main import main as cli_main
from vaft.plot import environment as plot_environment
from vaft.plot.environment import Environment

INLINE = "module://matplotlib_inline.backend_inline"
IPYMPL = "module://ipympl.backend_nbagg"


def _run(code: str, env: dict | None = None) -> None:
    subprocess.run([sys.executable, "-c", code], check=True, timeout=600, env=env)


def _clean_env(**extra) -> dict:
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND" and not k.startswith("VSCODE_")}
    env.update(extra)
    return env


# -- import contract -----------------------------------------------------------
def test_import_never_runs_setup_and_the_attribute_is_lazy():
    _run(
        "import sys, vaft; vaft.setup; "
        "loaded = sorted(m for m in sys.modules if m.startswith(('vaft.plot', 'matplotlib'))); "
        "assert not loaded, loaded",
        env=_clean_env(),
    )
    assert "setup" not in vaft.__all__ and "setup" in dir(vaft)
    assert vaft.setup is setup


def test_the_database_profile_imports_no_plotting_stack():
    _run(
        "import sys, vaft; vaft.setup('database'); "
        "loaded = sorted(m for m in sys.modules if m.startswith(('vaft.plot', 'matplotlib', 'omas', 'h5pyd'))); "
        "assert not loaded, loaded",
        env=_clean_env(),
    )


# -- outside a notebook: nothing changes -----------------------------------------
# ``backend`` and ``backend_fallback`` are what selecting a backend means
# (``matplotlib.use`` turns the fallback off); every other rcParams entry --
# the presentation and anything a user set -- must survive setup untouched.
_SNAPSHOT = (
    "import os, matplotlib, vaft; r = matplotlib.rcParams; "
    "snap = lambda: ({k: repr(v) for k, v in dict.items(r) if k not in ('backend', 'backend_fallback')}, "
    "r._get_backend_or_none(), dict(os.environ)); "
)


@pytest.mark.parametrize("profile", ["auto", "notebook"])
def test_outside_a_kernel_the_backend_is_left_alone(profile):
    _run(
        _SNAPSHOT + "before = snap(); "
        f"result = vaft.setup({profile!r}); "
        "assert snap() == before, 'state changed'; "
        "assert before[1] is None and not result.changed and result.environment == 'terminal', result; "
        f"assert vaft.setup({profile!r}) == result",
        env=_clean_env(),
    )


def test_batch_selects_agg_only_when_nothing_was_chosen():
    _run(
        _SNAPSHOT + "before = snap(); result = vaft.setup('batch'); after = snap(); "
        "assert result.changed and result.backend == 'agg', result; "
        "assert after[0] == before[0], 'rcParams other than backend changed'; "
        "assert after[2] == before[2], 'the environment changed'; "
        "again = vaft.setup('batch'); assert not again.changed and again.backend == 'agg', again",
        env=_clean_env(),
    )


def test_batch_leaves_the_callers_environment_alone():
    """Cold review 0.8.0 delta-absorb-2 F6: no ``MPLCONFIGDIR`` exported behind the caller's back.

    It was inert for the calling process but made every child process rebuild
    the font cache in a fresh temporary directory, unreported.
    """
    _run(
        "import os, vaft; "
        "assert 'MPLCONFIGDIR' not in os.environ; "
        "result = vaft.setup('batch'); "
        "assert result.changed and result.backend == 'agg', result; "
        "assert 'MPLCONFIGDIR' not in os.environ, os.environ['MPLCONFIGDIR']; "
        "assert result.details == (), result",
        env={k: v for k, v in _clean_env().items() if k != "MPLCONFIGDIR"},
    )


def test_batch_reports_the_config_dir_it_exports_when_the_default_is_unwritable(monkeypatch, tmp_path):
    import matplotlib
    import tempfile

    fallback = os.path.join(tempfile.gettempdir(), "matplotlib-vaft-test")
    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.setattr(_setup, "_kernel_kind", lambda: "terminal")
    monkeypatch.setattr(_setup, "_chosen_backend", lambda: None)
    monkeypatch.setattr(_setup, "_use_agg", lambda: None)
    monkeypatch.setattr(matplotlib, "get_configdir", lambda: fallback)
    result = setup("batch")
    assert os.environ["MPLCONFIGDIR"] == fallback
    assert dict(result.details)["MPLCONFIGDIR"] == fallback
    assert any("not writable" in reason for reason in result.reasons)
    assert "MPLCONFIGDIR" in str(result)
    # A writable default is never overridden: nothing is exported or reported.
    monkeypatch.delenv("MPLCONFIGDIR")
    monkeypatch.setattr(matplotlib, "get_configdir", lambda: str(tmp_path / ".matplotlib"))
    result = setup("batch")
    assert "MPLCONFIGDIR" not in os.environ and result.details == ()


def test_batch_and_notebook_keep_an_explicit_mplbackend():
    _run(
        _SNAPSHOT + "before = snap(); "
        "results = [vaft.setup(p) for p in ('batch', 'notebook', 'auto')]; "
        "assert not any(r.changed for r in results), results; "
        "assert snap() == before",
        env=_clean_env(MPLBACKEND="svg"),
    )


# -- in a notebook kernel (environment detection monkeypatched) -----------------
class FakeMatplotlib:
    """Stands in for the process backend so no real switch happens."""

    def __init__(self, monkeypatch, chosen, kind="jupyter", ipympl=True):
        self.chosen = chosen
        self.kind = kind
        self.switches = 0
        monkeypatch.setattr(_setup, "_kernel_kind", lambda: kind)
        monkeypatch.setattr(_setup, "_chosen_backend", lambda: self.chosen)
        monkeypatch.setattr(_setup, "_has_ipympl", lambda: ipympl)
        monkeypatch.setattr(_setup, "_switch_to_ipympl", self.switch)
        monkeypatch.setattr(_setup, "_use_agg", lambda: pytest.fail("batch must not run here"))
        monkeypatch.setattr(plot_environment, "detect_environment", self.detect)

    def switch(self):
        self.switches += 1
        self.chosen = IPYMPL

    def detect(self):
        backend = self.chosen or INLINE
        self.chosen = backend  # detection resolves the backend, as Matplotlib does
        live = backend.startswith("module://ipympl")
        return Environment(kind=self.kind, backend=backend, live_figures=live, widgets=True)


@pytest.fixture
def kernel_env(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", INLINE)  # what ipykernel exports
    return monkeypatch


@pytest.mark.parametrize("profile", ["auto", "notebook"])
def test_a_kernel_with_ipympl_gets_live_figures_once(profile, kernel_env):
    fake = FakeMatplotlib(kernel_env, chosen=None)
    result = setup(profile)
    assert result.changed and result.live_figures and result.backend == IPYMPL
    assert result.previous_backend == INLINE and fake.switches == 1
    assert any("ipympl is installed" in reason for reason in result.reasons)
    again = setup(profile)
    assert not again.changed and again.live_figures and fake.switches == 1


@pytest.mark.parametrize("name", ["widget", "ipympl", IPYMPL])
def test_ipympl_is_live_under_every_name_matplotlib_reports(name, kernel_env):
    # Matplotlib 3.9+ reports ``%matplotlib widget`` as "widget", which
    # vaft.plot.environment does not list as live.
    fake = FakeMatplotlib(kernel_env, chosen=None)

    def switch():
        fake.switches += 1
        fake.chosen = name

    kernel_env.setattr(_setup, "_switch_to_ipympl", switch)
    kernel_env.setattr(
        plot_environment,
        "detect_environment",
        lambda: Environment("jupyter", fake.chosen or INLINE, live_figures=False, widgets=True),
    )
    result = setup()
    assert result.changed and result.live_figures and result.backend == name
    again = setup()
    assert not again.changed and again.live_figures and fake.switches == 1
    assert "live" in again.reasons[0]


def test_a_kernel_without_ipympl_stays_inline_with_a_hint(kernel_env):
    fake = FakeMatplotlib(kernel_env, chosen=INLINE, ipympl=False)
    result = setup()
    assert not result.changed and result.live_figures is False and fake.switches == 0
    assert "pip install ipympl" in result.warnings[0]


def test_a_kernel_keeps_agg_and_an_explicit_backend(kernel_env):
    fake = FakeMatplotlib(kernel_env, chosen="agg")
    assert not setup().changed and fake.switches == 0
    fake.chosen = "qtagg"
    assert not setup("notebook").changed and fake.switches == 0
    fake.chosen = None
    kernel_env.setenv("MPLBACKEND", "Agg")
    result = setup("notebook")
    assert not result.changed and "MPLBACKEND=Agg" in result.reasons[0] and fake.switches == 0


def test_vscode_counts_as_a_kernel(kernel_env):
    fake = FakeMatplotlib(kernel_env, chosen=None, kind="vscode")
    assert setup().changed and fake.switches == 1


def test_an_ipython_terminal_is_not_a_notebook(kernel_env):
    kernel_env.delenv("MPLBACKEND")
    fake = FakeMatplotlib(kernel_env, chosen="tkagg", kind="ipython")
    result = setup()
    assert not result.changed and fake.switches == 0 and result.backend == "tkagg"


def test_a_broken_ipympl_install_falls_back_with_a_warning(kernel_env):
    fake = FakeMatplotlib(kernel_env, chosen=None)

    def broken():
        raise RuntimeError("'widget' is not a recognised GUI loop or backend name")

    kernel_env.setattr(_setup, "_switch_to_ipympl", broken)
    result = setup()
    assert not result.changed and result.live_figures is False and result.backend == INLINE
    assert "could not be enabled" in result.warnings[0] and fake.switches == 0


def test_batch_in_a_kernel_keeps_the_inline_default(kernel_env):
    FakeMatplotlib(kernel_env, chosen=INLINE)
    kernel_env.delenv("MPLBACKEND")
    result = setup("batch")
    assert not result.changed and "inline default" in result.reasons[0]


# -- a real kernel ---------------------------------------------------------------
_KERNEL_CODE = """
import json, os, vaft
from vaft.plot.environment import default_interaction_backend, detect_environment
first = vaft.setup()
second = vaft.setup()
env = detect_environment()
print("RESULT" + json.dumps({"first": first.as_dict(), "second": second.as_dict(),
       "live": env.live_figures, "interaction": default_interaction_backend(env)}))
"""


def _kernel_run(env_update: dict) -> dict:
    manager_module = pytest.importorskip("jupyter_client.manager")
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND" and not k.startswith("VSCODE_")}
    env.update(env_update)
    try:
        km, kc = manager_module.start_new_kernel(kernel_name="python3", env=env, startup_timeout=120)
    except Exception as error:  # noqa: BLE001 - no kernelspec in this environment
        pytest.skip(f"no Jupyter kernel available: {error}")
    output: list[str] = []

    def hook(message):
        if message["msg_type"] == "stream":
            output.append(message["content"]["text"])
        elif message["msg_type"] == "error":
            output.append("\n".join(message["content"]["traceback"]))

    try:
        kc.execute_interactive(_KERNEL_CODE, output_hook=hook, timeout=300)
    finally:
        kc.stop_channels()
        km.shutdown_kernel(now=True)
    text = "".join(output)
    assert "RESULT" in text, text
    return json.loads(text.split("RESULT", 1)[1])


def test_in_a_real_kernel_setup_gives_what_vaft_plots_see_as_live():
    import importlib.util

    data = _kernel_run({})
    first, second = data["first"], data["second"]
    assert first["environment"] == "jupyter" and not second["changed"]
    if importlib.util.find_spec("ipympl") is None:
        assert not first["changed"] and "pip install ipympl" in first["warnings"][0]
    else:
        # VAFT's own plot layer must agree that the canvas is live.
        assert first["changed"] and first["live_figures"]
        assert data["live"] and data["interaction"] == "matplotlib"


def test_in_a_real_kernel_an_exported_agg_is_kept():
    data = _kernel_run({"MPLBACKEND": "Agg"})
    assert not data["first"]["changed"] and not data["live"]
    assert "MPLBACKEND=Agg" in data["first"]["reasons"][0]


# -- database: diagnosis only ----------------------------------------------------
SECRETS = ("SEKRIT-PW-123", "SEKRIT-KEY-456")


def test_database_setup_diagnoses_without_writing_prompting_or_leaking(tmp_path, monkeypatch, capsys):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(work)
    for name in ("HS_ENDPOINT", "HS_USERNAME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HS_PASSWORD", SECRETS[0])
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("setup must not prompt"))
    monkeypatch.setattr("getpass.getpass", lambda *a: pytest.fail("setup must not prompt"))

    result = setup("database")
    assert dict(result.details)["hsds endpoint"] == "not set"
    assert any("vaft hsds configure" in reason for reason in result.reasons)
    assert list(home.iterdir()) == [] and list(work.iterdir()) == []

    (home / ".hscfg").write_text(
        f"hs_endpoint = https://hsds.example.org\nhs_username = student\nhs_password = {SECRETS[0]}\n"
        f"hs_api_key = {SECRETS[1]}\n",
        encoding="utf-8",
    )
    result = setup("database")
    assert dict(result.details)["hsds endpoint"].startswith("set")
    outputs = [str(result), result._repr_markdown_(), json.dumps(result.as_dict())]
    for fmt in ("text", "markdown", "json"):
        assert cli_main(["setup", "database", "--format", fmt]) == 0
        captured = capsys.readouterr()
        outputs += [captured.out, captured.err]
    for text in outputs:
        for forbidden in (*SECRETS, "HS_PASSWORD", "HS_API_KEY", "hs_password", "hs_api_key"):
            assert forbidden not in text


# -- result and CLI ----------------------------------------------------------------
def test_the_result_is_data_and_renders():
    result = VAFTSetup("notebook", "jupyter", backend=IPYMPL, previous_backend=INLINE,
                       live_figures=True, changed=True, reasons=("kernel detected",))
    text = str(result)
    assert f"{INLINE} -> {IPYMPL}" in text and "because: kernel detected" in text
    assert "live figures" in result._repr_markdown_()
    assert json.loads(json.dumps(result.as_dict()))["live_figures"] is True


def test_unknown_profile_is_an_error():
    with pytest.raises(ValueError, match="choose from"):
        setup("publication")


def test_setup_command(capsys, monkeypatch):
    monkeypatch.setattr(_setup, "_kernel_kind", lambda: "terminal")
    monkeypatch.setattr(_setup, "_chosen_backend", lambda: None)
    assert cli_main(["setup", "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["profile"] == "auto"
    with pytest.raises(SystemExit) as raised:
        cli_main(["setup", "publication"])
    assert raised.value.code == 2
