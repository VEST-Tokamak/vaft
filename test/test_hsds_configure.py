"""``vaft hsds configure`` and the ``.hscfg`` checks never expose a secret (#969).

Everything runs against a temporary HOME; no real credential file is read.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import os
from pathlib import Path
import stat
import sys

import pytest

from vaft.cli import hsds as cli
from vaft.cli._main import main as vaft_main
from vaft.database import hscfg

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
POSIX = os.name != "nt"
SECRET = "Sentinel-Pa55word-7f3a"
API_KEY = "Sentinel-ApiKey-91c2"


def _load_checker():
    path = REPOSITORY_ROOT / "install" / "check_vaft_environment.py"
    spec = importlib.util.spec_from_file_location("_check_vaft_environment_969", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


@pytest.fixture
def home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    # The checker also looks at the checkout's own .hscfg; point it at an
    # empty directory so a developer's real file is never read.
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(checker, "REPOSITORY_ROOT", repo)
    for name in ("HS_ENDPOINT", "HS_USERNAME", "HS_PASSWORD", "HS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    return home


def _args(**overrides):
    values = {"endpoint": None, "username": None, "password_stdin": False, "config": None}
    values.update(overrides)
    return argparse.Namespace(**values)


class _Prompts:
    """Scripted answers; records which reader asked which prompt."""

    def __init__(self, plain, secret):
        self.plain, self.secret = list(plain), list(secret)
        self.plain_prompts, self.secret_prompts = [], []

    def ask(self, prompt):
        self.plain_prompts.append(prompt)
        return self.plain.pop(0)

    def ask_secret(self, prompt):
        self.secret_prompts.append(prompt)
        return self.secret.pop(0)


def _run(arguments, prompts=None, stdin=None):
    out = io.StringIO()
    kwargs = {"out": out, "stdin": stdin}
    if prompts is not None:
        kwargs.update(ask=prompts.ask, ask_secret=prompts.ask_secret)
    else:
        def refuse(prompt):
            raise AssertionError(f"unexpected prompt {prompt!r}")

        kwargs.update(ask=refuse, ask_secret=refuse)
    code = cli.configure(arguments, **kwargs)
    return code, out.getvalue()


# ---------------------------------------------------------------------------
# Interactive prompting
# ---------------------------------------------------------------------------


def test_secrets_are_read_with_getpass_and_never_echoed(home, capsys):
    prompts = _Prompts(["http://hsds.example:5101", "student"], [SECRET, API_KEY])
    code, out = _run(_args(), prompts)
    captured = capsys.readouterr()
    assert code == 0
    assert [p.split(" ")[0] for p in prompts.secret_prompts] == ["Password", "API"]
    assert all("Password" not in p and "API" not in p for p in prompts.plain_prompts)
    for text in (out, captured.out, captured.err, *prompts.plain_prompts, *prompts.secret_prompts):
        assert SECRET not in text and API_KEY not in text
    values = hscfg.read_values(home / ".hscfg")
    assert values["hs_password"] == SECRET and values["hs_api_key"] == API_KEY
    assert values["hs_endpoint"] == "http://hsds.example:5101"


def test_existing_secret_shows_configured_and_empty_input_keeps_it(home, capsys):
    config = home / ".hscfg"
    config.write_text(
        f"hs_endpoint = http://old:5101\nhs_username = student\nhs_password = {SECRET}\n",
        encoding="utf-8",
    )
    prompts = _Prompts(["", ""], ["", ""])
    code, out = _run(_args(), prompts)
    captured = capsys.readouterr()
    assert code == 0
    assert "[configured]" in prompts.secret_prompts[0]
    assert "[configured]" not in prompts.secret_prompts[1]  # no API key stored
    for text in (out, captured.out, captured.err, *prompts.plain_prompts, *prompts.secret_prompts):
        assert SECRET not in text
    assert hscfg.read_values(config)["hs_password"] == SECRET
    assert "No changes" in out


def test_default_prompt_uses_getpass(home, monkeypatch):
    """The real entry point wires the secret prompts to getpass, not input."""
    calls = []
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": calls.append(prompt) or "")
    answers = iter(["http://hsds.example:5101", "student"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert vaft_main(["hsds", "configure"]) == 0
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# The file itself
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="file modes are not enforced on Windows")
def test_written_file_is_0600_even_under_umask_022(home):
    previous = os.umask(0o022)
    try:
        code, _ = _run(_args(endpoint="http://hsds.example:5101", username="student"))
    finally:
        os.umask(previous)
    assert code == 0
    assert stat.S_IMODE((home / ".hscfg").stat().st_mode) == 0o600


@pytest.mark.skipif(not POSIX, reason="file modes are not enforced on Windows")
def test_loose_existing_file_is_replaced_by_a_0600_temporary(home, monkeypatch):
    config = home / ".hscfg"
    config.write_text("hs_endpoint = http://old:5101\n", encoding="utf-8")
    config.chmod(0o664)
    seen = []
    real_replace = os.replace

    def watching_replace(source, destination):
        seen.append(stat.S_IMODE(os.stat(source).st_mode))
        return real_replace(source, destination)

    monkeypatch.setattr(hscfg.os, "replace", watching_replace)
    previous = os.umask(0o002)
    try:
        code, _ = _run(_args(password_stdin=True), stdin=io.StringIO(SECRET + "\n"))
    finally:
        os.umask(previous)
    assert code == 0
    assert seen == [0o600], "the temporary file must be 0600 before it is renamed into place"
    assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert not list(home.glob(".hscfg.*.tmp"))


def test_other_keys_and_comments_are_preserved(home):
    config = home / ".hscfg"
    original = (
        "# my HSDS setup\n"
        "hs_endpoint = http://old:5101\n"
        "\n"
        "# bucket for the lab\n"
        "hs_bucket = vest\n"
        "hs_username = student\n"
        "track_order = True\n"
    )
    config.write_text(original, encoding="utf-8")
    code, _ = _run(_args(endpoint="http://new:5101", password_stdin=True), stdin=io.StringIO(SECRET))
    assert code == 0
    text = config.read_text(encoding="utf-8")
    assert text.splitlines()[:7] == [
        "# my HSDS setup",
        "hs_endpoint = http://new:5101",
        "",
        "# bucket for the lab",
        "hs_bucket = vest",
        "hs_username = student",
        "track_order = True",
    ]
    assert text.splitlines()[7:] == [f"hs_password = {SECRET}"]


def test_value_h5pyd_would_truncate_is_refused_without_echo(home, capsys):
    code, out = _run(_args(password_stdin=True), stdin=io.StringIO("abc=def\n"))
    captured = capsys.readouterr()
    assert code == 2
    assert not (home / ".hscfg").exists()
    assert "abc=def" not in out + captured.out + captured.err
    assert "hs_password" in captured.err


def test_password_stdin_rejects_empty_input(home, capsys):
    code, _ = _run(_args(password_stdin=True), stdin=io.StringIO("\n"))
    assert code == 2
    assert not (home / ".hscfg").exists()


def test_secrets_are_not_accepted_as_arguments(capsys):
    with pytest.raises(SystemExit):
        cli.main(["configure", "--help"])
    text = capsys.readouterr().out
    assert "--password-stdin" in text
    assert "--password " not in text and "--api-key" not in text


@pytest.mark.parametrize(
    "argv",
    [
        ["configure", "--password", SECRET],
        ["configure", f"--password={SECRET}"],
        ["configure", "--pass", SECRET],
        ["configure", "--api-key", SECRET],
        ["configure", SECRET],
    ],
)
def test_a_secret_typed_as_an_argument_is_refused_without_echo(home, capsys, argv):
    """No prefix match onto --password-stdin, and the error never quotes it."""
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert SECRET not in captured.out + captured.err
    assert not (home / ".hscfg").exists()


def test_password_stdin_from_a_terminal_uses_the_hidden_prompt(home, monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr(sys, "stdin", Terminal("typed-with-echo\n"))
    calls = []
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": calls.append(prompt) or SECRET)
    assert cli.main(["configure", "--password-stdin"]) == 0
    assert calls, "a terminal stdin must be read through getpass"
    assert hscfg.read_values(home / ".hscfg")["hs_password"] == SECRET


def test_empty_endpoint_flag_is_refused(home):
    code, _ = _run(_args(endpoint="  "))
    assert code == 2 and not (home / ".hscfg").exists()


def test_invalid_endpoint_is_asked_again_before_the_secrets(home, capsys):
    prompts = _Prompts(["hsds.example", "http://hsds.example:5101", "student"], [SECRET, ""])
    code, _ = _run(_args(), prompts)
    assert code == 0
    assert len(prompts.plain_prompts) == 3
    assert hscfg.read_values(home / ".hscfg")["hs_endpoint"] == "http://hsds.example:5101"


def test_end_of_input_aborts_without_writing(home):
    def eof(prompt):
        raise EOFError

    code = cli.configure(_args(), ask=eof, ask_secret=eof, out=io.StringIO())
    assert code == 1 and not (home / ".hscfg").exists()


def test_non_ascii_value_is_refused(home, capsys):
    code, _ = _run(_args(password_stdin=True), stdin=io.StringIO("pässwort\n"))
    assert code == 2 and not (home / ".hscfg").exists()
    assert "pässwort" not in capsys.readouterr().err


def test_undecodable_file_is_reported_without_its_bytes(home, capsys):
    (home / ".hscfg").write_bytes(b"hs_password = caf\xe9-secret\n")
    code, _ = _run(_args(endpoint="http://x:5101"))
    err = capsys.readouterr().err
    assert code == 2 and "secret" not in err and "position" not in err


@pytest.mark.skipif(not POSIX, reason="symlinks need privileges on Windows")
def test_symlinked_file_is_updated_through_the_link(home):
    target = home / "dotfiles" / "hscfg"
    target.parent.mkdir()
    target.write_text("hs_endpoint = http://old:5101\n", encoding="utf-8")
    (home / ".hscfg").symlink_to(target)
    code, _ = _run(_args(password_stdin=True), stdin=io.StringIO(SECRET))
    assert code == 0
    assert (home / ".hscfg").is_symlink()
    assert hscfg.read_values(target)["hs_password"] == SECRET
    assert stat.S_IMODE(target.stat().st_mode) == 0o600


def test_h5pyd_parses_the_written_file(home):
    """What we write must read back identically through h5pyd's own parser."""
    h5pyd_config = pytest.importorskip("h5pyd.config")
    code, _ = _run(
        _args(endpoint="http://hsds.example:5101", username="student", password_stdin=True),
        stdin=io.StringIO(SECRET),
    )
    assert code == 0
    saved = dict(h5pyd_config.Config._cfg)
    h5pyd_config.Config._cfg.clear()
    try:
        parsed = h5pyd_config.Config(config_file=str(home / ".hscfg"))
        assert parsed["hs_endpoint"] == "http://hsds.example:5101"
        assert parsed["hs_username"] == "student"
        assert parsed["hs_password"] == SECRET
    finally:
        h5pyd_config.Config._cfg.clear()
        h5pyd_config.Config._cfg.update(saved)


# ---------------------------------------------------------------------------
# Environment check
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not POSIX, reason="file modes are not enforced on Windows")
def test_permission_check_warns_about_group_readable_file(home):
    config = home / ".hscfg"
    config.write_text(f"hs_endpoint = http://x:5101\nhs_password = {SECRET}\n", encoding="utf-8")
    config.chmod(0o664)
    result = checker.check_hscfg_permissions([config])
    assert result.status == checker.WARN
    assert f"chmod 600 {config}" in result.remediation
    assert "0664" in result.detail
    assert SECRET not in checker.format_report([result])


@pytest.mark.skipif(not POSIX, reason="file modes are not enforced on Windows")
def test_permission_check_finds_home_and_project_local_files(home):
    for config in (home / ".hscfg", Path.cwd() / ".hscfg"):
        config.write_text("hs_endpoint = http://x:5101\n", encoding="utf-8")
        config.chmod(0o600)
    assert checker.check_hscfg_permissions().status == checker.PASS
    (Path.cwd() / ".hscfg").chmod(0o644)
    result = checker.check_hscfg_permissions()
    assert result.status == checker.WARN
    assert str(Path.cwd() / ".hscfg") in result.detail
    assert str(home / ".hscfg") not in result.detail


def test_permission_check_is_skipped_on_windows(home):
    (home / ".hscfg").write_text("hs_endpoint = http://x:5101\n", encoding="utf-8")
    assert checker.check_hscfg_permissions(platform="nt").status == checker.SKIP


def test_environment_variables_satisfy_the_configuration_check(home):
    result = checker.check_hsds_configuration(
        required=True, environ={"HS_ENDPOINT": "http://x:5101", "HS_PASSWORD": SECRET}
    )
    assert result.status == checker.PASS
    assert "HS_ENDPOINT" in result.detail and "HS_PASSWORD" in result.detail
    assert SECRET not in checker.format_report([result])


def test_project_local_file_is_the_one_h5pyd_reads(home):
    (home / ".hscfg").write_text("hs_endpoint = http://home:5101\n", encoding="utf-8")
    local = Path.cwd() / ".hscfg"
    local.write_text("hs_username = x\n", encoding="utf-8")
    result = checker.check_hsds_configuration(environ={})
    assert str(local) in result.detail and result.status == checker.WARN


def test_probe_errors_are_scrubbed_of_credentials(home):
    (home / ".hscfg").write_text(
        f"hs_endpoint = http://x:5101\nhs_password = {SECRET}\nhs_api_key = {API_KEY}\n",
        encoding="utf-8",
    )

    def explode():
        raise OSError(
            f"HTTPConnectionPool: http://student:{SECRET}@x:5101/ failed; key={API_KEY}"
        )

    result = checker.check_hsds_connection(probe=explode)
    rendered = checker.format_report([result])
    assert result.failed
    assert SECRET not in rendered and API_KEY not in rendered
    assert "student" not in rendered  # URL user-info is dropped whole
    assert "http://***@x:5101/" in rendered


def test_probe_scrubs_encoded_forms_and_at_signs():
    secret = "p@ss/word 1"
    scrub = checker._secret_values(environ={"HS_USERNAME": "u", "HS_PASSWORD": secret}, paths=[])
    import base64
    from urllib.parse import quote

    text = (
        f"http://u:{secret}@host:5101/ q={quote(secret, safe='')} "
        f"r={secret!r} auth=Basic {base64.b64encode(f'u:{secret}'.encode()).decode()}"
    )
    rendered = checker.redact(text, scrub)
    for form in (secret, quote(secret, safe=""), "word 1", base64.b64encode(f"u:{secret}".encode()).decode()):
        assert form not in rendered
    assert "http://***@host:5101/" in rendered


def test_probe_scrubs_environment_secrets(home, monkeypatch):
    monkeypatch.setenv("HS_PASSWORD", SECRET)

    def explode():
        raise RuntimeError(f"auth failed for password {SECRET}")

    rendered = checker.format_report([checker.check_hsds_connection(probe=explode)])
    assert SECRET not in rendered
