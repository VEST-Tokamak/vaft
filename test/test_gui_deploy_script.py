"""``vaft/deploy/gui/test-instance.sh`` (#1862): dry runs only, nothing installed."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "vaft" / "deploy" / "gui" / "test-instance.sh"
pytestmark = [
    pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash"),
    # The script installs a systemd unit and an nginx site on the Linux GUI host. On a
    # Windows runner `bash` is WSL's launcher (no distribution, UTF-16 output), and the
    # POSIX paths, os.geteuid and the fake systemctl/nginx sandbox have no meaning there.
    pytest.mark.skipif(sys.platform == "win32", reason="deploys a Linux systemd/nginx host; no POSIX bash on Windows"),
]
SHA = "95c10bf6c4630560a2e0774624c40d10683664d2"


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
    result = _run("install", SHA, "--dry-run", tmp_path=tmp_path)
    assert result.returncode == 0, result.stderr
    assert f"vaft@{SHA}" in result.stdout and "NOT STARTED" in result.stdout
    assert "enable --now" not in result.stdout.replace("then: systemctl enable --now vaft-gui-test", "")


def test_install_starts_once_the_account_is_filled_in(tmp_path):
    env_file = tmp_path / "missing.env"
    env_file.write_text("HS_ENDPOINT=http://127.0.0.1:5101\nHS_USERNAME=gui-test-reader\nHS_PASSWORD=filled\n")
    result = _run("install", SHA, "--dry-run", tmp_path=tmp_path)
    assert "+ systemctl enable --now vaft-gui-test" in result.stdout


def test_publishing_needs_the_approved_ranges(tmp_path):
    result = _run("nginx", "--dry-run", tmp_path=tmp_path)
    assert result.returncode != 0 and "approved internal ranges" in result.stderr


def test_without_root_and_without_dry_run_nothing_runs(tmp_path):
    import os

    if os.geteuid() == 0:
        pytest.skip("running as root")
    result = _run("install", SHA)  # the real default paths, not as root
    assert result.returncode == 1 and "run as root" in result.stderr and "+ " not in result.stdout


def test_remove_leaves_the_production_service_alone():
    # The real default paths, printed only (--dry-run).
    result = _run("remove", "--dry-run")
    assert result.returncode == 0
    for line in result.stdout.splitlines():
        if line.startswith("+ "):
            assert "vaft-gui-test" in line or "nginx" in line or "daemon-reload" in line, line
            assert "vaft-gui.service" not in line and "/etc/vaft-gui.env" not in line and "/opt/vaft-gui " not in line


# -- real (non-dry) runs in a sandbox, with fake nginx/systemctl/useradd/id ------------------
def _gnu_sed() -> bool:
    try:
        return "GNU" in subprocess.run(["sed", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return False


_SITE = """server {
    listen 80;
    location /gui { return 301 https://$host$request_uri; }
}
server {
    listen 443 ssl http2;
    location = /gui/login { proxy_pass http://127.0.0.1:5006; }
    location /gui/ { proxy_pass http://127.0.0.1:5006; }
    location / { return 404; }
}
"""


@pytest.fixture
def sandbox(tmp_path):
    """A fake root's world: the site, snippets and conf.d copies, and stub commands."""
    import os
    import tempfile

    if not _gnu_sed():
        pytest.skip("the script targets GNU sed (Linux)")
    root = Path(tempfile.mkdtemp(prefix="vaft-gui-test-", dir="/tmp"))
    for name in ("snippets", "bin"):
        (root / name).mkdir()
    (root / "site.conf").write_text(_SITE)
    log = root / "calls.log"
    stubs = {
        # nginx -t fails when the file "nginx-fails" exists, or when an include points at a missing file
        "nginx": f"""#!/bin/bash
echo "nginx $*" >> {log}
[[ -e {root}/nginx-fails ]] && exit 1
for f in $(grep -ho 'include [^;]*' {root}/site.conf | cut -d' ' -f2); do [[ -e $f ]] || exit 1; done
exit 0""",
        "systemctl": f"#!/bin/bash\necho \"systemctl $*\" >> {log}\n[[ $1 == is-active ]] && exit 3\nexit 0",
        "id": "#!/bin/bash\n[[ $1 == -u ]] && { echo 0; exit 0; }\nexit 1",
        "useradd": f"#!/bin/bash\necho \"useradd $*\" >> {log}",
        "userdel": f"#!/bin/bash\necho \"userdel $*\" >> {log}",
    }
    for name, body in stubs.items():
        stub = root / "bin" / name
        stub.write_text(body + "\n")
        stub.chmod(0o755)
    env = {
        "PATH": f"{root / 'bin'}:/usr/bin:/bin",
        "VAFT_GUI_TEST_SANDBOX": str(root),
        "VAFT_GUI_TEST_SITE": str(root / "site.conf"),
        "VAFT_GUI_TEST_SNIPPETS": str(root / "snippets"),
        "VAFT_GUI_TEST_CONFD": str(root / "confd.conf"),
        "VAFT_GUI_TEST_UNIT": str(root / "unit.service"),
        "VAFT_GUI_TEST_ENV": str(root / "test.env"),
        "VAFT_GUI_TEST_VENV": str(root / "venv"),
        "VAFT_GUI_TEST_STATE": str(root / "state"),
    }
    yield root, env, log
    shutil.rmtree(root, ignore_errors=True)
    del os


def _real(env, *args):
    return subprocess.run(["bash", str(SCRIPT), *args], capture_output=True, text=True, env=env, timeout=60)


def test_publish_then_remove_round_trips_the_site(sandbox):
    root, env, log = sandbox
    result = _real(env, "nginx", "10.0.0.0/8", "147.46.36.0/24")
    assert result.returncode == 0, result.stderr
    site = (root / "site.conf").read_text()
    lines = site.splitlines()
    include = next(i for i, line in enumerate(lines) if "vaft-gui-test-locations.conf" in line)
    assert lines[include + 1].strip() == "location /gui/ { proxy_pass http://127.0.0.1:5006; }", "before the 443 /gui/"
    assert "allow 10.0.0.0/8;" in (root / "snippets" / "vaft-gui-test-locations.conf").read_text()
    assert "systemctl reload nginx" in log.read_text()

    result = _real(env, "remove")
    assert result.returncode == 0, result.stderr
    assert (root / "site.conf").read_text() == _SITE, "the site is exactly as before"
    assert not (root / "confd.conf").exists() and not any((root / "snippets").iterdir())


def test_a_failed_nginx_check_on_publish_restores_everything(sandbox):
    root, env, log = sandbox
    (root / "nginx-fails").touch()
    result = _real(env, "nginx", "10.0.0.0/8")
    assert result.returncode == 1 and "nothing reloaded" in result.stderr
    assert (root / "site.conf").read_text() == _SITE
    assert not (root / "confd.conf").exists() and not any((root / "snippets").iterdir())
    assert "reload" not in log.read_text()


def test_remove_never_deletes_what_a_failing_site_still_needs(sandbox):
    root, env, log = sandbox
    assert _real(env, "nginx", "10.0.0.0/8").returncode == 0
    # Hand-edited spelling of the include, and nginx now failing for another reason.
    site = (root / "site.conf").read_text().replace("include ", "include   ")
    (root / "site.conf").write_text(site)
    (root / "nginx-fails").touch()
    result = _real(env, "remove")
    assert result.returncode == 1 and "nothing removed" in result.stderr
    assert (root / "site.conf").read_text() == site, "restored, include still there"
    assert (root / "snippets" / "vaft-gui-test-locations.conf").exists(), "the snippet it includes is kept"


def test_publish_refuses_a_site_it_cannot_place_the_include_in(sandbox):
    root, env, _log = sandbox
    (root / "site.conf").write_text(_SITE.replace("location /gui/ {", "location ^~ /gui/ {"))
    result = _real(env, "nginx", "10.0.0.0/8")
    assert result.returncode == 1 and "not exactly one" in result.stderr


def test_bad_inputs_are_refused_before_anything_changes(sandbox):
    root, env, log = sandbox
    assert "not an address or CIDR" in _real(env, "nginx", "10.0.0.0/8;deny").stderr
    assert "40-hex commit" in _real(env, "install", "main").stderr
    assert not log.exists() or "reload" not in log.read_text()


def test_overrides_outside_a_tmp_sandbox_are_refused(sandbox):
    _root, env, _log = sandbox
    hostile = dict(env, VAFT_GUI_TEST_VENV="/opt/vaft-gui")
    result = _real(hostile, "remove")
    assert result.returncode == 1 and "outside the sandbox" in result.stderr
    no_sandbox = {key: value for key, value in env.items() if key != "VAFT_GUI_TEST_SANDBOX"}
    assert "only honoured with --dry-run" in _real(no_sandbox, "remove").stderr
