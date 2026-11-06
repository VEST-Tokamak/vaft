"""No server address is committed: hosts, logins and ports stay in site files outside the repository.

Scans the tracked source, documentation and configuration (data products and their recorded
provenance are not scanned) for public IPv4 addresses and ``user@<ip>`` / ``ssh -p <port>
user@host`` logins. Documentation and examples use placeholders (``<host>``, ``gui.example.org``)
or the reserved documentation ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24).
"""

from __future__ import annotations

import ipaddress
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNED = {".py", ".md", ".markdown", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".conf", ".service",
           ".example", ".env", ".txt", ".rst", ".js", ".html"}
#: Data and recorded provenance, not configuration.
SKIPPED_PREFIXES = ("vaft/data/", "test/data/", "docs/assets/", "notebooks/data/")
_IPV4 = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")
#: A dotted quad is an address only in an address's context (not a version or section number).
_CONTEXT = re.compile(r"(?i)\b(host|hosts|address|endpoint|origin|server|ssh|scp|ip|url|nginx|listen)\b")
_DOCUMENTATION = [ipaddress.ip_network(n) for n in ("192.0.2.0/24", "198.51.100.0/24", "203.0.113.0/24")]


def _tracked() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    this = Path(__file__).resolve()
    return [ROOT / name for name in out.splitlines()
            if not name.startswith(SKIPPED_PREFIXES) and Path(name).suffix in SCANNED
            and (ROOT / name).resolve() != this]


def _public(text: str) -> bool:
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False  # a version number such as 1.2.300.4
    if any(address in network for network in _DOCUMENTATION):
        return False
    return address.is_global


#: Logins in documentation examples: a generic user name, or a host name that is a placeholder.
_PLACEHOLDER_USERS = {"user", "you", "username", "someone"}


def _login(line: str) -> bool:
    if re.search(r"[\w.-]+@\d{1,3}(?:\.\d{1,3}){3}\b", line):
        return True
    command = re.search(r"\b(?:ssh|scp|rsync)\b[^#\n]*", line)
    if command is None:
        return False
    if re.search(r"\s-[pP]\s*\d{2,5}\b", command.group(0)):
        return True  # a real port number on a remote-shell line
    for user, host in re.findall(r"(?<![\w<])([\w.-]+)@([\w-]+(?:\.[\w-]+)*)", command.group(0)):
        if user not in _PLACEHOLDER_USERS and not host.endswith(("example.org", "example.com")):
            return True
    return False


def _is_address(line: str, text: str) -> bool:
    quoted = re.escape(text)
    return bool(re.search(rf"(://|@){quoted}|{quoted}[:/]\d", line) or _CONTEXT.search(line))


@pytest.mark.parametrize(("line", "flagged"), [
    ("Server endpoint: http://8.8.8.8:5101", True),
    ("ssh -p 2222 someone@login.example.org", True),
    ("opencv-python>=4.9.0.80", False),
    ("Sec. 1.3.5.17 (detection of outliers)", False),
    ("HS_ENDPOINT=http://127.0.0.1:5101", False),
    ("host: 203.0.113.7", False),
    ("scp -P 2222 user@host:/srv/x .", True),
    ("ssh someone@login.example.org -p 2222", True),
    ("ssh admin@lab-server", True),
    ("ssh -L 5006:localhost:5006 user@server", False),
    ("ssh vaft-server python3 - scan", False),
    ("ssh -L 8080:localhost:5006 ...", False),
    ("write to someone@example.org", False),
])
def test_the_scan_tells_addresses_from_version_numbers(line, flagged):
    hits = [m for m in _IPV4.findall(line) if _public(m) and _is_address(line, m)]
    assert bool(hits or _login(line)) is flagged


def test_no_public_address_or_login_is_committed():
    found = []
    for path in _tracked():
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(text.splitlines(), 1):
            addresses = [match for match in _IPV4.findall(line) if _public(match) and _is_address(line, match)]
            if addresses or _login(line):
                found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:120]}")
    assert not found, "\n".join(found)
