"""Configure HSDS (h5pyd) credentials without echoing secrets.

``vaft hsds configure`` replaces the upstream ``hsconfigure`` prompt, which reads
the password with plain ``input()`` and shows an existing password as the
prompt default. Here the endpoint and username are ordinary prompts; the
password and API key are read with :func:`getpass.getpass`, an existing secret
is shown only as ``[configured]``, and an empty answer keeps what is there.

The result is an ordinary h5pyd ``.hscfg`` (other keys and comments preserved),
written atomically with mode ``0600`` on POSIX. On Windows file modes are not
enforced; the file inherits the ACL of the user profile.

Non-interactive use (CI, HPC, containers): pass ``--endpoint``/``--username``
and ``--password-stdin``, or skip the file entirely and export ``HS_ENDPOINT``,
``HS_USERNAME``, ``HS_PASSWORD`` (or ``HS_API_KEY``), which h5pyd reads
directly. Secrets are never accepted as command-line arguments, where they would
land in shell history and the process table.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable
import getpass
from pathlib import Path
import sys

from vaft.database import hscfg


def _prompt_plain(label: str, current: str, reader: Callable[[str], str]) -> str | None:
    """Endpoint/username: the current value is not secret and is the default."""
    suffix = f" [{current}]" if current else ""
    answer = reader(f"{label}{suffix}: ").strip()
    return answer or None


def _prompt_secret(label: str, configured: bool, reader: Callable[[str], str]) -> str | None:
    """Password/API key: hidden input, and an existing value is never shown."""
    suffix = " [configured]" if configured else ""
    answer = reader(f"{label}{suffix} (input hidden, Enter keeps it): ").strip()
    return answer or None


def configure(
    arguments: argparse.Namespace,
    *,
    ask: Callable[[str], str] | None = None,
    ask_secret: Callable[[str], str] | None = None,
    stdin: Iterable[str] | None = None,
    out=None,
) -> int:
    # Resolved per call, not bound at import, so the secret prompt is always
    # whatever getpass.getpass is now (and tests can see that it is used).
    ask = ask if ask is not None else input
    ask_secret = ask_secret if ask_secret is not None else getpass.getpass
    out = out if out is not None else sys.stdout
    path = Path(arguments.config).expanduser() if arguments.config else hscfg.default_path()
    current = hscfg.read_values(path)
    updates: dict[str, str] = {}

    interactive = not (
        arguments.endpoint is not None
        or arguments.username is not None
        or arguments.password_stdin
    )
    if arguments.endpoint is not None:
        updates["hs_endpoint"] = arguments.endpoint.strip()
    if arguments.username is not None:
        updates["hs_username"] = arguments.username.strip()
    if arguments.password_stdin:
        source = stdin if stdin is not None else sys.stdin
        line = next(iter(source), "")
        password = line.rstrip("\r\n")
        if not password:
            print("error: --password-stdin read an empty line", file=sys.stderr)
            return 2
        updates["hs_password"] = password

    if interactive:
        print(f"Configuring HSDS credentials in {path}", file=out)
        for key, label in (("hs_endpoint", "Server endpoint"), ("hs_username", "Username")):
            answer = _prompt_plain(label, current.get(key, ""), ask)
            if answer is not None:
                updates[key] = answer
        for key, label in (("hs_password", "Password"), ("hs_api_key", "API key")):
            answer = _prompt_secret(label, bool(current.get(key)), ask_secret)
            if answer is not None:
                updates[key] = answer

    endpoint = updates.get("hs_endpoint", current.get("hs_endpoint", ""))
    if endpoint and not endpoint.startswith(("http://", "https://", "http+unix://")):
        print(f"error: endpoint must start with http:// or https:// (got {endpoint!r})", file=sys.stderr)
        return 2
    try:
        if updates:
            hscfg.update_hscfg(path, updates)
    except ValueError as error:  # messages name the key, never the value
        print(f"error: {error}", file=sys.stderr)
        return 2

    if updates:
        print(f"Updated {', '.join(sorted(updates))} in {path} (mode 0600).", file=out)
    else:
        print(f"No changes; {path} left as it was.", file=out)
    if path.is_file() and hscfg.insecure_permissions(path):
        # Only reachable when nothing was written (a write always leaves 0600).
        print(
            f"warning: {path} is readable by other users; run `chmod 600 {path}`.",
            file=sys.stderr,
        )
    active = hscfg.active_path()
    if active.resolve() != path.resolve() and path == hscfg.default_path():
        print(
            f"note: h5pyd run from this directory reads {active} instead of {path}.",
            file=out,
        )
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vaft hsds", description=__doc__.split("\n\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    setup = subparsers.add_parser(
        "configure",
        help="write h5pyd credentials to ~/.hscfg without echoing secrets",
        description=(
            "Prompt for the HSDS endpoint and username, and read the password "
            "and API key with hidden input. Existing secrets are shown only as "
            "[configured]; Enter keeps them. Writes an h5pyd-compatible .hscfg "
            "with mode 0600 (not enforced on Windows). Giving any of the options "
            "below skips the prompts."
        ),
    )
    setup.add_argument("--endpoint", help="HSDS endpoint URL (non-interactive)")
    setup.add_argument("--username", help="HSDS username (non-interactive)")
    setup.add_argument(
        "--password-stdin",
        action="store_true",
        help="read the password from the first line of standard input",
    )
    setup.add_argument(
        "--config",
        help="file to write instead of ~/.hscfg (h5pyd also reads ./.hscfg)",
    )
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    if arguments.command == "configure":
        return configure(arguments)
    parser.error(f"unknown command {arguments.command!r}")
    return 2


__all__ = ["configure", "main"]
