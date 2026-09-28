"""Read and write the h5pyd credential file (``.hscfg``) without exposing secrets.

h5pyd reads ``hs_endpoint``, ``hs_username``, ``hs_password`` and ``hs_api_key``
from ``./.hscfg`` if one exists in the working directory, otherwise from
``~/.hscfg``; an ``HS_*`` environment variable overrides the file. VAFT keeps
that file format and that lookup -- this module only makes writing it safe:

* the file is rewritten in place, so keys VAFT does not manage and comments
  survive;
* it is written to a temporary file in the same directory, created ``0600``
  from the start, and renamed over the original, so there is no moment at which
  the credentials sit in a group- or world-readable file;
* secret values are never returned for display -- :func:`configured_keys` reports
  names only.

On Windows ``os.open``'s mode sets only the read-only flag, so the ``0600``
guarantee does not apply there; the file inherits the ACL of its directory
(the user profile, by default).

The upstream ``hsconfigure`` command reads the password with ``input()`` and
shows an existing password as the prompt default; ``vaft hsds configure``
(:mod:`vaft.cli.hsds`) is the frontend built on this module.
"""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import secrets
import stat

__all__ = [
    "HSCFG_KEYS",
    "SECRET_KEYS",
    "default_path",
    "active_path",
    "read_values",
    "configured_keys",
    "render_update",
    "update_hscfg",
    "write_private",
    "insecure_permissions",
    "validate_value",
]

#: The keys h5pyd itself fills in, in the order hsconfigure writes them.
HSCFG_KEYS: tuple[str, ...] = ("hs_endpoint", "hs_username", "hs_password", "hs_api_key")
#: Keys whose values are never printed, echoed, or offered as a prompt default.
SECRET_KEYS: frozenset[str] = frozenset({"hs_password", "hs_api_key"})

_HEADER = "# HDFCloud configuration file (written by `vaft hsds configure`)\n"


def default_path() -> Path:
    """``~/.hscfg``, the file ``vaft hsds configure`` writes unless told otherwise."""
    return Path.home() / ".hscfg"


def active_path(cwd: Path | None = None) -> Path:
    """The file h5pyd will actually read: ``./.hscfg`` shadows ``~/.hscfg``."""
    local = Path(cwd if cwd is not None else Path.cwd()) / ".hscfg"
    return local if local.is_file() else default_path()


def _parse_line(line: str) -> tuple[str, str] | None:
    """``(key, value)`` exactly as h5pyd parses the line, or ``None``.

    h5pyd splits on every ``=`` and keeps only the second field, which is why
    :func:`validate_value` refuses a value containing ``=``.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    fields = stripped.split("=")
    if len(fields) < 2:
        return None
    return fields[0].strip(), fields[1].strip()


def read_values(path: Path) -> dict[str, str]:
    """Every ``key = value`` pair in *path*; the last occurrence wins, as in h5pyd.

    The result holds secrets. Use it to decide what to keep, never to print.
    """
    values: dict[str, str] = {}
    if not Path(path).is_file():
        return values
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        parsed = _parse_line(line)
        if parsed is not None:
            values[parsed[0]] = parsed[1]
    return values


def configured_keys(path: Path) -> tuple[str, ...]:
    """Names of the h5pyd keys *path* sets to a non-empty value. Never values."""
    values = read_values(path)
    return tuple(key for key in HSCFG_KEYS if values.get(key))


def validate_value(key: str, value: str) -> None:
    """Refuse a value h5pyd would read back differently from what was typed.

    A newline would start a new line; an ``=`` truncates the value, because
    h5pyd keeps only the text between the first and second ``=``; surrounding
    whitespace is stripped on read. The message never contains the value.
    """
    if "\n" in value or "\r" in value:
        raise ValueError(f"{key} must be a single line")
    if "=" in value:
        raise ValueError(
            f"{key} contains '=', which h5pyd's .hscfg parser truncates at; "
            f"set it through the HS_{key[3:].upper()} environment variable instead"
        )
    if value != value.strip():
        raise ValueError(f"{key} has leading or trailing whitespace, which h5pyd strips")


def render_update(original: str, updates: Mapping[str, str]) -> str:
    """*original* with each key in *updates* set, everything else untouched.

    An existing ``key = value`` line is rewritten where it stands (a key that
    occurs twice has every occurrence rewritten, so the last one -- the one
    h5pyd uses -- cannot keep the old value); comments, blank lines and unknown
    keys are kept verbatim; keys not yet present are appended in
    :data:`HSCFG_KEYS` order.
    """
    for key, value in updates.items():
        validate_value(key, value)
    lines = original.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        parsed = _parse_line(line)
        if parsed is not None and parsed[0] in updates:
            key = parsed[0]
            out.append(f"{key} = {updates[key]}\n")
            seen.add(key)
        else:
            out.append(line)
    if not out:
        out.append(_HEADER)
    ordered = [key for key in HSCFG_KEYS if key in updates]
    ordered += [key for key in updates if key not in HSCFG_KEYS]
    for key in ordered:
        if key not in seen:
            out.append(f"{key} = {updates[key]}\n")
    return "".join(out)


def write_private(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*, readable by the owner only.

    The temporary file is created by ``os.open(..., O_EXCL, 0o600)`` in the
    destination's directory (so ``os.replace`` is a same-filesystem rename) and
    ``fchmod``-ed to ``0600`` again, so the mode does not depend on the process
    umask having no odd bits. A symlink at *path* is replaced by a regular
    file rather than followed.
    """
    path = Path(path)
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".hscfg.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(temporary, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        if hasattr(os, "fchmod"):
            os.fchmod(fd, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def update_hscfg(path: Path, updates: Mapping[str, str]) -> None:
    """Set *updates* in the ``.hscfg`` at *path*, creating it ``0600`` if needed."""
    path = Path(path)
    original = path.read_text(encoding="utf-8") if path.is_file() else ""
    write_private(path, render_update(original, updates))


def insecure_permissions(path: Path) -> bool:
    """Whether *path* is group- or world-accessible. Always ``False`` on Windows."""
    if os.name == "nt":
        return False
    return bool(Path(path).stat().st_mode & 0o077)
