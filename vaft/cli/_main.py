"""Top-level VAFT command dispatcher."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Iterable

from vaft._help._registry import NAME, TAGLINE


_COMMANDS = {
    "filedb": (".filedb", "resolve and audit local FileDB layouts"),
    "shotlog": (".shotlog", "archive the VEST ShotLog and extract per-shot records"),
    "raw-redump": (".raw_redump", "serial, restartable VEST raw-DAQ exports"),
    "raw-upgrade": (".raw_upgrade", "in-place timebase upgrade for legacy raw dumps"),
    "raw-legacy-import": (".raw_legacy_import", "import pre-SQL per-field CSV shots into a FileDB"),
    "sxr-pack": (".sxr_pack", "pack soft X-ray digitizer CSVs into lossless HDF5"),
    "compare-ods": (".compare_ods", "compare two local ODS products"),
    "vest-upstream": (".vest_upstream", "run VEST upstream OMAS stages"),
    "summary": (".summary", "query and export preset database summaries"),
    "maintenance": (".maintenance", "repair already-published HSDS shots"),
    "plot": (".plot", "render a canonical plot for one or more shots"),
    "export": (".export", "export one shot as IMAS/OMAS/GEQDSK files"),
    "hsds": (".hsds", "configure HSDS credentials without echoing secrets"),
    "cache": (".cache", "inspect, prune and cap the local HSDS domain cache"),
    "pipeline-worker": (".pipeline_worker", "poll VEST SQL and run the routine pipeline on new shots"),
    "corrective-worker": (".corrective_worker", "watch Thomson/CES uploads and run the corrective pipeline"),
    "help": (".help", "what VAFT can do: topics, defaults and setup status"),
    "setup": (".setup", "report or prepare the runtime environment (never scientific settings)"),
    "mcp": (".mcp", "serve read-only VAFT discovery tools to an MCP client over stdio"),
    "gui": (".gui", "launch the browser GUI"),
}


#: How ``vaft --help`` and ``vaft help cli`` present the commands: general
#: research tools first, VEST's own data and operations tools last.  Every
#: command sits in exactly one group; the order within a group is display order.
_GROUPS = (
    ("Scientific analysis", ("plot", "compare-ods", "summary")),
    ("Data access and interoperability", ("export", "filedb", "hsds", "cache")),
    ("Research interfaces and environment", ("help", "setup", "mcp", "gui")),
    (
        "VEST infrastructure and operations",
        (
            "shotlog",
            "raw-redump",
            "raw-upgrade",
            "raw-legacy-import",
            "sxr-pack",
            "vest-upstream",
            "maintenance",
            "pipeline-worker",
            "corrective-worker",
        ),
    ),
)


def command_group(command: str) -> str:
    """The ``_GROUPS`` title ``command`` is listed under."""
    for title, names in _GROUPS:
        if command in names:
            return title
    raise KeyError(f"unknown command {command!r}; choose from: {', '.join(_COMMANDS)}")


def _command_listing() -> str:
    width = max(map(len, _COMMANDS)) + 2
    blocks = []
    for title, names in _GROUPS:
        rows = "\n".join(f"  {name:<{width}}{_COMMANDS[name][1]}" for name in names)
        blocks.append(f"{title}:\n{rows}")
    return "\n\n".join(blocks)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaft",
        description=f"{NAME}\n{TAGLINE}",
        epilog=(
            _command_listing()
            + "\n\nwhat VAFT can do: vaft help  |  a command's syntax: vaft <command> --help"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(_COMMANDS),
        metavar="command",
        help="one of the commands listed below",
    )
    return parser


def _tolerate_unencodable_output() -> None:
    """Never let a console encoding kill a command over a glyph.

    ``vaft plot --list`` draws its tree with box-drawing characters and the
    reference pages print em-dashes; a Windows console or a ``text=True`` pipe
    runs under the locale codec (cp1252, cp949), where ``print`` raises
    ``UnicodeEncodeError`` and the command exits 1 with its work done. The
    streams keep their encoding (a caller decoding the pipe with the same
    locale still reads it) and only the error handler changes, so an
    unencodable glyph becomes ``?`` instead of a crash.
    """
    lenient = {"replace", "backslashreplace", "namereplace", "xmlcharrefreplace", "ignore"}
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        encoding = (getattr(stream, "encoding", None) or "").replace("-", "").lower()
        # "surrogateescape" (what a Windows pipe carries) is not lenient: it
        # only round-trips undecodable input bytes and still raises on a glyph
        # the codec lacks, which is how the release's Windows leg found this.
        if encoding != "utf8" and getattr(stream, "errors", None) not in lenient:
            reconfigure(errors="replace")


def main(argv: Iterable[str] | None = None) -> int:
    _tolerate_unencodable_output()
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return 0
    command = arguments.pop(0)
    if command not in _COMMANDS:
        parser.error(
            f"invalid command {command!r}; choose from: {', '.join(_COMMANDS)}"
        )
    module_name, _description = _COMMANDS[command]
    module = import_module(module_name, __package__)
    return int(module.main(arguments))


__all__ = ["command_group", "main"]
