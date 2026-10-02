"""Top-level VAFT command dispatcher."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Iterable


_COMMANDS = {
    "filedb": (".filedb", "resolve and audit local FileDB layouts"),
    "shotlog": (".shotlog", "archive the VEST ShotLog and extract per-shot records"),
    "raw-redump": (".raw_redump", "serial, restartable VEST raw-DAQ exports"),
    "raw-upgrade": (".raw_upgrade", "in-place timebase upgrade for legacy raw dumps"),
    "sxr-pack": (".sxr_pack", "pack soft X-ray digitizer CSVs into lossless HDF5"),
    "compare-ods": (".compare_ods", "compare two local ODS products"),
    "vest-upstream": (".vest_upstream", "run VEST upstream OMAS stages"),
    "summary": (".summary", "query and export preset database summaries"),
    "maintenance": (".maintenance", "repair already-published HSDS shots"),
    "plot": (".plot", "render a canonical plot for one or more shots"),
    "export": (".export", "export one shot as IMAS/OMAS/GEQDSK files"),
    "hsds": (".hsds", "configure HSDS credentials without echoing secrets"),
    "pipeline-worker": (".pipeline_worker", "poll VEST SQL and run the routine pipeline on new shots"),
    "help": (".help", "what VAFT can do: topics, defaults and setup status"),
    "setup": (".setup", "report or prepare the runtime environment (never scientific settings)"),
    "mcp": (".mcp", "serve read-only VAFT discovery tools to an MCP client over stdio"),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli",
        description="VAFT command-line workflows",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=tuple(_COMMANDS),
        help="workflow to run",
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


__all__ = ["main"]
