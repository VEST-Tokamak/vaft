"""Top-level VAFT command dispatcher."""

from __future__ import annotations

import argparse
from importlib import import_module
import sys
from typing import Iterable


_COMMANDS = {
    "filedb": (".filedb", "resolve and audit local FileDB layouts"),
    "raw-redump": (".raw_redump", "serial, restartable VEST raw-DAQ exports"),
    "raw-upgrade": (".raw_upgrade", "in-place timebase upgrade for legacy raw dumps"),
    "compare-ods": (".compare_ods", "compare two local ODS products"),
    "vest-upstream": (".vest_upstream", "run VEST upstream OMAS stages"),
    "summary": (".summary", "query and export preset database summaries"),
    "maintenance": (".maintenance", "repair already-published HSDS shots"),
    "plot": (".plot", "render a canonical plot for one or more shots"),
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


def main(argv: Iterable[str] | None = None) -> int:
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
