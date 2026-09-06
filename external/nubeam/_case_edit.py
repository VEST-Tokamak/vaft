#!/usr/bin/env python3
"""Text edits the NUBEAM validation scripts need, done portably.

Called by ``run-local-validation.sh`` and ``run-local-vest.sh`` instead of
``sed``/``awk``. Those tools diverge between GNU and BSD in exactly the two
ways these scripts require:

* ``sed -i`` takes a mandatory backup suffix on BSD and none on GNU, so the
  ``sed -i ''`` that is correct on macOS makes GNU sed read ``''`` as the
  script and the real script as a filename.
* GNU spells a whole-line replacement ``2c\\TEXT`` while BSD wants the text on
  the following line, and rejects the GNU form outright.

Rather than branch on the platform in shell, the edits live here and reuse the
adapter's implementations, so the validation harness and ``vaft.code.nubeam``
cannot drift apart.

This requires the ``vaft`` environment (``install/macos.sh`` or
``install/linux.sh``); the build script ``macos.sh`` deliberately does not.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
from typing import NoReturn


def _die(message: str) -> NoReturn:
    print(f"_case_edit.py: {message}", file=sys.stderr)
    raise SystemExit(1)


def _load_adapter():
    try:
        from vaft.code import nubeam
    except ImportError as error:  # pragma: no cover - environment problem
        _die(
            "the vaft environment is required for this step "
            f"({error}). Run install/macos.sh or install/linux.sh first."
        )
    return nubeam


def cmd_set_equilibrium(args: argparse.Namespace) -> None:
    nubeam = _load_adapter()
    path = Path(args.inputf)
    original = path.read_text(encoding="utf-8")
    path.write_text(
        nubeam.rewrite_inputf_equilibrium(original, args.gfile), encoding="utf-8"
    )


def cmd_read_field(args: argparse.Namespace) -> None:
    nubeam = _load_adapter()
    text = Path(args.inputf).read_text(encoding="utf-8")
    if args.field == "state":
        print(nubeam.inputf_state_filename(text))
    else:
        print(nubeam.inputf_runid(text))


def cmd_set_particles(args: argparse.Namespace) -> None:
    from vaft.code.nubeam.inputs import _apply_particle_count

    for namelist in args.namelist:
        _apply_particle_count(Path(namelist), args.count)


def cmd_set_seed(args: argparse.Namespace) -> None:
    """Set ``nseed`` in a NUBEAM init namelist.

    Local to this file: holding the RNG seed is a property of how the
    validation harness compares runs, not of running NUBEAM, so the adapter
    has no equivalent. The same regex discipline applies -- the shipped
    namelists document every knob inline and a namelist round-trip would
    discard all of it.
    """
    path = Path(args.namelist)
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"^(?P<indent>[ \t]*)nseed(?P<gap>[ \t]*=[ \t]*)\d+", re.MULTILINE
    )
    updated, count = pattern.subn(
        lambda m: f"{m.group('indent')}nseed{m.group('gap')}{args.seed}", text
    )
    if count == 0:
        _die(f"no nseed entry found in {path}")
    path.write_text(updated, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("set-equilibrium", help="point inputf at a G-EQDSK")
    p.add_argument("inputf")
    p.add_argument("gfile")
    p.set_defaults(func=cmd_set_equilibrium)

    p = sub.add_parser("read-field", help="read a positional inputf field")
    p.add_argument("inputf")
    p.add_argument("field", choices=("state", "runid"))
    p.set_defaults(func=cmd_read_field)

    p = sub.add_parser("set-particles", help="set nptcls/nptclf in a namelist")
    p.add_argument("count", type=int)
    p.add_argument("namelist", nargs="+")
    p.set_defaults(func=cmd_set_particles)

    p = sub.add_parser("set-seed", help="set nseed in an init namelist")
    p.add_argument("seed", type=int)
    p.add_argument("namelist")
    p.set_defaults(func=cmd_set_seed)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
