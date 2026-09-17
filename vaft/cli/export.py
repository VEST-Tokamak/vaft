"""``vaft export``: write one database shot as portable local files.

A thin front on :func:`vaft.database.export`, which stages the shot once and
fans out every requested backend. Nothing heavier than ``argparse`` is
imported before the arguments are parsed, so ``vaft export --help`` works in a
bare install.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable


# Mirrors vaft.database._export.BACKENDS; test_cli checks the two agree.
BACKEND_HELP = {
    "imas-hdf5": "imas_<shot>_hdf5/  native IMAS HDF5 Data Entry (master.h5 + IDS images), copied as stored",
    "imas-nc": "imas_<shot>.nc     IMAS netCDF convention, written by IMAS-Python",
    "omas-json": "omas_<shot>.json   OMAS JSON",
    "omas-hdf5": "omas_<shot>.h5     OMAS single-file HDF5 -- NOT an IMAS Data Entry",
    "omas-nc": "omas_<shot>.nc     OMAS flat netCDF -- NOT the IMAS netCDF convention",
    "geqdsk": "geqdsk_<shot>/     one EFIT g-file per equilibrium time slice (equilibrium only)",
}


def _parser() -> argparse.ArgumentParser:
    epilog = "backends:\n" + "\n".join(
        f"  {name:<10} {text}" for name, text in BACKEND_HELP.items()
    ) + (
        "\n\nimas-hdf5 and omas-hdf5, and imas-nc and omas-nc, are different formats."
        "\nConverted backends export occurrence 0 only; imas-hdf5 keeps every occurrence."
    )
    parser = argparse.ArgumentParser(
        prog="python -m vaft.cli export",
        description="Export one shot from an HSDS source as local files, downloading it once.",
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--shot", type=int, required=True, help="shot number")
    parser.add_argument("--source", help="HSDS source (default: main)")
    parser.add_argument(
        "--backend", nargs="+", required=True, choices=tuple(BACKEND_HELP), metavar="BACKEND",
        help="one or more of: " + ", ".join(BACKEND_HELP),
    )
    parser.add_argument("--output", help="destination directory (default: current directory)")
    parser.add_argument("--overwrite", action="store_true", help="replace existing artifacts")
    parser.add_argument("--imas-version", help="IMAS DD version to read with (default: as stored)")
    parser.add_argument("--cache", default="auto", help="HSDS file cache: auto, off, or a directory")
    parser.add_argument(
        "--transport", default="auto", choices=("auto", "canonical", "h5image"),
        help="HSDS transport (default: auto)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    from vaft import database

    try:
        written = database.export(
            args.shot,
            args.source,
            backend=args.backend,
            output=args.output,
            overwrite=args.overwrite,
            imas_version=args.imas_version,
            cache=args.cache,
            transport=args.transport,
        )
    except KeyboardInterrupt:
        print("vaft export: interrupted", file=sys.stderr)
        return 130
    except (ImportError, KeyError, ValueError, RuntimeError, OSError) as error:
        message = error.args[0] if isinstance(error, KeyError) and error.args else error
        print(f"vaft export: error: {message}", file=sys.stderr)
        return 1
    for name, path in written.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
