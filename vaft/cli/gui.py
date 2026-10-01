"""Launch the VAFT browser GUI (needs ``pip install 'vaft[gui]'``)."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from pathlib import Path

_EPILOG = """\
The server binds to 127.0.0.1. On a remote host or cluster node, forward the
port and open http://localhost:PORT locally; under SSH the page asks for the
password printed at start (or $VAFT_GUI_PASSWORD):

  ssh -L 5006:localhost:5006 user@host        # then run `vaft gui` there

VS Code Remote-SSH forwards the port by itself (Ports panel).

To serve a team behind nginx with HTTPS, see the GUI guide (Host it for a team):

  VAFT_GUI_PASSWORD=... vaft gui --hosted --prefix /gui --no-show \
      --allow-websocket-origin vest.example.org
"""


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vaft gui", description=__doc__, epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", type=int, nargs="+", help="open packaged sample shots (offline); several are compared")
    source.add_argument("--file", type=Path, help="open a local ODS/IMAS/GEQDSK file")
    source.add_argument("--shot", type=int, nargs="+", help="open database shots; several are compared")
    parser.add_argument("--source", dest="namespace", help="database namespace for --shot (default: main)")
    parser.add_argument("--plot", help="plot to draw first (a name from available_plots)")
    parser.add_argument("--address", default="127.0.0.1", help="address to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=5006, help="port to serve on (default: %(default)s)")
    parser.add_argument(
        "--allow-websocket-origin", action="append", default=[], metavar="HOST[:PORT]",
        help="extra origin the browser may connect from (repeatable)",
    )
    parser.add_argument(
        "--auth", choices=("auto", "password", "hsds", "none"), default="auto",
        help="ask for a password: under SSH or off loopback (auto, default), always, or never; "
             "the password is $VAFT_GUI_PASSWORD or a random one printed at start. "
             "hsds: sign in with an HSDS account instead, checked against $HS_ENDPOINT",
    )
    parser.add_argument(
        "--hosted", action="store_true",
        help="serve readers who are not this server's user, behind a reverse proxy: samples and "
             "database shots only (no server files, no uploads); needs $VAFT_GUI_PASSWORD, or --auth hsds",
    )
    parser.add_argument("--prefix", help="URL path to serve under, e.g. /gui behind a proxy")
    show = parser.add_mutually_exclusive_group()
    show.add_argument("--show", dest="show", action="store_true", default=None, help="open a browser")
    show.add_argument("--no-show", dest="show", action="store_false", help="do not open a browser")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.hosted and args.file is not None:
        parser.error("--hosted opens no files; use --sample or --shot")
    if args.hosted and args.auth in ("auto", "password") and not os.environ.get("VAFT_GUI_PASSWORD"):
        # serve() refuses too; said here without a traceback.
        print("vaft gui: a hosted server needs its password set: export VAFT_GUI_PASSWORD", file=sys.stderr)
        return 1

    from ..gui import require_panel

    try:
        require_panel()
    except ImportError as error:
        print(f"vaft gui: {error}", file=sys.stderr)
        return 1
    from ..gui.app import serve

    if args.auth == "hsds":
        from ..gui.auth import hsds_endpoint

        try:
            hsds_endpoint()
        except ValueError as error:
            print(f"vaft gui: {error}", file=sys.stderr)
            return 1

    serve(
        address=args.address,
        port=args.port,
        show=args.show,
        websocket_origin=args.allow_websocket_origin,
        auth=args.auth,
        hosted=args.hosted,
        prefix=args.prefix,
        sample=args.sample,
        file=None if args.file is None else str(args.file),
        shot=args.shot,
        namespace=args.namespace,
        plot=args.plot,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
