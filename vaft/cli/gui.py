"""Launch the VAFT browser GUI (needs ``pip install 'vaft[gui]'``)."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable
from pathlib import Path

_EPILOG = """\
The server binds to 127.0.0.1. On a remote host or cluster node, forward the
port and open http://localhost:PORT locally; under SSH the page asks for the
password printed at start (or $VAFT_GUI_PASSWORD):

  ssh -L 5006:localhost:5006 user@host        # then run `vaft gui` there

VS Code Remote-SSH forwards the port by itself (Ports panel).
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
        "--auth", choices=("auto", "password", "none"), default="auto",
        help="ask for a password: under SSH or off loopback (auto, default), always, or never; "
             "the password is $VAFT_GUI_PASSWORD or a random one printed at start",
    )
    show = parser.add_mutually_exclusive_group()
    show.add_argument("--show", dest="show", action="store_true", default=None, help="open a browser")
    show.add_argument("--no-show", dest="show", action="store_false", help="do not open a browser")
    args = parser.parse_args(list(argv) if argv is not None else None)

    from ..gui import require_panel

    try:
        require_panel()
    except ImportError as error:
        print(f"vaft gui: {error}", file=sys.stderr)
        return 1
    from ..gui.app import serve

    serve(
        address=args.address,
        port=args.port,
        show=args.show,
        websocket_origin=args.allow_websocket_origin,
        auth=args.auth,
        sample=args.sample,
        file=None if args.file is None else str(args.file),
        shot=args.shot,
        namespace=args.namespace,
        plot=args.plot,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
