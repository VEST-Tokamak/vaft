"""Make the flat helper modules (fixtures, helpers, spec) importable.

The contract tests import them as top-level modules (``from fixtures import
...``), which only works when this directory is on ``sys.path`` — e.g. when
pytest is invoked from inside it. Add it explicitly so the suite also collects
from the repository root.
"""

import sys
from pathlib import Path

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
