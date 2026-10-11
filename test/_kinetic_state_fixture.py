"""Test-side import of ``workflow/kinetic_state/archive_to_ods.py`` (issue #1837).

The converter lives beside the archive it reads; this shim puts that
directory on ``sys.path`` once so tests can ``from _kinetic_state_fixture import ...``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow" / "kinetic_state"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from archive_to_ods import (  # noqa: E402,F401
    ARCHIVE, CASES, E, LINEAGES, KineticStateCase, kinetic_state_case, kinetic_state_ods, load_archive,
)
