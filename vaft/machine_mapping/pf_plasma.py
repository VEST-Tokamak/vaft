"""
`pf_plasma` IDS mapping helpers (stub).

Schema reference:
https://gafusion.github.io/omas/schema.html
"""

from __future__ import annotations

from typing import Optional

from omas import ODS


def pf_plasma(ods: ODS, source: str, options: Optional[dict] = None) -> None:
    if options is None:
        options = {}
    raise NotImplementedError("pf_plasma is not implemented yet.")

