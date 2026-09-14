"""Reading a stored constraints product that predates the coilset change (#708).

The packaged samples were built when the constraint tree carried the legacy
twenty-six PF channels -- the solenoid in eight segments plus every other
circuit split upper/lower -- and the k-file writer selected sixteen of them by
position.  That selection is gone (PR #701), so the tree a writer is handed
must now already be the coilset the Green table describes.

A stored twenty-six-channel tree is therefore **not** something to truncate.
The first sixteen channels are PF1-1..PF1-8 followed by PF2, PF3 and PF4, where
the table's last eight slots belong to PF5, PF6, PF9 and PF10 -- different
coils, at different radii, silently in the wrong place.  `generate_kfile`
refuses the mismatch for exactly that reason.

Tests that want to use those samples select the right channels here, by name.
Doing it by name rather than by position is also the shape of the eventual fix:
once the table names its own groups (`fcname`, which the regenerated `mhdin.dat`
carries and the shipped legacy one does not), the writer can do this itself and
stored products become readable again.
"""

from __future__ import annotations

import copy
from typing import Any, Sequence

#: The groups the shipped 129x129 table describes, in its order. Not a new
#: convention: this is what `nfsum = 16` in `vaft/data/efit/mhdin.dat` means.
TABLE_GROUPS: tuple[str, ...] = (
    "PF1-1", "PF1-2", "PF1-3", "PF1-4", "PF1-5", "PF1-6", "PF1-7", "PF1-8",
    "PF5U", "PF5L", "PF6U", "PF6L", "PF9U", "PF9L", "PF10U", "PF10L",
)


def select_table_coilset(
    ods: Any, slice_index: int = 0, *, groups: Sequence[str] = TABLE_GROUPS
) -> Any:
    """Rewrite one slice's ``pf_current`` to the groups the table describes.

    Returns ``ods`` for chaining. A tree that already matches is left alone, so
    this is safe to apply to a freshly built product as well as a stored one.
    Raises if a wanted group is absent, because quietly writing a k-file short
    of a coil is the failure this exists to prevent.
    """
    root = f"equilibrium.time_slice.{slice_index}.constraints.pf_current"
    count = len(ods[root])
    by_name = {str(ods[f"{root}.{index}.source"]): index for index in range(count)}
    if tuple(by_name) == tuple(groups):
        return ods

    missing = [name for name in groups if name not in by_name]
    if missing:
        raise ValueError(
            f"the stored tree has no channel for {', '.join(missing)}; "
            f"it carries {', '.join(by_name)}"
        )

    kept = [copy.deepcopy(ods[f"{root}.{by_name[name]}"]) for name in groups]
    del ods[root]
    for index, channel in enumerate(kept):
        ods[f"{root}.{index}"] = channel
    return ods
