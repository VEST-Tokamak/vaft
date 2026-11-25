"""The stages of pipeline 2 and what each needs first.

One place for the answer to "which stages does the corrective pipeline build,
and in what dependency order", so the Snakefile (``workflow/automatic_pipeline_2_corrective_data_update``)
and the corrective worker (:mod:`vaft.database.worker.corrective`) cannot
disagree about it.  A stage must also be registered in
:data:`vaft.database.sources.STAGE_REPLICATION`; the Snakefile checks that.

The relation is the one the *products* have, not the order they replicate in:
``composition``, ``ti_inferred`` and ``resistive_zeff`` all own ``core_profiles``
in ``main`` and replicate one after another, which the Snakefile orders itself.
Pipeline 1's EFIT is consulted by several of these stages as a parameter, not
a product they wait for, so it is not an upstream here.
"""

from __future__ import annotations

from typing import Iterable, Mapping

#: Every stage the Thomson lineage builds, in a dependency-respecting order.
CORRECTIVE_LINEAGE = (
    "thomson",
    "ces",
    "core_profiles",
    "composition",
    "ti_inferred",
    "resistive_zeff",
    "electron_efit",
    "kinetic_efit",
)

#: What a run builds when ``stages`` is not given: the measured profiles with the
#: stated assumptions (``<Z_eff> = 2`` composition, ``T_i = T_e``) and the kinetic
#: EFITs.  ``ti_inferred`` and ``resistive_zeff`` are opt-in: the pressure
#: partition returns a ``T_i/T_e`` near 2 on VEST, and the resistive ``Z_eff`` is
#: identifiable on few shots, so neither is written to ``main`` unasked.
DEFAULT_STAGES = (
    "thomson", "ces", "core_profiles", "composition", "electron_efit", "kinetic_efit",
)

#: The stages whose product each stage reads.  The kinetic lineages read
#: ``composition`` and never ``ti_inferred`` (their pressure would then be
#: partitioned back into the temperature it was built from); ``resistive_zeff``
#: is built on ``ti_inferred`` so that replacing ``core_profiles`` keeps it.
UPSTREAM: Mapping[str, tuple[str, ...]] = {
    "thomson": (),
    "ces": (),
    "core_profiles": ("thomson", "ces"),
    "composition": ("core_profiles",),
    "ti_inferred": ("composition",),
    "resistive_zeff": ("ti_inferred",),
    "electron_efit": ("composition",),
    "kinetic_efit": ("composition",),
}


def stage_scope(value: Iterable[str] | str | None) -> frozenset[str] | None:
    """The stages a run builds and judges, from the configuration's ``stages`` key.

    ``None`` (the key absent) is :data:`DEFAULT_STAGES`; a list names the scope
    exactly, so ``ti_inferred`` and ``resistive_zeff`` are requested by name.  A list narrows it: a stage
    outside the scope is never scheduled and never judged, so a worker told to
    build ``[thomson, ces]`` does not fail at a profile stage it was never asked
    for.  This is scope, decided before a run -- a stage that runs and finds its
    data missing records ``unavailable`` instead.

    A stage whose upstream is left out is refused rather than silently pulling
    that upstream in unjudged, as in the routine pipeline's ``stage_scope``.
    ``thomson`` and ``ces`` are independent roots; ``core_profiles`` needs both,
    because the ion diagnostic decides which kinetic lineage a shot belongs to.

    Raises:
        ValueError: An unknown stage, an empty list, or a stage without its upstream.
    """
    if value is None:
        return frozenset(DEFAULT_STAGES)
    names = [value] if isinstance(value, str) else [str(name) for name in value]
    if not names:
        raise ValueError("stages is empty; omit it to build the whole lineage")
    unknown = sorted(set(names) - set(CORRECTIVE_LINEAGE))
    if unknown:
        raise ValueError(
            f"Unknown stage(s) in stages: {', '.join(unknown)}; expected some of "
            + ", ".join(CORRECTIVE_LINEAGE)
        )
    scope = set(names)
    for stage in CORRECTIVE_LINEAGE:
        missing = [needed for needed in UPSTREAM[stage] if stage in scope and needed not in scope]
        if missing:
            raise ValueError(f"stages includes {stage!r} but not {', '.join(map(repr, missing))}, which it needs")
    return frozenset(scope)


def scoped(scope: frozenset[str] | None) -> tuple[str, ...]:
    """The lineage's stages inside ``scope``, in lineage order (all of them for ``None``)."""
    return tuple(stage for stage in CORRECTIVE_LINEAGE if scope is None or stage in scope)


__all__ = ["CORRECTIVE_LINEAGE", "DEFAULT_STAGES", "UPSTREAM", "scoped", "stage_scope"]
