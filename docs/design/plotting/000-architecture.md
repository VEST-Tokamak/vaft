# 000 — Semantic plotting architecture (milestone 16, issue #250)

Status: **accepted** (decisions ratified 2026-09-01)

This document records the architecture-level decisions for the semantic plotting
redesign (umbrella issue #250). Per-phase design documents live next to this one:

| Doc | Issue | Topic |
|---|---|---|
| `001-taxonomy.md` | #251 | subject/view taxonomy, aliases, naming migration |
| `002-display-policy.md` | #256 | units, notation, titles, uncertainty, validity |
| `003-channel-selection.md` | #259 | semantic channel selection, representatives |
| `004-layout-contract.md` | #260 | overlay / subplots / grouped layouts |
| `005-discovery.md` | #262 | semantic `available_plots()` |
| `006-overview-composition.md` | #261 | overview, cross-source, interactive, projection |

Each doc is written as: Context → numbered Decisions → API sketch → test plan →
migration notes. Decisions are settled interactively with the maintainer before
the phase's implementation lands; deviations discovered during implementation
amend the doc in the same PR.

## Decisions

### A1 — Design records live in `docs/design/plotting/`

One numbered markdown doc per phase, committed alongside (or ahead of) the
implementation it governs. After a phase's decisions settle, a short summary is
posted as a comment on the corresponding GitHub issue.

### A2 — Layering

```
vaft/machine_mapping   geometry semantics (R/Z thresholds, device metadata)
        ↓
vaft/omas              ODS interpretation: extraction recipes, adapters,
                       channel resolution, availability
        ↓
vaft/plot              rendering: typed view models, renderers, registry,
                       taxonomy, display policy   (ODS-free, matplotlib-owning)
```

- `vaft/plot` must not import OMAS/IMAS types; the view models already reject
  them (`models.py` `_REJECTED_TYPE_NAMES`). The `test_no_pyplot_outside_plot`
  boundary stays.
- **Shared vocabulary types are `vaft/plot`-layer types populated by the data
  layers**: the subject taxonomy (`vaft/plot/taxonomy.py`), the display policy
  (`DisplaySpec`, phase C), and the resolved-channel record (`ChannelRef`,
  phase D) are defined in `vaft/plot` so that `vaft.imas` / `vaft.code`
  adapters (issue #63) and discovery (#262) can use them without importing
  `vaft.omas`, and without circular imports.
- `domain` on `PlotSpec` keeps meaning "where the data lives" (IDS ownership,
  selective loading via `ids` / `required_paths`); the new `subject` means
  "what the data physically represents" (issue #251).

### A3 — Issue #63 scope split

The `vaft.imas` and `vaft.code` adapter halves of #63 are in this milestone
(phase H, after the semantic surface is frozen). The `vaft.database` half stays
blocked on #51 (selective HSDS loading) and #56 (named `source` API) and is
deferred; noted on the issue.

### A4 — Deprecation policy

Renamed/replaced public plot names keep working for **two minor releases**,
emitting a `DeprecationWarning` that names the canonical replacement (same
window as the #63 adapter compatibility promise). The mechanism is the existing
migration machinery: rows in `vaft/plot/_migration.py` + thin warning wrappers,
enforced by `test/test_plot_migration.py`. Duplicate `status="legacy"` registry
specs are **not** used (they would double the registry and pollute discovery).

## Phase order

```
A #250 architecture → B #251 taxonomy → C #256 display → D #259 selection
  → E #260 layout → F #262 discovery → G #261 overview/… → H #63 partial
```

Rationale: the taxonomy renames the public surface, so it goes first; display
policy touches every recipe and fixes a live correctness bug (silent
relabel-without-rescale), so it lands before the new selection/layout kwargs;
discovery reports the frozen surface; overviews compose everything prior.
