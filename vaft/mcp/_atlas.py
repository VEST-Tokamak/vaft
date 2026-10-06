"""The campaign atlas tables the MCP tools may read, and how to read each (#188 Phase 2).

Each lane writes its table, a schema and a README under one atlas directory
(``VAFT_ATLAS_DIR`` on the server, ``~/runs/campaign/atlas`` on vestserver).
The lanes describe their columns in three different formats, and two of them
still spell the state key the pre-contract way.  This module is the registry
that turns them into one answer: column meanings, units, the row key, the
lane's rules and caveats, and rows filtered by value.

Nothing here computes a physics quantity, joins two tables or aggregates rows
beyond a count; the lane products are returned as written, after only the
spelling normalisation the State key contract v1 (#1454) defines.  Rules that
the lanes state as "never combine" become required filters: a stability query
names one toroidal mode number, a sensitivity query one TGLF configuration.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Pre-contract spellings, normalised on read (State key contract v1, #1454).
LINEAGE_SPELLINGS = {"magnetics-only": "magnetics", "electron-kinetic": "electron_kinetic"}
COLUMN_SPELLINGS = {"efit_label": "efit_quality"}

STATE_KEY = ("shot", "time_efit_s", "efit_lineage")
STATE_LABELS = ("efit_quality",)


@dataclass(frozen=True)
class AtlasTable:
    """One lane product: where it is, how its schema reads, and what must not be mixed."""

    name: str
    lane: str
    issue: str
    path: str
    summary: str
    key: tuple[str, ...]
    schema: str | None = None
    #: json_schema | columns | column_dictionary | manifest_units
    schema_kind: str | None = None
    #: The schema entry holding the column descriptions, for ``columns`` schemas.
    columns_entry: str = "columns"
    readme: str | None = None
    manifest: str | None = None
    #: Columns a query must pin with ``==``: the lane forbids mixing their values.
    pinned: tuple[str, ...] = ()
    default_columns: tuple[str, ...] = ()
    rules: tuple[str, ...] = ()
    caveats: tuple[str, ...] = field(default=())


_K_CAVEATS = (
    "efit_quality 'good' includes the Thomson band p_e <= p <= 3 p_e, so p/p_e statistics on good rows "
    "are censored to [1, 3]; report statistics per label.",
    "Unreconstructible slices never appear; electron_kinetic rows exist only where the paired magnetics "
    "slice is good or admissible.",
    "rho_tor_norm never silently falls back to sqrt(psi_N): rho_coordinate says when it is unavailable.",
    "429xx magnetics-EFIT stored energy is biased ~1.6x high (Lane D), which also moves r_w and beta_N.",
)

TABLES: tuple[AtlasTable, ...] = (
    AtlasTable(
        name="state",
        lane="K",
        issue="#1454",
        path="v1/state.csv",
        summary="Tier A equilibrium states: one row per (shot, time_efit_s, efit_lineage) with EFIT quality, "
        "Thomson matching, pressure consistency (r_sum, r_w) and global quantities.",
        key=STATE_KEY,
        schema="v1/schema/state.schema.json",
        schema_kind="json_schema",
        manifest="v1/MANIFEST.json",
        default_columns=STATE_KEY + STATE_LABELS + (
            "kinetic_admissible", "ts_status", "r_sum", "r_w", "betap", "li", "q95", "wmhd_j", "ip_measured_a"),
        rules=("Match states by (shot, time_efit_s, efit_lineage), never by slice index.",),
        caveats=_K_CAVEATS,
    ),
    AtlasTable(
        name="kinetic_profiles",
        lane="K",
        issue="#1454",
        path="v1/profiles.csv",
        summary="Thomson electron profiles mapped onto each state's equilibrium, per channel.",
        key=STATE_KEY + ("channel",),
        schema="v1/schema/profiles.schema.json",
        schema_kind="json_schema",
        manifest="v1/MANIFEST.json",
        rules=("Match states by (shot, time_efit_s, efit_lineage), never by slice index.",),
        caveats=_K_CAVEATS,
    ),
    AtlasTable(
        name="transport",
        lane="T",
        issue="#1453",
        path="transport/atlas.csv",
        summary="Local TGLF, NEO and classical fluxes per state and r/a, with every TGLF input.",
        key=STATE_KEY + ("r_over_a",),
        schema="transport/schema.json",
        schema_kind="columns",
        default_columns=STATE_KEY + STATE_LABELS + (
            "r_over_a", "tglf_config", "tglf_status", "qe_gb", "qi_gb", "q_tot_gb", "gamma_max_ion_scale",
            "f_e", "neo_status", "f_neo_qi", "ti_lineage"),
        rules=(
            "The table is one named TGLF configuration (tglf_config column); it is not a default.",
            "Every row is conditional on Ti = Te, H+/C6+ at Z_eff 2 and no ExB shear; a/L_Ti equals a/L_Te.",
        ),
        caveats=(
            "The SAT rule alone changes predicted flux by ~5x at the median surface (transport_sensitivity); "
            "any flux magnitude carries that uncertainty and no configuration is preferred.",
            "r/a 0.7-0.8 lies beyond the Thomson span.",
            "34 of 81 states use shape_kind=derived (#1458).",
        ),
    ),
    AtlasTable(
        name="transport_summary",
        lane="T",
        issue="#1453",
        path="transport/radial_summary.csv",
        summary="One row per transport state: fraction electron-dominated, peak ion-scale growth rate and its r/a, "
        "median gyro-Bohm flux and NEO ion fraction.",
        key=STATE_KEY,
        caveats=("Same TGLF configuration and assumptions as the transport table.",),
    ),
    AtlasTable(
        name="transport_sensitivity",
        lane="T",
        issue="#1453",
        path="transport_sensitivity/sensitivity.csv",
        summary="TGLF fluxes on representative states for 8 configurations (SAT 0-3 x electrostatic / "
        "electromagnetic-bper).",
        key=STATE_KEY + ("r_over_a", "tglf_config"),
        schema="transport_sensitivity/schema.json",
        schema_kind="columns",
        pinned=("tglf_config",),
        default_columns=STATE_KEY + STATE_LABELS + (
            "r_over_a", "tglf_config", "sat_rule", "field_model", "status", "qe_gb", "qi_gb", "q_tot_gb",
            "qe_over_qi", "gamma_max_ion_scale"),
        rules=(
            "Query one tglf_config at a time; compare configurations with transport_sensitivity_pairs.",
            "Model output; no configuration is preferred.",
        ),
    ),
    AtlasTable(
        name="transport_sensitivity_pairs",
        lane="T",
        issue="#1453",
        path="transport_sensitivity/sensitivity_pairs.csv",
        summary="Per surface: the spread of each flux over SAT rules within a field model "
        "(<q>_sat_spread_<model>) and electromagnetic vs electrostatic per SAT rule (<q>_em_vs_es_sat<k>).",
        key=STATE_KEY + ("r_over_a",),
        schema="transport_sensitivity/schema.json",
        schema_kind="columns",
        columns_entry="pair_columns",
        rules=(
            "delta = (A - B) / max(|A|, |B|, 1e-3 gyro-Bohm), in [-2, 2]; an empty cell means not run, not zero.",
        ),
    ),
    AtlasTable(
        name="stability",
        lane="N",
        issue="#1448",
        path="stability/atlas_n.csv",
        summary="Ideal (DCON) and tearing (RDCON, STRIDE) stability per state and toroidal mode number n_tor.",
        key=STATE_KEY + ("n_tor",),
        schema="stability/schema.json",
        schema_kind="column_dictionary",
        readme="stability/README.md",
        pinned=("n_tor",),
        default_columns=STATE_KEY + STATE_LABELS + (
            "n_tor", "equilibrium_status", "dcon_full_status", "dcon_trunc_status", "ideal_stable_full_edge",
            "ideal_stable_truncated_edge", "ideal_unstable_full_edge", "ideal_unstable_trunc_edge",
            "rdcon_status", "stride_status", "atlas_version"),
        rules=(
            "Never combine values across n_tor: query one n_tor at a time.",
            "Never combine DCON with RDCON, or RDCON with STRIDE, into one scalar.",
            "Keep dcon_full_* (full edge) and dcon_trunc_* (truncated edge) apart; full-edge W_t is psihigh-sensitive.",
            "DCON energies are normalised eigenvalues, not joules; VALID_* means mpsi 256 and 512 agree in sign.",
            "{s}_status is not a tearing verdict; delta' is the diagonal classical PEST3 value.",
        ),
        caveats=(
            "There is no |W_t| marginal band; MARGINAL is reserved and never produced.",
            "RDCON and STRIDE are NOT_APPLICABLE for n >= 3.",
            "Atlas version 2 adds raw / QA / physical layers; version 1 columns stay for one version.",
        ),
    ),
    AtlasTable(
        name="stability_surfaces",
        lane="N",
        issue="#1448",
        path="stability/atlas_surfaces.csv",
        summary="Tearing stability per rational surface: delta', resistive interchange and Mercier terms.",
        key=STATE_KEY + ("n_tor", "solver", "m"),
        schema="stability/schema.json",
        schema_kind="column_dictionary",
        readme="stability/README.md",
        pinned=("n_tor", "solver"),
        rules=(
            "Never combine values across n_tor or across solvers: query one n_tor and one solver.",
            "Summaries use resolved interior surfaces only (resolved / two_resolution_consistent).",
        ),
    ),
    AtlasTable(
        name="zeff_windows",
        lane="Z",
        issue="#1486",
        path="zeff/zeff.csv",
        summary="Resistive Z_eff per discharge window from loop-voltage power balance, with its uncertainty, "
        "status and the conductivity model.",
        key=("shot", "t_start_s", "window_kind"),
        schema="zeff/schema/zeff.schema.json",
        schema_kind="json_schema",
        manifest="zeff/MANIFEST.json",
        default_columns=(
            "shot", "t_start_s", "t_end_s", "window_kind", "window_class", "status", "reason", "zeff",
            "zeff_uncertainty", "conductivity_model", "zeff_spitzer_nrl", "zeff_sauter", "zeff_redl"),
        rules=(
            "Always read zeff together with conductivity_model: Spitzer gives ~3x larger values.",
            "Rows are windows, not states.",
        ),
        caveats=(
            "Zeff_resistive is model-inferred, not a composition measurement; assumes no non-inductive or "
            "bootstrap current and no smoothing.",
            "The flattop windows of 39915-39917 are early current decay (#1514).",
        ),
    ),
    AtlasTable(
        name="zeff_slices",
        lane="Z",
        issue="#1486",
        path="zeff/slices.csv",
        summary="Per-state loop-voltage decomposition behind the Z_eff windows.",
        key=STATE_KEY,
        schema="zeff/schema/slices.schema.json",
        schema_kind="json_schema",
        manifest="zeff/MANIFEST.json",
    ),
    AtlasTable(
        name="confinement",
        lane="D",
        issue="#1490",
        path="confinement/table.csv",
        summary="Energy confinement per magnetics state: tau_E, W, P_ohm, regime and the selection evidence.",
        key=STATE_KEY,
        schema="confinement/schema/table.schema.json",
        schema_kind="json_schema",
        manifest="confinement/MANIFEST.json",
        default_columns=STATE_KEY + STATE_LABELS + (
            "regime", "i_p_A", "b_t_T", "n_e_line_avg_m3", "p_loss_W", "w_th_J", "tau_e_th_s", "li_3",
            "beta_normal", "accepted", "quality_status"),
        rules=(
            "'accepted' is provisional: filter by the evidence and rule_* columns instead.",
            "The IPB98 H factor locates VEST; it is not a performance claim.",
        ),
        caveats=(
            "Primary selection: 59 states over 19 shots, ohmic.",
            "P_rad is not measured (NaN); p_loss_W is P_net. tau_e_kin_s is unreliable.",
            "n_e_line_avg_m3 is the Thomson chord average, present on 64 rows only.",
            "The B_T dependence is not identifiable.",
        ),
    ),
    AtlasTable(
        name="op_space_base",
        lane="V",
        issue="#1456",
        path="lane_v/efit_base.csv",
        summary="Operating-space coordinates per state: li_3, beta_N, q95, normalised current, elongation.",
        key=STATE_KEY,
        schema="lane_v/MANIFEST.json",
        schema_kind="manifest_units",
        manifest="lane_v/MANIFEST.json",
        rules=("Match states by (shot, time_efit_s, efit_lineage), never by slice index.",),
    ),
)

BY_NAME = {table.name: table for table in TABLES}

#: Manifest keys worth carrying as provenance; commands and absolute inputs stay out.
PROVENANCE_KEYS = (
    "contract", "contract_version", "generated_at", "vaft_git", "vaft_dirty", "criteria_sha", "atlas_version",
)


def normalised_column(name: str) -> str:
    return COLUMN_SPELLINGS.get(name, name)


def normalise_frame(frame):
    """``frame`` with the pre-contract column names and lineage values in v1 spelling."""
    renames = {old: new for old, new in COLUMN_SPELLINGS.items() if old in frame.columns and new not in frame.columns}
    if renames:
        frame = frame.rename(columns=renames)
    if "efit_lineage" in frame.columns:
        frame["efit_lineage"] = frame["efit_lineage"].replace(LINEAGE_SPELLINGS)
    return frame


def normalised_value(column: str, value: Any) -> Any:
    if column == "efit_lineage" and isinstance(value, str):
        return LINEAGE_SPELLINGS.get(value, value)
    return value


def _pattern_regex(template: str, patterns: dict[str, str]) -> re.Pattern | None:
    if "{" not in template:
        return None
    regex = re.escape(template)
    for placeholder, choices in patterns.items():
        options = "|".join(re.escape(choice.strip()) for choice in str(choices).split("|"))
        regex = regex.replace(re.escape(placeholder), f"(?:{options})")
    return re.compile(f"^{regex}$")


def column_descriptions(table: AtlasTable, schema: dict | None) -> dict[str, dict[str, Any]]:
    """Column name -> {unit, definition, ...} as the lane's schema states it, in v1 spelling."""
    if not schema:
        return {}
    kind = table.schema_kind
    described: dict[str, dict[str, Any]] = {}
    if kind == "json_schema":
        for name, entry in (schema.get("properties") or {}).items():
            row = {"definition": entry.get("description", "")}
            if "type" in entry:
                row["type"] = entry["type"]
            if "enum" in entry:
                row["values"] = entry["enum"]
            described[name] = row
    elif kind == "columns":
        for name, entry in (schema.get(table.columns_entry) or {}).items():
            described[name] = dict(entry) if isinstance(entry, dict) else {"definition": str(entry)}
    elif kind == "column_dictionary":
        for name, entry in (schema.get("column_dictionary") or {}).items():
            row = dict(entry) if isinstance(entry, dict) else {"definition": str(entry)}
            if "meaning" in row:
                row["definition"] = row.pop("meaning")
            described[name] = row
    elif kind == "manifest_units":
        for name, unit in (schema.get("units") or {}).items():
            described[name] = {"unit": unit}
    return {normalised_column(name): row for name, row in described.items()}


def describe_column(name: str, described: dict[str, dict], patterns: dict[str, str]) -> dict[str, Any]:
    """One column's description; templated entries (``dcon_{t}_{r}_W_t``) match by pattern."""
    if name in described:
        return described[name]
    for template, row in described.items():
        regex = _pattern_regex(template, patterns)
        if regex is not None and regex.match(name):
            return {**row, "pattern": template}
    return {}


def file_sha256(path: Path, _cache: dict = {}) -> str:  # noqa: B006 - per-process cache by (path, mtime, size)
    stat = path.stat()
    token = (str(path), stat.st_mtime_ns, stat.st_size)
    if token not in _cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
        _cache[token] = digest.hexdigest()
    return _cache[token]


def read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def manifest_provenance(manifest: dict | None) -> dict[str, Any]:
    if not manifest:
        return {}
    return {key: manifest[key] for key in PROVENANCE_KEYS if key in manifest}
