"""Population V&V of the kinetic-state atlas: input contract, evidence tables, summaries.

:mod:`vaft.validation.kinetic_state` computes the per-slice comparison of
Thomson and EFIT pressure under the atlas state key ``(shot, time_efit_s,
efit_lineage)``. This module takes the *population* those rows
form -- the atlas ``state`` and ``profiles`` tables and the virial-identity
table keyed by the same state -- and does two things a V&V notebook would
otherwise do inline with ``assert``:

**A. The population input contract** (:func:`assemble_kinetic_population`).
The tables are joined under explicit checks, and every finding is a row of an
*issues* table (key, check, status, detail) rather than an assertion. A broken
contract -- a duplicate state key, a missing required column -- raises
:class:`PopulationContractError`; everything else (a profile key the state
does not hold, a matched state without channels, a Thomson time outside the
tolerance, a virial row of another EFIT product, a manifest row count that
disagrees) is recorded, and with ``strict=True`` any failure raises.

**B. Evidence tables and summaries** (:func:`pressure_evidence`,
:func:`virial_evidence`, :func:`population_evidence_summary`). One row per
(state, quantity), each with a status of ``available``, ``unavailable`` (the
input does not exist, e.g. no Thomson match) or ``failed`` (the input exists
but violates the contract), so a summary counts every state and not only the
successful joins. Three things are kept apart throughout:

* **Pressure support.** ``r_sum`` (Thomson channel sums), ``r_w`` (volume
  integral over the flux span the channels cover) and ``r_w_full`` (the whole
  plasma, extrapolating the profile fit) are distinct quantities with their
  support named, never merged into one ratio.
* **Validation role.** A magnetics-only EFIT row compares a reconstruction
  with Thomson data it never used (``held_out_thomson``); an electron-kinetic
  row was fitted *to* Thomson, so it is a consistency check
  (``thomson_constrained_consistency``). The role is on every row.
* **Virial evidence.** The direct identity residuals E1/E2/E3 and their RMS,
  the volume-integral and pair-13 closure values of beta_p and l_i, and the
  study's ``virial_status`` verdict stay separate fields. Nothing here grades:
  no threshold, no aggregate score (Open: #547 -- no qualified virial
  threshold exists yet).

FileDB traversal and snapshot production stay in ``workflow/kinetic_state``;
this module reads tables only.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from vaft.validation.kinetic_state import INDEPENDENT_LINEAGES, LINEAGES

__all__ = [
    "STATE_KEY",
    "VALIDATION_ROLES",
    "PRESSURE_RATIOS",
    "VIRIAL_QUANTITIES",
    "CONDITIONING_QUANTITIES",
    "EVIDENCE_STATUSES",
    "ISSUE_COLUMNS",
    "PopulationContractError",
    "KineticPopulation",
    "validation_role",
    "assemble_kinetic_population",
    "read_kinetic_population",
    "pressure_evidence",
    "virial_evidence",
    "population_evidence_summary",
]

#: The atlas state key: one row per reconstructed slice and lineage.
STATE_KEY = ("shot", "time_efit_s", "efit_lineage")

#: Validation role of a lineage. The magnetics reconstruction never saw the
#: Thomson data, so comparing them is held-out validation; the electron-kinetic
#: reconstruction was fitted to Thomson, so agreement is fit consistency.
VALIDATION_ROLES: Mapping[str, str] = {
    lineage: ("held_out_thomson" if lineage in INDEPENDENT_LINEAGES else "thomson_constrained_consistency")
    for lineage in LINEAGES
}

#: Pressure ratios p_EFIT / p_e and the support each one is taken over.
PRESSURE_RATIOS: Mapping[str, str] = {
    "r_sum": "Thomson channel sums",
    "r_w": "volume over the flux span the Thomson channels cover",
    "r_w_full": "whole plasma, extrapolating the profile fit",
}

#: Virial quantities by family: the direct identities (verification of the
#: reconstruction), and beta_p / l_i from the volume integral and from the
#: pair-13 closure (two estimates of one quantity, not a verdict).
VIRIAL_QUANTITIES: Mapping[str, str] = {
    "e1_normalized": "identity",
    "e2_normalized": "identity",
    "e3_normalized": "identity",
    "rms": "identity",
    "beta_p_volume": "volume_integral",
    "beta_p_pair_13": "pair_13_closure",
    "li_volume": "volume_integral",
    "li_pair_13": "pair_13_closure",
}

# Optional in older Atlas exports. These are evidence of inversion geometry,
# not acceptance criteria; the formula layer supplies every value.
CONDITIONING_QUANTITIES = (
    "alpha",
    *(f"{closure}_denominator" for closure in ("pair_13", "pair_23", "lao_li", "full_123")),
    *(f"{closure}_distance_to_singularity" for closure in ("pair_13", "pair_23", "lao_li", "full_123")),
    "rt_denominator_ratio", "rt_over_r0_available",
)

EVIDENCE_STATUSES = ("available", "unavailable", "failed")
#: ``scope`` is ``state`` (keyed) or ``population`` (empty key: a manifest, schema
#: or table-level finding), so a key-wise groupby cannot silently drop the latter.
ISSUE_COLUMNS = ("scope", *STATE_KEY, "check", "status", "detail")

#: Required columns and the unit each carries (``""`` text, ``"-"`` dimensionless).
STATE_COLUMNS: Mapping[str, str] = {
    "shot": "-", "time_efit_s": "s", "efit_lineage": "", "efit_quality": "",
    "efit_product": "", "efit_product_sha256": "", "ts_status": "",
    "dt_ts_efit_s": "s", "ts_tolerance_s": "s",
    "r_sum": "-", "r_w": "-", "r_w_full": "-", "virial_status": "",
}
PROFILE_COLUMNS: Mapping[str, str] = {
    "shot": "-", "time_efit_s": "s", "efit_lineage": "", "channel": "-",
    "p_e_pa": "Pa", "p_efit_pa": "Pa", "r_p": "-",
}
VIRIAL_COLUMNS: Mapping[str, str] = {
    "shot": "-", "time_efit_s": "s", "efit_lineage": "", "efit_product": "", "efit_product_sha256": "",
    "identities_available": "-", **{name: "-" for name in VIRIAL_QUANTITIES},
}

_TIME_DIGITS = 4  # the atlas rounds time_efit_s to 1e-4 s (kinetic_state.state_time)
_UNIT_IN_DESCRIPTION = re.compile(r"\[([^\]]+)\]\s*$")


class PopulationContractError(ValueError):
    """The atlas tables break the population contract (duplicate keys, missing columns, or a strict failure)."""


@dataclass(frozen=True)
class KineticPopulation:
    """The checked population: the joined state table, its profiles, and every finding.

    ``population`` has one row per atlas state, the virial columns joined on
    the state key, ``validation_role`` and ``virial_join`` (``joined``,
    ``not_supplied``, ``missing`` or ``product_mismatch``; virial values are
    NaN unless ``joined``). ``profiles`` holds the Thomson channels whose key
    is a matched state. ``issues`` has :data:`ISSUE_COLUMNS`; a
    population-level finding has an empty key. ``provenance`` records the
    manifest identity, the row counts and the tolerances used.
    """

    population: Any
    profiles: Any
    issues: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        """True when any finding has status ``fail``."""
        return bool((self.issues["status"] == "fail").any())


def validation_role(lineage: str) -> str:
    """The validation role of an EFIT lineage; raises for an unknown one."""
    try:
        return VALIDATION_ROLES[str(lineage)]
    except KeyError:
        raise ValueError(f"unknown efit_lineage {lineage!r}; expected one of {', '.join(LINEAGES)}") from None


def _frame(table: Any, name: str):
    import pandas as pd

    if table is None:
        return None
    if isinstance(table, (str, Path)):
        return pd.read_csv(table)
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame(table)


def _require(frame, columns: Mapping[str, str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise PopulationContractError(f"{name} table lacks required column(s): {', '.join(missing)}")


def _normalise_key(frame, name: str) -> None:
    """Integer shot, time rounded to the atlas key (``kinetic_state.state_time``: 1e-4 s), text lineage.

    Producers write rounded times; a time that was not rounded and sits on a
    rounding boundary would land on a neighbouring key and be reported missing.
    """
    import pandas as pd

    for column in ("shot", "time_efit_s"):
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise PopulationContractError(f"{name} table has a missing or non-numeric {column} in the state key")
        frame[column] = values
    frame["shot"] = frame["shot"].astype(int)
    frame["time_efit_s"] = frame["time_efit_s"].astype(float).round(_TIME_DIGITS)
    frame["efit_lineage"] = frame["efit_lineage"].astype(str)


def _reject_duplicates(frame, name: str) -> None:
    duplicated = frame.duplicated(list(STATE_KEY), keep=False)
    if duplicated.any():
        keys = sorted({tuple(row) for row in frame.loc[duplicated, list(STATE_KEY)].itertuples(index=False)})
        raise PopulationContractError(f"duplicate {name} state key(s): {keys[:5]}" + (" ..." if len(keys) > 5 else ""))


def _key(row) -> tuple:
    return (int(row["shot"]), float(row["time_efit_s"]), str(row["efit_lineage"]))


def _text(value: Any) -> str:
    """A table cell as text, ``""`` for an empty cell (NaN or None after a CSV read)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _schema_issues(schema: Mapping[str, Any], columns: Mapping[str, str], table: str) -> list[dict]:
    properties = schema.get("properties", {}) if isinstance(schema, Mapping) else {}
    issues = []
    for column, unit in columns.items():
        if column not in properties:
            issues.append(_issue(None, "schema_declares_column", "fail", f"{table}.{column} is not in the schema"))
            continue
        found = _UNIT_IN_DESCRIPTION.search(str(properties[column].get("description", "")))
        if found and unit not in ("", "-") and found.group(1).strip() != unit:
            issues.append(_issue(None, "unit", "fail",
                                 f"{table}.{column}: schema says [{found.group(1).strip()}], contract expects [{unit}]"))
    return issues


def _issue(key: Optional[tuple], check: str, status: str, detail: str) -> dict:
    shot, time_s, lineage = key if key is not None else (None, None, None)
    return {"scope": "state" if key is not None else "population", "shot": shot, "time_efit_s": time_s,
            "efit_lineage": lineage, "check": check, "status": status, "detail": detail}


def assemble_kinetic_population(
    state: Any,
    profiles: Any,
    virial: Any = None,
    *,
    manifest: Optional[Mapping[str, Any]] = None,
    schemas: Optional[Mapping[str, Mapping[str, Any]]] = None,
    strict: bool = False,
) -> KineticPopulation:
    """Join the atlas state, profile and virial tables under the population contract.

    Parameters
    ----------
    state, profiles : pandas.DataFrame or path
        The atlas ``state`` and ``profiles`` tables (state key contract v1).
    virial : pandas.DataFrame or path, optional
        The virial-identity table keyed by the state key and carrying the EFIT
        product it was computed from; ``None`` records every virial value as
        not supplied.
    manifest : mapping, optional
        The atlas ``MANIFEST.json``; its ``rows`` and ``contract_version`` are checked.
    schemas : mapping, optional
        ``{"state": ..., "profiles": ...}`` JSON schemas; required columns must
        be declared, and a unit in a description's trailing ``[...]`` must match.
    strict : bool, optional
        Raise :class:`PopulationContractError` when any finding is a failure.

    Returns
    -------
    KineticPopulation
        The joined population, the matched profiles, the issues table and provenance.

    Raises
    ------
    PopulationContractError
        A required column is missing, a state key is duplicated in ``state`` or
        ``virial``, or (``strict``) any finding failed.
    """
    import pandas as pd

    state_df, profile_df, virial_df = _frame(state, "state"), _frame(profiles, "profiles"), _frame(virial, "virial")
    _require(state_df, STATE_COLUMNS, "state")
    _require(profile_df, PROFILE_COLUMNS, "profiles")
    _normalise_key(state_df, "state")
    _normalise_key(profile_df, "profiles")
    _reject_duplicates(state_df, "state")
    unknown = sorted(set(state_df["efit_lineage"]) - set(LINEAGES))
    if unknown:
        raise PopulationContractError(f"unknown efit_lineage value(s) in state: {unknown}")

    issues: list[dict] = []
    state_keys = {_key(row) for _, row in state_df.iterrows()}
    matched = {_key(row) for _, row in state_df.loc[state_df["ts_status"] == "matched"].iterrows()}

    # Profiles: every channel must belong to a matched state; every matched state needs channels.
    profile_keys = [_key(row) for _, row in profile_df.iterrows()]
    keep = []
    for key in profile_keys:
        if key not in state_keys:
            keep.append(False)
        elif key not in matched:
            keep.append(False)
        else:
            keep.append(True)
    for key in sorted(set(k for k, ok in zip(profile_keys, keep) if not ok)):
        reason = ("profile_key_in_state", "no such state") if key not in state_keys else \
            ("profile_of_unmatched_state", "the state is not Thomson-matched")
        issues.append(_issue(key, reason[0], "fail", reason[1]))
    profile_df = profile_df.loc[keep].reset_index(drop=True)
    with_channels = set(map(tuple, profile_df[list(STATE_KEY)].itertuples(index=False)))
    for key in sorted(matched - with_channels):
        issues.append(_issue(key, "missing_channels", "fail", "Thomson-matched state without profile channels"))
    for key, group in profile_df.groupby(list(STATE_KEY)):
        bad = int((~np.isfinite(group[["p_e_pa", "p_efit_pa"]].to_numpy(float))).any(axis=1).sum())
        if bad:
            issues.append(_issue(tuple(key), "channel_values", "not_available",
                                 f"{bad} of {len(group)} channels lack a finite p_e or p_EFIT"))

    # Thomson time within the tolerance the atlas recorded for the match.
    tol_fail = set()
    for _, row in state_df.loc[state_df["ts_status"] == "matched"].iterrows():
        dt, tol = row["dt_ts_efit_s"], row["ts_tolerance_s"]
        if not (_finite(dt) and _finite(tol)) or abs(float(dt)) > float(tol) + 1e-12:
            tol_fail.add(_key(row))
            issues.append(_issue(_key(row), "ts_time_tolerance", "fail", f"|dt_ts_efit_s| = {dt} > {tol} s"))

    # Virial join: one-to-one on the state key, same EFIT product and hash.
    joined = state_df.copy()
    for column in ("efit_product_virial", "efit_product_sha256_virial"):
        joined[column] = pd.Series([None] * len(joined), index=joined.index, dtype=object)
    for column in ("identities_available", *VIRIAL_QUANTITIES, *CONDITIONING_QUANTITIES):
        joined[column] = np.nan
    joined["virial_join"] = "not_supplied"
    if virial_df is None:
        issues.append(_issue(None, "virial_join", "not_available", "no virial table supplied"))
    else:
        _require(virial_df, VIRIAL_COLUMNS, "virial")
        for column in CONDITIONING_QUANTITIES:
            if column not in virial_df:
                virial_df[column] = np.nan
        virial_df["rt_over_r0_available"] = virial_df["rt_over_r0_available"].map(
            lambda value: {"true": 1, "false": 0}.get(str(value).lower(), value))
        _normalise_key(virial_df, "virial")
        virial_df["identities_available"] = pd.to_numeric(virial_df["identities_available"], errors="coerce")
        for column in (*VIRIAL_QUANTITIES, *CONDITIONING_QUANTITIES):
            virial_df[column] = pd.to_numeric(virial_df[column], errors="coerce").astype(float)
        _reject_duplicates(virial_df, "virial")
        by_key = {_key(row): row for _, row in virial_df.iterrows()}
        for key in sorted(set(by_key) - state_keys):
            issues.append(_issue(key, "virial_key_in_state", "fail", "virial row for a key the state does not hold"))
        for i, row in joined.iterrows():
            key = _key(row)
            other = by_key.get(key)
            if other is None:
                joined.at[i, "virial_join"] = "missing"
                issues.append(_issue(key, "virial_join", "not_available", "no virial row for this state"))
                continue
            joined.at[i, "efit_product_virial"] = other["efit_product"]
            joined.at[i, "efit_product_sha256_virial"] = other["efit_product_sha256"]
            identity = [_text(other["efit_product"]), _text(other["efit_product_sha256"]),
                        _text(row["efit_product"]), _text(row["efit_product_sha256"])]
            if not all(identity) or identity[:2] != identity[2:]:  # an absent hash proves nothing
                joined.at[i, "virial_join"] = "product_mismatch"
                issues.append(_issue(key, "efit_product_identity", "fail",
                                     f"virial from {other['efit_product']} ({str(other['efit_product_sha256'])[:12]}), "
                                     f"state from {row['efit_product']} ({str(row['efit_product_sha256'])[:12]})"))
                continue
            joined.at[i, "virial_join"] = "joined"
            for column in ("identities_available", *VIRIAL_QUANTITIES, *CONDITIONING_QUANTITIES):
                joined.at[i, column] = other[column]
            missing = [c for c in VIRIAL_QUANTITIES if not _finite(other[c])]
            if missing:
                issues.append(_issue(key, "virial_values", "not_available", "not finite: " + ", ".join(missing)))
    joined["validation_role"] = joined["efit_lineage"].map(VALIDATION_ROLES)
    joined["ts_time_within_tolerance"] = [
        (None if _key(row) not in matched else _key(row) not in tol_fail) for _, row in joined.iterrows()
    ]
    channels = profile_df.groupby(list(STATE_KEY)).size().to_dict()
    joined["ts_channels"] = [int(channels.get(_key(row), 0)) for _, row in joined.iterrows()]
    for _, row in joined.loc[joined["ts_status"] != "matched"].iterrows():
        stray = [ratio for ratio in PRESSURE_RATIOS if _finite(row[ratio])]
        if stray:
            issues.append(_issue(_key(row), "ratio_without_match", "fail",
                                 f"{', '.join(stray)} finite on a state with ts_status {row['ts_status']}"))

    # Manifest and schemas.
    provenance: dict[str, Any] = {"rows": {"state": int(len(state_df)), "profiles_matched": int(len(profile_df)),
                                           "virial": None if virial_df is None else int(len(virial_df))},
                                  "time_key_digits": _TIME_DIGITS}
    if manifest is not None:
        rows = manifest.get("rows", {})
        for table, actual in (("state", len(state_df)), ("profiles", len(profile_keys))):
            if table in rows and int(rows[table]) != int(actual):
                issues.append(_issue(None, "manifest_rows", "fail",
                                     f"{table}: manifest says {rows[table]}, table has {actual}"))
        version = manifest.get("contract_version")
        if version is not None and "contract_version" in state_df.columns:
            versions = {str(v) for v in state_df["contract_version"].dropna().unique()}
            if versions and versions != {str(version)}:
                issues.append(_issue(None, "contract_version", "fail",
                                     f"manifest {version}, state rows {sorted(versions)}"))
        provenance.update({k: manifest.get(k) for k in ("contract_version", "generated_at", "vaft_git", "criteria_sha")})
    for table, columns in (("state", STATE_COLUMNS), ("profiles", PROFILE_COLUMNS)):
        if schemas and table in schemas:
            issues.extend(_schema_issues(schemas[table], columns, table))

    issue_df = pd.DataFrame(issues, columns=list(ISSUE_COLUMNS))
    result = KineticPopulation(population=joined.reset_index(drop=True), profiles=profile_df, issues=issue_df,
                               provenance=provenance)
    if strict and result.failed:
        failed = issue_df.loc[issue_df["status"] == "fail"]
        raise PopulationContractError(
            f"{len(failed)} contract failure(s): " + "; ".join(
                f"{r.check} {'' if r.scope == 'population' else (r.shot, r.time_efit_s, r.efit_lineage)}: {r.detail}"
                for r in failed.head(5).itertuples())
        )
    return result


def read_kinetic_population(
    atlas: str | Path,
    *,
    virial: Any = None,
    strict: bool = False,
) -> KineticPopulation:
    """Read an atlas directory (``v1/state.csv``, ``profiles.csv``, ``MANIFEST.json``, ``schema/``) and assemble it.

    ``virial`` defaults to ``v1/virial_identity.csv`` when that file exists.
    Only the published atlas layout is read; producing a virial table from a
    FileDB is the workflow's job (``workflow/kinetic_state``).
    """
    root = Path(atlas).expanduser() / "v1"
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8")) if (root / "MANIFEST.json").is_file() \
        else None
    schemas = {name: json.loads((root / "schema" / f"{name}.schema.json").read_text(encoding="utf-8"))
               for name in ("state", "profiles") if (root / "schema" / f"{name}.schema.json").is_file()}
    if virial is None and (root / "virial_identity.csv").is_file():
        virial = root / "virial_identity.csv"
    return assemble_kinetic_population(root / "state.csv", root / "profiles.csv", virial, manifest=manifest,
                                       schemas=schemas or None, strict=strict)


def _outside_tolerance(flag: Any) -> bool:
    """The tolerance flag is None for an unmatched state, else a (possibly numpy) bool."""
    return flag is not None and not (isinstance(flag, float) and math.isnan(flag)) and not bool(flag)


def pressure_evidence(population: KineticPopulation):
    """One row per (state, pressure ratio): value, support, validation role and status.

    Status ``unavailable`` where the state has no Thomson match (``reason``
    is the atlas ``ts_reason`` when present), ``failed`` where it was matched
    outside the tolerance, has no profile channels, or the ratio is not a
    finite positive number, else ``available``. ``value`` is NaN unless
    available. ``r_sum``, ``r_w`` and ``r_w_full`` stay separate rows.
    """
    import pandas as pd

    pop = population.population
    rows = []
    for _, row in pop.iterrows():
        matched = row["ts_status"] == "matched"
        for ratio, support in PRESSURE_RATIOS.items():
            value = row[ratio]
            if not matched:
                status, reason = "unavailable", _text(row.get("ts_reason")) or f"ts_status {row['ts_status']}"
            elif _outside_tolerance(row["ts_time_within_tolerance"]):
                status, reason = "failed", "Thomson time outside the matching tolerance"
            elif int(row["ts_channels"]) == 0:
                status, reason = "failed", "Thomson-matched state without profile channels"
            elif not (_finite(value) and float(value) > 0):
                status = "failed"
                reason = (_text(row.get("r_w_reason")) if ratio != "r_sum" else "") or "not a finite positive ratio"
            else:
                status, reason = "available", ""
            rows.append({**{k: row[k] for k in STATE_KEY}, "efit_quality": row["efit_quality"],
                         "validation_role": row["validation_role"], "family": "pressure", "quantity": ratio,
                         "support": support, "value": float(value) if status == "available" else np.nan,
                         "status": status, "reason": reason})
    return pd.DataFrame(rows)


def virial_evidence(population: KineticPopulation):
    """One row per (state, virial quantity): value, family, the study verdict, validation role and status.

    ``family`` is ``identity`` (E1/E2/E3 and their RMS), ``volume_integral``
    or ``pair_13_closure``, with optional ``conditioning`` evidence (RT
    availability is 1/0, not a pass/fail verdict); ``virial_status`` is the study's own verdict,
    carried alongside and never recomputed. Status ``unavailable`` where no
    virial row was supplied or joined or the value is not finite, ``failed``
    where the virial row came from another EFIT product.
    """
    import pandas as pd

    pop = population.population
    rows = []
    for _, row in pop.iterrows():
        join = row["virial_join"]
        quantities = {**VIRIAL_QUANTITIES, **dict.fromkeys(CONDITIONING_QUANTITIES, "conditioning")}
        for quantity, family in quantities.items():
            value = row[quantity]
            if join == "product_mismatch":
                status, reason = "failed", "virial row from another EFIT product"
            elif join != "joined":
                status, reason = "unavailable", f"virial {join.replace('_', ' ')}"
            elif not _finite(value):
                status, reason = "unavailable", "not finite"
            else:
                status, reason = "available", ""
            verdict = row["virial_status"]
            rows.append({**{k: row[k] for k in STATE_KEY}, "efit_quality": row["efit_quality"],
                         "validation_role": row["validation_role"], "family": family, "quantity": quantity,
                         "value": float(value) if status == "available" else np.nan,
                         "identities_available": row["identities_available"],
                         "virial_status": verdict if isinstance(verdict, str) else "not_available",
                         "status": status, "reason": reason})
    return pd.DataFrame(rows)


def population_evidence_summary(population: KineticPopulation, *, by: Sequence[str] = ("efit_lineage",)):
    """Counts and quartiles of every evidence quantity, by lineage (or ``by``) and validation role.

    Every state is counted: ``states = available + unavailable + failed``.
    ``median``, ``q25`` and ``q75`` are over the available values only.
    """
    import pandas as pd

    evidence = pd.concat([pressure_evidence(population), virial_evidence(population)], ignore_index=True)
    keys = [*by, "validation_role", "family", "quantity"]
    rows = []
    for group_key, group in evidence.groupby(keys, sort=True):
        values = group.loc[group["status"] == "available", "value"].astype(float)
        counts = group["status"].value_counts()
        rows.append({**dict(zip(keys, group_key)), "states": int(len(group)),
                     **{status: int(counts.get(status, 0)) for status in EVIDENCE_STATUSES},
                     "median": float(values.median()) if len(values) else np.nan,
                     "q25": float(values.quantile(0.25)) if len(values) else np.nan,
                     "q75": float(values.quantile(0.75)) if len(values) else np.nan})
    return pd.DataFrame(rows)
