"""Public multi-machine databases mapped into common VAFT semantics (#1205).

Source-specific readers (:mod:`.itpa_hmode`, :mod:`.tcv_lh`, :mod:`.itpa_tc26`) normalise
published databases into canonical tables (:mod:`.schema`); profile databases
(:mod:`.itpa_profile`, PR08) map into ODS instead; VEST enters the same tables through
:mod:`.vest_confinement`; :mod:`.analysis` and :mod:`vaft.plot.population`
work on the canonical tables only.  Files are fetched on demand with a pinned
checksum (:mod:`._fetch`) and are never shipped with VAFT.
"""

from importlib import import_module

__all__ = [
    "MHD_STATE_COLUMNS",
    "PR08_MACHINES",
    "PROVENANCE_KINDS",
    "fetch_pr08_population",
    "mhd_state_coverage",
    "pr08_inventory",
    "pr08_mhd_state_table",
    "pr08_release_inventory",
    "projection_coverage",
    "read_pr08_zero_d",
    "CONFINEMENT_COLUMNS",
    "ChecksumError",
    "FetchError",
    "OPTIONAL_CONFINEMENT_COLUMNS",
    "SOURCES",
    "TRANSITION_COLUMNS",
    "confinement_coverage",
    "empty_confinement_table",
    "empty_transition_table",
    "fetch_pr08",
    "fetch_source",
    "h_factor",
    "load_vest_tier_a_confinement",
    "normalize_db5",
    "normalize_tc26",
    "normalize_tcv_lh",
    "pr08_mapping_coverage",
    "pr08_to_omas",
    "predict_confinement_time",
    "read_db5",
    "read_pr08",
    "read_tc26",
    "read_tcv_lh",
    "transition_margin",
    "validate_confinement_table",
    "validate_transition_table",
    "vest_ods_to_confinement_rows",
    "vest_summary_to_confinement_table",
]

_EXPORT_MAP = {
    "MHD_STATE_COLUMNS": (".pr08_mhd_state", "MHD_STATE_COLUMNS"),
    "PR08_MACHINES": (".pr08_mhd_state", "PR08_MACHINES"),
    "PROVENANCE_KINDS": (".pr08_mhd_state", "PROVENANCE_KINDS"),
    "fetch_pr08_population": (".pr08_mhd_state", "fetch_pr08_population"),
    "mhd_state_coverage": (".pr08_mhd_state", "mhd_state_coverage"),
    "pr08_inventory": (".pr08_mhd_state", "pr08_inventory"),
    "pr08_mhd_state_table": (".pr08_mhd_state", "pr08_mhd_state_table"),
    "pr08_release_inventory": (".pr08_mhd_state", "pr08_release_inventory"),
    "projection_coverage": (".pr08_mhd_state", "projection_coverage"),
    "read_pr08_zero_d": (".pr08_mhd_state", "read_pr08_zero_d"),
    "CONFINEMENT_COLUMNS": (".schema", "CONFINEMENT_COLUMNS"),
    "OPTIONAL_CONFINEMENT_COLUMNS": (".schema", "OPTIONAL_CONFINEMENT_COLUMNS"),
    "empty_confinement_table": (".schema", "empty_confinement_table"),
    "validate_confinement_table": (".schema", "validate_confinement_table"),
    "TRANSITION_COLUMNS": (".schema", "TRANSITION_COLUMNS"),
    "empty_transition_table": (".schema", "empty_transition_table"),
    "validate_transition_table": (".schema", "validate_transition_table"),
    "read_tcv_lh": (".tcv_lh", "read_tcv_lh"),
    "read_tc26": (".itpa_tc26", "read_tc26"),
    "read_pr08": (".itpa_profile", "read_pr08"),
    "fetch_pr08": (".itpa_profile", "fetch_pr08"),
    "pr08_to_omas": (".itpa_profile", "pr08_to_omas"),
    "pr08_mapping_coverage": (".itpa_profile", "pr08_mapping_coverage"),
    "normalize_tc26": (".itpa_tc26", "normalize_tc26"),
    "normalize_tcv_lh": (".tcv_lh", "normalize_tcv_lh"),
    "transition_margin": (".analysis", "transition_margin"),
    "ChecksumError": ("._fetch", "ChecksumError"),
    "FetchError": ("._fetch", "FetchError"),
    "SOURCES": ("._fetch", "SOURCES"),
    "fetch_source": ("._fetch", "fetch_source"),
    "read_db5": (".itpa_hmode", "read_db5"),
    "normalize_db5": (".itpa_hmode", "normalize_db5"),
    "vest_ods_to_confinement_rows": (".vest_confinement", "vest_ods_to_confinement_rows"),
    "load_vest_tier_a_confinement": (".vest_confinement", "load_vest_tier_a_confinement"),
    "vest_summary_to_confinement_table": (".vest_confinement", "vest_summary_to_confinement_table"),
    "predict_confinement_time": (".analysis", "predict_confinement_time"),
    "h_factor": (".analysis", "h_factor"),
    "confinement_coverage": (".analysis", "confinement_coverage"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORT_MAP[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
