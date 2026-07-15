"""Lazy namespace for external fusion-code adapters."""

from importlib import import_module

__all__ = [
    "CodeConfig",
    "CodeInputs",
    "CodeResult",
    "CodeRunner",
    "CHEASEConfig",
    "CHEASEInputs",
    "CHEASEResult",
    "EFITConfig",
    "EFITInputs",
    "EFITResult",
    "GPECCaseInputs",
    "GPECModuleRun",
    "GPECSuiteConfig",
    "GPECSuiteResult",
    "base",
    "chease",
    "collect_efit_outputs",
    "collect_gpec_suite_outputs",
    "efit",
    "format_gfile_header_for_gpec",
    "find_chease_executable",
    "generate_constraints_ods",
    "generate_kfile",
    "gfile_to_omas",
    "gpec",
    "init_snakemake_logger",
    "prepare_efit_inputs",
    "prepare_chease_inputs",
    "prepare_gpec_suite_case",
    "refine_equilibrium",
    "run_efit",
    "run_chease",
    "run_gpec",
    "run_gpec_suite_case",
    "snakemake",
    "tes",
    "TESConfig",
    "TESInputs",
    "TESResult",
    "prepare_tes_inputs",
    "run_tes",
    "collect_tes_outputs",
    "parse_result_scalars",
    "parse_result_coils",
    "scan_tes",
]

_EXPORT_MAP = {
    "CodeConfig": (".base", "CodeConfig"),
    "CodeInputs": (".base", "CodeInputs"),
    "CodeResult": (".base", "CodeResult"),
    "CodeRunner": (".base", "CodeRunner"),
    "CHEASEConfig": (".chease", "CHEASEConfig"),
    "CHEASEInputs": (".chease", "CHEASEInputs"),
    "CHEASEResult": (".chease", "CHEASEResult"),
    "EFITConfig": (".efit", "EFITConfig"),
    "EFITInputs": (".efit", "EFITInputs"),
    "EFITResult": (".efit", "EFITResult"),
    "GPECCaseInputs": (".gpec", "GPECCaseInputs"),
    "GPECModuleRun": (".gpec", "GPECModuleRun"),
    "GPECSuiteConfig": (".gpec", "GPECSuiteConfig"),
    "GPECSuiteResult": (".gpec", "GPECSuiteResult"),
    "find_chease_executable": (".chease", "find_chease_executable"),
    "collect_efit_outputs": (".efit", "collect_efit_outputs"),
    "collect_gpec_suite_outputs": (".gpec", "collect_gpec_suite_outputs"),
    "format_gfile_header_for_gpec": (".gpec", "format_gfile_header_for_gpec"),
    "generate_constraints_ods": (".efit", "generate_constraints_ods"),
    "generate_kfile": (".efit", "generate_kfile"),
    "gfile_to_omas": (".efit", "gfile_to_omas"),
    "init_snakemake_logger": (".snakemake", "init_snakemake_logger"),
    "prepare_chease_inputs": (".chease", "prepare_chease_inputs"),
    "prepare_efit_inputs": (".efit", "prepare_efit_inputs"),
    "prepare_gpec_suite_case": (".gpec", "prepare_gpec_suite_case"),
    "refine_equilibrium": (".chease", "refine_equilibrium"),
    "run_efit": (".efit", "run_efit"),
    "run_chease": (".chease", "run_chease"),
    "run_gpec": (".gpec", "run_gpec"),
    "run_gpec_suite_case": (".gpec", "run_gpec_suite_case"),
    "TESConfig": (".tes", "TESConfig"),
    "TESInputs": (".tes", "TESInputs"),
    "TESResult": (".tes", "TESResult"),
    "prepare_tes_inputs": (".tes", "prepare_tes_inputs"),
    "run_tes": (".tes", "run_tes"),
    "collect_tes_outputs": (".tes", "collect_tes_outputs"),
    "parse_result_scalars": (".tes", "parse_result_scalars"),
    "parse_result_coils": (".tes", "parse_result_coils"),
    "scan_tes": (".tes", "scan_tes"),
}


def __getattr__(name: str):
    if name in {"base", "efit", "gpec", "chease", "snakemake", "tes"}:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORT_MAP[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + __all__)
