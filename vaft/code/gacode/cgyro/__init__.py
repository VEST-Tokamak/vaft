"""CGYRO: the GACODE suite's local delta-f continuum gyrokinetic solver (#1354).

    GACODEProfile -> TGLFInput(r/a) -> CGYROInput -> input.cgyro -> CGYRO -> CgyroOutputs
                                                                              |
                                                      solver-native source of truth

CGYRO's local input is a renaming of TGLF's (:mod:`~vaft.code.gacode.cgyro.inputs`), so
the two codes see one surface and a CGYRO-TGLF comparison isolates the model, not the
projection. CGYRO's own ``PROFILE_MODEL=2`` projection is the oracle the renaming is
held against (:func:`compare_with_oracle`).

Formalism (#1353 §14): delta-f, local flux tube, continuum, closed flux surface, Miller
geometry, kinetic electrons; the field model and linear/nonlinear regime come from the
configuration and are recorded with every run (:func:`formalism`).

Locality QA (:mod:`~vaft.code.gacode.cgyro.locality`): every run can be judged for box
adequacy (``l_corr/L_x``) and for the local approximation itself
(``epsilon_local = l_corr / min(L_Ti, L_Te, L_n, L_q)``), separately -- a box comparable
to the minor radius on VEST (``a/rho_s ~ 50``) is not by itself a verdict.

Importing this package does not need GACODE.
"""

from __future__ import annotations

from ._types import (
    FIELD_MODELS,
    MAX_SPECIES,
    TGLF_FIELD_MODEL,
    CGYROConfig,
    CGYROResult,
    formalism,
)
from .inputs import (
    CGYROInput,
    CGYROInputs,
    LocalConversionError,
    cgyro_input_from_tglf,
    cgyro_parameters,
    compare_with_oracle,
    input_sha256,
    prepare_cgyro_case,
    prepare_cgyro_input,
    stage_cgyro_case,
    write_input_cgyro,
)
from .outputs import (
    FIELD_NAMES,
    FLUX_MOMENTS,
    FREQUENCY_SIGN_CONVENTION,
    SCHEMA,
    SCHEMA_VERSION,
    CgyroOutputs,
    collect_cgyro_outputs,
)
from .locality import (
    BOX_THRESHOLDS,
    LOCALITY_THRESHOLDS,
    locality_report,
    radial_correlation_length,
    scale_lengths,
)
from .runner import (
    CGYROExecutionError,
    gacode_revision,
    read_cgyro_case,
    run_cgyro,
    run_cgyro_case,
)

__all__ = [
    "BOX_THRESHOLDS",
    "LOCALITY_THRESHOLDS",
    "locality_report",
    "radial_correlation_length",
    "scale_lengths",
    "CGYROConfig",
    "CGYROExecutionError",
    "CGYROInput",
    "CGYROInputs",
    "CGYROResult",
    "CgyroOutputs",
    "FIELD_MODELS",
    "FIELD_NAMES",
    "FLUX_MOMENTS",
    "FREQUENCY_SIGN_CONVENTION",
    "LocalConversionError",
    "MAX_SPECIES",
    "SCHEMA",
    "SCHEMA_VERSION",
    "TGLF_FIELD_MODEL",
    "cgyro_input_from_tglf",
    "cgyro_parameters",
    "collect_cgyro_outputs",
    "compare_with_oracle",
    "formalism",
    "gacode_revision",
    "input_sha256",
    "prepare_cgyro_case",
    "prepare_cgyro_input",
    "read_cgyro_case",
    "run_cgyro",
    "run_cgyro_case",
    "stage_cgyro_case",
    "write_input_cgyro",
]
