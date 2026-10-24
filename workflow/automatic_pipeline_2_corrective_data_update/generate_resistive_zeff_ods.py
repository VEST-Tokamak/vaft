#!/usr/bin/env python3
"""Add the Romero resistive Z_eff of one shot's window to core_profiles.global_quantities.

Reads the inferred-T_i stage's product (the composed profiles, plus T_i where it
could be inferred) and pipeline 1's magnetic EFIT, and runs Lane Z's estimator:
Romero's transformer balance over the current-carrying window gives the observed
plasma resistance, and the one Z_eff that makes the conductivity model match it
is stored as ``core_profiles.global_quantities.z_eff_resistive`` at each usable
kinetic slice, with ``resistive_*`` provenance in ``zeff_fit.parameters``.

The estimate is one scalar for the window; per-slice solutions go to the
manifest as a labelled diagnostic. A fit on a bound, or one that is not
identifiable, is recorded and not stored. The composition Z_eff is not touched,
and the manifest cross-checks the two through Lane L's projection.

Every outcome is written and the exit status is 0: read the manifest ``status``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import traceback

from vaft.omas.vest_upstream import RESISTIVE_MODELS, build_resistive_zeff_ods, write_stage_product

LOGGER = logging.getLogger("vaft.generate_resistive_zeff_ods")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shot", required=True, type=int, help="VEST shot number.")
    parser.add_argument("--core-profiles-product", required=True, type=Path, help="The inferred-T_i stage product.")
    parser.add_argument("--efit-product", required=True, type=Path, help="Pipeline 1's magnetic EFIT product.")
    parser.add_argument("--model", default="redl", choices=RESISTIVE_MODELS, help="Parallel conductivity model.")
    parser.add_argument("--ln-lambda", default="sauter", help="Coulomb logarithm: a number, or 'sauter'.")
    parser.add_argument("--bounds", default=(1.0, 8.0), type=float, nargs=2, help="Z_eff search interval.")
    parser.add_argument("--i-ni", default=0.0, type=float, help="Non-inductive current [A]; 0 asserts a purely Ohmic window.")
    parser.add_argument("--smoothing-window-s", default=None, type=float,
                        help="Local-polynomial window [s]; omit for no smoothing.")
    parser.add_argument("--smoothing-order", default=2, type=int, help="Local-polynomial order.")
    parser.add_argument("--max-excluded-current", default=0.05, type=float,
                        help="Largest share of j_tor outside the profiles a state may have.")
    parser.add_argument("--inductive-max", default=1.0, type=float, help="States with |V_I|/|V_B| at or above this are dropped.")
    parser.add_argument("--output", required=True, type=Path, help="Output core_profiles ODS JSON.")
    parser.add_argument("--metadata", required=True, type=Path, help="Output stage manifest JSON.")
    parser.add_argument("--run", default=1, type=int, help="Dataset run number.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
    )
    try:
        ln_lambda = float(args.ln_lambda) if args.ln_lambda != "sauter" else "sauter"
        ods, manifest = build_resistive_zeff_ods(
            shot=args.shot,
            core_profiles_product=args.core_profiles_product,
            efit_product=args.efit_product,
            model=args.model,
            ln_lambda=ln_lambda,
            bounds=tuple(args.bounds),
            i_ni_a=args.i_ni,
            smoothing_window_s=args.smoothing_window_s,
            smoothing_order=args.smoothing_order,
            max_excluded_current=args.max_excluded_current,
            inductive_max=args.inductive_max,
            run=args.run,
        )
    except Exception as error:  # noqa: BLE001 - the containment is the point
        LOGGER.exception("resistive_zeff stage failed for shot %s; recording", args.shot)
        from omas import ODS

        from vaft.database._local import load_ods

        # Nothing is replicated for a failed stage; keep the input profiles in the
        # product anyway, so it is never an empty file that looks like data.
        try:
            ods, _ = load_ods(args.core_profiles_product)
        except Exception:  # noqa: BLE001 - an unreadable input leaves nothing to pass on
            ods = ODS(consistency_check=False)
        manifest = {
            "schema_version": 1,
            "stage": "resistive_zeff",
            "shot": int(args.shot),
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }

    write_stage_product(ods, manifest, output=args.output, metadata=args.metadata)
    LOGGER.info(
        "resistive_zeff ODS saved to %s (status=%s, inference=%s)",
        args.output, manifest["status"], manifest.get("inference"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
