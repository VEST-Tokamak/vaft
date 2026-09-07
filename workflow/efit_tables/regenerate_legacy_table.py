"""Generate a fresh EFIT Green table for one VEST machine era (issue #194).

    PYTHONPATH=$PWD EFITHOME=~/git/efit/vaft-install \\
        python workflow/efit_tables/regenerate_legacy_table.py --output /scratch/tables/legacy-39915

The chain is the one the provenance record needs to be able to state::

    build_static_ods(era) -> efund_geometry_from_static -> mhdin.dat -> efund -> tables + manifest

Nothing here touches ``vaft/data/efit``: the output directory is yours, and
``compare_tables.py`` / ``ab_efit_table.py`` are how a generated table is
judged against the bundled one.  The era is named explicitly (default: the
legacy era shot 39915 belongs to) and is never derived from a shot here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from vaft.code.efit.efund import (
    EFUNDConfig,
    prepare_efund_inputs,
    run_efund,
    write_table_manifest,
)
from vaft.code.efit.toolchain import resolve_toolchain, toolchain_identities
from vaft.omas.vest_upstream import VEST_MACHINE_ERAS, build_static_ods

DEFAULT_ERA = "vest-pre-43017-pf1906"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="directory to generate the table into")
    parser.add_argument("--era", default=DEFAULT_ERA, choices=[era.name for era in VEST_MACHINE_ERAS])
    parser.add_argument("--nw", type=int, default=129)
    parser.add_argument("--nh", type=int, default=None, help="defaults to --nw")
    parser.add_argument("--label", default=None, help="a short name recorded in the manifest")
    parser.add_argument("--efit-home", default=None, help="sets EFITHOME for this run")
    parser.add_argument("--efund", default=None, help="explicit efund executable (wins over EFITHOME)")
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--ivesel", type=int, default=1)
    parser.add_argument("--iecoil", type=int, default=0)
    args = parser.parse_args(argv)

    if args.efit_home:
        os.environ["EFITHOME"] = str(Path(args.efit_home).expanduser())
    resolved = resolve_toolchain(efund_executable=args.efund)
    identities = toolchain_identities(resolved)
    print("toolchain:")
    for role, identity in identities.items():
        if identity is None:
            print(f"  {role}: not configured")
        else:
            print(f"  {role}: {identity['path']}  sha256 {identity['sha256'][:12]}  {identity.get('build_revision')}")

    output = args.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    config = EFUNDConfig(
        workdir=output,
        nw=args.nw,
        nh=args.nh or args.nw,
        ivesel=args.ivesel,
        iecoil=args.iecoil,
        executable=args.efund,
        timeout=args.timeout,
    )
    started = time.perf_counter()
    ods, manifest = build_static_ods(args.era)
    inputs = prepare_efund_inputs(ods, config, manifest=manifest)
    print(f"input: {inputs.mhdin} sha256 {inputs.mhdin_sha256[:12]} counts {inputs.counts}")
    result = run_efund(inputs, config)
    elapsed = time.perf_counter() - started
    print(f"efund: status {result.status} returncode {result.returncode} in {elapsed:.0f} s")
    if not result.ok:
        print(f"  reason: {result.reason}", file=sys.stderr)
        return 1
    path = write_table_manifest(
        result,
        inputs,
        config,
        label=args.label or f"{args.era}-{config.table_suffix}",
        extra={"generator": "workflow/efit_tables/regenerate_legacy_table.py", "seconds": elapsed},
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"manifest: {path}")
    print(f"table identity: {payload['table']['identity']}")
    for name, record in payload["table"]["files"].items():
        print(f"  {name:14s} {record['size']:>12d} bytes  sha256 {record['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
