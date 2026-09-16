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
from vaft.code.efit.config import IGNORE_CRITERION
from vaft.machine_mapping.efund_geometry import vest_acceptance_envelope
from vaft.omas.vest_upstream import VEST_MACHINE_ERAS, build_static_ods

DEFAULT_ERA = "vest-pre-43017-pf1906"


#: Era-independent files EFIT reads from the table directory alongside the
#: Green tables EFUND writes. Copied from the packaged directory so a
#: generated table can actually be run.
RUNTIME_COMPANIONS = ("lim.dat", "dprobe.dat")


def _copy_runtime_companions(output: Path) -> list[str]:
    """Place EFIT's non-EFUND inputs beside a freshly generated table."""
    import shutil

    from vaft.data.resources import data_path

    source = Path(data_path("efit"))
    copied: list[str] = []
    for name in RUNTIME_COMPANIONS:
        origin = source / name
        target = output / name
        if origin.is_file() and not target.exists():
            shutil.copy2(origin, target)
            copied.append(name)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="directory to generate the table into")
    parser.add_argument("--era", default=DEFAULT_ERA, choices=[era.name for era in VEST_MACHINE_ERAS])
    parser.add_argument("--nw", type=int, default=129)
    parser.add_argument("--nh", type=int, default=None, help="defaults to --nw")
    # The computational box. EFIT has no independent notion of it: it reads
    # `rgrid`/`zgrid` straight out of the Green table (`tables.F90:158`), so
    # the domain is whatever EFUND baked in here, and studying an alternate
    # one (#459) means generating a table for it.
    parser.add_argument("--rleft", type=float, default=None, help="inner edge of the box, m")
    parser.add_argument("--rright", type=float, default=None, help="outer edge of the box, m")
    parser.add_argument("--zbotto", type=float, default=None, help="bottom of the box, m")
    parser.add_argument("--ztop", type=float, default=None, help="top of the box, m")
    parser.add_argument(
        "--no-acceptance-envelope",
        action="store_true",
        help="omit the &incheck block, leaving EFIT on its compiled-in DIII-D bounds",
    )
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
    box = {
        name: getattr(args, name)
        for name in ("rleft", "rright", "zbotto", "ztop")
        if getattr(args, name) is not None
    }
    config = EFUNDConfig(
        workdir=output,
        nw=args.nw,
        nh=args.nh or args.nw,
        ivesel=args.ivesel,
        iecoil=args.iecoil,
        executable=args.efund,
        timeout=args.timeout,
        **box,
    )
    print(
        f"grid: {config.nw} x {config.nh}, box R {config.rleft}-{config.rright} m, "
        f"Z {config.zbotto}-{config.ztop} m, cells "
        f"{(config.rright - config.rleft) / (config.nw - 1) * 1e3:.2f} x "
        f"{(config.ztop - config.zbotto) / (config.nh - 1) * 1e3:.2f} mm"
    )
    started = time.perf_counter()
    ods, manifest = build_static_ods(args.era)

    # EFUND ignores `&incheck`; EFIT reads it from the same file at run time and
    # rejects a reconstruction against it. Deriving it from the machine is the
    # whole point of `vest_acceptance_envelope`, and until #649 the generated
    # table carried no `&incheck` at all -- so EFIT fell back to bounds that are
    # DIII-D's, and the packaged table carried a hand-preserved legacy block.
    #
    # Measured on 39915, same Green tables, only this block changed: as
    # shipped, every plasma slice is rejected (fourteen of fourteen, all on
    # failure #21, the virial `delbp`). Disabling the virial checks alone
    # changes nothing -- the geometric bounds reject instead, on #5 `aminor`
    # and #8 `rcurrt`. With the derived envelope, ten of fourteen are accepted
    # and the remaining four fail chi-square, which is a fit result rather than
    # a bound.
    envelope = (
        None
        if args.no_acceptance_envelope
        else vest_acceptance_envelope(ods, nw=config.nw, rleft=config.rleft, rright=config.rright)
    )
    if envelope is not None:
        print(
            f"acceptance envelope: aminor {envelope.aminor_min:.2f}-{envelope.aminor_max:.2f} cm, "
            f"virial checks {'on' if envelope.delbp_diff < IGNORE_CRITERION else 'disabled'}"
        )
    inputs = prepare_efund_inputs(ods, config, manifest=manifest, envelope=envelope)
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
    # EFUND produces the Green tables; EFIT also reads a limiter contour and a
    # probe description from the same directory, and neither depends on the
    # era. Without them the generated directory is not runnable: EFIT dies in
    # `read_limiter.f90` with a Fortran runtime error naming `lim.dat`, which
    # is a confusing way to learn that a table is incomplete (#805).
    copied = _copy_runtime_companions(output)
    if copied:
        print("companions: " + ", ".join(sorted(copied)))

    payload = json.loads(path.read_text(encoding="utf-8"))
    print(f"manifest: {path}")
    print(f"table identity: {payload['table']['identity']}")
    for name, record in payload["table"]["files"].items():
        print(f"  {name:14s} {record['size']:>12d} bytes  sha256 {record['sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
