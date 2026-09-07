"""The reference set the EFIT-quality studies run on, and the evidence for it.

Issue #171 asks for a curated reference shot set, and #196, #468, #459 and
#579 all inherit it. A set is only useful if each shot is there for a stated
reason and the reason is checked rather than remembered, so this script
declares the set and then verifies every claim made about each shot against
the files:

    PYTHONPATH=$PWD python workflow/efit_reference_set/build_reference_set.py \\
        --table test/data/efit_reference_set.json --markdown /tmp/ref.md

Two arms have to be covered, and no single VEST shot covers both well:

* **magnetics / input quality** — what the reconstruction is constrained by,
  assessed by ``workflow/magnetics_quality``. Varying it is how a study
  separates "the fit is poor" from "the input was poor".
* **independent kinetic information** — Thomson scattering, charge exchange
  and fitted profiles, against which a reconstruction's pressure and stored
  energy can be checked without using the magnetics that produced it.

Nothing here runs EFIT or changes a product. It reads what is packaged, says
what each shot can support, and refuses to claim an arm it cannot evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

SCHEMA = 1
REPOSITORY = Path(__file__).resolve().parents[2]
DEFAULT_TABLE = REPOSITORY / "test" / "data" / "efit_reference_set.json"
MAGNETICS_SCAN = REPOSITORY / "workflow" / "magnetics_quality" / "scan_magnetics_quality.py"
MAGNETICS_TABLE = REPOSITORY / "test" / "data" / "magnetics_quality.json"


@dataclass(frozen=True)
class ReferenceShot:
    """One shot, why it is in the set, and where its evidence lives."""

    shot: int
    role: str
    #: Path relative to ``vaft/data`` holding the pre-EFIT product, if any.
    product: str | None = None
    #: Path relative to ``vaft/data`` holding the Thomson MAT file, if any.
    thomson_mat: str | None = None
    #: Path relative to ``vaft/data`` holding a kinetic ODS, if any.
    kinetic_ods: str | None = None
    #: Stored EFIT outputs, relative to ``vaft/data``.
    equilibrium_reference: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The set, declared. Order is the order a study should read it in: the
#: dual-arm anchor first, then the magnetics variation, then the kinetic
#: cross-check, then the spare.
REFERENCE_SET: tuple[ReferenceShot, ...] = (
    ReferenceShot(
        shot=39915,
        role="dual-arm anchor: magnetics assessed and independent kinetic data inside the EFIT window",
        product="samples/39915/source/pipeline-until-efit.json.gz",
        thomson_mat="legacy/NeTe_Shot39915_v9_rev.mat",
        equilibrium_reference=("efit/a039915.00319", "efit/g039915.00319", "efit/g039915.00317"),
        notes=(
            "the only packaged shot with both arms; the stored 2023 reconstruction is the "
            "reference every local run is compared against (#119)",
            "its magnetics hold their last value from 0.340 s, outside the routine window",
        ),
    ),
    ReferenceShot(
        shot=41524,
        role="magnetics variation: three condemned probes, a later and shorter window",
        product="samples/41524/source/pipeline-until-efit.json.gz",
        notes=("no kinetic data is packaged for this shot",),
    ),
    ReferenceShot(
        shot=41672,
        role="magnetics variation: six condemned probes, the longest window of the three",
        product="samples/41672/source/pipeline-until-efit.json.gz",
        notes=("no kinetic data is packaged for this shot",),
    ),
    ReferenceShot(
        shot=48224,
        role="kinetic cross-check: Thomson, charge exchange and fitted profiles with a kinetic-EFIT reference",
        thomson_mat="kineticEfit/NeTe_48224.mat",
        kinetic_ods="kineticEfit/ods_48224_300ms.json",
        equilibrium_reference=(
            "kineticEfit/g048224.00300",
            "kineticEfit/g048224.00300.kinetic_efit",
            "kineticEfit/g048224.00300.chease",
        ),
        notes=(
            "no magnetics are packaged, so its input quality cannot be assessed offline and "
            "its reconstruction cannot be rebuilt from constraints here",
            "issue #317: the packaged kinetic ODS is not trustworthy within psi_N < 0.05 -- "
            "an unphysical near-axis dvolume_dpsi ramp traceable to a q[0] outlier",
        ),
    ),
    ReferenceShot(
        shot=46051,
        role="spare kinetic candidate: Thomson only, no equilibrium or magnetics packaged",
        thomson_mat="legacy/46051_NeTe.mat",
        notes=("held in the set so a second kinetic case exists if 48224's near-axis defect blocks a study",),
    ),
)


def _magnetics_module():
    """The magnetics sweep, loaded by path and registered.

    Registration is not optional: a dataclass resolves its annotations through
    ``sys.modules[cls.__module__].__dict__``.
    """
    spec = importlib.util.spec_from_file_location("scan_magnetics_quality", MAGNETICS_SCAN)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _data_path(relative: str) -> Path:
    from vaft.data.resources import data_path

    return Path(data_path(relative))


def _present(relative: str | None) -> dict[str, Any] | None:
    if relative is None:
        return None
    path = _data_path(relative)
    return {"path": relative, "exists": path.is_file(), "bytes": path.stat().st_size if path.is_file() else 0}


def thomson_arm(entry: ReferenceShot, window: tuple[float, float] | None) -> dict[str, Any]:
    """What Thomson scattering this shot offers, and whether it lands in the window.

    A profile measured outside the reconstructed window cannot check that
    reconstruction, so the overlap is the property that matters, not the mere
    presence of a file.
    """
    if entry.thomson_mat is None:
        return {"available": False, "reason": "no Thomson file is packaged for this shot"}
    path = _data_path(entry.thomson_mat)
    if not path.is_file():
        return {"available": False, "reason": f"{entry.thomson_mat} is not in this checkout"}

    from omas import ODS

    from vaft.machine_mapping.thomson_scattering import thomson_scattering

    ods = ODS(consistency_check=False)
    try:
        thomson_scattering(ods, entry.shot, data_root=str(path))
    except Exception as error:
        return {"available": False, "reason": f"{type(error).__name__}: {error}"[:200]}
    channels = len(ods["thomson_scattering.channel"])
    times = np.asarray(ods["thomson_scattering.time"], dtype=float).reshape(-1)
    positions = [float(ods[f"thomson_scattering.channel.{i}.position.r"]) for i in range(channels)]
    record: dict[str, Any] = {
        "available": True,
        "source": entry.thomson_mat,
        "channels": channels,
        "times": [float(times[0]), float(times[-1])],
        "samples": int(times.size),
        "radii": positions,
    }
    if window is not None:
        inside = [float(value) for value in times if window[0] <= value <= window[1]]
        record["window"] = [float(window[0]), float(window[1])]
        record["samples_in_window"] = len(inside)
        record["overlaps_window"] = bool(inside)
    return record


def kinetic_ods_arm(entry: ReferenceShot) -> dict[str, Any]:
    """The packaged kinetic ODS, if any: which IDSs it carries and how much."""
    if entry.kinetic_ods is None:
        return {"available": False, "reason": "no kinetic ODS is packaged for this shot"}
    path = _data_path(entry.kinetic_ods)
    if not path.is_file():
        return {"available": False, "reason": f"{entry.kinetic_ods} is not in this checkout"}

    from omas import load_omas_json

    ods = load_omas_json(str(path), consistency_check=False)
    present = sorted({key.split(".")[0] for key in ods.keys()})
    record: dict[str, Any] = {"available": True, "source": entry.kinetic_ods, "ids": present}
    if "thomson_scattering" in ods:
        record["thomson_channels"] = len(ods["thomson_scattering.channel"])
    if "charge_exchange" in ods:
        record["charge_exchange_channels"] = len(ods["charge_exchange.channel"])
    if "core_profiles" in ods:
        fit = "core_profiles.profiles_1d.0.electrons.temperature_fit.measured"
        record["core_profiles"] = {
            "fitted_points": int(np.asarray(ods["core_profiles.profiles_1d.0.electrons.temperature"]).size),
            "measured_points": int(np.asarray(ods[fit]).size) if fit in ods else 0,
        }
    if "equilibrium" in ods:
        record["equilibrium_times"] = [float(value) for value in np.asarray(ods["equilibrium.time"], dtype=float)]
    return record


def magnetics_arm(entry: ReferenceShot, rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """The magnetics verdict for this shot, from the magnetics-quality table."""
    if entry.product is None:
        return {"available": False, "reason": "no pre-EFIT product is packaged, so the magnetics cannot be assessed"}
    row = rows.get(entry.shot)
    if row is None:
        return {"available": False, "reason": "the shot is not in test/data/magnetics_quality.json"}
    if row.get("status") != "assessed":
        return {"available": False, "reason": row.get("reason", row.get("status"))}
    return {
        "available": True,
        "verdict": row["verdict"],
        "window": [row["window"]["start"], row["window"]["end"]],
        "slices": row["window"]["slices"],
        "condemned": row["condemned"],
        "min_usable_fraction": row["decisions"]["min_usable_fraction"],
        "reasons": row["reasons"],
    }


def build(rows: dict[int, dict[str, Any]]) -> dict[str, Any]:
    entries = []
    for entry in REFERENCE_SET:
        magnetics = magnetics_arm(entry, rows)
        window = tuple(magnetics["window"]) if magnetics.get("available") else None
        record = {
            **entry.as_dict(),
            "files": {
                "product": _present(entry.product),
                "thomson_mat": _present(entry.thomson_mat),
                "kinetic_ods": _present(entry.kinetic_ods),
                "equilibrium_reference": [_present(name) for name in entry.equilibrium_reference],
            },
            "magnetics": magnetics,
            "thomson": thomson_arm(entry, window),
            "kinetic_ods_contents": kinetic_ods_arm(entry),
        }
        record["arms"] = sorted(
            name
            for name, ok in (
                ("magnetics", bool(magnetics.get("available"))),
                ("kinetic", bool(record["thomson"].get("available")) or bool(record["kinetic_ods_contents"].get("available"))),
            )
            if ok
        )
        entries.append(record)

    missing = [
        f"{item['shot']}:{name['path']}"
        for item in entries
        for name in [value for value in item["files"].values() if isinstance(value, dict)]
        + [value for value in item["files"]["equilibrium_reference"]]
        if isinstance(name, dict) and not name["exists"]
    ]
    return {
        "schema_version": SCHEMA,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": {
            "shots": len(entries),
            "both_arms": [item["shot"] for item in entries if item["arms"] == ["kinetic", "magnetics"]],
            "magnetics_only": [item["shot"] for item in entries if item["arms"] == ["magnetics"]],
            "kinetic_only": [item["shot"] for item in entries if item["arms"] == ["kinetic"]],
            "no_arm": [item["shot"] for item in entries if not item["arms"]],
            "missing_files": sorted(set(missing)),
        },
        "shots": entries,
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = ["# The EFIT-quality reference set", ""]
    summary = payload["summary"]
    lines.append(
        f"{summary['shots']} shots: {len(summary['both_arms'])} covering both arms, "
        f"{len(summary['magnetics_only'])} magnetics only, {len(summary['kinetic_only'])} kinetic only."
    )
    if summary["missing_files"]:
        lines.append("")
        lines.append(f"**Missing files:** {', '.join(summary['missing_files'])}")
    lines.append("")
    lines.append("| shot | arms | magnetics verdict | window [s] | condemned | Thomson | in window | kinetic ODS |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for item in payload["shots"]:
        magnetics = item["magnetics"]
        thomson = item["thomson"]
        kinetic = item["kinetic_ods_contents"]
        window = (
            f"{magnetics['window'][0]:.3f}–{magnetics['window'][1]:.3f}" if magnetics.get("available") else "–"
        )
        ts = (
            f"{thomson['channels']} ch, {thomson['times'][0]:.3f}–{thomson['times'][1]:.3f}"
            if thomson.get("available")
            else "–"
        )
        overlap = (
            f"{thomson.get('samples_in_window', 0)}/{thomson['samples']}"
            if thomson.get("available") and "samples_in_window" in thomson
            else "–"
        )
        lines.append(
            f"| {item['shot']} | {', '.join(item['arms']) or 'none'} "
            f"| {magnetics.get('verdict', '–')} | {window} "
            f"| {len(magnetics.get('condemned', [])) if magnetics.get('available') else '–'} "
            f"| {ts} | {overlap} "
            f"| {'yes' if kinetic.get('available') else '–'} |"
        )
    lines.append("")
    for item in payload["shots"]:
        lines.append(f"## {item['shot']}")
        lines.append("")
        lines.append(f"- role: {item['role']}")
        for note in item["notes"]:
            lines.append(f"- {note}")
        magnetics = item["magnetics"]
        if magnetics.get("available"):
            lines.append(
                f"- magnetics: **{magnetics['verdict']}**, {magnetics['slices']} slices, "
                f"minimum usable fraction {magnetics['min_usable_fraction']:.2f}"
            )
            for reason in magnetics["reasons"]:
                lines.append(f"  - {reason}")
        else:
            lines.append(f"- magnetics: not available ({magnetics['reason']})")
        thomson = item["thomson"]
        if thomson.get("available"):
            radii = ", ".join(f"{value:.3f}" for value in thomson["radii"])
            lines.append(
                f"- Thomson: {thomson['channels']} channels at R = {radii} m, "
                f"{thomson['samples']} times {thomson['times'][0]:.3f}–{thomson['times'][1]:.3f} s"
                + (
                    f", {thomson['samples_in_window']} inside the EFIT window"
                    if "samples_in_window" in thomson
                    else ""
                )
            )
        else:
            lines.append(f"- Thomson: not available ({thomson['reason']})")
        kinetic = item["kinetic_ods_contents"]
        if kinetic.get("available"):
            lines.append(
                f"- kinetic ODS: {', '.join(kinetic['ids'])}"
                + (f"; {kinetic.get('charge_exchange_channels')} charge-exchange channels" if kinetic.get("charge_exchange_channels") else "")
                + (
                    f"; core profiles fitted on {kinetic['core_profiles']['fitted_points']} points from "
                    f"{kinetic['core_profiles']['measured_points']} measurements"
                    if kinetic.get("core_profiles")
                    else ""
                )
            )
        references = [record["path"] for record in item["files"]["equilibrium_reference"] if record and record["exists"]]
        if references:
            lines.append(f"- stored equilibria: {', '.join(references)}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _magnetics_rows(table: Path, *, rescan: bool) -> dict[int, dict[str, Any]]:
    if rescan or not table.is_file():
        module = _magnetics_module()
        policy = module.FitnessPolicy()
        rows = [
            module.scan_shot(shot, source=None, packaged=True, policy=policy, tstep=0.001)
            for shot in module.PACKAGED_SHOTS
        ]
        return {int(row["shot"]): row for row in rows}
    payload = json.loads(table.read_text(encoding="utf-8"))
    return {int(row["shot"]): row for row in payload["rows"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--magnetics-table", type=Path, default=MAGNETICS_TABLE)
    parser.add_argument(
        "--rescan",
        action="store_true",
        help="re-run the magnetics sweep instead of reading its committed table",
    )
    args = parser.parse_args(argv)

    rows = _magnetics_rows(args.magnetics_table, rescan=args.rescan)
    payload = build(rows)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    text = markdown(payload)
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    else:
        print(text)
    print(f"table: {args.table}")
    return 1 if payload["summary"]["missing_files"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
