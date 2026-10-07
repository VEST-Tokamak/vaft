"""Export virial identity residuals and pair-13 closure for an Atlas state table.

Run this against the FileDB that produced the state table. Product SHA-256 and
slice times are checked before any result is written. The CSV is keyed by the
Atlas state key and retains the product hash for later joins.

Example::

    python workflow/kinetic_state/export_virial_identity.py \
        --state ~/runs/campaign/atlas/v1/state.csv \
        --filedb ~/runs/campaign/filedb \
        --out virial_identity.csv
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import sys
import tempfile
from pathlib import Path

from omas import load_omas_json

from vaft.omas.process_wrapper import compute_virial_equilibrium_quantities_ods
from vaft.validation.kinetic_state import slice_at_time


KEY = ("shot", "time_efit_s", "efit_lineage")
COLUMNS = (*KEY, "efit_product", "efit_product_sha256", "e1_normalized",
           "e2_normalized", "e3_normalized", "rms", "identities_available",
           "beta_p_volume", "beta_p_pair_13", "li_volume", "li_pair_13")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_product(path: Path):
    if path.suffix != ".gz":
        return load_omas_json(str(path), consistency_check=False)
    with gzip.open(path, "rt", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
        "w", suffix=".json", encoding="utf-8", delete=False
    ) as staged:
        staged.write(source.read())
        staged_path = Path(staged.name)
    try:
        return load_omas_json(str(staged_path), consistency_check=False)
    finally:
        staged_path.unlink(missing_ok=True)


def export(state: Path, filedb: Path) -> list[dict[str, object]]:
    with state.open(newline="", encoding="utf-8") as stream:
        states = list(csv.DictReader(stream))
    products = {}
    seen = set()
    output = []
    for row in states:
        key = (int(row["shot"]), float(row["time_efit_s"]), row["efit_lineage"])
        if key in seen:
            raise ValueError(f"duplicate Atlas state key: {key}")
        seen.add(key)
        relative = Path(row["efit_product"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"product path must be inside FileDB: {relative}")
        product_path = filedb / relative
        if relative not in products:
            actual_hash = _sha256(product_path)
            products[relative] = (actual_hash, _load_product(product_path))
        actual_hash, ods = products[relative]
        expected_hash = row["efit_product_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"product SHA-256 mismatch for {relative}")
        index, _ = slice_at_time(ods, key[1], tolerance_s=0.6e-3)
        virial = compute_virial_equilibrium_quantities_ods(ods, time_slice=index)[index]
        identity = virial["identity"]
        output.append({
            "shot": key[0], "time_efit_s": row["time_efit_s"],
            "efit_lineage": key[2], "efit_product": str(relative),
            "efit_product_sha256": actual_hash,
            **{name: identity[name] for name in ("e1_normalized", "e2_normalized",
                                                  "e3_normalized", "rms", "identities_available")},
            "beta_p_volume": virial["volume"]["beta_p"],
            "beta_p_pair_13": virial["pair_13"]["beta_p"],
            "li_volume": virial["volume"]["li"],
            "li_pair_13": virial["pair_13"]["li"],
        })
    if len(output) != len(states):
        raise AssertionError("virial export lost an Atlas state")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--filedb", required=True, type=Path)
    parser.add_argument("--out", required=True, help="CSV path or - for stdout")
    args = parser.parse_args()
    rows = export(args.state, args.filedb)
    if args.out == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    else:
        with Path(args.out).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
    print(f"Exported {len(rows)} rows from {len({r['efit_product'] for r in rows})} products", file=sys.stderr)


if __name__ == "__main__":
    main()
