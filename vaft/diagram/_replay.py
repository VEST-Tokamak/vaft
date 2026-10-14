"""Opt-in scientific Scene export for the two documentation galleries.

No solver runs while generating the catalog or building the site. Recipes here
are the single declaration of animation capabilities, not a second science API.
The browser selects stored float64 states, never interpolates or integrates.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from ._scene import Arrow, Label, Marker, Polyline, Scene

SCHEMA_VERSION = 1
ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs/assets/animations"
RECIPES = (
    {
        "builder": "magnetic_island",
        "variant": "phase_scan",
        "title": "Magnetic-island helical phase",
        "poster": "magnetic_island_phase_scan.svg",
        "driver": {"name": "phase", "kind": "phase", "unit": "rad"},
        "parameters": {
            "m": 2,
            "n": 1,
            "width": 0.24,
            "projection": "poloidal",
            "labels": False,
        },
    },
    {
        "builder": "exb_drift",
        "variant": "particle_motion",
        "title": "E×B particle and guiding-centre motion",
        "poster": "exb_drift.svg",
        "driver": {"name": "time", "kind": "physical_time", "unit": "normalized time"},
        "parameters": {
            "mass_ratio": 4.0,
            "projection": "perpendicular",
            "labels": False,
        },
    },
)
PLAYER_BUDGET = 100 * 1024
STATE_BUDGET = 300 * 1024


def encode(data):
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def atomic_write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as f:
        temporary = Path(f.name)
        f.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def scientific_sources(builder):
    """Hash the model, primitive definitions and actual formulas, not just the recipe."""
    from .docs_catalog import formula_references
    import inspect
    import vaft.diagram as diagram

    function = getattr(diagram, builder)
    path = Path(inspect.getsourcefile(function)).resolve()
    paths = {path, Path(__file__).resolve(), ROOT / "vaft/diagram/_scene.py"}
    for name in formula_references(path, function.__name__):
        paths.add(ROOT / ("/".join(name.split(".")[:-1]) + ".py"))
    return [
        {"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p.read_bytes())}
        for p in sorted(paths)
    ]


def element_rows(scene):
    counts = Counter()
    rows = []
    for item in scene.items:
        if not isinstance(item, (Polyline, Marker, Arrow, Label)):
            raise ValueError(f"Unsupported replay primitive: {type(item).__name__}")
        role = item.role or type(item).__name__.lower()
        counts[role] += 1
        rows.append(
            {
                "id": f"{role}-{counts[role]}",
                "type": type(item).__name__.lower(),
                **asdict(item),
            }
        )
    return rows


def scene_payload(
    builder, variant, driver, values, scenes, *, parameters, fps=8, observables=None
):
    rows = [element_rows(s) for s in scenes]
    if len(rows) != len(values) or not rows:
        raise ValueError("One scene is required for every scientific state")
    ids = [r["id"] for r in rows[0]]
    if any([r["id"] for r in state] != ids for state in rows):
        raise ValueError("Scene semantic IDs must remain stable across states")
    static, dynamic = [], []
    for index, row in enumerate(rows[0]):
        if all(encode(state[index]) == encode(row) for state in rows):
            static.append(row)
        else:
            dynamic.append(index)
    points = []
    for state in rows:
        for row in state:
            points.extend(row.get("points", []))
            points.extend([row[k] for k in ("at", "start", "end") if k in row])
    xy = np.asarray(points, dtype=float)
    low, high = xy.min(axis=0) - 0.4, xy.max(axis=0) + 0.4
    result = {
        "schema_version": SCHEMA_VERSION,
        "builder": builder,
        "variant": variant,
        "driver": {**driver, "values": [float(v) for v in values]},
        "coordinates": {"system": "scene_cartesian", "unit": "cm", "dtype": "float64"},
        "parameters": parameters,
        "fps": fps,
        "interpolation": "none",
        "bounds": [float(low[0]), float(low[1]), float(high[0]), float(high[1])],
        "order": ids,
        "static": static,
        "states": [[state[i] for i in dynamic] for state in rows],
        "observables": observables or [],
        "sources": scientific_sources(builder),
    }
    validate_payload(result)
    return result


def make_payload(recipe):
    import vaft.diagram as diagram

    parameters = recipe["parameters"]
    if recipe["builder"] == "magnetic_island":
        # The finite supported sequence is explicit. There is no interpolated phase.
        values = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        scenes = [
            diagram.magnetic_island(phase=float(p), **parameters).scene for p in values
        ]
        return scene_payload(
            recipe["builder"],
            recipe["variant"],
            recipe["driver"],
            values,
            scenes,
            parameters=parameters,
            fps=2,
        )
    figure = diagram.exb_drift(**parameters)
    model = figure.model
    ion_dt, electron_dt = model.parameters["ion_dt"], model.parameters["electron_dt"]
    times = np.arange(len(model.orbits["ion"])) * ion_dt
    electron_indices = np.rint(times / electron_dt).astype(int)
    if not np.allclose(electron_indices * electron_dt, times, rtol=0, atol=1e-12):
        raise ValueError(
            "E×B recipe requires shared stored ion/electron times; interpolation is forbidden"
        )
    scenes = []
    # Both paths are static; markers use the exact same scaled model coordinates.
    base = figure.scene
    for i, (t, j) in enumerate(zip(times, electron_indices)):
        markers = []
        for name, index in (("ion", i), ("electron", j)):
            markers.append(
                Marker(
                    tuple(0.45 * model.orbits[name][index, :2]),
                    "o",
                    f"orbit {name}",
                    f"{name}_particle",
                )
            )
            gc = (
                model.vectors[f"{name}_guiding_centre"]
                + model.vectors[f"{name}_drift"] * t
            )
            markers.append(
                Marker(tuple(0.45 * gc[:2]), "x", f"orbit {name}", f"{name}_gc")
            )
        scenes.append(base + Scene(tuple(markers)))
    observables = [
        {
            "name": "Ion x",
            "unit": "normalized length",
            "values": model.orbits["ion"][:, 0].tolist(),
        },
        {
            "name": "Electron x",
            "unit": "normalized length",
            "values": model.orbits["electron"][electron_indices, 0].tolist(),
        },
    ]
    return scene_payload(
        recipe["builder"],
        recipe["variant"],
        recipe["driver"],
        times,
        scenes,
        parameters={**parameters, "model": model.parameters},
        fps=30,
        observables=observables,
    )


def validate_payload(data):
    if (
        data.get("schema_version") != SCHEMA_VERSION
        or data.get("interpolation") != "none"
    ):
        raise ValueError("Unsupported replay version or interpolation")
    driver = data.get("driver", {})
    if driver.get("kind") not in (
        "phase",
        "physical_time",
        "parameter",
        "process_stage",
        "iteration",
    ) or not driver.get("unit"):
        raise ValueError("Driver kind and unit must be explicit")
    coordinates = data.get("coordinates", {})
    if coordinates != {"system": "scene_cartesian", "unit": "cm", "dtype": "float64"}:
        raise ValueError("Unsupported coordinate frame, units or precision")
    values = driver.get("values", [])
    if not driver.get("name") or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) for v in values
    ):
        raise ValueError("Scientific driver values must be numeric")
    if len(values) < 2 or len(data.get("states", [])) != len(values):
        raise ValueError("State/driver shape mismatch")
    if driver["kind"] == "physical_time" and any(
        b <= a for a, b in zip(values, values[1:])
    ):
        raise ValueError("Physical times must increase")
    ids = data.get("order", [])
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Duplicate or missing semantic IDs")
    if not np.isfinite(data.get("fps", float("nan"))) or data["fps"] <= 0:
        raise ValueError("Presentation fps must be positive")
    bounds = data.get("bounds", [])
    if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
        raise ValueError("Invalid scene bounds")

    def finite(value):
        if isinstance(value, (float, int)) and not np.isfinite(value):
            raise ValueError("Non-finite scientific value")
        if isinstance(value, dict):
            for v in value.values():
                finite(v)
        if isinstance(value, (list, tuple)):
            for v in value:
                finite(v)

    finite(data)
    fixed = {r["id"] for r in data.get("static", [])}
    if len(fixed) != len(data.get("static", [])):
        raise ValueError("Duplicate static IDs")
    for state in data["states"]:
        combined = [r["id"] for r in data["static"] + state]
        if len(combined) != len(ids) or set(combined) != set(ids):
            raise ValueError("State semantic IDs do not match scene order")
        for row in data["static"] + state:
            if row["type"] not in ("polyline", "marker", "arrow", "label"):
                raise ValueError("Unsupported scene element")

            def point(value):
                return (
                    isinstance(value, (list, tuple))
                    and len(value) == 2
                    and all(
                        isinstance(v, (int, float)) and not isinstance(v, bool)
                        for v in value
                    )
                )

            if row["type"] == "polyline":
                valid = bool(row.get("points")) and all(point(p) for p in row["points"])
            elif row["type"] == "arrow":
                valid = point(row.get("start")) and point(row.get("end"))
            else:
                valid = point(row.get("at"))
            if not valid:
                raise ValueError("Invalid scene coordinate shape")
    for observable in data.get("observables", []):
        if len(observable["values"]) != len(values) or not observable.get("unit"):
            raise ValueError("Observable/driver shape mismatch")
    encode(data)  # JSON rejects non-finite values, including in unexpected fields.


def generate(output=ASSETS):
    output = Path(output)
    records = []
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    for recipe in RECIPES:
        payload = make_payload(recipe)
        raw = encode(payload)
        size = len(gzip.compress(raw, mtime=0))
        if size > STATE_BUDGET:
            raise ValueError(f"{recipe['builder']}: gzip state budget exceeded: {size}")
        name = f"{recipe['builder']}--{recipe['variant']}.json"
        atomic_write(output / name, raw)
        records.append(
            {
                **recipe,
                "id": name[:-5],
                "artifact": f"assets/animations/{name}",
                "sha256": digest(raw),
                "bytes": len(raw),
                "gzip_bytes": size,
                "count": len(payload["states"]),
                "fps": payload["fps"],
                "sources": payload["sources"],
                "generated_commit": commit,
                "representation": "svg_state"
                if recipe["builder"] == "magnetic_island"
                else "trajectory_replay",
                "interpolation": "none",
                "capabilities": ["seek", "step", "speed", "enlarge"],
                "recipe_source": "vaft/diagram/_replay.py",
            }
        )
    atomic_write(
        output / "manifest.json",
        encode({"schema_version": SCHEMA_VERSION, "animations": records}),
    )
    return records


def catalog_rows(root=ROOT):
    root = Path(root)
    directory = root / "docs/assets/animations"
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported animation manifest version")
    rows = manifest["animations"]
    expected = {f"{r['builder']}--{r['variant']}": r for r in RECIPES}
    if len({r["id"] for r in rows}) != len(rows) or {r["id"] for r in rows} != set(
        expected
    ):
        raise ValueError("Missing, duplicate or undeclared animation variant")
    artifacts = set()
    for row in rows:
        recipe = expected[row["id"]]
        for key, value in recipe.items():
            if row.get(key) != value:
                raise ValueError(
                    f"Animation recipe drift: {row['id']} {key}; regenerate assets"
                )
        path = Path(row["artifact"])
        if (
            path.is_absolute()
            or ".." in path.parts
            or path.parent.as_posix() != "assets/animations"
        ):
            raise ValueError("Animation artifact must be a local Pages path")
        raw = (root / "docs" / path).read_bytes()
        if digest(raw) != row["sha256"] or len(raw) != row["bytes"]:
            raise ValueError(f"Animation integrity mismatch: {row['id']}")
        if (
            len(gzip.compress(raw, mtime=0)) != row["gzip_bytes"]
            or row["gzip_bytes"] > STATE_BUDGET
        ):
            raise ValueError("Animation size budget mismatch")
        data = json.loads(raw)
        validate_payload(data)
        if (
            data["builder"],
            data["variant"],
            len(data["states"]),
            data["fps"],
            data["sources"],
        ) != (row["builder"], row["variant"], row["count"], row["fps"], row["sources"]):
            raise ValueError("Manifest and scientific state disagree")
        if not (root / "docs/assets/diagrams" / row["poster"]).is_file():
            raise ValueError("Missing canonical animation poster")
        for source in row["sources"]:
            p = Path(source["path"])
            if (
                p.is_absolute()
                or ".." in p.parts
                or digest((root / p).read_bytes()) != source["sha256"]
            ):
                raise ValueError(
                    f"Stale animation scientific source: {source['path']}; regenerate assets"
                )
        artifacts.add(path.name)
    if {p.name for p in directory.glob("*.json")} - {"manifest.json"} != artifacts:
        raise ValueError("Orphan animation state artifact")
    javascript_bytes = sum(
        len(gzip.compress((directory / name).read_bytes(), mtime=0))
        for name in ("player.js", "loader.js")
        if (directory / name).exists()
    )
    if javascript_bytes > PLAYER_BUDGET:
        raise ValueError("Combined animation JavaScript exceeds gzip budget")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ASSETS)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(f"Validated {len(catalog_rows())} animation variants")
    else:
        for row in generate(args.output):
            print(f"{row['id']}: {row['count']} states, {row['gzip_bytes']} gzip bytes")


if __name__ == "__main__":
    main()
