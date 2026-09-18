"""Pipeline 1's Snakefile parses and builds its per-product stages.

Nothing else in the suite loads this file. `test_pipeline1_paths.py` covers
`PipelinePaths` on its own, which cannot see how the Snakefile *calls* it -- and
that is where #787 broke: `plot_mhd_linear` and `build_gpec_ideal` resolved
their paths through `shot_pattern`, which cannot name the stability product
those stages now require, so `FileDBPathError` was raised while Snakemake was
still reading the file. Every invocation under the canonical `filedb` layout
failed before building a DAG, replication or not, and the suite stayed green.

A Snakefile raises at parse time for every rule in it, including rules no
target reaches, so the first test needs no target at all. The second asks for
the per-product stage products by name, which is what proves the patterns the
producing rules declare are the ones `PipelinePaths` resolves -- a parse alone
would accept a rule whose output no request can ever match.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

WORKFLOW = (
    Path(__file__).resolve().parents[1]
    / "workflow"
    / "automatic_pipeline_1_routine_data_processing"
)
SHOT = 39915


def _snakemake_importable() -> bool:
    try:
        import snakemake  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = [
    pytest.mark.skipif(
        not WORKFLOW.exists(), reason="workflow scripts are not part of the distribution"
    ),
    pytest.mark.skipif(
        shutil.which("snakemake") is None and not _snakemake_importable(),
        reason="snakemake is not installed",
    ),
]


def _config(tmp_path: Path, layout: str = "filedb") -> Path:
    config = tmp_path / "config.yaml"
    config.write_text(
        json.dumps(
            {
                "base_dir": str(tmp_path / "filedb"),
                "layout": layout,
                "shots": [SHOT],
                "raw": {"mode": "sql"},
                # Replication and every stability module on: the rules they gate
                # are exactly the ones a narrower configuration would not read.
                # (Replication has no home in a shot-first tree; see #89, #138.)
                "hsds": {"replicate": layout == "filedb"},
                "gpec": {"modules": ["dcon", "rdcon", "stride", "gpec"], "modes": [1, 2]},
                "conda": None,
            }
        ),
        encoding="utf-8",
    )
    return config


def _dry_run(
    tmp_path: Path, targets: list[str] | None = None, *, layout: str = "filedb"
):
    # config.yaml interpolates these; a dry run never executes them.
    env = dict(os.environ)
    for name in ("VAFT_FILEDB_DIR", "VAFT_DATA_DIR", "EFIT", "CHEASE", "GPECHOME"):
        env.setdefault(name, str(tmp_path / name.lower()))
    return subprocess.run(
        [
            sys.executable, "-m", "snakemake",
            "--snakefile", str(WORKFLOW / "Snakefile"),
            "--configfile", str(_config(tmp_path, layout)),
            "--directory", str(tmp_path),
            "--cores", "1", "-n",
            *(targets or []),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=WORKFLOW,
    )


def _paths(tmp_path: Path, layout: str = "filedb"):
    sys.path.insert(0, str(WORKFLOW))
    try:
        from paths import PipelinePaths
    finally:
        sys.path.remove(str(WORKFLOW))
    return PipelinePaths(str(tmp_path / "filedb"), layout)


def test_the_snakefile_parses_under_the_canonical_layout(tmp_path):
    result = _dry_run(tmp_path)
    assert result.returncode == 0, result.stderr[-3000:]


def test_the_per_product_stages_are_reachable_by_the_paths_that_name_them(tmp_path):
    paths = _paths(tmp_path)
    targets = [
        paths.gpec_ideal_ods(SHOT, "gpec"),
        paths.gpec_ideal_manifest(SHOT, "gpec"),
        *(paths.mhd_linear_ods(SHOT, module) for module in ("dcon", "rdcon", "stride")),
    ]

    result = _dry_run(tmp_path, targets)

    assert result.returncode == 0, result.stderr[-3000:]
    assert "build_gpec_ideal" in result.stdout
    assert "build_mhd_linear" in result.stdout


def test_the_full_target_set_resolves_once_preflight_has_run(tmp_path):
    """`rule all` asks for most of its targets only after the preflight checkpoint.

    Its input functions -- the validation plots, the replication records, every
    per-product stage -- run when Snakemake re-evaluates the DAG with the
    checkpoint's output in hand, not while it reads the file. A dry run with no
    preflight output stops at the checkpoint and never calls them, so both tests
    above pass while a path those functions build can still be unresolvable.
    Writing the checkpoint's output first is what makes the dry run reach them.
    """
    paths = _paths(tmp_path)
    raw_dump = Path(paths.raw_dump(SHOT))
    raw_dump.parent.mkdir(parents=True, exist_ok=True)
    raw_dump.write_bytes(b"")
    Path(paths.raw_manifest(SHOT)).write_text("{}", encoding="utf-8")
    eligible = Path(paths.preflight_eligible())
    eligible.parent.mkdir(parents=True, exist_ok=True)
    eligible.write_text(json.dumps({"eligible_shots": [SHOT]}), encoding="utf-8")
    Path(paths.preflight_excluded()).write_text(
        json.dumps({"excluded_shots": []}), encoding="utf-8"
    )

    result = _dry_run(tmp_path)

    assert result.returncode == 0, result.stderr[-3000:]
    assert "build_gpec_ideal" in result.stdout
    assert "plot_mhd_linear" in result.stdout


# --------------------------------------------------------------------------- #
# layout: shot_first (cold review data F4)
# --------------------------------------------------------------------------- #


def test_the_snakefile_parses_under_the_shot_first_layout(tmp_path):
    """The `PipelinePaths` default and the documented legacy-diff layout.

    It could not start: `impa_selection` named an unimported `Path`, and past
    that the per-product rules asked `product_pattern` for a `{product}`
    wildcard in paths the shot-first layout resolved without the product.
    """
    result = _dry_run(tmp_path, layout="shot_first")
    assert result.returncode == 0, result.stderr[-3000:]


def test_shot_first_reaches_the_per_product_stage_by_the_path_that_names_it(tmp_path):
    paths = _paths(tmp_path, "shot_first")
    targets = [
        paths.mhd_linear_ods(SHOT, module) for module in ("dcon", "rdcon", "stride")
    ]
    assert len(set(targets)) == 3, "one product owns one mhd_linear"

    result = _dry_run(tmp_path, targets, layout="shot_first")

    assert result.returncode == 0, result.stderr[-3000:]
    assert "build_mhd_linear" in result.stdout
