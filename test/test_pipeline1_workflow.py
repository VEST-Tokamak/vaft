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


def _config(tmp_path: Path, layout: str = "filedb", overrides: dict | None = None) -> Path:
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
                **(overrides or {}),
            }
        ),
        encoding="utf-8",
    )
    return config


def _dry_run(
    tmp_path: Path, targets: list[str] | None = None, *, layout: str = "filedb",
    extra: list[str] | None = None, config_overrides: dict | None = None,
):
    # config.yaml interpolates these; a dry run never executes them.
    env = dict(os.environ)
    for name in ("VAFT_FILEDB_DIR", "VAFT_DATA_DIR", "EFIT", "CHEASE", "GPECHOME"):
        env.setdefault(name, str(tmp_path / name.lower()))
    return subprocess.run(
        [
            sys.executable, "-m", "snakemake",
            "--snakefile", str(WORKFLOW / "Snakefile"),
            "--configfile", str(_config(tmp_path, layout, config_overrides)),
            "--directory", str(tmp_path),
            "--cores", "1", "-n",
            # Targets before the options: `--config` takes every following
            # word as a name=value pair.
            *(targets or []),
            *(extra or []),
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


def _kfile_command(stdout: str) -> str:
    """The k-file stage's printed shell command, up to its log redirect.

    Whitespace is collapsed: Snakemake 7 prints a ``{param}`` substitution
    with the template's indentation around it (``--preset  statistical_891``),
    Snakemake 9 does not, and the assertions are about the words.
    """
    start = stdout.index("generate_kfile.py")
    return " ".join(stdout[start:stdout.index(">", start)].split())


def test_an_efit_preset_reaches_the_kfile_stage_in_place_of_the_basis(tmp_path):
    """#891: `efit.preset` selects a named configuration for every shot's k-files."""
    target = [_paths(tmp_path).kfile_manifest(SHOT)]
    preset = _dry_run(tmp_path, target, extra=["-p", "--config", 'efit={"preset": "statistical_891"}'])
    assert preset.returncode == 0, preset.stderr[-3000:]
    command = _kfile_command(preset.stdout)
    assert "--preset statistical_891" in command
    assert "--npprime" not in command  # the constraints stage still takes its own
    # Naming nothing is the library default (statistical_891 since 2026-10-01):
    # no basis and no preset flag; generate_kfile.py records the default itself.
    default = _dry_run(tmp_path, target, extra=["-p"])
    assert default.returncode == 0, default.stderr[-3000:]
    command = _kfile_command(default.stdout)
    assert "--preset" not in command and "--npprime" not in command


def test_the_routine_preset_is_an_explicit_opt_out(tmp_path):
    target = [_paths(tmp_path).kfile_manifest(SHOT)]
    result = _dry_run(tmp_path, target, extra=["-p", "--config", 'efit={"preset": "routine"}'])
    assert result.returncode == 0, result.stderr[-3000:]
    command = _kfile_command(result.stdout)
    assert "--preset routine" in command and "--npprime" not in command


def test_an_unknown_efit_preset_fails_the_run_before_any_job(tmp_path):
    result = _dry_run(tmp_path, extra=["--config", 'efit={"preset": "statistical"}'])
    assert result.returncode != 0
    assert "unknown EFIT preset" in (result.stderr + result.stdout)


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


def _preflight_done(tmp_path):
    paths = _paths(tmp_path)
    raw_dump = Path(paths.raw_dump(SHOT))
    raw_dump.parent.mkdir(parents=True, exist_ok=True)
    raw_dump.write_bytes(b"")
    Path(paths.raw_manifest(SHOT)).write_text("{}", encoding="utf-8")
    eligible = Path(paths.preflight_eligible())
    eligible.parent.mkdir(parents=True, exist_ok=True)
    eligible.write_text(json.dumps({"eligible_shots": [SHOT]}), encoding="utf-8")
    Path(paths.preflight_excluded()).write_text(json.dumps({"excluded_shots": []}), encoding="utf-8")


def test_a_stage_scope_narrows_rule_all(tmp_path):
    """`stages: [raw, diagnostics, eddy]` asks for nothing of EFIT and after (#58).

    Not even the constraint step, which refuses a vacuum shot by failing: a
    scope that left it in would still fail every vacuum shot.
    """
    _preflight_done(tmp_path)

    result = _dry_run(tmp_path, extra=["--config", "stages=[raw,diagnostics,eddy]"])

    assert result.returncode == 0, result.stderr[-3000:]
    for rule in ("generate_diagnostics_ods", "generate_eddy_ods",
                 "replicate_diagnostics_to_hsds", "replicate_eddy_to_hsds", "plot_eddy"):
        assert rule in result.stdout, rule
    for rule in ("generate_constraints_ods", "generate_kfile", "run_efit_reconstruction",
                 "replicate_efit_to_hsds", "run_chease", "plot_mhd_linear", "build_gpec_ideal"):
        assert rule not in result.stdout, rule


def test_a_stability_scope_runs_only_its_own_gpec_cells(tmp_path):
    """`mhd_linear` without `gpec_ideal` must not run the ideal-GPEC solver (cold review)."""
    _preflight_done(tmp_path)

    result = _dry_run(tmp_path, extra=["-p", "--config",
                                       "stages=[raw,diagnostics,eddy,efit,chease,mhd_linear]"])

    assert result.returncode == 0, result.stderr[-3000:]
    assert "plot_mhd_linear" in result.stdout
    assert "build_gpec_ideal" not in result.stdout
    assert "product=ideal-gpec" not in result.stdout


@pytest.mark.parametrize(
    ("stages", "message"),
    [("[raw,diagnostic]", "Unknown stage"), ("[raw,diagnostics,efit]", "but not 'eddy'")],
)
def test_a_stage_scope_that_cannot_be_run_fails_before_any_job(tmp_path, stages, message):
    result = _dry_run(tmp_path, extra=["--config", f"stages={stages}"])
    assert result.returncode != 0
    assert message in result.stdout + result.stderr


def _shell_args(stdout: str, flag: str) -> set[str]:
    import re

    return set(re.findall(re.escape(flag) + r" +(\S*)", " ".join(stdout.split())))


def test_a_run_directory_does_not_hide_the_workflow_config(tmp_path):
    """#1530: `configfile: "config.yaml"` resolved against `--directory`.

    A run launched with `--directory <run dir> --configfile <partial>` re-read
    its own partial config and never loaded the workflow's, so EFIT ran with
    `run=false`, `args 65`, and the eddy solve had no plasma filament. The
    partial config here names neither section; the workflow defaults must win.
    """
    _preflight_done(tmp_path)

    result = _dry_run(tmp_path, extra=["-p", "--config", "stages=[raw,diagnostics,eddy,efit]"])

    assert result.returncode == 0, result.stderr[-3000:]
    assert _shell_args(result.stdout, "--filament-r") == {'"0.35,0.35,0.35"'}
    assert _shell_args(result.stdout, "--filament-z") == {'"0.25,0.0,-0.25"'}
    assert "true" in _shell_args(result.stdout, "--run")
    assert "false" not in _shell_args(result.stdout, "--run")
    assert _shell_args(result.stdout, "--args") == {'"129"'}
    assert _shell_args(result.stdout, "--detect-broken") == {"true"}


def test_the_run_config_still_overrides_the_workflow_config(tmp_path):
    """The run's `--configfile` is merged over the workflow's, not under it.

    The run config switches EFIT off and that reaches the rules, while the EFIT
    keys it leaves out (`args 129`) still come from the workflow: a deep merge,
    not a replacement -- before #1530 they fell back to code defaults (`65`).
    """
    _preflight_done(tmp_path)

    result = _dry_run(
        tmp_path,
        extra=["-p", "--config", "stages=[raw,diagnostics,eddy,efit]"],
        config_overrides={"efit": {"run": False}},
    )

    assert result.returncode == 0, result.stderr[-3000:]
    assert "replicate_eddy_to_hsds" in result.stdout
    run_flags = _shell_args(result.stdout, "--run")
    assert "false" in run_flags and "true" not in run_flags
    assert _shell_args(result.stdout, "--args") == {'"129"'}


def test_the_deployment_guide_states_what_a_partial_run_config_inherits():
    """cold review 0.8.0 delta-absorb-16 F5: #1687 made a run config that omits
    `efit.run` inherit the workflow's `true` (the 10-06 redeploy had to pin it
    off), and nothing documented which `vest.magnetics.processing` keys a run
    config may still override after #1541/#1731. The guide, the worker example
    and the workflow config must say both."""
    guide = (WORKFLOW / "DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "A run config that omits `efit.run` inherits `true`" in guide
    assert "What a run config may override in the equilibrium magnetics" in guide
    assert "`time_start`, `time_end`, `sample_count` unless `window_override`" in guide
    assert "vest_magnetics_processing_effective" in guide
    example = (WORKFLOW / "worker.example.yaml").read_text(encoding="utf-8")
    assert "`efit.run: true` included" in example
    workflow_config = (WORKFLOW / "config.yaml").read_text(encoding="utf-8")
    assert "inherits `true` when it omits the key" in workflow_config


@pytest.mark.parametrize("missing", ["base_dir", "shots"])
def test_a_run_config_must_say_where_and_which_shots(tmp_path, missing):
    """Loading the workflow defaults must not turn a forgotten base_dir into a production write."""
    payload = json.loads(_config(tmp_path).read_text(encoding="utf-8"))
    payload.pop(missing)
    partial = tmp_path / "partial.yaml"
    partial.write_text(json.dumps(payload), encoding="utf-8")
    env = dict(os.environ)
    for name in ("VAFT_FILEDB_DIR", "VAFT_DATA_DIR", "EFIT", "CHEASE", "GPECHOME"):
        env.setdefault(name, str(tmp_path / name.lower()))
    result = subprocess.run(
        [sys.executable, "-m", "snakemake", "--snakefile", str(WORKFLOW / "Snakefile"),
         "--configfile", str(partial), "--directory", str(tmp_path), "--cores", "1", "-n"],
        capture_output=True, text=True, env=env, cwd=WORKFLOW,
    )
    assert result.returncode != 0
    assert f"does not set {missing}" in result.stdout + result.stderr


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


# --------------------------------------------------------------------------- #
# CHEASE refinement: a run of solver verdicts is a result, not a missing one
# --------------------------------------------------------------------------- #
def _run_chease_refinement(tmp_path, monkeypatch, run_chease):
    """Run the refinement script in-process against a fake `run_chease`."""
    import runpy
    from types import SimpleNamespace

    from vaft.code import chease as chease_module

    gfile = tmp_path / "g039915.00316"
    gfile.write_text("not read: prepare_chease_inputs is stubbed\n", encoding="utf-8")
    manifest = tmp_path / "gfiles.txt"
    manifest.write_text(f"{gfile}\n", encoding="utf-8")
    monkeypatch.setattr(chease_module, "find_chease_executable", lambda config: Path("/stub/chease"))
    monkeypatch.setattr(chease_module, "prepare_chease_inputs", lambda gfile, config: SimpleNamespace(gfile=gfile))
    monkeypatch.setattr(chease_module, "run_chease", run_chease)
    output = tmp_path / "chease" / "refined.txt"
    status = tmp_path / "chease" / "status.txt"
    monkeypatch.setattr(sys, "argv", [
        "run_chease_refinement.py", "--shot", str(SHOT), "--gfile-manifest", str(manifest),
        "--output", str(output), "--status", str(status), "--run", "true", "--timeout", "1",
        "--create-plot", "false", "--plot-dir", str(tmp_path / "plot"),
    ])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(WORKFLOW / "run_chease_refinement.py"), run_name="__main__")
    return exit_info.value.code, output, status, json.loads((output.parent / "chease_runs.json").read_text())


def test_an_all_timeout_chease_run_is_a_recorded_result(tmp_path, monkeypatch):
    """cold review 0.8.0 workflows F5: exit 1 made Snakemake drop the outputs and
    the worker re-run every slice until it gave up on the same verdict."""
    from types import SimpleNamespace

    def timed_out(inputs, config):
        return SimpleNamespace(returncode=None, refined_geqdsk=None, comparison={}, stderr="timed out")

    code, output, status, runs = _run_chease_refinement(tmp_path, monkeypatch, timed_out)
    assert code == 0
    assert output.read_text(encoding="utf-8") == ""
    assert status.read_text(encoding="utf-8").startswith("failed: refined_gfiles=0; failed=1")
    (record,) = runs["records"]
    assert record["status"] == "failed" and record["returncode"] is None


def test_a_chease_run_that_raised_still_fails_the_rule(tmp_path, monkeypatch):
    def broken(inputs, config):
        raise RuntimeError("the work directory vanished")

    code, _output, status, runs = _run_chease_refinement(tmp_path, monkeypatch, broken)
    assert code == 1
    assert status.read_text(encoding="utf-8").startswith("failed:")
    assert runs["records"][0]["status"] == "error"
