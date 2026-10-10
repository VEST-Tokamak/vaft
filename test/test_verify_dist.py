"""``test/verify_dist.py`` and the packaging rules it enforces (#1755, #1756).

The distribution policy is spread over three files that must agree:
``[tool.setuptools.package-data]`` decides what the wheel carries,
``MANIFEST.in`` what the sdist carries, and ``verify_dist.REQUIRED_FILES``
what a built distribution is refused without.  These tests pin the files that
moved under the package so a PyPI install has them.
"""

from __future__ import annotations

import fnmatch
import sys
import zipfile
from importlib.resources import files
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
import verify_dist  # noqa: E402

tomllib = pytest.importorskip("tomllib")

#: The hosted-GUI deployment templates (#1755), relative to ``vaft/``.
DEPLOY_FILES = (
    "deploy/gui/vaft-gui.service",
    "deploy/gui/vaft-gui.env.example",
    "deploy/gui/nginx-vaft-gui.conf",
    "deploy/gui/vaft-gui-proxy.conf",
)


def _package_data_globs() -> list[str]:
    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["setuptools"]["package-data"]["vaft"]


def _manifest_includes() -> list[str]:
    return [
        line.split(None, 1)[1].strip()
        for line in (ROOT / "MANIFEST.in").read_text(encoding="utf-8").splitlines()
        if line.startswith("include ")
    ]


def _wheel_of(tmp_path: Path, names: set[str], tag: str) -> Path:
    path = tmp_path / f"vaft-0.0-{tag}-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as archive:
        for name in sorted(names):
            archive.writestr(name, "x")
    return path


# ---------------------------------------------------------------------------
# #1755: deploy/gui lives inside the package and ships
# ---------------------------------------------------------------------------


def test_the_gui_deployment_files_moved_under_the_package():
    assert not (ROOT / "deploy").exists(), "deploy/ must not come back at the repository root"
    for relative in DEPLOY_FILES:
        assert (ROOT / "vaft" / relative).is_file(), relative
    assert (ROOT / "vaft" / "deploy" / "__init__.py").is_file()
    assert (ROOT / "vaft" / "deploy" / "gui" / "__init__.py").is_file()


@pytest.mark.parametrize("relative", DEPLOY_FILES)
def test_every_packaging_rule_ships_the_gui_deployment_file(relative):
    globs = _package_data_globs()
    assert any(fnmatch.fnmatch(relative, pattern) for pattern in globs), (
        f"no [tool.setuptools.package-data] glob matches {relative}; the wheel would omit it"
    )
    assert any(fnmatch.fnmatch(f"vaft/{relative}", pattern) for pattern in _manifest_includes()), (
        f"MANIFEST.in does not name {relative}"
    )
    assert f"vaft/{relative}" in verify_dist.REQUIRED_FILES, (
        "test/verify_dist.py would accept a distribution without it"
    )


def test_verify_dist_refuses_a_wheel_without_the_gui_deployment_files(tmp_path):
    verify_dist._verify_distribution(_wheel_of(tmp_path, verify_dist.REQUIRED_FILES, "all"))
    for relative in DEPLOY_FILES:
        name = f"vaft/{relative}"
        with pytest.raises(ValueError, match=name.replace(".", r"\.")):
            verify_dist._verify_distribution(
                _wheel_of(tmp_path, verify_dist.REQUIRED_FILES - {name}, Path(relative).stem)
            )


def test_the_gui_deployment_files_are_reachable_as_package_resources():
    """Code and documentation read them through ``importlib.resources``, never
    through a repository-relative path, so a wheel install serves them too."""
    directory = files("vaft.deploy.gui")
    for relative in DEPLOY_FILES:
        resource = directory.joinpath(Path(relative).name)
        assert resource.is_file(), relative
        assert resource.read_text(encoding="utf-8").strip()
    unit = directory.joinpath("vaft-gui.service").read_text(encoding="utf-8")
    assert "vaft gui --hosted" in unit


def test_no_documentation_links_the_old_deployment_path():
    pages = list((ROOT / "docs").rglob("*.md")) + [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    offenders = [
        page.relative_to(ROOT).as_posix()
        for page in pages
        if "tree/develop/deploy/gui" in page.read_text(encoding="utf-8")
    ]
    assert offenders == []


# ---------------------------------------------------------------------------
# #1756: the issue studies live under vaft/validation/studies; only scripts ship
# ---------------------------------------------------------------------------

STUDIES = ROOT / "vaft" / "validation" / "studies"
STUDY_FOLDERS = ("ec_launcher_cad_266", "fixed_free_1608", "nice_issue_666", "nice_issue_666_superseded")


def test_the_issue_studies_moved_under_the_validation_package():
    assert not (ROOT / "validation").exists(), "validation/ must not come back at the repository root"
    for folder in STUDY_FOLDERS:
        assert (STUDIES / folder).is_dir(), folder
    # every folder with scripts is a regular package, so `python -m` reaches it
    for script in STUDIES.rglob("*.py"):
        assert (script.parent / "__init__.py").is_file(), script.relative_to(ROOT).as_posix()


def test_the_shipped_study_scripts_are_required_and_the_rest_excluded():
    shipped = {f"vaft/validation/studies/{name}" for name in verify_dist._SHIPPED_STUDY_SCRIPTS}
    assert shipped <= verify_dist.REQUIRED_FILES
    on_disk = {
        path.relative_to(ROOT).as_posix()
        for folder in verify_dist._SHIPPED_STUDIES
        for path in (STUDIES / folder).glob("*.py")
        if path.name != "__init__.py"
    }
    assert on_disk == shipped, "a shipped study folder gained or lost a script; update verify_dist"

    config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    excluded = config["tool"]["setuptools"]["packages"]["find"]["exclude"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for folder in STUDY_FOLDERS:
        if folder in verify_dist._SHIPPED_STUDIES:
            assert not any(fnmatch.fnmatch(f"vaft.validation.studies.{folder}", p) for p in excluded), folder
            continue
        assert any(fnmatch.fnmatch(f"vaft.validation.studies.{folder}", p) for p in excluded), (
            f"{folder} is not excluded from the wheel in [tool.setuptools.packages.find]"
        )
        assert f"prune vaft/validation/studies/{folder}" in manifest, f"{folder} is not pruned from the sdist"
    # notes and run records of the shipped folder stay in the repository
    for name in ("README.md", "measured_matrix.json"):
        assert (STUDIES / "fixed_free_1608" / name).is_file()
        assert f"exclude vaft/validation/studies/fixed_free_1608/{name}" in manifest, name


def test_verify_dist_refuses_repository_only_study_files(tmp_path):
    base = verify_dist.REQUIRED_FILES
    verify_dist._verify_distribution(_wheel_of(tmp_path, base, "studies-ok"))
    for extra in (
        "vaft/validation/studies/fixed_free_1608/README.md",
        "vaft/validation/studies/fixed_free_1608/measured_matrix.json",
        "vaft/validation/studies/nice_issue_666/__init__.py",
        "vaft/validation/studies/nice_issue_666/synthetic_equilibria/run_solovev.py",
        "vaft/validation/studies/ec_launcher_cad_266/extract_launcher_geometry.py",
        "vaft/validation/studies/nice_issue_666_superseded/README.md",
    ):
        with pytest.raises(ValueError, match="repository-only issue-study"):
            verify_dist._verify_distribution(_wheel_of(tmp_path, base | {extra}, Path(extra).stem))
    with pytest.raises(ValueError, match="missing required files"):
        verify_dist._verify_distribution(
            _wheel_of(tmp_path, base - {"vaft/validation/studies/fixed_free_1608/direct_fit.py"}, "no-direct-fit")
        )


def test_no_reference_to_the_old_study_paths_remains():
    import subprocess

    out = subprocess.run(
        ["git", "grep", "-l", "-E", r"(^|[^/a-z_])validation/(ec_launcher_cad_266|fixed_free_1608|nice_issue_666)"],
        cwd=ROOT, capture_output=True, text=True,
    )
    if out.returncode == 128:  # not a git checkout (an sdist): nothing to scan
        pytest.skip(out.stderr.strip())
    offenders = [line for line in out.stdout.split() if not line.startswith("vaft/validation/studies/")]
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# release 0.8.0: samples/39915/imas.nc is repository-only; only the OMAS form ships
# ---------------------------------------------------------------------------

IMAS_TWIN = "vaft/data/samples/39915/imas.nc"
OMAS_SAMPLE = "vaft/data/samples/39915/omas.json.gz"


def test_the_39915_imas_twin_is_in_no_packaging_rule():
    """The wheel was 26.04 MiB against a 26 MiB cap with both forms of the same
    product; the IMAS netCDF form is repository-only and the OMAS form ships."""
    globs = _package_data_globs()
    assert not any(fnmatch.fnmatch("data/samples/39915/imas.nc", pattern) for pattern in globs)
    assert any(fnmatch.fnmatch("data/samples/39915/omas.json.gz", pattern) for pattern in globs)
    includes = _manifest_includes()
    assert not any(fnmatch.fnmatch(IMAS_TWIN, pattern) for pattern in includes)
    assert any(fnmatch.fnmatch(OMAS_SAMPLE, pattern) for pattern in includes)
    assert IMAS_TWIN not in verify_dist.REQUIRED_FILES
    assert "samples/39915/imas.nc" not in verify_dist._ALLOWED_DATA_FILES
    assert OMAS_SAMPLE in verify_dist.REQUIRED_FILES
    # the build hook no longer swaps the twin into the build output either
    setup_py = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert '"imas.nc"' not in setup_py
    assert '"omas.json.gz"' in setup_py


def test_sample_manifests_mark_the_39915_imas_twin_repository_only():
    for root in ("samples", "wheel_samples"):
        manifest = (ROOT / "vaft" / "data" / root / "39915" / "manifest.yaml").read_text(encoding="utf-8")
        representations = yaml.safe_load(manifest)["representations"]
        assert representations["omas"]["package"] == "wheel-and-sdist", root
        assert representations["imas"]["package"] == "repository-only", root
        assert "imas" in representations["omas"]["compatible_adapters"], root


def test_verify_dist_refuses_a_distribution_carrying_the_39915_imas_twin(tmp_path):
    base = verify_dist.REQUIRED_FILES
    verify_dist._verify_distribution(_wheel_of(tmp_path, base, "no-imas-twin"))
    with pytest.raises(ValueError, match="repository-only data.*samples/39915/imas\\.nc"):
        verify_dist._verify_distribution(_wheel_of(tmp_path, base | {IMAS_TWIN}, "imas-twin"))
    with pytest.raises(ValueError, match="missing required files.*omas\\.json\\.gz"):
        verify_dist._verify_distribution(_wheel_of(tmp_path, base - {OMAS_SAMPLE}, "no-omas"))


WHEEL_SAMPLE_HOOK_FILES = {
    "vaft/data/wheel_samples/39915/manifest.yaml",
    "vaft/data/wheel_samples/39915/omas.json.gz",
}
WHEEL_SAMPLE_IMAS_TWIN = "vaft/data/wheel_samples/39915/imas.nc"


def _sdist_of(tmp_path: Path, names: set[str], tag: str) -> Path:
    import io
    import tarfile

    path = tmp_path / f"vaft-0.0-{tag}.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name in sorted(names):
            info = tarfile.TarInfo(f"vaft-0.0/{name}")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    return path


def test_the_sdist_carries_only_the_wheel_sample_files_the_build_hook_reads(tmp_path):
    """The compact 39915 variant exists for setup.py's build_py hook, which
    reads manifest.yaml and omas.json.gz from wheel_samples/; its imas.nc was
    9.5 MiB of sdist that nothing reads once the hook stops copying it."""
    includes = _manifest_includes()
    for name in WHEEL_SAMPLE_HOOK_FILES:
        assert any(fnmatch.fnmatch(name, pattern) for pattern in includes), name
    assert not any(fnmatch.fnmatch(WHEEL_SAMPLE_IMAS_TWIN, pattern) for pattern in includes)
    assert {name.removeprefix("vaft/data/") for name in WHEEL_SAMPLE_HOOK_FILES} == verify_dist._SDIST_ONLY_DATA_FILES
    # nothing under vaft/data/wheel_samples/ is package data for the wheel
    assert not any("wheel_samples" in pattern for pattern in _package_data_globs())

    base = verify_dist.REQUIRED_FILES
    verify_dist._verify_distribution(_sdist_of(tmp_path, base | WHEEL_SAMPLE_HOOK_FILES, "hook-files"))
    with pytest.raises(ValueError, match="repository-only data.*wheel_samples/39915/imas\\.nc"):
        verify_dist._verify_distribution(
            _sdist_of(tmp_path, base | WHEEL_SAMPLE_HOOK_FILES | {WHEEL_SAMPLE_IMAS_TWIN}, "imas-twin")
        )
    with pytest.raises(ValueError, match="missing the compact sample the build hook reads"):
        verify_dist._verify_distribution(_sdist_of(tmp_path, base, "no-hook-files"))
    # and no wheel_samples/ path ships in the wheel at all
    for name in sorted(WHEEL_SAMPLE_HOOK_FILES | {WHEEL_SAMPLE_IMAS_TWIN}):
        with pytest.raises(ValueError, match="wheel_samples"):
            verify_dist._verify_distribution(_wheel_of(tmp_path, base | {name}, Path(name).stem))
