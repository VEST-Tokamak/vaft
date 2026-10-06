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
