"""Rate reductions must be explicit, not a side effect of ``np.interp`` (#425).

Interpolating a signal onto a coarser grid silently folds everything above the
new Nyquist frequency back into the band.  VEST makes this easy to do by
accident: the fast DAQ runs at 250 kHz (``FAST_DT``) while every processed time
grid in ``vest.yaml`` is 25 kHz, so any fast channel written onto a policy grid
is a 10x decimation.

So inside the mapping and processing layers, a time-domain interpolation must
either go through :func:`vaft.process.signal_processing.resample_to_time` --
which anti-aliases when the rate really drops and is bit-for-bit ``np.interp``
when it does not -- or carry an ``# anti-alias:`` comment recording which of the
audit's categories the site falls into.  The audit table lives in
``docs/_guide/Processing.md``; this test is what keeps it true.

Modules that interpolate over *space* (psi, rho, R-Z) rather than time are
exempt wholesale: there is no sampling rate to reduce.
"""

import ast
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "vaft"
SCANNED_DIRS = ("machine_mapping", "process")

#: Interpolating constructors that can silently perform a rate reduction.
INTERPOLATORS = {"interp", "interp1d", "CubicSpline", "PchipInterpolator"}

#: The marker a call site uses to record its audit classification.
MARKER = "# anti-alias:"

#: How far above a call the marker may sit, in lines.  A call spread over a
#: ``set_path(...)`` block still has its comment within reach of the statement.
MARKER_LOOKBACK = 8

#: Modules whose interpolation is over a spatial or flux coordinate, not time.
#: This set must only shrink.  ``signal_processing.py`` is exempt because it
#: *is* the primitive.
SPATIAL_ONLY_MODULES = {
    "process/_equilibrium_parametric.py",
    "process/atomic.py",
    "process/equilibrium.py",
    "process/profile.py",
    "process/signal_processing.py",
    "process/soft_x_rays.py",
}


def _scanned_files():
    for directory in SCANNED_DIRS:
        yield from sorted((PACKAGE_ROOT / directory).rglob("*.py"))


def _interpolating_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            name = func.attr
        elif isinstance(func, ast.Name):
            name = func.id
        else:
            continue
        if name in INTERPOLATORS:
            found.append((node.lineno, name))
    return found


def _unjustified(path: Path) -> list[str]:
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    offenders = []
    for lineno, name in sorted(set(_interpolating_calls(ast.parse(source, filename=str(path))))):
        start = max(0, lineno - 1 - MARKER_LOOKBACK)
        context = "\n".join(lines[start:lineno])
        if MARKER in context:
            continue
        offenders.append(f"line {lineno}: {name}")
    return offenders


@pytest.mark.parametrize(
    "path", list(_scanned_files()), ids=lambda p: str(p.relative_to(PACKAGE_ROOT))
)
def test_time_domain_interpolation_is_classified(path):
    relative = str(path.relative_to(PACKAGE_ROOT))
    if relative in SPATIAL_ONLY_MODULES:
        pytest.skip(f"{relative} interpolates over space, not time")
    offenders = _unjustified(path)
    assert not offenders, (
        f"{relative} interpolates without recording whether it reduces the sample "
        f"rate: {offenders}. Route it through resample_to_time(), or add an "
        f"'{MARKER} ...' comment saying why a bare interpolation is correct here."
    )


def test_spatial_allowlist_entries_still_exist_and_still_interpolate():
    for relative in sorted(SPATIAL_ONLY_MODULES):
        path = PACKAGE_ROOT / relative
        assert path.exists(), f"{relative} is allowlisted but missing"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        assert _interpolating_calls(tree), (
            f"{relative} no longer interpolates; remove it from SPATIAL_ONLY_MODULES"
        )


def test_the_marker_is_not_accepted_from_an_unrelated_distance(tmp_path):
    # Guards the guard: a marker far above an interpolation must not launder it.
    module = tmp_path / "far.py"
    module.write_text(
        f"{MARKER} unrelated\n" + "x = 1\n" * (MARKER_LOOKBACK + 2) + "y = np.interp(a, b, c)\n",
        encoding="utf-8",
    )
    assert _unjustified(module)
