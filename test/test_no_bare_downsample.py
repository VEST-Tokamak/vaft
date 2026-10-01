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
    "path", list(_scanned_files()), ids=lambda p: p.relative_to(PACKAGE_ROOT).as_posix()
)
def test_time_domain_interpolation_is_classified(path):
    # SPATIAL_ONLY_MODULES is keyed by POSIX path, so compare in that
    # grammar -- str() yields "process\atomic.py" on Windows and never matches.
    relative = path.relative_to(PACKAGE_ROOT).as_posix()
    if relative in SPATIAL_ONLY_MODULES:
        pytest.skip(f"{relative} interpolates over space, not time")
    offenders = _unjustified(path)
    assert not offenders, (
        f"{relative} interpolates without recording whether it reduces the sample "
        f"rate: {offenders}. Route it through resample_to_time(), or add an "
        f"'{MARKER} ...' comment saying why a bare interpolation is correct here."
    )


#: Abscissa names that say an interpolation runs over time, whatever module
#: it sits in.
TIME_ABSCISSAE = {"t", "tt", "time", "times", "t_new", "t_out", "t_grid", "time_s"}


def _time_abscissa_calls(tree: ast.AST) -> list[tuple[int, str]]:
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        first = node.args[0]
        if name in INTERPOLATORS and isinstance(first, ast.Name) and first.id in TIME_ABSCISSAE:
            found.append((node.lineno, f"{name}({first.id}, ...)"))
    return found


@pytest.mark.parametrize("relative", sorted(SPATIAL_ONLY_MODULES))
def test_a_time_domain_interpolation_in_an_exempt_module_still_carries_the_marker(relative):
    """The exemption says the module interpolates over space.  A call whose
    abscissa is named like a time axis contradicts that and must justify
    itself like any scanned module would (cold review 0.8.0
    equilibrium-representation F3: `integrate_romero_closure` upsamples its
    time histories onto RK45 sub-steps behind the exemption)."""
    path = PACKAGE_ROOT / relative
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    offenders = []
    for lineno, name in sorted(set(_time_abscissa_calls(ast.parse(source, filename=str(path))))):
        context = "\n".join(lines[max(0, lineno - 1 - MARKER_LOOKBACK):lineno])
        if MARKER not in context:
            offenders.append(f"line {lineno}: {name}")
    assert not offenders, (
        f"{relative} is exempt as spatial-only but interpolates over a time axis "
        f"without a '{MARKER} ...' comment: {offenders}"
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
