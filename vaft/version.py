# Version information
__version__ = "0.6.2"


# ────────────────────────────────────────────────────────
# patch notes
# ────────────────────────────────────────────────────────
# 0.6.2
# - Windows portability hotfix. install/README.md calls native Windows a
#   first-class path, but the Linux-only CI had never exercised it: a full
#   suite run on Windows failed 25 of 3082 tests, and two defects stopped the
#   suite being collected at all. Linux and macOS are unaffected.
# - `import fcntl` at module scope made `vaft raw-redump`/`raw-upgrade`
#   unimportable on Windows and aborted pytest collection for the whole
#   repository; file locking now uses msvcrt there
# - `omas.omas_imas` reads os.environ["HOME"] in a default argument, so it
#   raised KeyError at import time on a stock Windows install; vaft.compat now
#   establishes HOME from USERPROFILE before any optional dependency loads
# - four NamedTemporaryFile sites handed a still-open path to omas.ODS.load /
#   ODS.save, which Windows refuses with PermissionError; this broke
#   vaft.omas.save to .json.gz and silently disabled the derived-ODS cache
# - imas_core keeps IDS files over ~10 MB open after DBEntry.close(), so
#   scratch cleanup failed work that had already succeeded; cleanup is now
#   best-effort on Windows and still strict on POSIX
# - GPEC namelists were quoted with json.dumps, which escapes a backslash and
#   so doubled every separator in a Windows path GPEC then failed to resolve
# - pipeline rule paths are serialized in Snakemake's slash grammar rather
#   than the host's native separator
# - `os.access(X_OK)` is meaningless on Windows, so external-code guards could
#   not reject a non-executable; replaced by a real probe
# - content-addressed fixtures are pinned to LF: a CRLF checkout changed their
#   sha256 and failed their own offline verification
# - Package CI gains a windows-latest leg so none of this regresses silently
# 0.6.1
# - regenerate the packaged wheel sample so the bundled 39915 reference
#   carries the IMPA b_field_tor_probe toroidal_angle the mapper writes
#   (pi/2); 0.6.0 shipped the pre-fix 0.0
# - reconcile the release line with develop: the 0.6.0 stabilization fixes,
#   the docs/ site and the regenerated samples now live on one branch
# - vaft.formula catalog and its generated reference pages ship for the first
#   time, declared in docs/generators.yml
# - move the compact wheel sample to vaft/data/wheel_samples/39915; it ships
#   in the sdist for the build hook and is kept out of the wheel
# 0.6.0
# - canonical FileDB layout, staged Snakemake pipeline, and post-generation
#   validation plots for raw/static/diagnostics/eddy/EFIT/CHEASE stages
# - raw-field-first diagnostics calibration with shot-era rules in vest.yaml
#   (plasma current, PF currents, magnetics, Langmuir, interferometers, IMPA,
#   fluctuation Mirnov, limiter shunts, SXR)
# - corrected b_field_pol_probe poloidal_angle convention (3*pi/2, DD-clockwise)
# - DCON/RDCON/STRIDE adapters with source-verified mhd_linear/ntms mapping
# - TokaMaker forward free-boundary adapter with vessel/eddy v2
# - parametric equilibrium representations and derived descriptors
# - EM Green-function/response-matrix foundation and EQDSK geometry derivation
# - database summary presets, SQL shot discovery, lazy HSDS access
# - plot renderer contracts and registry; notebook and GH Pages docs updates
# 0.5.0
# - public PyPI release
# 0.1.0
# - initial release with basic functions
# ────────────────────────────────────────────────────────

