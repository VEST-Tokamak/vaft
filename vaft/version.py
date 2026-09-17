# Version information
__version__ = "0.7.0"


# ────────────────────────────────────────────────────────
# patch notes
# ────────────────────────────────────────────────────────
# 0.7.0
# - development release line 2026-09-03 .. 2026-09-16 merged into main: 261
#   pull requests; the detailed notes are on the release pull request and the
#   GitHub release. Headlines:
# - discharge timing is detected, not assumed: H-alpha onset, loop-voltage
#   zero crossing, one main discharge window shared by every consumer
#   (EFIT constraint times, eddy window, plasma features, shot class); the
#   legacy onset detectors are retired and the evidence corpus ships as data
#   (#409, #842)
# - plotting: every plot has plot_*/dd_*/extract_* on one DD path grammar,
#   computed views declare what they read and most run natively on IMAS and
#   lazy HSDS handles, backend="plotly", interactive=True controls, `vaft plot`
#   from the shell, screen-sized figures by default, theme/colour semantics,
#   overview panels chosen live (#63, #434, #439, #480-#484, #689, #709-#712)
# - equilibrium: the packaged shot stores psi in DD weber and declares its
#   COCOS, psi maps take units=, five validated radial coordinates on the
#   profiles, q from the 2-D map, GS residual and virial closures as a
#   framework, triangularity at the vertical extrema (#478, #479, #365)
# - EFIT: Green-table provenance with build/generate/compare/A-B and the
#   regenerated table, one channel decision per slice formed upstream, the
#   VEST-quality reference set and VEST envelope for accepting a fit,
#   profile-model and constraint-identifiability studies, k-file A/B harness,
#   readable EFIT logs and per-regime failure records (#171, #194, #296,
#   #579, #588, #649, #695, #708, #76)
# - stability: DCON summary/profile/edge-scan parsers, energy eigenmodes and
#   eigenfunctions mapped to IMAS, GPEC profile, singcoup and spectral-field
#   readers, resonant response and island overlap, each stability product in
#   its own mhd_linear, segment-wise wall eigenmode basis with a
#   response-ranked reduced order (#473, #494)
# - new code adapters: NUBEAM (beam, fast-ion distributions, native Windows),
#   TRANSP readers, GACODE NEO/TGLF/TGLF-NN with Sauter/Redl comparison and
#   input.gacode confirmed COCOS 2, FLARE interoperability core, MARS
#   PROF*.IN, Osborne p-files (#550, #553, #741-#744)
# - diagnostics: toroidal angles derived from the VEST port clock and the
#   measured mode-number sign convention, zero-phase SXR filtering, IMPA
#   defaults reconciled with vest.yaml (Hall gain -2/15 T/V), soft X-ray and
#   camera data consolidated with a publication contract, FAST-camera
#   fluctuation analysis phase 1 (#161, #624, #625, #626, #638, #718)
# - database: every OMAS stage product gzipped and holding only the IDS its
#   stage owns, a shot's master written last and merged before it lands,
#   HSDS source hierarchy, constant-cost source probes (#813, #787)
# - process contract: every processing module documented under one docstring
#   contract with a shared parser and catalog; kinetic-profile radial
#   coordinate is a named choice (#417-#421, #420)
# - startup formula layer: breakdown chain through the avalanche, vacuum-field
#   view with |E_phi|, breakdown figure and Lloyd margin (#783, #676)
# - platform: CHEASE, DCON/GPEC, EFIT/EFUND, NUBEAM build and run natively on
#   Windows; one recipe set under install/ builds CHEASE, GPEC, GACODE and
#   NUBEAM on Linux and macOS; GPEC pins its netCDF; TokaMaker is an optional
#   extra (#226 closed out)
# - CI: develop asks for development confidence (core selection), main for
#   release confidence (full suite on Linux and Windows, tutorials); main's
#   branch protection is code (#515, #471)
# - tutorials: sessions 02, 03 and 04 with one QMD presentation source; the
#   docs' code samples are executed
# - deprecated: current_density_from_psi (sign fixed, #355); the legacy onset
#   detectors and the chease-mhd-stability product are retired behind gates;
#   the old figure sizes remain as format="legacy"
# - known limitations: TGLF-NN has no VEST-trained model and refuses
#   out-of-domain input; os.access(X_OK) still says nothing on Windows
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
# - repository files are read and written as UTF-8 rather than at the locale's
#   encoding, which raised UnicodeDecodeError on a cp949 host
# - content-addressed fixtures are pinned to LF: a CRLF checkout changed their
#   sha256 and failed their own offline verification
# - known limitation: `os.access(X_OK)` is meaningless on Windows, so an
#   external-code executable that cannot actually be launched is reported as
#   an opaque WinError 193 rather than a named error. Left alone here because
#   a real probe costs the platform-independent resolution-precedence tests
#   their Windows coverage; to be fixed with issue #226.
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

