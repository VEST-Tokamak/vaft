# Version information
__version__ = "0.8.0"

__all__ = ["__version__"]


# ────────────────────────────────────────────────────────
# patch notes
# ────────────────────────────────────────────────────────
# 0.8.0
# - development release line 2026-09-18 .. 2026-10-06 merged into main: ~190
#   pull requests; the detailed notes are on the release pull request and
#   the GitHub release. Headlines:
# - equilibrium representations: Fourier, Miller with inboard indentation,
#   Solov'ev fit and X-point topology, Guazzotto-Freidberg, MXH-Chebyshev
#   compact forms, the Contour interchange, toroidal current-density
#   moments, the Grad-Shafranov residual in poloidal harmonics, and one
#   coordinate/projection/representation taxonomy (#941-#945, #948, #1148,
#   #1149, #1166, #1201); Solov'ev f_sign is derived from f_boundary and a
#   contradictory pair is refused (BREAKING, #1307)
# - flux conventions: weber against weber-per-radian decided from contour q
#   as well as Ampere's law, flux helpers take their family explicitly,
#   from_equilibrium writes a per-radian g-file and a COCOS 11 ODS no longer
#   gains 2*pi on core_profiles grid.psi, the field interpolator reads the
#   record's convention, packaged samples declare their COCOS (#354, #1292,
#   #1313, #1372); every volume average takes the plasma from the LCFS
#   outline, not psi_N <= 1, so <p>, mu_i and W change (#1004)
# - synthetic plasma states and CHEASE synthesis: L/H/ITB presets, kinetic
#   profiles at fidelity 0-3, self-consistent equilibrium-kinetic iteration,
#   fixed-boundary equilibria from 0D descriptors meeting several targets at
#   once, pedestal/ITB source shapes, independent NZBOX, NIDEAL from the
#   GEQDSK contract (#120, #122, #123, #459, #1045, #1166)
# - EFIT: opt-in Picard iteration history, convergence against an
#   independent GS residual, standard_deviation means the sigma EFIT fits
#   against, the magnetic-sigma scan and the (2,1) working setting, named
#   presets in pipelines 1 and 2, Ip chi-square with the prescribed vessel
#   current, the Green table built per machine era (#891, #918, #924,
#   #1038, #1379)
# - magnetics and machine history: probe wiring, the +0.06 probe, wall 2409
#   and the PF boundary versioned from raw signals (#956); probe C4-04 and
#   array-contradicted probes excluded (#977); the diamagnetic channel's
#   real history (#993); limiter faces off the EFIT grid lines (#965);
#   stored diamagnetic flux is paramagnetic-POSITIVE (#1196): packaged
#   samples were rewritten in place and method_name carries the token,
#   products written before 0.8.0 keep the old sign
# - startup: Romero's exact plasma-transformer identities and first-order
#   current-diffusion closure, loop voltage at the boundary with its
#   inductive part, Townsend coefficients for H2/He/Ar, the burn-through
#   barrier, mean and effective charge, the lumped circuit, Tutorial 02
#   reduced models (#652, #676, #781-#783)
# - stability and 3-D fields: analytic islands integrated along SXR chords,
#   GPEC perturbation renderers, the toroidal-phase convention audit, DCON
#   products in the ideal-GPEC cell, PENTRC torque parts, resistive-layer
#   metrics, FLARE field-line products as IMAS stand-ins, kink/sawtooth/
#   stochastic-layer/separatrix-lobe phenomena on an equilibrium (#886,
#   #1209)
# - operational boundaries: vaft.formula.boundaries with Greenwald, Hugill,
#   Murakami, the L-H threshold family (Martin 2008, Ryter, Takizuka ITPA04),
#   a sourced Troyon limit, canonical public tables (ITPA DB5.2.3, TCV,
#   TC-26, PR08 profiles) and the VEST density-limit recipe (#350, #636,
#   #1066-#1068, #1205); the unsourced stability heuristics and
#   virial_stability_criterion are deprecated (#366, #350)
# - formula families: definitions as data with a Jupyter card (#889),
#   geometry, toroidicity and ripple, single-particle motion and
#   guiding-centre invariants, neoclassical/NTV scales, cold-plasma waves,
#   NBI attenuation, plasma-wall interaction, disruption and VDE
#   quantities, edge/SOL, blobs and MARFE (#951, #1041, #1042, #1047,
#   #1062, #1070, #1092, #1111, #1113, #1136, #1211)
# - vaft.diagram: reproducible TikZ-rendered schematics for islands,
#   tokamak geometry and coordinates, tearing and ballooning, harmonics,
#   disruptions, VDEs, the edge, wall conditioning, spectroscopy, waves and
#   integrated modelling; 156 canonical SVGs hash-pinned in docs (#890,
#   #1039, #1051, #1052, #1063, #1071-#1075, #1085, #1088, #1093, #1215)
# - plotting: 3-D scenes in Plotly, VTK/ParaView and K3D with the vtk and
#   jupyter3d extras (#1087), recipes label the coordinate they draw, the
#   core-profile map at rho_tor (#335), colours by intent (#748), the camera
#   overlay never bridges NaN gaps (#1314), committed thumbnails with a
#   freshness manifest (#1097); animation=True renders a plot sequence to
#   mp4/webm/gif through a private PyAV backend, time_range= sets the
#   window and the *_animation_frames helpers are deprecated (#1049,
#   #1050); time_range= is honoured on a time axis or refused, never
#   accepted and ignored
# - database, HSDS and ShotLog: h5pyd pinned at 0.24.0 with `vaft hsds
#   configure` (#969), the VEST ShotLog as a FileDB archive and
#   pulse_schedule (#995), SXR digitizer CSVs packed into lossless HDF5
#   (#1186), a hard_x_rays prototype (#1160), bulk prefetch for the lazy
#   HSDS store (#1331)
# - execution: one ExecutionBackend for TES, GACODE, GPEC, NUBEAM, CHEASE,
#   FLARE, EFUND, EFIT and NICE, a Slurm backend, an in-process memory
#   guard (#671, #1017, #1146); a timeout is a failed result, not an
#   exception, for CHEASE, GACODE, NUBEAM, FLARE, TES, NICE and GENRAY
#   (EFIT/EFUND/GPEC unchanged until after 2026-10-06), and a local launch
#   stops the whole process tree on timeout, Ctrl-C or SIGTERM (#1016); a
#   VEST-server worker polls for new shots and runs the pipeline (#58)
# - new adapters: GENRAY EC ray tracing (#264), an experimental NICE
#   reconstruction (#666), the provisional 6 kW ECH launch from CAD (#266),
#   vaft.process.ml with vaft-nn resolution and the ml extra (#669);
#   NUBEAM's Plasma State is built from public NTCC sources only
# - packaging and Python: 3.14 canonical, 3.10-3.14 supported (#1008);
#   freeze-era pins replaced by a documented policy, astropy/fortranformat/
#   imageio/pyjwt/requests-unixsocket/setuptools/urllib3 dropped, numba in
#   the accel extra (#1007, #1012); omas 0.95.2 (an ODS is unhashable);
#   vaft.help() and `vaft help` (#1203); `vaft shotlog`, `vaft hsds
#   configure`, `vaft pipeline-worker`
# - docs and tutorials: source-synchronised API reference with
#   revision-pinned source links (#162, #1069), generated plot/diagram
#   catalogs with a coverage gate, the Formula / Process / Code layers and
#   the optional Actor contract documented (#1078), the framework concept
#   diagrams (#1090), __all__ declared across vaft.omas, vaft.imas and
#   vaft.machine_mapping (#1382), tutorials 02-06 revamped (#783, #952,
#   #1005, #1023, #1052, #1091)
# - removed: current_density_from_psi,
#   bremsstrahlung_power_density_from_Z_eff_n_e_T_e and the seven renamed
#   constants of vaft.formula (promised for 0.8.0); the legacy and renamed
#   vaft.plot names now say 0.9.0 and a test holds every promise to its date
# - cold review of this line (16 slices, every finding re-verified):
#   113 findings, 0 critical, 9 major, all fixed before the merge (#1410-
#   #1412, #1417-#1419, #1432-#1434): EFIT diagnostic-fit grades were never
#   produced, separatrix lobes traced a 2*pi-wrong pitch on per-radian
#   records, two 3-D coil plots refused time= on the public path, NICE
#   tolerances never reached param.xml, documentation added to redirect
#   stubs never rendered; and three defects older than this line:
#   diamagnetism normalised by half the plasma volume, W_th with 2/3
#   (#1282), a power-balance cache that served stale results
# - known issues: products for shots >= 43017 built before 0.8.0 differ
#   from the release mapper (regeneration note in DEPLOYMENT.md); the
#   Takizuka L-H gamma is not reachable through boundary_value; the ion
#   volume averages only now exist, so older summaries lack them; #926,
#   #927, #825 and #1338 remain open
# 0.7.1
# - bug-fix release: the verified findings of the 0.7.0 cold review that
#   shipped as known issues (#920), one commit each with a failing-before
#   test; no new features. Detailed notes on the release pull request and
#   the GitHub release.
# - EFIT: outputs at sub-millisecond times are read back at their own time
#   through one shared slice-name codec (g0SHOT.00306_320 was 306.32 s and
#   overwrote equilibrium.time; the kinetic k-file selection never picked a
#   sub-ms file); a cached profile-model result is reused only for the run
#   that produced it; log parsing, decision/slice time checks (#928)
# - database maintenance: the retirement gate no longer calls an unreadable
#   source deletable; strip_impa_from_source writes the master last and a
#   re-run repairs a stage-only master; a failed shot-folder listing no
#   longer replaces the union master; filedb relocate leaves what it cannot
#   attribute in place and repairs 0.7.0's mis-filed tree; a resumed
#   consolidation verifies an existing target; pipeline 1 parses under
#   layout: shot_first (#929)
# - installers: an uninstall removes only what its install recorded, never
#   a whole prefix; the installer refuses a non-empty prefix it did not
#   create (a 0.7.0 prefix keeps its directory); NUBEAM Windows -Uninstall
#   checks the tree and manifest first; POSIX installers exit with the
#   checker's status; nubeam/macos.sh derives the GCC major; the Windows
#   CHEASE installer accepts a fresh clone; ldconfig | grep -q under
#   pipefail no longer reports libraries missing (#930)
# - kernels: collapse_time no longer lands in the noise tail; finite peak
#   prominence with a search mask; n_rejected no longer saturates; clockwise
#   SFL-angle wrap; non-increasing kinetic coordinates refused; shot blocks
#   found under int and string keys; NBI beamlet position is the source
#   centre; NUBEAM path budget and stale Plasma State; ion Z/mass defaults;
#   run_neo(check=False); stable cells not re-solved; rmatch inputs
#   validated; FLARE output decoded as UTF-8 (#931)
# - plotting: time= is resolved to the nearest slice by time or refused,
#   never ignored; GPEC resonant plots pair by (n, time_slice); the
#   core_profiles slice control and the core-profile 2-D map pair by time
#   and draw in DD orientation; interactive refusals keep the canvas;
#   min_wall_authority/show_uncertainty/source are valid options (#933)
# - documentation: the guide's code samples run under a test (94 executed,
#   141 marked with a reason); four TypeError samples and their pre-fix
#   prose, an AttributeError after import vaft and 18 calls through the
#   wrong machine_mapping name fixed; documented defaults match the code;
#   langmuir_probe_positions.csv ships in the wheel (#934)
# - still open: FWTFC=0 frees an unenergised coil (#926, needs a k-file
#   A/B); core_profiles product ownership (#927); probe C2-05 published at
#   two toroidal angles, which keeps tutorial session 04's angle cell and
#   two sample checks as expected failures (#825); the remaining minors in
#   #920
# 0.7.0
# - development release line 2026-09-03 .. 2026-09-17 merged into main: 266
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
#   PROF*.IN, Osborne p-files (#550, #553, #741-#744); the GACODE input
#   writer carries a sqrt(psi_N) proxy on the core_profiles grid through
#   psi_N, takes SIGN_IT from the signed q, and writes the ZEFF its species
#   list realises
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
# - `vaft export` and vaft.database.export(): a shot leaves as a portable
#   file, refusing what the format would drop and never deleting the last
#   copy on overwrite (#450)
# - startup formula layer: breakdown chain through the avalanche, vacuum-field
#   view with |E_phi|, breakdown figure and Lloyd margin (#783, #676); the
#   2.45 GHz ECR field, the 6 kW ECH power mapped into ec_launchers as an
#   optional component, startup proxies, midplane profiles and vacuum field
#   lines on the camera view (#165, #888)
# - platform: CHEASE, DCON/GPEC and EFIT/EFUND build and run natively on
#   Windows (the NUBEAM Windows recipe is experimental); one recipe set under install/ builds CHEASE, GPEC, GACODE and
#   NUBEAM on Linux and macOS; GPEC pins its netCDF; TokaMaker is an optional
#   extra (#226 closed out)
# - CI: develop asks for development confidence (core selection), main for
#   release confidence (full suite on Linux and Windows, tutorials); main's
#   branch protection is code (#515, #471)
# - tutorials: session 02 redesigned to follow a startup from the light to
#   the field lines (#888); sessions 03 and 04 with one QMD presentation source; the
#   docs' code samples are checked to name real API
# - deprecated: current_density_from_psi (#355); the legacy onset
#   detectors and the chease-mhd-stability product are retired behind gates;
#   the old figure sizes remain as format="legacy"
# - records three samples too short for filtfilt are refused (#893)
# - known limitations: TGLF-NN has no VEST-trained model and refuses
#   out-of-domain input; os.access(X_OK) still says nothing on Windows
# - known issues, carried to 0.7.1 (#920, from the release's cold review):
#   EFIT outputs at sub-millisecond times are read back with the wrong time;
#   FWTFC=0 leaves an unenergised coil free rather than pinned; plots accept
#   time= but most draw slice 0 (use time_slice=); GPEC resonant plots pair
#   slices by n only; the external-code uninstallers remove the whole
#   --prefix, so never point them at a directory you did not create for
#   them; the database retirement, IMPA-strip and relocate tools are not yet
#   safe against partial failures; tutorial session 04 and the plotting sample
#   notebook still describe the 39915 sample as it was before its regeneration
#   (probe angles, source-flagged channels)
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

