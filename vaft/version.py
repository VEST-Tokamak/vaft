# Version information
__version__ = "0.8.0"

__all__ = ["__version__"]


# ────────────────────────────────────────────────────────
# patch notes
# ────────────────────────────────────────────────────────
# unreleased
# - vaft.process.core_q_context binds the documented function: the
#   submodule is renamed q_profile_context so its key no longer shadows the
#   function it exports (BREAKING for `import vaft.process.core_q_context`;
#   the function names are unchanged), and no submodule key may equal an
#   exported name; the reference page moved to
#   /reference/process/q_profile_context/ with a redirect (#1838)
# - core_q_context: a NaN psi_n sample no longer drops the finite sample
#   after it from shear, q_min, low-shear regions and q95; the context
#   records the slice time; one interior maximum is single_maximum, not
#   multi_extremum (#1838)
# 0.8.0
# - development release line 2026-09-18 .. 2026-10-08 merged into main: 488
#   pull requests (first-parent merges on develop since v0.7.1); the
#   detailed notes are on the release pull request and the GitHub release.
#   Headlines:
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
#   #1313, #1372); a weber-family g-file (COCOS 11-18) is read without a
#   second 2*pi, so its flux-derived quantities change (#294, #1684); every
#   volume average takes the plasma from the LCFS outline, not psi_N <= 1,
#   so <p>, mu_i and W change (#1004)
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
#   #1038, #1379); BREAKING: the #891 working setting statistical_891 is
#   the library default for every EFIT run that names no configuration
#   (statistical sigma with a 2 % floor, probes x3.62, loops x2.15, Ip x4,
#   diamagnetic flux x16, KPPCUR 2 / KFFCUR 1, psi-only exit, up to 514
#   iterations, pipeline-1 efit.timeout 3600 s); the previous production
#   configuration is kept byte-for-byte as the routine preset; every new
#   product names its configuration in code.parameters and the summary's
#   efit_configuration column, and products written before 0.8.0 read
#   unrecorded -- routine and statistical products coexist on the server
#   until regeneration and must not be compared as one population (#1440)
# - equilibrium quality: per-slice cohort table with rule verdicts, census
#   and crosswalk, representative cases chosen reproducibly, the quality x
#   confinement funnel, study criteria shipped (#1644); the #579 EFIT
#   sensitivity ensemble (#1663); the fixed-to-free closure study (#1608);
#   a vacuum shot records EFIT as not applicable instead of failing (#205)
# - magnetics and machine history: probe wiring, the +0.06 probe, wall 2409
#   and the PF boundary versioned from raw signals (#956); probe C4-04 and
#   array-contradicted probes excluded (#977); the diamagnetic channel's
#   real history (#993); limiter faces off the EFIT grid lines (#965);
#   stored diamagnetic flux is paramagnetic-POSITIVE (#1196): packaged
#   samples were rewritten in place and method_name carries the token,
#   products written before 0.8.0 keep the old sign; class-shot QA with
#   Rogowski validity, fault records and the TF excursion repair (#1616),
#   lower-inboard flux-loop faults #12-#15 (#1793), pickup-only pulses are
#   not Plasma (#1733), PF gaps keep the acquired circuits (#1568);
#   BREAKING: a magnetics processing override applies ON TOP of the shot's
#   era policy; a shadowed legacy key set off its default, and time_start/
#   time_end/sample_count without window_override, are refused (#1541,
#   #1729, #1766); VFIT Wkin maps to energy_mhd, the GSE li_3 is built
#   from Wmag and the FEM li_3 is renormalised to R0 and to the slice ip,
#   so VFIT-derived W and li_3 change (#1771, #1775, #1812); the EFIT
#   not-applicable verdict judges CUTIP on the box-averaged current
#   (#1792); flux loops #14/#15 are faults during plasma (#1796)
# - startup: Romero's exact plasma-transformer identities and first-order
#   current-diffusion closure, loop voltage at the boundary with its
#   inductive part, Townsend coefficients for H2/He/Ar, the burn-through
#   barrier, mean and effective charge, the lumped circuit, Tutorial 02
#   reduced models (#652, #676, #781-#783); the Romero balance plot (#1590)
# - stability and 3-D fields: analytic islands integrated along SXR chords,
#   GPEC perturbation renderers, the toroidal-phase convention audit, DCON
#   products in the ideal-GPEC cell, PENTRC torque parts, resistive-layer
#   metrics, FLARE field-line products as IMAS stand-ins, kink/sawtooth/
#   stochastic-layer/separatrix-lobe phenomena on an equilibrium (#886,
#   #1209); the DCON payload in mhd_linear.code.parameters v2 (#940),
#   DCON/RDCON/STRIDE numerical health in the validation layer (#142),
#   the RDCON-vs-STRIDE Delta' benchmark (#143), reduced interchange and
#   kink kernels (#1635), asymptotic orderings and the timescale hierarchy
#   (#1627), GPEC memory stops named as such (#1460)
# - operational boundaries: vaft.formula.boundaries with Greenwald, Hugill,
#   Murakami, the L-H threshold family (Martin 2008, Ryter, Takizuka ITPA04),
#   a sourced Troyon limit, canonical public tables (ITPA DB5.2.3, TCV,
#   TC-26, PR08 profiles) and the VEST density-limit recipe (#350, #636,
#   #1066-#1068, #1205); the unsourced stability heuristics and
#   virial_stability_criterion are deprecated (#366, #350); the l_i-q
#   family, Wesson 1989 and Cheng-Furth-Boozer 1987 side by side (#1422,
#   #1603), the ST Hugill diagram after Sykes 2000 (#1602), estimated q95
#   without an equilibrium (#1583), the volume beta (#1691), each drawn
#   boundary's applicability to the plotted population (#1628), the PR08
#   MHD-state population table and the multi-machine op-space notebook
#   (#1620, #1736), dimensionless-similarity spaces (#1624)
# - formula families: definitions as data with a Jupyter card (#889),
#   geometry, toroidicity and ripple, single-particle motion and
#   guiding-centre invariants, neoclassical/NTV scales, cold-plasma waves,
#   NBI attenuation, plasma-wall interaction, disruption and VDE
#   quantities, edge/SOL, blobs and MARFE (#951, #1041, #1042, #1047,
#   #1062, #1070, #1092, #1111, #1113, #1136, #1211); exact Buckingham-Pi
#   analysis (#1621); sensitivity, linearisation and UQ kernels (#1642)
# - impurities and Z_eff: mixture algebra, the VEST preset and one
#   resolver, the composition written into core_profiles, radial Z_eff from
#   OpenADAS charge states, the resistive projection and the TGLF/CGYRO
#   sensitivity (#1565, #1566), the species/population state (#1567),
#   atomic identity under vaft.data (#1711), a bundled ion's radial charge
#   moments read back and lumped for GACODE (#1769); BREAKING: the VEST
#   line-radiation fractions are 1/86, not 0.01 (+16.3 % P_rad_line)
# - vaft.diagram: reproducible TikZ-rendered schematics for islands,
#   tokamak geometry and coordinates, tearing and ballooning, harmonics,
#   disruptions, VDEs, the edge, wall conditioning, spectroscopy, waves and
#   integrated modelling (#890, #1039, #1051, #1052, #1063, #1071-#1075,
#   #1085, #1088, #1093, #1215); spatial vocabulary (#1101), workflow
#   spine (#1585), data platform (#1550), MHD mode map (#1574), research
#   concepts (#1636-#1645, #1698), current-profile topology (#1604), an
#   animated magnetic_island (#1053); 230 canonical SVGs hash-pinned
# - plotting: 3-D scenes in Plotly, VTK/ParaView and K3D with the vtk and
#   jupyter3d extras (#1087), recipes label the coordinate they draw, the
#   core-profile map at rho_tor (#335), colours by intent (#748), the camera
#   overlay never bridges NaN gaps (#1314) and overlay="equilibrium_section"
#   projects the same-shot equilibrium section onto FAST camera frames
#   (#1830), committed thumbnails with a
#   freshness manifest (#1097); animation=True renders a plot sequence to
#   mp4/webm/gif through a private PyAV backend, time_range= sets the
#   window and the *_animation_frames helpers are deprecated (#1049,
#   #1050); time_range= is honoured on a time axis or refused, never
#   accepted and ignored; plot composition (#1467), selection against
#   validity, dense time_index navigation and one sequence contract
#   (#1380), canonical visuals (#497), diagram formats (#1097); figure
#   options, slide and poster formats (#1421), table and text views (#1180)
#   and the validation verdict table (#474), rational surfaces (#506), the
#   registered gyrokinetics family with include_unconverged (#1591), the
#   plot docstring contract and vaft.plot.documentation (#1505)
# - GUI: `vaft gui` (Panel) with a workspace shell, discovery-driven
#   explorer, Figure Options, player and video export, a Routine
#   Diagnostics workspace and a hosted mode with HSDS sign-in (#1086,
#   #1174, #1172, #1421, #1380, #1400, #1348); templates in vaft.deploy.gui
#   (#1755); Panel and Bokeh are CORE dependencies (about 150 MB of
#   install; `vaft[gui]` stays as an empty extra)
# - database, HSDS and ShotLog: h5pyd pinned at 0.24.0 with `vaft hsds
#   configure` (#969), the VEST ShotLog as a FileDB archive and
#   pulse_schedule (#995), SXR digitizer CSVs packed into lossless HDF5
#   (#1186), a hard_x_rays prototype (#1160), bulk prefetch for the lazy
#   HSDS store (#1331); concurrent replications of one shot no longer lose
#   master.h5 links: one per-(source, shot) lock covers the upload, the
#   re-read merge and the master replace (#913), and the master audit tells
#   stubs, NaN-only and data files apart; a no_output stage is skipped, an
#   EFIT no_output with g-files is a fault (#1540); round trips read
#   without the HSDS cache (#1758); summary() stops at the first
#   unreachable source (#1765); `vaft raw-legacy-import` (#1613); GX-8
#   camera headers (#1589); a read-only MCP server (#1423, #188); the
#   unified VEST diagnostics fixture ships in the wheel (#1609, #1610)
# - confinement and operational space: the VEST confinement-time table,
#   regression against the ITER scalings, Kadomtsev closures, extensions
#   and figures (#548, #351), the operational-space view through
#   vaft.formula.boundaries (#1425), a resistive Z_eff inferred from the
#   Spitzer loop voltage (#1214), a Thomson consistency band, the
#   inboard side-limited state in TokaMaker, DCON at high n in the
#   stability atlas (#1429); the Tier A loader, ohmic/L-mode scalings in
#   vaft.formula (#670), energy basis and global columns (#1713, #1280),
#   per-row sigma_W ODR (#579), the conference atlas (#1456), inferred T_i
#   opt-in only (#1426); BREAKING: h_factor returns NaN with a warning for
#   a global/unaudited scaling unless tau_e_global_s or thermal_as_global
#   is given; the two ITER97-L kappa paths (H 1.07 vs 0.99) are tagged
# - transport and gyrokinetics: the transport atlas with classical,
#   neoclassical and turbulent summaries (#1427, #1654, #1655), SAT-rule
#   sensitivity (#1482), the CGYRO adapter (#1354), TGLF spectra and the
#   audited gyrokinetics_local mapping (#1591), the MITIM adapter with
#   TGLF/NEO cross-checks (#1588); TGLF outputs serialise non-finite values
#   (#1770); BREAKING: cross_spectral_matrix without sample_rate is bounded
#   by the SLOWEST record's Nyquist, not the fastest (#1611)
# - execution: one ExecutionBackend for TES, GACODE, GPEC, NUBEAM, CHEASE,
#   FLARE, EFUND, EFIT and NICE, a Slurm backend, an in-process memory
#   guard (#671, #1017, #1146); a timeout is a failed result, not an
#   exception, for CHEASE, GACODE, NUBEAM, FLARE, TES, NICE and GENRAY
#   (EFIT/EFUND/GPEC unchanged until after 2026-10-06), and a local launch
#   stops the whole process tree on timeout, Ctrl-C or SIGTERM (#1016); a
#   VEST-server worker polls for new shots and runs the pipeline (#58),
#   with a stage scope and a disk guard that pauses and resumes above a
#   margin (#1730); each pipeline loads its own config.yaml (#1530); a
#   remote ssh+Slurm backend for any adapter that takes backend= (#1599);
#   TokaMaker can impose the eddy-stage wall currents (#1534)
# - new adapters: GENRAY EC ray tracing (#264), an experimental NICE
#   reconstruction (#666), the provisional 6 kW ECH launch from CAD (#266),
#   vaft.process.ml with vaft-nn resolution and the ml extra (#669);
#   NUBEAM's Plasma State is built from public NTCC sources only
# - validation: the credibility taxonomy and ordering-margin applicability
#   (#1639), the sensitivity/UQ contract (#1642), studies under
#   vaft.validation.studies (#1756)
# - packaging and Python: 3.14 canonical, 3.10-3.14 supported (#1008);
#   freeze-era pins replaced by a documented policy, astropy/fortranformat/
#   imageio/pyjwt/requests-unixsocket/setuptools/urllib3 dropped, numba in
#   the accel extra (#1007, #1012); omas 0.95.2 (an ODS is unhashable);
#   vaft.help() and `vaft help` (#1203); `vaft shotlog`, `vaft hsds
#   configure`, `vaft pipeline-worker`, `vaft gui`, `python -m vaft.mcp`;
#   the architecture extra (grimp, #1646); the 39915 sample ships in its
#   OMAS form only, the IMAS netCDF twin is repository-only, so the wheel
#   is ~16.6 MiB under the 26 MiB cap (#1806)
# - docs and tutorials: source-synchronised API reference with
#   revision-pinned source links (#162, #1069), generated plot/diagram
#   catalogs with a coverage gate, the Formula / Process / Code layers and
#   the optional Actor contract documented (#1078), the framework concept
#   diagrams (#1090), __all__ declared across vaft.omas, vaft.imas and
#   vaft.machine_mapping (#1382), tutorials 02-06 revamped (#783, #952,
#   #1005, #1023, #1052, #1091); Tutorial 03 around equilibrium inference
#   (#1714), the parameter inference and orderings pages (#1601, #1627),
#   interactive dependency and pipeline explorers (vendored Cytoscape +
#   dagre, MIT, docs only; #1646, #1647), the coherent fluctuation
#   workflow (#1611)
# - removed: current_density_from_psi,
#   bremsstrahlung_power_density_from_Z_eff_n_e_T_e and the seven renamed
#   constants of vaft.formula (promised for 0.8.0); the legacy and renamed
#   vaft.plot names now say 0.9.0 and a test holds every promise to its date
# - cold review of this line (16 slices + 16 delta reviews, every finding
#   re-verified): 113 primary findings, 0 critical, 9 major, all fixed
#   before the merge (#1410-#1412, #1417-#1419, #1432-#1434): EFIT
#   diagnostic-fit grades were never produced, separatrix lobes traced a
#   2*pi-wrong pitch on per-radian records, two 3-D coil plots refused
#   time= on the public path, NICE tolerances never reached param.xml,
#   documentation added to redirect stubs never rendered; and three defects
#   older than this line: diamagnetism normalised by half the plasma
#   volume, W_th with 2/3 (#1282), a power-balance cache that served stale
#   results; the absorb-16/17 deltas (16 majors, 0 critical) fixed by
#   #1764-#1767, #1769, #1770, #1773, #1774, #1776, #1778, #1780, #1781,
#   #1788-#1790: a NaN cell counted as Thomson-consistent, the kinetic
#   overview mapped through slice 0, a bundled ion read as a fixed charge,
#   ten plots without their contract docstrings, gate-red tests
# - known issues: products for shots >= 46404 generated 2026-10-02..10-06
#   carry slow-DAQ magnetics and must be regenerated with everything
#   derived from them (#1731); products for shots >= 43017 built before
#   0.8.0 differ from the release mapper (regeneration note in
#   DEPLOYMENT.md); EFIT dirs mixed across runs and 210 HSDS products
#   with stale-instant slices (#1786; generate_kfile now keeps one
#   superseded generation per shot, #1812); a forwarded remote-backend
#   variable equal to the laptop's is dropped from job.sh (decision
#   pending); five
#   non-era-coupled legacy magnetics keys still apply; the EC launcher is
#   absent from the top and 3-D views; CHEASE not-launchable is recorded
#   skipped; the inferred-T_i sigma floor borrows #874's 17 %; the #1644
#   notebook cell 12 is stale; the pipeline-3 sheets were regenerated for
#   the release (#1843) but main holds EFIT equilibria only up to shot
#   45000 and its core_profiles predate the #1786/#1793 equilibrium
#   regenerations (regenerated in 0.8.1, #1842); the Takizuka L-H gamma
#   is not reachable through
#   boundary_value; the delta-19b review findings
#   are carried to #1838: vaft.process.core_q_context binds the submodule,
#   not the function (#1810), the camera equilibrium-section overlay
#   projects probes through the provisional pose frame (#746) and draws no
#   sensor markers on the lazy database path (#1830), README reference
#   links point at the develop docs track (#1777); #926, #927, #825 and
#   #1338 remain open

