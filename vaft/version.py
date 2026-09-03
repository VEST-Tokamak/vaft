# Version information
__version__ = "0.6.1"


# ────────────────────────────────────────────────────────
# patch notes
# ────────────────────────────────────────────────────────
# 0.6.1
# - regenerate the packaged wheel sample so the bundled 39915 reference
#   carries the IMPA b_field_tor_probe toroidal_angle the mapper writes
#   (pi/2); 0.6.0 shipped the pre-fix 0.0
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

