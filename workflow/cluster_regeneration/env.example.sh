# Copy to $REGEN_ROOT/env.sh and adapt the tool paths; every job sources it.
# Resolves the current pin ($REGEN_ROOT/{src,env} symlinks written by setup_env.sh).
REGEN_ROOT=${REGEN_ROOT:?set REGEN_ROOT (site.example.env)}
export REGEN_ROOT
REGEN_SRC=$(readlink -f "$REGEN_ROOT/src")
REGEN_ENV=$(readlink -f "$REGEN_ROOT/env")
export REGEN_SRC REGEN_ENV
export REGEN_SHA=$(cat "$REGEN_ROOT/PIN_SHA")
unset PYTHONPATH PYTHONHOME PYTHONSTARTUP MPLBACKEND CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL OMAS_DEBUG_TOPIC LD_LIBRARY_PATH
# sbatch exports the login environment: drop every VAFT_* / XDG_* setting from it (cache, lock and
# source overrides) before setting the ones this run uses.
for v in $(compgen -e | grep -E '^(VAFT_|XDG_)'); do unset "$v"; done
export VIRTUAL_ENV=$REGEN_ENV PATH=$REGEN_ENV/bin:/usr/bin:/bin
export PYTHONNOUSERSITE=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
# Inputs
export VAFT_DATA_DIR=$REGEN_SRC/vaft/data
# The production server's EFIT build (3b5dae5 ipfix) with its own glibc 2.35 loader; sha256 of
# efit.bin/efund.bin in $EFITHOME/SHA256SUMS == the server's install.
export EFITHOME=${EFITHOME:-$HOME/opt/efit-bundle}
export EFIT=$EFITHOME/bin/efit
# CHEASE fb46366 and GPEC e68d7ac2 (dcon/match/rdcon/rmatch/stride/gpec), built on the cluster
# with system gfortran 11.5 like the server's (11.4); GPEC deps RPATHed beside them.
# RDCON is the STOCK build (gal.f i2.2 crashes at msing >= 25) -- the same stock binary
# the server's pipeline GPECHOME carries; a patched RDCON is a separate study's.
export CHEASEHOME=${CHEASEHOME:-$HOME/git/CHEASE/vaft-install} GPECHOME=${GPECHOME:-$HOME/git/GPEC-upstream/vaft-install}
export CHEASE=$CHEASEHOME/bin/chease
# HSDS isolation: unroutable endpoint, no credentials, private HOME (no ~/.hscfg).
export HS_ENDPOINT=http://127.0.0.1:9 HS_USERNAME= HS_PASSWORD= HS_API_KEY=
export HOME=$REGEN_ROOT/home
mkdir -p "$HOME"
export VAFT_FILEDB_DIR=/srv/vest.filedb
# CPU-kernel parity with the production server (an AVX2 CPU): same OpenBLAS kernel family and no
# AVX-512 numpy loops, so floating-point results match the server's bit for bit where the
# Python stack matches (REGEN_CPU_PARITY=0 to run native kernels).
if [ "${REGEN_CPU_PARITY:-1}" = 1 ]; then
  export OPENBLAS_CORETYPE=Haswell
  export NPY_DISABLE_CPU_FEATURES="AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL AVX512_SPR"
fi
