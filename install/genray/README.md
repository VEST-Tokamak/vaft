# GENRAY build support

This directory holds what `install/install_genray.sh` adds to an upstream GENRAY
build. It contains no GENRAY source.

- `pgplot_stub.f`: no-op versions of the PGPLOT routines GENRAY calls. Linking
  them instead of `-lpgplot -lX11` turns off GENRAY's diagnostic plots and leaves
  the rays and `genray.nc` unchanged. VAFT reads only `genray.nc`. If an upstream
  revision calls a PGPLOT routine that is not listed here, the build fails at
  link time; add the routine with its PGPLOT argument list and an empty body.
  This is fixed-form Fortran, so keep every line within column 72.

For usage, see "GENRAY" under Per-code notes in `install/README.md`.
