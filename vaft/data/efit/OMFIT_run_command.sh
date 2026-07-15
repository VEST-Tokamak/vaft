#!/bin/bash -l
# Example invocation of EFIT's EFUND Green's-function table generator.
#
# Point EFUND at the executable in your own EFIT build before running, e.g.
#   export EFUND=/path/to/efit/build/green/efund
set -euo pipefail
"${EFUND:?set EFUND to the path of the efund executable}"
