#!/usr/bin/env bash
# Build the serial NTCC NUBEAM distribution natively on Windows.
#
# Usage (from an MSYS2 UCRT64 shell):
#   bash external/nubeam/windows.sh --nubeam-root PATH --accept-ntcc-terms
#
# Most people should run external/nubeam/windows.ps1 instead, which finds
# MSYS2, converts paths, wires $NUBEAMHOME, and calls this script. This file
# owns the build recipe; the PowerShell wrapper owns the Windows integration.
#
# VAFT does not vendor the NUBEAM source: NTCC requires each user to accept its
# licence before downloading it. This script operates on a NUBEAM tree you
# already hold, named by --nubeam-root, and never modifies it beyond writing
# the generated share/Make.local every NTCC build needs.
#
# The acceptance flag is required before this script downloads the NTCC
# dependency modules (PSPLINE, PREACT, XPLASMA). It signifies that the person
# running it has read and accepted
# https://w3.pppl.gov/NTCC/NUBEAM/downloads.shtml. The script never accepts the
# agreement implicitly.
#
# The resulting installation prefix is <root>/local, which is what $NUBEAMHOME
# should point at for vaft.code.nubeam.

set -euo pipefail
IFS=$'\n\t'

NUBEAM_ROOT="${NUBEAM_SOURCE_DIR:-}"
NETCDF_HOME="${VAFT_NETCDF_HOME:-}"
ACCEPT_NTCC_TERMS=0
RESUME=0

usage() {
  cat <<'EOF'
Usage: bash external/nubeam/windows.sh --nubeam-root PATH --accept-ntcc-terms

  --nubeam-root PATH   the NUBEAM source tree to build (or set NUBEAM_SOURCE_DIR)
  --netcdf-home PATH   prefix holding libnetcdff.a (or set VAFT_NETCDF_HOME)
  --accept-ntcc-terms  you have read and accepted the NTCC agreement
  --resume             reuse a partially completed build

Relative to the NUBEAM tree: the installation prefix is ./local, generated
output is ./build/windows-x86_64, and downloaded NTCC sources stay in
./vendor/ntcc. Point $NUBEAMHOME at ./local afterwards.

--netcdf-home must name a netCDF built without the S3 and NCZarr backends.
MSYS2's own mingw-w64-ucrt-x86_64-netcdf pulls in the AWS C++ SDK, whose
atexit handler deadlocks on Windows after the program has finished -- the
same defect that hung DCON. install/install_gpec_windows.ps1
-BuildDependencies produces a suitable prefix.
EOF
}

die() {
  printf 'windows.sh: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '==> %s\n' "$*"
}

while (($#)); do
  case "$1" in
    --nubeam-root) (($# >= 2)) || die '--nubeam-root needs a path'; NUBEAM_ROOT="$2"; shift 2 ;;
    --netcdf-home) (($# >= 2)) || die '--netcdf-home needs a path'; NETCDF_HOME="$2"; shift 2 ;;
    --accept-ntcc-terms) ACCEPT_NTCC_TERMS=1; shift ;;
    --resume) RESUME=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown option: $1 (use --help)" ;;
  esac
done

[[ -n "$NUBEAM_ROOT" ]] ||
  die "--nubeam-root is required: the NUBEAM source tree is not vendored in VAFT. Obtain it from https://w3.pppl.gov/NTCC/NUBEAM/ and pass its path (or set NUBEAM_SOURCE_DIR)."
ROOT_DIR="$(cd "$NUBEAM_ROOT" 2>/dev/null && pwd -P)" ||
  die "NUBEAM source tree does not exist: $NUBEAM_ROOT"
[[ -f "$ROOT_DIR/Makefile" && -d "$ROOT_DIR/nubeam_comp_exec" ]] ||
  die "not a NUBEAM source tree (no Makefile and nubeam_comp_exec/): $ROOT_DIR"

# Everything this script generates stays inside the NUBEAM tree, never in the
# VAFT checkout.
PREFIX="$ROOT_DIR/local"
BUILD_DIR="$ROOT_DIR/build/windows-x86_64"
NTCC_SOURCE_DIR="$ROOT_DIR/vendor/ntcc"
MANIFEST="$ROOT_DIR/.nubeam-install-manifest"
COMPAT_DIR="$BUILD_DIR/compat"
LOG_FILE="$BUILD_DIR/install.log"

case "$PREFIX" in "$ROOT_DIR"/*) ;; *) die "refusing path outside source tree: $PREFIX" ;; esac
case "$BUILD_DIR" in "$ROOT_DIR"/*) ;; *) die "refusing path outside source tree: $BUILD_DIR" ;; esac
case "$NTCC_SOURCE_DIR" in "$ROOT_DIR"/*) ;; *) die "refusing path outside source tree: $NTCC_SOURCE_DIR" ;; esac

[[ "$(uname -o 2>/dev/null)" == "Msys" ]] ||
  die "this installer runs inside MSYS2; start the UCRT64 shell, or use external/nubeam/windows.ps1"
[[ "${MSYSTEM:-}" == "UCRT64" ]] ||
  die "MSYSTEM is '${MSYSTEM:-unset}'; NUBEAM must be built in the UCRT64 environment so it links the same C runtime as CPython"
for tool in gfortran gcc g++ make python3 curl ar ranlib objdump; do
  command -v "$tool" >/dev/null || die "$tool is required: pacman -S --needed mingw-w64-ucrt-x86_64-toolchain make python curl"
done

((ACCEPT_NTCC_TERMS)) ||
  die "the build requires NTCC dependency sources. Read https://w3.pppl.gov/NTCC/NUBEAM/downloads.shtml, then rerun with --accept-ntcc-terms"

if [[ -z "$NETCDF_HOME" ]]; then
  for candidate in "${LOCALAPPDATA:-}/vaft/external/gpec/deps" "$HOME/AppData/Local/vaft/external/gpec/deps"; do
    [[ -n "$candidate" && -f "$candidate/lib/libnetcdff.a" ]] && { NETCDF_HOME="$candidate"; break; }
  done
fi
[[ -n "$NETCDF_HOME" ]] ||
  die "no netCDF prefix found. Build one with 'powershell -File install/install_gpec_windows.ps1 <gpec-source> -BuildDependencies', or pass --netcdf-home."
NETCDF_HOME="$(cd "$NETCDF_HOME" 2>/dev/null && pwd -P)" || die "--netcdf-home does not exist: $NETCDF_HOME"
[[ -f "$NETCDF_HOME/lib/libnetcdff.a" ]] ||
  die "no libnetcdff.a under $NETCDF_HOME/lib; --netcdf-home must name a static netCDF-Fortran prefix"

if [[ -e "$MANIFEST" ]]; then
  ((RESUME)) || die "an existing NUBEAM installation manifest was found; remove $PREFIX and $BUILD_DIR first, or resume the interrupted installer with --resume"
elif [[ -e "$PREFIX" ]]; then
  die "installation prefix already exists without a manifest: $PREFIX"
elif [[ -e "$BUILD_DIR" ]]; then
  die "generated build directory already exists without a manifest: $BUILD_DIR"
fi
if [[ -e "$ROOT_DIR/share/Make.local" ]] && ! grep -q 'Generated by VAFT external/nubeam' "$ROOT_DIR/share/Make.local"; then
  die "refusing to overwrite an existing user Make.local: $ROOT_DIR/share/Make.local"
fi

mkdir -p "$BUILD_DIR" "$COMPAT_DIR" "$PREFIX"
: > "$MANIFEST"
exec > >(tee -a "$LOG_FILE") 2>&1

# fpp.py, NUBEAM's Fortran preprocessor driver, runs its subcommands through
# os.popen, which on a MinGW Python goes by way of cmd.exe. MSYS2 leaves
# COMSPEC empty, and the preprocessor then fails on every .F90 in the tree.
export COMSPEC="$(cygpath -w "${SYSTEMROOT:-C:/Windows}/System32/cmd.exe")"
export PATH="$PATH:${SYSTEMROOT:-/c/Windows}/System32"

note "NUBEAM source:   $ROOT_DIR"
note "install prefix:  $PREFIX"
note "netCDF prefix:   $NETCDF_HOME"
note "build log:       $LOG_FILE"

# --------------------------------------------------------------------------
# Compatibility sources
#
# Four gaps separate MinGW-w64 from the POSIX userland NTCC was written for.
# Each is filled here, in the generated build directory, so the NUBEAM source
# tree itself is never edited.
# --------------------------------------------------------------------------

write_compat_sources() {
  note "writing Windows compatibility sources to $COMPAT_DIR"
  mkdir -p "$COMPAT_DIR/sys" "$COMPAT_DIR/netinet" "$COMPAT_DIR/arpa" "$COMPAT_DIR/bits"

  cat > "$COMPAT_DIR/vaft_win32_posix.h" <<'EOF'
/*
  Force-included into every NTCC C compile by VAFT's Windows recipe.

  The system headers come first so their own declarations are seen unmangled;
  the macros below then rewrite the call sites. Include guards make a later
  #include <sys/stat.h> in NTCC's own sources a no-op, so the ordering holds
  no matter what each file includes.
*/
#ifndef VAFT_WIN32_POSIX_H
#define VAFT_WIN32_POSIX_H
#if defined(_WIN32)

#include <direct.h>
#include <io.h>
#include <sys/types.h>
#include <sys/stat.h>

/* MinGW's mkdir() carries no mode argument; Windows has no POSIX mode bits. */
#undef mkdir
#define mkdir(path, mode) _mkdir(path)

/*
  setenv(3) has no MinGW equivalent. _putenv_s is the Windows spelling, and
  unlike putenv it copies its arguments rather than adopting the caller's
  buffer. portlib's f77_setenv.c is the one caller.
*/
#include <stdlib.h>
#undef setenv
#define setenv(name, value, overwrite) \
  (((overwrite) || getenv(name) == NULL) ? _putenv_s((name), (value)) : 0)
#undef unsetenv
#define unsetenv(name) _putenv_s((name), "")

#endif /* _WIN32 */
#endif /* VAFT_WIN32_POSIX_H */
EOF

  cat > "$COMPAT_DIR/sys/socket.h" <<'EOF'
/*
  BSD sockets over Winsock, for NTCC sources that include <sys/socket.h>.

  This header only makes the declarations available. It deliberately does not
  redirect close/read/write, even though a Winsock handle rejects all three:
  sglib's sgsys.c includes this header and calls those functions on ordinary
  file descriptors, so a macro here would quietly turn its file I/O into
  socket I/O. The redirection lives in vaft_trsocket_win32.h, which is
  force-included when compiling the one file that needs it.
*/
#ifndef VAFT_SYS_SOCKET_H
#define VAFT_SYS_SOCKET_H

#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdint.h>

/*
  Winsock has no SO_REUSEPORT. SO_REUSEADDR carries the meaning portlib wants
  -- rebinding a listening port still in TIME_WAIT -- so both names resolve to
  it, and the SO_REUSEADDR|SO_REUSEPORT the source passes stays correct.
*/
#ifndef SO_REUSEPORT
#define SO_REUSEPORT SO_REUSEADDR
#endif

#endif /* VAFT_SYS_SOCKET_H */
EOF

  cat > "$COMPAT_DIR/vaft_trsocket_win32.h" <<'EOF'
/*
  Force-included when compiling portlib's trsocket.c on Windows.

  A Winsock handle is not a file descriptor, so close(), read() and write()
  all fail on one. trsocket.c uses those three on sockets exclusively, which
  is what makes the redirection safe here and unsafe anywhere else.
*/
#ifndef VAFT_TRSOCKET_WIN32_H
#define VAFT_TRSOCKET_WIN32_H

#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#undef close
#undef read
#undef write
#define close(fd)        closesocket((SOCKET)(fd))
#define read(fd, b, n)   recv((SOCKET)(fd), (char *)(b), (int)(n), 0)
#define write(fd, b, n)  send((SOCKET)(fd), (const char *)(b), (int)(n), 0)

#endif /* VAFT_TRSOCKET_WIN32_H */
EOF

  cat > "$COMPAT_DIR/sys/un.h" <<'EOF'
/*
  Windows carries AF_UNIX and struct sockaddr_un in afunix.h rather than in a
  sys/un.h, so this header is the POSIX spelling of that one.
*/
#ifndef VAFT_SYS_UN_H
#define VAFT_SYS_UN_H
#include <winsock2.h>
#include <afunix.h>
#endif
EOF

  cat > "$COMPAT_DIR/netinet/in.h" <<'EOF'
#ifndef VAFT_NETINET_IN_H
#define VAFT_NETINET_IN_H
#include <sys/socket.h>
#endif
EOF

  cat > "$COMPAT_DIR/arpa/inet.h" <<'EOF'
#ifndef VAFT_ARPA_INET_H
#define VAFT_ARPA_INET_H
#include <sys/socket.h>
#endif
EOF

  cat > "$COMPAT_DIR/termios.h" <<'EOF'
/*
  A terminal-attribute shim for sglib's interactive input path.

  sgsys.c turns off canonical mode and echo while it reads a keystroke from
  the controlling terminal. Windows exposes that through the console API, not
  through termios, and NUBEAM's batch entry points never reach the code. The
  calls are therefore reported as unsupported rather than emulated: sgsys.c
  checks the return value, and a failure leaves the terminal as it found it.
*/
#ifndef VAFT_TERMIOS_H
#define VAFT_TERMIOS_H

#include <errno.h>

typedef unsigned int tcflag_t;
typedef unsigned char cc_t;

#define NCCS 32
#define TCSANOW   0
#define TCSADRAIN 1
#define TCSAFLUSH 2
#define ICANON 0x0002
#define ECHO   0x0008
#define VMIN   6
#define VTIME  5

struct termios {
  tcflag_t c_iflag, c_oflag, c_cflag, c_lflag;
  cc_t     c_cc[NCCS];
};

static __inline int tcgetattr(int fd, struct termios *t) {
  (void) fd; (void) t; errno = ENOTSUP; return -1;
}
static __inline int tcsetattr(int fd, int action, const struct termios *t) {
  (void) fd; (void) action; (void) t; errno = ENOTSUP; return -1;
}

#endif /* VAFT_TERMIOS_H */
EOF

  cat > "$COMPAT_DIR/endian.h" <<'EOF'
/*
  glibc's byte-order names. MinGW is little-endian on every target it builds
  for here, so the constants are fixed rather than probed.
*/
#ifndef VAFT_ENDIAN_H
#define VAFT_ENDIAN_H
#define __LITTLE_ENDIAN 1234
#define __BIG_ENDIAN    4321
#define __PDP_ENDIAN    3412
#define __BYTE_ORDER    __LITTLE_ENDIAN
#define LITTLE_ENDIAN   __LITTLE_ENDIAN
#define BIG_ENDIAN      __BIG_ENDIAN
#define BYTE_ORDER      __BYTE_ORDER
#endif
EOF

  cat > "$COMPAT_DIR/bits/byteswap.h" <<'EOF'
/*
  glibc's byte-swap builtins, spelled with the intrinsics gcc already has.
*/
#ifndef VAFT_BITS_BYTESWAP_H
#define VAFT_BITS_BYTESWAP_H
#define __bswap_16(x) __builtin_bswap16(x)
#define __bswap_32(x) __builtin_bswap32(x)
#define __bswap_64(x) __builtin_bswap64(x)
#define bswap_16(x) __bswap_16(x)
#define bswap_32(x) __bswap_32(x)
#define bswap_64(x) __bswap_64(x)
#endif
EOF

  cat > "$COMPAT_DIR/sys/wait.h" <<'EOF'
/*
  Windows has no wait(2) family. The macros are provided so that sources
  including this header still compile; the one file that used the functions
  themselves, portlib's c_execsystem.c, is replaced wholesale.
*/
#ifndef VAFT_SYS_WAIT_H
#define VAFT_SYS_WAIT_H
#define WIFEXITED(s)    (1)
#define WEXITSTATUS(s)  (s)
#define WIFSIGNALED(s)  (0)
#define WTERMSIG(s)     (0)
#define WIFSTOPPED(s)   (0)
#define WSTOPSIG(s)     (0)
#define WNOHANG         1
#define WUNTRACED       2
#endif
EOF

  cat > "$COMPAT_DIR/vaft_get_proc_mem.c" <<'EOF'
/*
  get_proc_mem_ for Windows.

  portlib's get_proc_mem.c holds two implementations, one reading
  /proc/<pid>/stat and one calling the Mach task API, and __WIN32 compiles both
  out. What survives is the get_proc_mem() wrapper, which still calls the
  underscored entry point -- so the archive references a symbol nothing in it
  defines, and the final link fails.

  GetProcessMemoryInfo answers the same two questions. WorkingSetSize is the
  resident set; PagefileUsage is the private commit, which is what "virtual
  memory used by this process" means on Windows. Both are returned in KB as
  floats, matching the POSIX implementations.
*/
#include <windows.h>
#include <psapi.h>

void get_proc_mem_(float *v_mem, float *r_mem)
{
  PROCESS_MEMORY_COUNTERS counters;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof counters)) {
    *r_mem = (float) (counters.WorkingSetSize / 1024.0);
    *v_mem = (float) (counters.PagefileUsage / 1024.0);
  }
  else {
    *r_mem = 0.0f;
    *v_mem = 0.0f;
  }
}
EOF

  cat > "$COMPAT_DIR/vaft_wsa_init.c" <<'EOF'
/*
  Winsock needs WSAStartup before any socket call and portlib, written for
  POSIX, never makes one. A constructor runs it at load time so the socket
  entry points behave the way their callers assume.
*/
#include <winsock2.h>

static void __attribute__((constructor)) vaft_winsock_startup_ctor(void)
{
  WSADATA data;
  WSAStartup(MAKEWORD(2, 2), &data);
}

void vaft_winsock_startup(void) { vaft_winsock_startup_ctor(); }
EOF

  cat > "$COMPAT_DIR/c_execsystem_win32.c" <<'EOF'
/*
  c_execsystem for Windows.

  portlib's own c_execsystem.c builds the child with fork() plus execvp/execve
  and clears the environment with clearenv(). None of those exist on MinGW-w64,
  and the file also reaches for <sys/wait.h> and O_NONBLOCK, so it cannot be
  compiled there at all.

  _spawnvpe(_P_WAIT, ...) is the same operation in one call: it searches the
  path for the executable, hands the child an argument vector and an
  environment block, waits, and returns the child's exit status. That is
  precisely the contract portlib documents, so this file keeps the documented
  status codes rather than inventing its own.

  The shell-server entry points return their documented failure code. That
  feature keeps a helper shell alive on a named FIFO to avoid repeated process
  creation; it is a POSIX optimisation, it is reached only when iexec==2, and
  execsystem.F90 falls back to an ordinary spawn when connecting fails.
*/

#include "fpreproc/f77name.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>
#include <process.h>

#define VAFT_EXEC_SPAWN_FAILED   40  /* portlib: fork failed        */
#define VAFT_EXEC_NO_MEMORY      42  /* portlib: logic error        */
#define VAFT_EXEC_NO_SHELLSERVER 46  /* portlib: server unavailable */

/* Count and index the null-terminated strings packed into a block. */
static char **vaft_unpack(const char *block, size_t count, const char *lead)
{
  size_t extra = (lead != NULL) ? 1 : 0;
  char **out = (char **) malloc((count + extra + 1) * sizeof(char *));
  size_t i;
  const char *p = block;

  if (out == NULL) return NULL;
  if (lead != NULL) out[0] = (char *) lead;
  for (i = 0; i < count; i++) {
    out[i + extra] = (char *) p;
    p += strlen(p) + 1;
  }
  out[count + extra] = NULL;
  return out;
}

int F77NAME(c_execsystem)(const int *iexec, const int *kverbose,
                          const char *execute,
                          const int *margs, const char *args,
                          const int *menvs, const char *envs)
{
  char **argv = NULL;
  char **envp = NULL;
  intptr_t status;
  FILE *vstd = NULL;

  if (*kverbose > 0) vstd = stdout;
  else if (*kverbose < 0) vstd = stderr;

  /* argv[0] is the program name, as execvp would have been given. */
  argv = vaft_unpack(args, (size_t) (*margs > 0 ? *margs : 0), execute);
  if (argv == NULL) return VAFT_EXEC_NO_MEMORY;

  /* menvs < 0 means "inherit", which _spawnvpe spells as a null block. */
  if (*menvs >= 0) {
    envp = vaft_unpack(envs, (size_t) *menvs, NULL);
    if (envp == NULL) { free(argv); return VAFT_EXEC_NO_MEMORY; }
  }

  if (vstd != NULL) {
    fprintf(vstd, "%%c_execsystem: %s\n", execute);
    fflush(vstd);
  }

  status = _spawnvpe(_P_WAIT, execute, (const char *const *) argv,
                     (const char *const *) envp);

  free(argv);
  if (envp != NULL) free(envp);

  if (status == -1) {
    if (vstd != NULL) {
      fprintf(vstd, "%%c_execsystem: could not start %s: %s\n",
              execute, strerror(errno));
      fflush(vstd);
    }
    return VAFT_EXEC_SPAWN_FAILED;
  }
  return (int) status;
}

/*
  The shell server needs a named FIFO and a long-lived helper process.
  Reporting it unavailable is what execsystem.F90 already handles: it spawns
  the command itself instead.
*/
int F77NAME(c_connect_shell_server)(int *ioptarg, const char *command,
                                    const char *result, const char *msg)
{
  (void) ioptarg; (void) command; (void) result; (void) msg;
  return VAFT_EXEC_NO_SHELLSERVER;
}

int F77NAME(c_shell_server_send)(const char *com)
{
  (void) com;
  return VAFT_EXEC_NO_SHELLSERVER;
}
EOF
}

# --------------------------------------------------------------------------
# Build configuration
# --------------------------------------------------------------------------

write_make_local() {
  local dir="$1"
  mkdir -p "$dir/share"
  cat > "$dir/share/Make.local" <<EOF
# Generated by VAFT external/nubeam/windows.sh. Not part of the NTCC
# distribution, and removed by the uninstall path.
#
# MinGW-w64 is a Windows target with a POSIX-shaped userland. NUBEAM's build
# system only knows the OS names it shipped with, and every one of its
# OS-conditional sources is guarded by __WIN32 rather than by the OS variable.
# LINUX therefore selects a working set of build rules -- it is what puts
# nblkfac.F on its 4-byte branch and gives linux_hostnm.c a real hostnm --
# while -D__WIN32 selects the Windows branch inside the sources themselves.
MACHINE = LINUX
OS = LINUX
FORTRAN_VARIANT = GCC
DEFS = -D__WIN32
CDEFS = -D__WIN32
FC = gfortran
FC90 = gfortran
CC = gcc
CXX = g++
USEFC = Y
FORTLIBS = -lgfortran -lquadmath -lm
# gethostname lives in ws2_32 on Windows, not in libc.
CLIBS = -lstdc++ -lws2_32
FFLAGS = -c -O -m64 -fno-range-check -fdollar-ok -cpp -fno-common -std=legacy -fallow-argument-mismatch -fallow-invalid-boz
DFFLAGS = -c -g -m64 -fno-range-check -fdollar-ok -cpp -fno-common -std=legacy -fallow-argument-mismatch -fallow-invalid-boz
# NTCC's C predates C99 declarations, which gcc 14 and later reject by default.
override CFLAGS = -c -O -std=gnu89 -Wno-implicit-int -Wno-implicit-function-declaration -I$COMPAT_DIR -include $COMPAT_DIR/vaft_win32_posix.h
PREFIX = $PREFIX
NETCDF_DIR = $NETCDF_HOME
NETCDF_FORTRAN_HOME = $NETCDF_HOME
NETCDF_C_HOME = $NETCDF_HOME
PSPLINE_HOME = $PREFIX
LIBROOT = /ucrt64
BLAS = -L/ucrt64/lib -lopenblas
LAPACK = -L/ucrt64/lib -lopenblas
NO_MDSPLUS = 1
NO_EDITLIBS = 1
# NUBEAM documents FFTW 2.1.5, which MSYS2 does not package; FFTW 3 is not
# ABI-compatible. Leave it unset unless a compatible build is supplied.
FFTW = \$(NUBEAM_FFTW_FLAGS)
EOF
}

# --------------------------------------------------------------------------
# portlib
# --------------------------------------------------------------------------

# Two portlib sources need Windows treatment, and both are handled after make
# has built everything else. Seeding the archive first is not an option:
# portlib's libs target is satisfied by the archive file existing, so an
# archive placed there ahead of make suppresses all 140-odd other members.
#
#   c_execsystem.c  cannot compile at all -- fork, execvp, clearenv,
#                   O_NONBLOCK and <sys/wait.h> are all absent -- so the
#                   Windows replacement is compiled in its place.
#   trsocket.c      compiles, but its close/read/write act on sockets, which
#                   on Windows are not file descriptors. It is recompiled with
#                   those three redirected to their Winsock spellings.
seed_portlib() {
  local root="$1" obj="$2"
  [[ -d "$root/portlib" ]] || return 0
  # Only when pass 1 actually built portlib here. A module whose Makefile
  # found an installed libportlib.a reports "USING" it and builds nothing,
  # and creating an archive in that case would leave a three-member stub
  # that install_ntcc_artifacts then copies over the complete one.
  [[ -f "$obj/lib/libportlib.a" ]] || return 0
  mkdir -p "$obj/obj/portlib"
  local cflags="-c -O -std=gnu89 -Wno-implicit-int -Wno-implicit-function-declaration"
  gcc $cflags -I"$COMPAT_DIR" -I"$root" -I"$root/include" \
      -o "$obj/obj/portlib/c_execsystem.o" "$COMPAT_DIR/c_execsystem_win32.c"
  gcc -c -O -o "$obj/obj/portlib/vaft_wsa_init.o" "$COMPAT_DIR/vaft_wsa_init.c"
  gcc -c -O -o "$obj/obj/portlib/vaft_get_proc_mem.o" "$COMPAT_DIR/vaft_get_proc_mem.c"
  gcc $cflags -I"$COMPAT_DIR" -I"$root/portlib" -I"$root/include" \
      -include "$COMPAT_DIR/vaft_trsocket_win32.h" \
      -o "$obj/obj/portlib/trsocket.o" "$root/portlib/trsocket.c"
  # -U, not the default: binutils writes deterministic archives, which zero
  # every member's timestamp. make would then find each member older than its
  # source and rebuild it -- including the one that cannot compile.
  ar rU "$obj/lib/libportlib.a" \
     "$obj/obj/portlib/c_execsystem.o" \
     "$obj/obj/portlib/vaft_wsa_init.o" \
     "$obj/obj/portlib/vaft_get_proc_mem.o" \
     "$obj/obj/portlib/trsocket.o"
  ranlib "$obj/lib/libportlib.a"
}

# Pass 1 keeps going past the one file that cannot compile, so everything else
# in the archive is built. Pass 2 is the real gate: it runs without -k, so any
# other failure still stops the install rather than hiding behind -k.
make_libs() {
  local root="$1" obj="$2"
  make -C "$root" libs "OBJ=$obj" -k >/dev/null 2>&1 || true
  seed_portlib "$root" "$obj"
  make -C "$root" libs "OBJ=$obj" || die "library build failed in $root; see $LOG_FILE"
}

# --------------------------------------------------------------------------
# NTCC dependency modules
# --------------------------------------------------------------------------

download_ntcc_module() {
  local module="$1"
  local destination="$NTCC_SOURCE_DIR/$module"
  local archive="$BUILD_DIR/$module.tar.gz"
  local stage="$BUILD_DIR/$module.extract"
  local url="https://w3.pppl.gov/rib/repositories/NTCC/files/$module.tar.gz"

  [[ -d "$destination" ]] && return 0

  note "downloading NTCC $module source after explicit agreement acceptance"
  mkdir -p "$NTCC_SOURCE_DIR" "$stage"
  curl --fail --location --show-error --silent "$url" -o "$archive" ||
    die "NTCC did not provide the $module download; obtain it manually from https://w3.pppl.gov/NTCC/ and place its extracted source in $destination"
  file "$archive" | grep -qi 'HTML' &&
    die "NTCC returned an HTML page instead of $module source. Download it manually and extract it to $destination"
  tar -tf "$archive" >/dev/null 2>&1 ||
    die "unrecognized NTCC archive for $module; extract it manually to $destination"
  tar -xf "$archive" -C "$stage"

  local candidate
  if [[ -f "$stage/Makefile" && -d "$stage/share" ]]; then
    candidate="$stage"
  else
    candidate="$(find "$stage" -type f -name Makefile -print | sed 's#/Makefile$##' | head -n 1 || true)"
  fi
  [[ -n "$candidate" ]] ||
    die "downloaded $module archive has no recognizable Makefile; extract it manually to $destination"
  mkdir -p "$destination"
  cp -R "$candidate"/. "$destination"/
  rm -rf "$stage" "$archive"
}

install_ntcc_artifacts() {
  local module_root="$1"
  local module_build="$2"
  local library

  mkdir -p "$PREFIX/include" "$PREFIX/lib" "$PREFIX/mod"
  shopt -s nullglob
  for library in "$module_build/lib"/*.a; do cp "$library" "$PREFIX/lib/"; done
  # NUBEAM 2021 links this historical SGLIB archive as jclib, while the current
  # PREACT distribution ships it as libjc.a. Windows has no dependable symlink,
  # so both link names are real files.
  if [[ -f "$PREFIX/lib/libjc.a" && ! -e "$PREFIX/lib/libjclib.a" ]]; then
    cp "$PREFIX/lib/libjc.a" "$PREFIX/lib/libjclib.a"
  fi
  for library in "$module_build/mod"/*; do
    [[ -f "$library" ]] && cp "$library" "$PREFIX/mod/"
  done
  [[ -d "$module_root/include" ]] && cp -R "$module_root/include"/. "$PREFIX/include/"
  # PSPLINE publishes its C-API headers one directory below include/, while the
  # 2021 NUBEAM sources include them without the cpp/ prefix. Keep the original
  # hierarchy and export the public headers at include/ too.
  for library in "$module_root/include/cpp"/*.h; do
    [[ -f "$library" ]] && cp "$library" "$PREFIX/include/"
  done
  shopt -u nullglob
}

# Four modules ship an archive called libportlib.a, and install_ntcc_artifacts
# copies each over the last, so the prefix keeps whichever module was installed
# most recently and silently loses every member the others had. The loss only
# surfaces at the final link, as undefined references to uupper_, find_io_unit_
# and d1mach_.
#
# Back-fill rather than reorder: every member the prefix already has keeps the
# implementation that was linked against, so this cannot change what any
# previously linked executable resolves. It only adds members nothing provided.
backfill_shared_archives() {
  local base target source missing scratch
  for base in libportlib.a portlib.a; do
    target="$PREFIX/lib/$base"
    [[ -f "$target" ]] || continue
    for source in "$BUILD_DIR"/*/lib/"$base"; do
      [[ -f "$source" ]] || continue
      [[ "$source" -ef "$target" ]] && continue
      # ar here is the MinGW build and terminates its listing lines with CR.
      # Left in place every name carries a stray carriage return, and ar x
      # then reports "no entry <member> in archive" for all of them.
      missing="$(comm -23 <(ar t "$source" | tr -d '\r' | sort -u) <(ar t "$target" | tr -d '\r' | sort -u) | tr '\n' ' ')"
      [[ -n "${missing// /}" ]] || continue
      note "back-filling $(wc -w <<<"$missing" | tr -d ' ') member(s) into $base from ${source#"$BUILD_DIR"/}"
      scratch="$(mktemp -d)"
      ( cd "$scratch" && ar x "$source" $missing && ar rU "$target" $missing )
      ranlib "$target"
      rm -rf "$scratch"
    done
  done
}

build_ntcc_module() {
  local module="$1"
  local module_root="$NTCC_SOURCE_DIR/$module"
  [[ -f "$module_root/Makefile" ]] ||
    module_root="$(find "$NTCC_SOURCE_DIR/$module" -maxdepth 2 -type f -name Makefile -print | sed 's#/Makefile$##' | head -n 1)"
  [[ -n "$module_root" && -f "$module_root/Makefile" ]] ||
    die "could not identify a build root for NTCC module $module under $NTCC_SOURCE_DIR"
  note "building NTCC dependency $module from $module_root"
  write_make_local "$module_root"
  make_libs "$module_root" "$BUILD_DIR/$module"
  install_ntcc_artifacts "$module_root" "$BUILD_DIR/$module"
  backfill_shared_archives
}

# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

write_compat_sources

download_ntcc_module pspline
download_ntcc_module preact
download_ntcc_module xplasma

build_ntcc_module pspline
build_ntcc_module preact
build_ntcc_module xplasma

note "building serial NUBEAM libraries"
write_make_local "$ROOT_DIR"
make_libs "$ROOT_DIR" "$BUILD_DIR/nubeam"
install_ntcc_artifacts "$ROOT_DIR" "$BUILD_DIR/nubeam"
backfill_shared_archives

# --------------------------------------------------------------------------
# Linking
#
# Two things about the upstream link line do not survive on Windows.
#
# Order. It lists -lpspline first, ahead of every library that calls into it.
# ld on macOS searches archives repeatedly; GNU ld makes a single pass and
# discards an archive whose members resolve nothing yet, so the same line
# leaves several hundred spline symbols undefined. Restating the archives
# inside --start-group/--end-group makes the pass order irrelevant, and the
# group is built from what is actually in the two library directories rather
# than from a hard-coded list, so it follows the distribution.
#
# netCDF. The line carries -L/ucrt64/lib from BLAS, so a plain -lnetcdff finds
# MSYS2's shared netCDF -- the one built with the AWS C++ S3 SDK -- before the
# S3-free static build. Naming the archives by path bypasses the search.
# --------------------------------------------------------------------------

NETCDF_STATIC="$NETCDF_HOME/lib/libnetcdff.a $NETCDF_HOME/lib/libnetcdf.a"
for extra in libhdf5_hl.a libhdf5.a; do
  [[ -f "$NETCDF_HOME/lib/$extra" ]] && NETCDF_STATIC="$NETCDF_STATIC $NETCDF_HOME/lib/$extra"
done
NETCDF_STATIC="$NETCDF_STATIC -lsz -lz"

link_libraries() {
  local archive name group="-L$PREFIX/lib -L$BUILD_DIR/nubeam/lib -Wl,--start-group"
  shopt -s nullglob
  for archive in "$PREFIX/lib"/lib*.a "$BUILD_DIR/nubeam/lib"/lib*.a; do
    name="${archive##*/}"
    name="${name#lib}"
    name="${name%.a}"
    # netCDF and HDF5 are named by path below, never found by -l search.
    case "$name" in netcdff|netcdf|hdf5|hdf5_hl) continue ;; esac
    case " $group " in *" -l$name "*) continue ;; esac
    group="$group -l$name"
  done
  shopt -u nullglob
  # psapi provides GetProcessMemoryInfo for the get_proc_mem replacement.
  printf '%s -Wl,--end-group %s -lstdc++ -lws2_32 -lpsapi' "$group" "$NETCDF_STATIC"
}

note "linking nubeam_comp_exec"
make -C "$ROOT_DIR/nubeam_comp_exec" exec "OBJ=$BUILD_DIR/nubeam" \
  "NETCDF=$NETCDF_STATIC" "CLIBS=$(link_libraries)" ||
  die "nubeam_comp_exec did not link; see $LOG_FILE"
NUBEAM_EXEC="$BUILD_DIR/nubeam/test/nubeam_comp_exec.exe"
[[ -f "$NUBEAM_EXEC" ]] || die "serial executable was not created: $NUBEAM_EXEC"
mkdir -p "$PREFIX/bin"
cp "$NUBEAM_EXEC" "$PREFIX/bin/nubeam_comp_exec.exe"
build_aux_executables() {
  local preact_root preact_build generator_build source_file

  note "building update_state"
  # update_state ends its link line with $(LAPACK) and never reaches
  # $(CLIBS), so that is where the group has to go.
  make -C "$ROOT_DIR/update_state" exec "OBJ=$BUILD_DIR/nubeam" \
    "NETCDF=$NETCDF_STATIC" "LAPACK=$(link_libraries) -L/ucrt64/lib -lopenblas" ||
    die "update_state did not link; see $LOG_FILE"
  [[ -f "$BUILD_DIR/nubeam/test/update_state.exe" ]] ||
    die "update_state was not created; see $LOG_FILE"
  cp "$BUILD_DIR/nubeam/test/update_state.exe" "$PREFIX/bin/update_state.exe"

  # preact_init initializes PREACTDIR; preactinit(1) invokes it at the end.
  preact_root="$NTCC_SOURCE_DIR/preact/preact"
  preact_build="$BUILD_DIR/preact"
  note "building preact_init"
  # The rule names the target without a suffix; MinGW writes preact_init.exe
  # from it. preact_init ends its line with $(FORTLIBS).
  make -C "$preact_root" "$preact_build/test/preact_init" \
    "OBJ=$preact_build" "THISLIB=$preact_build/lib/libpreact.a" \
    "NETCDF=$NETCDF_STATIC" "FORTLIBS=-lgfortran -lquadmath -lm $(link_libraries) -L/ucrt64/lib -lopenblas" ||
    die "preact_init did not link; see $LOG_FILE"
  [[ -f "$preact_build/test/preact_init.exe" ]] ||
    die "preact_init was not created; see $LOG_FILE"
  cp "$preact_build/test/preact_init.exe" "$PREFIX/bin/preact_init.exe"

  # The Plasma State generator. The NTCC archives ship no main program for it;
  # the complete source is plasma_state_test.f90 in the 2021 server tree, which
  # arrives with a full NUBEAM distribution rather than with the modules this
  # script downloads. Its own Makefile is unusable (absolute /home paths,
  # MDSplus, termcap), so compile and link it directly.
  local src="$ROOT_DIR/vendor/server-ntcc-2021/plasma_state_test"
  if [[ ! -f "$src/plasma_state_test.f90" ]]; then
    note "skipping plasma_state_test: no source at $src"
    note "  Cases that generate a Plasma State from scratch need it; cases that"
    note "  read an existing state do not. It ships with the full NUBEAM"
    note "  distribution, not with the NTCC dependency modules."
    return 0
  fi
  note "building plasma_state_test"
  generator_build="$BUILD_DIR/generator"
  mkdir -p "$generator_build"
  ( cd "$generator_build"
    for source_file in ps_momtest.F90 plasma_state_test.f90; do
      gfortran -c -O -m64 -fno-range-check -fdollar-ok -cpp -fno-common \
        -std=legacy -fallow-argument-mismatch -fallow-invalid-boz \
        -I"$PREFIX/mod" -I"$PREFIX/include" -I"$NETCDF_HOME/include" \
        -o "${source_file%.*}.o" "$src/$source_file"
    done
    # The same group and the same static netCDF as every other link here.
    gfortran -o plasma_state_test.exe plasma_state_test.o ps_momtest.o \
      $(link_libraries) -L/ucrt64/lib -lopenblas )
  [[ -f "$generator_build/plasma_state_test.exe" ]] ||
    die "plasma_state_test was not created; see $LOG_FILE"
  cp "$generator_build/plasma_state_test.exe" "$PREFIX/bin/plasma_state_test.exe"
}

# nubeam_comp_exec requires both PREACTDIR and ADASDIR and calls bad_exit when
# either is unset, so a usable installation has to ship them. Both are runtime
# data directories that the table code writes back into: PREACT and ADAS
# generate missing reaction tables on demand and cache them here.
stage_reaction_databases() {
  local preact_root preact_data preact_dir adas_dir

  preact_root="$NTCC_SOURCE_DIR/preact"
  preact_data="$preact_root/preact"
  preact_dir="$PREFIX/share/preact"
  adas_dir="$PREFIX/share/adas"

  [[ -f "$preact_data/ORNL6086.DAT" ]] ||
    die "PREACT Aladdin cross-section data not found: $preact_data/ORNL6086.DAT"
  [[ -d "$preact_root/data" ]] ||
    die "ADAS data tree not found: $preact_root/data"

  note "initializing PREACT reaction-table directory: $preact_dir"
  mkdir -p "$preact_dir"
  # preactinit builds the tables/{cx,fs,ii,ei,sv}/... skeleton, copies
  # ORNL6086.DAT into data/, then runs preact_init to populate it.
  ( cd "$preact_data" &&
    PREACTDIR="$preact_dir" PATH="$PREFIX/bin:$PATH" \
      ./preactinit DATA "$preact_data" PATH "$PREFIX/bin" )
  [[ -f "$preact_dir/data/ORNL6086.DAT" ]] ||
    die "preactinit did not populate $preact_dir; see $LOG_FILE"

  # adas_mod.f90 composes paths as $ADASDIR/data/adf02/... for the shipped
  # cross-section archives and $ADASDIR/tables/... for the generated tables,
  # so ADASDIR needs a data/ tree and a writable tables/. macos.sh symlinks the
  # data tree; on Windows a symlink needs a privilege an ordinary account does
  # not have, so it is copied. The tree is read-only reference data, and the
  # OPEN-ADAS terms permit this local copy but not redistribution.
  note "staging ADAS data directory: $adas_dir"
  mkdir -p "$adas_dir/tables"
  if [[ ! -e "$adas_dir/data" ]]; then
    cp -R "$preact_root/data" "$adas_dir/data"
  fi
}

build_aux_executables
stage_reaction_databases

note "NUBEAM installed successfully"
note "binaries:    $PREFIX/bin"
note "libraries:   $PREFIX/lib"
note "PREACT tables: $PREFIX/share/preact"
note "ADAS data:     $PREFIX/share/adas"
note "point \$NUBEAMHOME at: $PREFIX"
