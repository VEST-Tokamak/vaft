#!/usr/bin/env bash
# Install, publish or remove the isolated vaft gui TEST instance (#1862).
#
#   sudo bash test-instance.sh install <commit>          # step B: user, venv, env file, unit
#   sudo bash test-instance.sh nginx <cidr> [<cidr>...]  # step D: conf.d zone + include, allow ranges
#   sudo bash test-instance.sh remove                    # rollback of B and D
#   bash test-instance.sh status                         # read-only
#   ... --dry-run                                        # print the commands, change nothing
#
# Run it from the directory holding the files of vaft/deploy/gui/ (this one).
# It never touches vaft-gui.service, /etc/vaft-gui.env, /opt/vaft-gui or HSDS:
# the read-only HSDS account (step A) is created by hand through the server's
# HSDS account procedure first, and its secret goes into /etc/vaft-gui-test.env
# with an editor, never through this script or a command line.
set -euo pipefail

# The paths can be overridden (VAFT_GUI_TEST_*) only to rehearse on copies.
SITE=${VAFT_GUI_TEST_SITE:-/etc/nginx/sites-available/vaft-gui}   # the live production site (443)
UNIT=${VAFT_GUI_TEST_UNIT:-/etc/systemd/system/vaft-gui-test.service}
ENV_FILE=${VAFT_GUI_TEST_ENV:-/etc/vaft-gui-test.env}
VENV=${VAFT_GUI_TEST_VENV:-/opt/vaft-gui-test}
STATE=${VAFT_GUI_TEST_STATE:-/var/lib/vaft-gui-test}
SNIPPETS=${VAFT_GUI_TEST_SNIPPETS:-/etc/nginx/snippets}
CONFD=${VAFT_GUI_TEST_CONFD:-/etc/nginx/conf.d/vaft-gui-test.conf}
INCLUDE="include $SNIPPETS/vaft-gui-test-locations.conf;"
HERE=$(cd "$(dirname "$0")" && pwd)

DRY=0
ARGS=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then DRY=1; else ARGS+=("$arg"); fi
done
set -- "${ARGS[@]:-}"

run() {
  echo "+ $*"
  if [[ $DRY == 0 ]]; then "$@"; fi
}

need_root() {
  if [[ $DRY == 0 && $(id -u) != 0 ]]; then
    echo "run as root (sudo), or add --dry-run to only print the commands" >&2
    exit 1
  fi
}

install_instance() {
  local commit=${1:?usage: install <commit>}
  need_root
  id vaft-gui-test >/dev/null 2>&1 || \
    run useradd --system --home-dir "$STATE" --create-home --shell /usr/sbin/nologin vaft-gui-test
  [[ -d $VENV ]] || run python3 -m venv "$VENV"
  run "$VENV/bin/pip" install --quiet "vaft[gui] @ git+https://github.com/VEST-Tokamak/vaft@$commit"
  if [[ ! -e $ENV_FILE ]]; then
    run install -m 0600 -o root -g root "$HERE/vaft-gui-test.env.example" "$ENV_FILE"
    echo "NOTE: edit $ENV_FILE (sudoedit) with the read-only HSDS account before starting"
  fi
  run install -m 0644 "$HERE/vaft-gui-test.service" "$UNIT"
  run systemctl daemon-reload
  if [[ ! -e $ENV_FILE ]] || grep -q '^HS_PASSWORD=change-me' "$ENV_FILE"; then
    echo "NOT STARTED: fill in $ENV_FILE (the read-only HSDS account), then: systemctl enable --now vaft-gui-test"
  else
    run systemctl enable --now vaft-gui-test
  fi
}

publish_nginx() {
  [[ $# -ge 1 ]] || { echo "usage: nginx <cidr> [<cidr>...] (the approved internal ranges)" >&2; exit 1; }
  need_root
  local tmp
  tmp=$(mktemp)
  # The locations file with its allow lines filled in; deny all stays last.
  awk -v ranges="$*" '
    /# allow <approved internal CIDR>;/ { n = split(ranges, r, " "); for (i = 1; i <= n; i++) print "    allow " r[i] ";"; next }
    { print }
  ' "$HERE/nginx-vaft-gui-test-locations.conf" > "$tmp"
  run install -m 0644 "$tmp" "$SNIPPETS/vaft-gui-test-locations.conf"
  run install -m 0644 "$HERE/vaft-gui-test-proxy-headers.conf" "$SNIPPETS/vaft-gui-test-proxy-headers.conf"
  run install -m 0644 "$HERE/nginx-vaft-gui-test-http.conf" "$CONFD"
  rm -f "$tmp"
  if ! grep -qF "$INCLUDE" "$SITE"; then
    run cp -p "$SITE" "$SITE.before-gui-test"
    # Insert the include just before the /gui/ location of the 443 server.
    run sed -i "0,/^\(\s*\)location \/gui\/ {/s##\1$INCLUDE\n&#" "$SITE"
  fi
  if [[ $DRY == 0 ]] && ! nginx -t; then
    echo "nginx -t failed: restoring the site" >&2
    cp -p "$SITE.before-gui-test" "$SITE"
    rm -f "$CONFD"
    exit 1
  fi
  run systemctl reload nginx
}

remove_instance() {
  need_root
  if grep -qF "$INCLUDE" "$SITE" 2>/dev/null; then
    run sed -i "\|$INCLUDE|d" "$SITE"
  fi
  run rm -f "$CONFD" "$SNIPPETS/vaft-gui-test-locations.conf" "$SNIPPETS/vaft-gui-test-proxy-headers.conf"
  if [[ $DRY == 0 ]]; then nginx -t && systemctl reload nginx; else echo "+ nginx -t && systemctl reload nginx"; fi
  run systemctl disable --now vaft-gui-test || true
  run rm -f "$UNIT" "$ENV_FILE"
  run systemctl daemon-reload
  run rm -rf "$VENV" "$STATE"
  id vaft-gui-test >/dev/null 2>&1 && run userdel vaft-gui-test || true
  echo "removed; the HSDS account is removed separately (ACL rows, then login, then restart)"
}

status() {
  systemctl is-active vaft-gui-test 2>/dev/null || true
  [[ -x $VENV/bin/vaft ]] && "$VENV/bin/python" -c "import vaft; print('vaft', vaft.__version__)" || echo "no venv"
  ls -l "$ENV_FILE" 2>/dev/null || echo "no env file"
  grep -qF "$INCLUDE" "$SITE" 2>/dev/null && echo "nginx include: present" || echo "nginx include: absent"
  ss -ltnH 2>/dev/null | grep -q '127.0.0.1:5016 ' && echo "listening on 127.0.0.1:5016" || echo "not listening"
}

case ${1:-} in
  install) shift; install_instance "$@" ;;
  nginx) shift; publish_nginx "$@" ;;
  remove) remove_instance ;;
  status) status ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
