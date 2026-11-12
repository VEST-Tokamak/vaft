#!/usr/bin/env bash
# Install, publish or remove the isolated vaft gui TEST instance.
#
#   sudo bash test-instance.sh install <40-hex commit>   # step B: user, venv, env file, unit
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
#
# Every nginx change is checked with `nginx -t` before nginx is reloaded; a
# failed check restores what this run changed and stops with a non-zero exit.
set -euo pipefail

DRY=0
ARGS=()
for arg in "$@"; do
  if [[ $arg == --dry-run ]]; then DRY=1; else ARGS+=("$arg"); fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

die() { echo "ERROR: $*" >&2; exit 1; }

# Path overrides exist only to rehearse on copies: with --dry-run, or inside a
# sandbox under a temporary directory (VAFT_GUI_TEST_SANDBOX). A real run with
# an inherited override pointing anywhere else is refused rather than trusted.
if [[ $DRY == 0 ]] && env | grep -q '^VAFT_GUI_TEST_'; then
  sandbox=${VAFT_GUI_TEST_SANDBOX:-}
  [[ $sandbox =~ ^(/tmp|/private/tmp|/var/folders|/private/var/folders)/[^/].*$ && $sandbox != *..* ]] || \
    die "VAFT_GUI_TEST_* overrides are only honoured with --dry-run or a VAFT_GUI_TEST_SANDBOX under /tmp; unset them"
  while IFS='=' read -r name value; do
    [[ $name == VAFT_GUI_TEST_SANDBOX ]] && continue
    [[ $value == "$sandbox"/* ]] || die "$name=$value is outside the sandbox $sandbox"
  done < <(env | grep '^VAFT_GUI_TEST_')
fi
SITE=${VAFT_GUI_TEST_SITE:-/etc/nginx/sites-available/vaft-gui}   # the live production site (443)
UNIT=${VAFT_GUI_TEST_UNIT:-/etc/systemd/system/vaft-gui-test.service}
ENV_FILE=${VAFT_GUI_TEST_ENV:-/etc/vaft-gui-test.env}
VENV=${VAFT_GUI_TEST_VENV:-/opt/vaft-gui-test}
STATE=${VAFT_GUI_TEST_STATE:-/var/lib/vaft-gui-test}
SNIPPETS=${VAFT_GUI_TEST_SNIPPETS:-/etc/nginx/snippets}
CONFD=${VAFT_GUI_TEST_CONFD:-/etc/nginx/conf.d/vaft-gui-test.conf}
LOCATIONS_NAME=vaft-gui-test-locations.conf
HEADERS_NAME=vaft-gui-test-proxy-headers.conf
INCLUDE="include $SNIPPETS/$LOCATIONS_NAME;"
HERE=$(cd "$(dirname "$0")" && pwd)

run() {
  echo "+ $*"
  if [[ $DRY == 0 ]]; then "$@"; fi
}

need_root() {
  if [[ $DRY == 0 && $(id -u) != 0 ]]; then
    die "run as root (sudo), or add --dry-run to only print the commands"
  fi
}

nginx_check() {
  if [[ $DRY == 1 ]]; then echo "+ nginx -t"; return 0; fi
  nginx -t
}

install_instance() {
  local commit=${1:-}
  [[ $commit =~ ^[0-9a-f]{40}$ ]] || die "install needs a full 40-hex commit SHA (a branch name would move); got '${commit}'"
  need_root
  if ! id vaft-gui-test >/dev/null 2>&1; then
    run useradd --system --home-dir "$STATE" --create-home --shell /usr/sbin/nologin vaft-gui-test
  fi
  [[ -d $STATE ]] || run install -d -o vaft-gui-test -g vaft-gui-test -m 0750 "$STATE"
  [[ -d $VENV ]] || run python3 -m venv "$VENV"
  run "$VENV/bin/pip" install --quiet "vaft[gui] @ git+https://github.com/VEST-Tokamak/vaft@$commit"
  if [[ ! -e $ENV_FILE ]]; then
    run install -m 0600 -o root -g root "$HERE/vaft-gui-test.env.example" "$ENV_FILE"
  else
    run chown root:root "$ENV_FILE"
    run chmod 0600 "$ENV_FILE"
  fi
  run install -m 0644 "$HERE/vaft-gui-test.service" "$UNIT"
  run systemctl daemon-reload
  if [[ ! -e $ENV_FILE ]] || grep -q '^HS_PASSWORD=change-me' "$ENV_FILE"; then
    echo "NOT STARTED: fill in $ENV_FILE (the read-only HSDS account), then: systemctl enable --now vaft-gui-test"
  else
    run systemctl enable --now vaft-gui-test
  fi
}

valid_cidr() {
  local value=$1
  [[ $value =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}(/([0-9]|[12][0-9]|3[0-2]))?$ ]] && return 0
  [[ $value == *:* && $value =~ ^[0-9a-fA-F:]+(/([0-9]|[1-9][0-9]|1[01][0-9]|12[0-8]))?$ ]] && return 0
  return 1
}

publish_nginx() {
  [[ $# -ge 1 ]] || die "usage: nginx <cidr> [<cidr>...] (the approved internal ranges)"
  local cidr
  for cidr in "$@"; do valid_cidr "$cidr" || die "not an address or CIDR range: '$cidr'"; done
  need_root
  [[ -f $SITE ]] || die "no site at $SITE"
  local matches
  matches=$(grep -c '^[[:space:]]*location /gui/ {' "$SITE" || true)
  if ! grep -qF "$LOCATIONS_NAME" "$SITE" && [[ $matches != 1 ]]; then
    die "$SITE has $matches 'location /gui/ {' lines, not exactly one: add '$INCLUDE' by hand inside its 443 server"
  fi

  local backup target tmp
  local added=()
  backup="$SITE.before-gui-test.$(date +%Y%m%d-%H%M%S)"
  run cp -p "$SITE" "$backup"
  for target in "$SNIPPETS/$LOCATIONS_NAME" "$SNIPPETS/$HEADERS_NAME" "$CONFD"; do
    [[ -e $target ]] || added+=("$target")
  done
  restore() {
    echo "restoring $SITE from $backup and removing the files this run added" >&2
    if [[ $DRY == 0 ]]; then
      cp -p "$backup" "$SITE"
      rm -f "${added[@]+"${added[@]}"}"
      nginx -t || echo "WARNING: nginx -t still fails after the restore; do not reload, check by hand" >&2
    fi
  }

  tmp=$(mktemp)
  # The locations file with its allow lines filled in; deny all stays last.
  awk -v ranges="$*" '
    /# allow <approved internal CIDR>;/ { n = split(ranges, r, " "); for (i = 1; i <= n; i++) print "    allow " r[i] ";"; next }
    { print }
  ' "$HERE/nginx-vaft-gui-test-locations.conf" > "$tmp"
  run install -m 0644 "$tmp" "$SNIPPETS/$LOCATIONS_NAME"
  rm -f "$tmp"
  run install -m 0644 "$HERE/$HEADERS_NAME" "$SNIPPETS/$HEADERS_NAME"
  run install -m 0644 "$HERE/nginx-vaft-gui-test-http.conf" "$CONFD"
  if ! grep -qF "$LOCATIONS_NAME" "$SITE"; then
    # Insert the include just before the single /gui/ location (the 443 server's).
    run sed -i "0,/^\(\s*\)location \/gui\/ {/s##\1$INCLUDE\n&#" "$SITE"
    if [[ $DRY == 0 ]] && ! grep -qF "$INCLUDE" "$SITE"; then
      restore
      die "the include did not land in $SITE"
    fi
  fi
  if ! nginx_check; then
    restore
    die "nginx -t failed with the test instance's files; nothing reloaded"
  fi
  run systemctl reload nginx
  echo "published at /gui-test/ for: $*  (backup: $backup)"
}

remove_instance() {
  need_root
  # 1. Take the include out of the site, whatever its exact spelling, and
  #    check nginx BEFORE deleting anything the site might still need.
  if [[ -f $SITE ]] && grep -qF "$LOCATIONS_NAME" "$SITE"; then
    local backup
    backup="$SITE.before-gui-test-remove.$(date +%Y%m%d-%H%M%S)"
    run cp -p "$SITE" "$backup"
    run sed -i "/$LOCATIONS_NAME/d" "$SITE"
    if ! nginx_check; then
      if [[ $DRY == 0 ]]; then cp -p "$backup" "$SITE"; fi
      die "nginx -t failed without the include; restored $SITE, nothing removed"
    fi
    run systemctl reload nginx
  fi
  # 2. Only now remove the test instance's own nginx files, and check again.
  run rm -f "$CONFD" "$SNIPPETS/$LOCATIONS_NAME" "$SNIPPETS/$HEADERS_NAME"
  nginx_check || die "nginx -t fails after removing the test files; check by hand before any reload"
  run systemctl reload nginx
  # 3. The service, then its files and user; any failure stops here.
  if [[ $DRY == 1 || -e $UNIT ]]; then
    run systemctl disable --now vaft-gui-test.service
  fi
  if [[ $DRY == 0 ]] && systemctl is-active --quiet vaft-gui-test.service; then
    die "vaft-gui-test is still running; stop it, then run remove again"
  fi
  run rm -f "$UNIT" "$ENV_FILE"
  run systemctl daemon-reload
  run rm -rf "$VENV" "$STATE"
  if id vaft-gui-test >/dev/null 2>&1; then
    run userdel vaft-gui-test
  fi
  echo "removed; the HSDS account is removed separately (ACL rows, then login, then restart)"
}

status() {
  systemctl is-active vaft-gui-test 2>/dev/null || true
  if [[ -x $VENV/bin/vaft ]]; then "$VENV/bin/python" -c "import vaft; print('vaft', vaft.__version__)"; else echo "no venv"; fi
  ls -l "$ENV_FILE" 2>/dev/null || echo "no env file"
  if grep -qF "$LOCATIONS_NAME" "$SITE" 2>/dev/null; then echo "nginx include: present"; else echo "nginx include: absent"; fi
  if ss -ltnH 2>/dev/null | grep -q '127.0.0.1:5016 '; then echo "listening on 127.0.0.1:5016"; else echo "not listening"; fi
}

case ${1:-} in
  install) shift; install_instance "$@" ;;
  nginx) shift; publish_nginx "$@" ;;
  remove) remove_instance ;;
  status) status ;;
  *) sed -n '2,17p' "$0"; exit 1 ;;
esac
