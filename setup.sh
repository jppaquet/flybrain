#!/usr/bin/env bash
# flybrain setup — run once after `git clone`. Safe to re-run: finished steps are skipped.
#
#   1. finds Python >= 3.12
#   2. downloads the MaleCNS v1.0 connectome (Janelia FlyEM / Google Research, CC-BY, ~1.1 GB)
#      and checks every file against the MD5 published by the server (interrupted downloads resume)
#   3. creates .venv with pinned, hash-checked dependencies (requirements.txt)
#   4. converts the connectome into a compact graph (malecns_graph.npz, neurons.csv)
#
# Environment variables:
#   PYTHON=/path/to/python3   interpreter used to create .venv (default: first python3.x >= 3.12 on PATH)
#   MIN_WEIGHT=3              minimum synapse count to keep a connection (forces a new conversion)
set -euo pipefail
cd "$(dirname "$0")"

BASE=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome
FILES=(
  body-annotations-male-cns-v1.0-minconf-0.5.feather
  body-neurotransmitters-male-cns-v1.0.feather
  connectome-weights-male-cns-v1.0-minconf-0.5.feather
)

step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31mError:\033[0m %s\n' "$*" >&2; exit 1; }
py_ok() { "$1" -c 'import sys; sys.exit(sys.version_info < (3, 12))' 2>/dev/null; }

# ----------------------------------------------------------------- 1. Python
step "Python"
command -v curl >/dev/null 2>&1 || die "curl is required."
PY=""
if [ -n "${PYTHON:-}" ]; then
  py_ok "$PYTHON" || die "PYTHON=$PYTHON is not a Python >= 3.12 interpreter."
  PY=$PYTHON
else
  for c in python3.14 python3.13 python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1 && py_ok "$c"; then PY=$(command -v "$c"); break; fi
  done
fi
[ -n "$PY" ] || die "Python >= 3.12 not found.
       macOS: brew install python@3.13   ·   Debian/Ubuntu: sudo apt install python3.12 python3.12-venv
       or point to an interpreter: PYTHON=/path/to/python3 ./setup.sh"
ok "$("$PY" --version) ($PY)"

# ------------------------------------------------------------ 2. Data files
step "Connectome data (MaleCNS v1.0, ~1.1 GB)"
md5_b64() {   # base64 MD5, the format of Google Cloud Storage's x-goog-hash header
  "$PY" - "$1" <<'EOF'
import base64, hashlib, sys
h = hashlib.md5()
with open(sys.argv[1], "rb") as f:
    for chunk in iter(lambda: f.read(1 << 20), b""):
        h.update(chunk)
print(base64.b64encode(h.digest()).decode())
EOF
}
missing=0
for f in "${FILES[@]}"; do [ -f "$f" ] || missing=1; done
if [ "$missing" = 1 ]; then
  free_kb=$(df -Pk . | awk 'NR == 2 { print $4 }')
  [ "$free_kb" -gt 1700000 ] || die "about 1.7 GB of free disk space is needed (data + converted graph)."
fi
for f in "${FILES[@]}"; do
  headers=$(curl -sfIL "$BASE/$f" | tr -d '\r') || headers=""
  want=$(printf '%s\n' "$headers" | sed -n 's/^[Xx]-[Gg]oog-[Hh]ash: md5=//p' | tail -n 1)
  if [ -z "$want" ]; then
    if [ -f "$f" ]; then warn "$f present, but the server is unreachable: checksum not verified"; continue; fi
    die "cannot reach $BASE — check your network connection."
  fi
  if [ -f "$f" ] && [ "$(md5_b64 "$f")" = "$want" ]; then ok "$f (checksum verified)"; continue; fi
  for attempt in 1 2; do
    echo "   downloading $f"
    curl -# -fL -C - -o "$f" "$BASE/$f" || true        # -C - resumes a partial file
    if [ -f "$f" ] && [ "$(md5_b64 "$f")" = "$want" ]; then ok "$f (checksum verified)"; break; fi
    if [ "$attempt" = 2 ]; then die "$f is still corrupted after a fresh download."; fi
    warn "checksum mismatch for $f: downloading it again from scratch"
    rm -f "$f"
  done
done

# ------------------------------------------------------- 3. Python environment
step "Python environment (.venv)"
if [ -x .venv/bin/python ] && py_ok .venv/bin/python; then
  ok "existing .venv ($(.venv/bin/python --version))"
else
  if [ -e .venv ]; then warn ".venv exists but is unusable (broken or Python < 3.12): recreating it"; rm -rf .venv; fi
  "$PY" -m venv .venv
  ok "created .venv"
fi
.venv/bin/python -m pip install --quiet --disable-pip-version-check --require-hashes -r requirements.txt
ok "$(.venv/bin/python -c 'import numpy, pyarrow; print(f"numpy {numpy.__version__}, pyarrow {pyarrow.__version__}")')"

# ---------------------------------------------------------------- 4. Conversion
step "Compact graph (malecns_graph.npz, neurons.csv)"
uptodate=1
[ -f malecns_graph.npz ] && [ -f neurons.csv ] || uptodate=0
for f in "${FILES[@]}"; do [ malecns_graph.npz -nt "$f" ] || uptodate=0; done
if [ -z "${MIN_WEIGHT:-}" ] && [ "$uptodate" = 1 ]; then
  ok "already up to date"
else
  .venv/bin/python convert.py . | sed 's/^/   /'
  rm -f meta_cache.npz                                    # soma-position cache, rebuilt at first start
  ok "converted"
fi

step "Done"
cat <<'EOF'
   Start the server:  ./run.sh          (options: --port 8765 --host 127.0.0.1)
   Then open:         http://127.0.0.1:8765        connectome dashboard
                      http://127.0.0.1:8765/fly    3D fly
EOF
