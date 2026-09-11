#!/bin/bash
# Télécharge le connectome MaleCNS v1.0 (Janelia / Google Research, CC-BY)
# et le convertit en graphe compact. À lancer depuis le Terminal du Mac.
set -e
cd "$(dirname "$0")"
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome

echo "== Téléchargement (~1,2 Go) =="
curl -# -L -C - -O "$B/body-annotations-male-cns-v1.0-minconf-0.5.feather"
curl -# -L -C - -O "$B/body-neurotransmitters-male-cns-v1.0.feather"
curl -# -L -C - -O "$B/connectome-weights-male-cns-v1.0-minconf-0.5.feather"

echo "== Environnement Python =="
if [ ! -d .venv ]; then python3 -m venv .venv; fi
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q pyarrow numpy

echo "== Conversion =="
./.venv/bin/python convert.py .
echo
echo "Terminé. Dis-le moi et je prends la suite."
