#!/usr/bin/env bash
# Entrena un clasificador con probabilidades menos saturadas (suavizado de etiquetas + mejor punto en
# validación) en models/clasificador_v3 y su meta-clasificador en models/meta_v3.json, sin tocar los
# que usa la aplicación. Uso:  PY=/ruta/a/python bash research/pipeline_clasificador_v3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

echo "== Clasificador v3"
$PY -u research/train_classifier.py --label-smoothing 0.1 --val-frac 0.15 --out models/clasificador_v3
echo "== Meta-clasificador v3"
IADECCION_CLASIFICADOR_DIR=models/clasificador_v3 IADECCION_META=models/meta_v3.json \
  $PY -u research/train_meta.py --recompute
echo "FIN"
