#!/usr/bin/env bash
# Calibra y evalúa Binoculars en inglés con AuTexTification (en). Requiere:
#   python research/download_data.py --lang en
# Uso:  PY=/ruta/a/python bash research/pipeline_ingles.sh   (registro en data/pipeline_ingles.log)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

echo "== Calibración (train)"
$PY -u research/calibrate.py --lang en --data data/autextification_en_train.parquet --n 2000
echo "== Evaluación (test, dominios no vistos)"
$PY -u research/calibrate.py --lang en --data data/autextification_en_test.parquet --n 3000 --eval-only
echo "FIN"
