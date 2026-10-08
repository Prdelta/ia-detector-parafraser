#!/usr/bin/env bash
# Pasos finales tras reentrenar el clasificador: meta-clasificador, auditoría de falsos positivos
# y calibración en inglés. Uso:  PY=/ruta/a/python bash research/pipeline_final.sh
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8

echo "== Meta-clasificador"
$PY -u research/train_meta.py --recompute
echo "== Auditoría de falsos positivos"
$PY -u research/audit_fp.py
echo "== Inglés"
bash research/pipeline_ingles.sh
echo "FIN"
