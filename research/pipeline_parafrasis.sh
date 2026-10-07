#!/usr/bin/env bash
# Refuerza la detección de paráfrasis: genera paráfrasis con modelos locales, reconstruye el corpus
# y reentrena clasificador y meta-clasificador. Es reanudable (cada paso retoma lo ya hecho).
# Uso:  PY=/ruta/a/python bash research/pipeline_parafrasis.sh   (registro en data/pipeline_parafrasis.log)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
N=${N:-350}

for model in BSC-LT/salamandra-2b-instruct utter-project/EuroLLM-1.7B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  echo "== Paráfrasis con $model"
  $PY -u research/corpus/generate_ai.py --provider local --model "$model" --domain academico \
      --n "$N" --tareas parafrasear --batch-size 8
done

echo "== Corpus"
$PY -u research/corpus/build_dataset.py

echo "== Clasificador"
# Con las paráfrasis la IA ya iguala a los humanos en el corpus: sin sobremuestreo.
$PY -u research/train_classifier.py --ai-oversample 1

echo "== Meta-clasificador"
$PY -u research/train_meta.py --recompute

echo "== Auditoría de falsos positivos"
$PY -u research/audit_fp.py
echo "FIN"
