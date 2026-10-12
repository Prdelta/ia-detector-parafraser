#!/usr/bin/env bash
# Segundo lote de paráfrasis, ahora con las cuatro variantes de enunciado (el primero usó solo una),
# y reentrenamiento en models/clasificador_v4 sin tocar el clasificador que usa la aplicación.
# Al final se recalcula el meta del clasificador actual con el mismo conjunto de prueba, para comparar.
# Uso:  PY=/ruta/a/python bash research/pipeline_parafrasis_v2.sh   (registro en data/pipeline_parafrasis_v2.log)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
export PYTHONIOENCODING=utf-8
N=${N:-250}

for model in BSC-LT/salamandra-2b-instruct utter-project/EuroLLM-1.7B-Instruct Qwen/Qwen2.5-1.5B-Instruct; do
  echo "== Paráfrasis (variantes) con $model"
  $PY -u research/corpus/generate_ai.py --provider local --model "$model" --domain academico \
      --n "$N" --tareas parafrasear --sufijo variantes --batch-size "${BATCH:-4}"
done

echo "== Corpus"
$PY -u research/corpus/build_dataset.py

echo "== Clasificador v4"
$PY -u research/train_classifier.py --ai-oversample 1 --out models/clasificador_v4

echo "== Meta v4"
IADECCION_CLASIFICADOR_DIR=models/clasificador_v4 IADECCION_META=models/meta_v4.json \
  $PY -u research/train_meta.py --recompute

echo "== Meta del clasificador actual con el nuevo conjunto de prueba (comparación)"
IADECCION_META=data/meta_actual_nuevo_test.json $PY -u research/train_meta.py --recompute
echo "FIN"
