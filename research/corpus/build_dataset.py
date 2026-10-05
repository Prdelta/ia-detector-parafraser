"""Une textos humanos y de IA en conjuntos train/test listos para calibrar y evaluar.

La división se hace por texto humano de origen: un texto humano y sus versiones de IA
quedan siempre en el mismo conjunto, para que el test no comparta temas con el train.

Uso:  python research/corpus/build_dataset.py [--test-size 0.3] [--holdout-model NOMBRE]
Salida: data/corpus/train.parquet, data/corpus/test.parquet
"""

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_human import is_spanish  # noqa: E402

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--holdout-model", help="modelo cuyos textos van solo al test (generalización)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    human = pd.read_json(CORPUS / "human.jsonl", lines=True)
    human = human[human.text.map(is_spanish)]
    human = human.assign(label=0, model="humano", task="humano", source_id=human["id"])
    ai_files = sorted((CORPUS / "ai").glob("*.jsonl"))
    ai = pd.concat([pd.read_json(f, lines=True) for f in ai_files], ignore_index=True)
    ai = ai.assign(label=1)

    cols = ["id", "source_id", "text", "label", "domain", "model", "task", "words"]
    df = pd.concat([human[cols], ai[cols]], ignore_index=True).drop_duplicates("text")

    sources = sorted(human["id"])
    random.Random(args.seed).shuffle(sources)
    test_sources = set(sources[: int(len(sources) * args.test_size)])
    is_test = df.source_id.isin(test_sources)
    if args.holdout_model:
        held = df.model.str.contains(args.holdout_model, regex=False)
        is_test = (is_test & ~held) | held
        # Evitar que el train vea textos humanos emparejados con el modelo reservado.
        is_test |= df.source_id.isin(df[held].source_id) & (df.label == 0)

    for name, part in [("train", df[~is_test]), ("test", df[is_test])]:
        path = CORPUS / f"{name}.parquet"
        part.reset_index(drop=True).to_parquet(path, index=False)
        print(f"{name}: {len(part)} textos (IA={part.label.mean():.0%}) -> {path}")
        print(part.groupby(["model", "task"]).size().to_string(), "\n")

    summary = {"archivos_ia": [f.name for f in ai_files], "humanos": len(human), "ia": len(ai)}
    (CORPUS / "resumen.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
