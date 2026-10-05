"""Descarga el corpus AuTexTification 2023 (IberLEF) para detección humano/IA.

Uso:  python research/download_data.py [--lang es]
Licencia de los datos: CC BY-NC-SA 4.0 (solo uso no comercial).
"""

import argparse
from pathlib import Path

import pandas as pd

BASE = "https://huggingface.co/datasets/symanto/autextification2023/resolve/main/data"
OUT = Path(__file__).resolve().parents[1] / "data"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="es", choices=["es", "en"])
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for split in ("train", "test"):
        url = f"{BASE}/{split}/subtask_1/{args.lang}/{split}.tsv"
        df = pd.read_csv(url, sep="\t")
        df["label"] = (df["label"] == "generated").astype(int)  # 1 = IA
        path = OUT / f"autextification_{args.lang}_{split}.parquet"
        df[["id", "text", "label", "domain", "model"]].to_parquet(path, index=False)
        print(f"{split}: {len(df)} textos ({df.label.mean():.0%} IA) -> {path}")


if __name__ == "__main__":
    main()
