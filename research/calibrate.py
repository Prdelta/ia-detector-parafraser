"""Evalúa Binoculars en un corpus etiquetado y genera models/calibration.json.

Uso:
  python research/calibrate.py --data data/autextification_es_train.parquet --n 2000
  python research/calibrate.py --data data/autextification_es_test.parquet --eval-only

Métricas: AUROC y TPR con FPR del 1 % (la que importa para no acusar a inocentes).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import config  # noqa: E402
from backend.app.detectors.binoculars import Binoculars  # noqa: E402
from backend.app.detectors.calibration import load as load_calibration  # noqa: E402


def tpr_at_fpr(labels, ai_scores, target_fpr=0.01):
    fpr, tpr, thr = roc_curve(labels, ai_scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    return float(tpr[idx]), float(thr[idx])


def report(name, labels, bino_scores):
    # Binoculars: menor = IA, así que se invierte el signo para las métricas.
    auroc = roc_auc_score(labels, -bino_scores)
    tpr, thr = tpr_at_fpr(labels, -bino_scores)
    print(f"[{name}] n={len(labels)}  AUROC={auroc:.3f}  TPR@1%FPR={tpr:.3f}  umbral={-thr:.4f}")
    return auroc, tpr, -thr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=2000, help="muestras (balanceadas) a evaluar")
    parser.add_argument("--min-words", type=int, default=40)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save-scores", help="CSV donde guardar la puntuación de cada texto")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df = df[df.text.str.split().str.len() >= args.min_words]
    per_class = args.n // 2
    df = (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), per_class), random_state=args.seed))
        .reset_index(drop=True)
    )
    print(f"Evaluando {len(df)} textos de {args.data} (IA={df.label.mean():.0%})")

    det = Binoculars()
    det.load()
    t0 = time.time()
    scores = []
    for i, text in enumerate(df.text):
        scores.append(det.score(text))
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)}  ({(time.time() - t0) / (i + 1):.2f} s/texto)")
    df["score"] = scores
    df = df.dropna(subset=["score"])
    labels, s = df.label.to_numpy(), df.score.to_numpy()

    auroc, tpr, threshold = report("global", labels, s)
    for domain, g in df.groupby("domain"):
        if g.label.nunique() == 2:
            report(domain, g.label.to_numpy(), g.score.to_numpy())

    # Cada modelo generador y cada tarea frente a todos los textos humanos.
    humans = df[df.label == 0]
    for col in ("model", "task"):
        if col not in df or df[col].nunique() < 3:
            continue
        for value, g in df[df.label == 1].groupby(col):
            both = pd.concat([humans, g])
            report(f"{col}={value}", both.label.to_numpy(), both.score.to_numpy())

    if args.save_scores:
        df.drop(columns=["text"]).to_csv(args.save_scores, index=False)

    if args.eval_only:
        calib = load_calibration(det.model_id)
        preds = np.array([calib.probability(x) for x in s]) >= 0.5
        print(f"Exactitud con calibración actual ({'calibrada' if calib.calibrated else 'por defecto'}): "
              f"{(preds == labels).mean():.3f}")
        return

    lr = LogisticRegression().fit(s.reshape(-1, 1), labels)
    out = {
        "model_id": det.model_id,
        "slope": float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
        "threshold_low_fpr": round(threshold, 4),
        "metrics": {"auroc": round(auroc, 4), "tpr_at_1pct_fpr": round(tpr, 4), "n": int(len(df))},
        "data": Path(args.data).name,
    }
    config.CALIBRATION_FILE.parent.mkdir(exist_ok=True)
    config.CALIBRATION_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Calibración guardada en {config.CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
