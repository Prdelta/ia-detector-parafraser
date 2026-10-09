"""Aprende cómo combinar Binoculars, estilometría y el clasificador (genera models/meta.json).

Se entrena con textos que el clasificador NO vio al entrenar, para que sus probabilidades
no estén infladas:
  - Corpus propio, conjunto test (modelos actuales, textos largos, incluye paráfrasis)
  - AuTexTification 2023 (es, test): noticias y reseñas, ≥60 palabras

Las métricas se estiman con validación cruzada agrupada por tema (``source_id``), así un texto
humano y sus versiones de IA nunca quedan repartidos entre entrenamiento y evaluación.

Uso:  python research/train_meta.py [--autext-n 1500] [--folds 5]
Salida: models/meta.json y caché de rasgos en data/meta_features_<clasificador>.parquet
(IADECCION_CLASIFICADOR_DIR e IADECCION_META permiten evaluar otro clasificador sin tocar el actual)
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
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import config  # noqa: E402
from backend.app.detectors import meta  # noqa: E402
from backend.app.detectors import stylometry  # noqa: E402
from backend.app.detectors.binoculars import Binoculars  # noqa: E402
from backend.app.detectors.calibration import load as load_calibration  # noqa: E402
from backend.app.detectors.classifier import Classifier  # noqa: E402
from backend.app.segment import normalize, split_sentences  # noqa: E402



def tpr_at_fpr(labels, probs, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, probs)
    return float(tpr[np.searchsorted(fpr, target_fpr, side="right") - 1])


def load_data(autext_n: int, seed: int) -> pd.DataFrame:
    corpus = pd.read_parquet(ROOT / "data" / "corpus" / "test.parquet")
    corpus["origen"] = "corpus"
    autext = pd.read_parquet(ROOT / "data" / "autextification_es_test.parquet")
    autext = autext[autext.text.str.split().str.len() >= 60]
    autext = autext.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), autext_n // 2), random_state=seed))
    autext = autext.assign(id="at_" + autext.id.astype(str), task="autext", origen="autext")
    autext["source_id"] = autext.id
    cols = ["id", "source_id", "text", "label", "domain", "model", "task", "origen"]
    return pd.concat([corpus[cols], autext[cols]], ignore_index=True)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    det, clf = Binoculars(), Classifier()
    det.load()
    clf.load()
    rows, t0 = [], time.time()
    for i, text in enumerate(df.text):
        text = normalize(text)
        style = stylometry.analyze(text, split_sentences(text))
        rows.append({"binoculars": det.score(text), "estilo": style.score, "clasificador": clf.probability(text)})
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(df)}  ({(time.time() - t0) / (i + 1):.2f} s/texto)", flush=True)
    out = pd.concat([df.drop(columns=["text"]).reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return out.dropna(subset=["binoculars"])


def design(df: pd.DataFrame) -> np.ndarray:
    return np.array([meta.features(b, s, c) for b, s, c in zip(df.binoculars, df.estilo, df.clasificador)])


def fit(X, y):
    mean, std = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(class_weight="balanced").fit((X - mean) / std, y)
    return meta.Meta(lr.coef_[0].tolist(), float(lr.intercept_[0]), mean.tolist(), std.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--autext-n", type=int, default=1500)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--recompute", action="store_true", help="ignorar la caché de rasgos")
    parser.add_argument("--fpr-objetivo", type=float, default=0.01,
                        help="FPR máximo con el umbral 0.65 (0 = sin ajustar el intercepto)")
    args = parser.parse_args()

    det_id = f"{config.OBSERVER_MODEL}|{config.PERFORMER_MODEL}"
    clf_id = config.CLASSIFIER_DIR.name
    CACHE = ROOT / "data" / f"meta_features_{clf_id}.parquet"
    if CACHE.exists() and not args.recompute:
        feats = pd.read_parquet(CACHE)
        print(f"Rasgos leídos de {CACHE} ({len(feats)} textos)")
    else:
        data = load_data(args.autext_n, args.seed)
        print(f"Calculando rasgos de {len(data)} textos (IA={data.label.mean():.0%})")
        feats = compute_features(data)
        feats.to_parquet(CACHE)

    X, y, groups = design(feats), feats.label.to_numpy(), feats.source_id.to_numpy()
    calib = load_calibration(det_id)

    # Probabilidades fuera de muestra para cada método.
    oof = {
        "binoculars": np.array([calib.probability(s) for s in feats.binoculars]),
        "clasificador": feats.clasificador.to_numpy(),
        "pesos fijos": 0.5 * np.array([calib.probability(s) for s in feats.binoculars])
        + 0.3 * feats.clasificador.to_numpy() + 0.2 * feats.estilo.to_numpy(),
        "meta": np.zeros(len(feats)),
    }
    for tr, te in GroupKFold(args.folds).split(X, y, groups):
        m = fit(X[tr], y[tr])
        oof["meta"][te] = [m.probability(list(x)) for x in X[te]]

    # La regresión se entrena con clases equilibradas, así que 0.65 no implica pocos falsos positivos.
    # Se desplaza el intercepto para que 0.65 ("probablemente IA") marque como mucho el FPR objetivo
    # de los textos humanos fuera de muestra.
    shift = 0.0
    if args.fpr_objetivo > 0:
        q = np.quantile(oof["meta"][y == 0], 1 - args.fpr_objetivo)
        shift = meta.logit(0.65) - meta.logit(q)
        p = np.clip(oof["meta"], 1e-6, 1 - 1e-6)
        oof["meta ajustado"] = 1 / (1 + np.exp(-(np.log(p / (1 - p)) + shift)))

    print(f"\n{'método':<14}{'subconjunto':<22}{'n':>6}{'AUROC':>8}{'TPR@1%':>8}{'FPR≥0.65':>10}{'TPR≥0.65':>10}")
    subsets = {"todo": np.ones(len(feats), bool)}
    for origen in feats.origen.unique():
        subsets[origen] = (feats.origen == origen).to_numpy()
    humans = (feats.label == 0).to_numpy()
    for task in ("redactar", "humanizar", "parafrasear"):
        subsets[f"corpus/{task}"] = humans & (feats.origen == "corpus").to_numpy() | (feats.task == task).to_numpy()
    for name, p in oof.items():
        for sub, mask in subsets.items():
            yy, pp = y[mask], p[mask]
            if len(np.unique(yy)) < 2:
                continue
            print(f"{name:<14}{sub:<22}{mask.sum():>6}{roc_auc_score(yy, pp):>8.3f}{tpr_at_fpr(yy, pp):>8.3f}"
                  f"{(pp[yy == 0] >= 0.65).mean():>10.3%}{(pp[yy == 1] >= 0.65).mean():>10.1%}")

    final = fit(X, y)
    final.intercept += shift
    final_oof = oof["meta ajustado"] if shift else oof["meta"]
    out = {
        "features": list(meta.FEATURES),
        "binoculars": det_id,
        "clasificador": clf_id,
        "coef": final.coef,
        "intercept": final.intercept,
        "mean": final.mean,
        "std": final.std,
        "metricas_cv": {
            "auroc": round(roc_auc_score(y, final_oof), 4),
            "tpr_at_1pct_fpr": round(tpr_at_fpr(y, final_oof), 4),
            "fpr_at_065": round(float((final_oof[y == 0] >= 0.65).mean()), 4),
            "tpr_at_065": round(float((final_oof[y == 1] >= 0.65).mean()), 4),
            "n": int(len(y)),
        },
        "ajuste_intercepto": round(float(shift), 4),
    }
    config.META_FILE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nCoeficientes (estandarizados): {dict(zip(meta.FEATURES, np.round(final.coef, 3)))}")
    print(f"Meta-clasificador guardado en {config.META_FILE}")


if __name__ == "__main__":
    main()
