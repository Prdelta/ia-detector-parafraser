"""Calibra el Binoculars del navegador (modelos ONNX q4f16, frontend/local) y lo compara con el servidor.

La cuantización cambia las puntuaciones, así que el modo local necesita su propia calibración.
Las puntuaciones se calculan con el mismo código JavaScript que usa el navegador
(research/browser/score.mjs, en Node con CPU).

  - Calibración: AuTexTification (es, train), como models/calibration.json
  - Evaluación: conjunto test del corpus propio, frente a las puntuaciones del servidor

Uso:  python research/calibrate_local.py [--n 1000]
Requiere:  cd research/browser && npm install
Salida: frontend/local/calibration.json (las puntuaciones se guardan en data/local_scores.jsonl
y se reutilizan si se interrumpe)
"""

import argparse
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from calibrate import tpr_at_fpr  # noqa: E402

SCORES = ROOT / "data" / "local_scores.jsonl"
OUT = ROOT / "frontend" / "local" / "calibration.json"
MODEL_ID = "onnx-community/Qwen2.5-0.5B|onnx-community/Qwen2.5-0.5B-Instruct|q4f16"  # = binoculars.js


def load_data(n: int, seed: int) -> pd.DataFrame:
    autext = pd.read_parquet(ROOT / "data" / "autextification_es_train.parquet")
    autext = autext[autext.text.str.split().str.len() >= 40]
    autext = autext.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), n // 2), random_state=seed))
    autext = autext.assign(id="at_" + autext.id.astype(str), conjunto="calibracion", task="autext")
    corpus = pd.read_parquet(ROOT / "data" / "corpus" / "test.parquet").assign(conjunto="evaluacion")
    cols = ["id", "text", "label", "task", "conjunto"]
    return pd.concat([autext[cols], corpus[cols]], ignore_index=True)


def score_missing(df: pd.DataFrame) -> pd.DataFrame:
    done = {}
    if SCORES.exists():
        done = {r["id"]: r["score"] for r in map(json.loads, SCORES.read_text(encoding="utf-8").splitlines())}
    todo = df[~df.id.isin(done)]
    if len(todo):
        print(f"Puntuando {len(todo)} textos con el Binoculars del navegador (ya hechos: {len(done)})", flush=True)
        payload = "\n".join(json.dumps({"id": i, "text": t}, ensure_ascii=False) for i, t in zip(todo.id, todo.text))
        with SCORES.open("a", encoding="utf-8") as out:
            proc = subprocess.Popen(["node", "score.mjs"], cwd=ROOT / "research" / "browser", stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, text=True, encoding="utf-8")
            # Escribir en otro hilo: así cada puntuación se guarda en cuanto llega.
            def feed():
                proc.stdin.write(payload)
                proc.stdin.close()

            threading.Thread(target=feed, daemon=True).start()
            for line in proc.stdout:
                out.write(line)
                out.flush()
                r = json.loads(line)
                done[r["id"]] = r["score"]
            if proc.wait():
                sys.exit("score.mjs falló")
    return df.assign(score=df.id.map(done)).dropna(subset=["score"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="textos de AuTexTification para calibrar")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = score_missing(load_data(args.n, args.seed))
    cal = df[df.conjunto == "calibracion"]
    y, s = cal.label.to_numpy(), cal.score.to_numpy()
    tpr, thr = tpr_at_fpr(y, -s)
    lr = LogisticRegression().fit(s.reshape(-1, 1), y)
    out = {
        "model_id": MODEL_ID,
        "slope": float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
        "threshold_low_fpr": round(float(-thr), 4),
        "metrics": {"auroc": round(roc_auc_score(y, -s), 4), "tpr_at_1pct_fpr": round(tpr, 4), "n": int(len(cal))},
        "data": "autextification_es_train.parquet",
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[calibración] n={len(cal)}  AUROC={out['metrics']['auroc']:.3f}  TPR@1%FPR={tpr:.3f}")
    print(f"Guardado en {OUT}")

    ev = df[df.conjunto == "evaluacion"]
    server = pd.read_csv(ROOT / "data" / "corpus" / "all_scores.csv", usecols=["id", "score"])
    ev = ev.merge(server.rename(columns={"score": "servidor"}), on="id", how="left")
    print(f"\nCorrelación navegador/servidor: {ev[['score', 'servidor']].corr().iloc[0, 1]:.3f}")
    print(f"{'subconjunto':<14}{'n':>5}{'AUROC nav.':>12}{'AUROC serv.':>13}{'TPR@1% nav.':>13}{'TPR@1% serv.':>14}")
    humans = ev[ev.label == 0]
    subsets = {"todo": ev, **{t: pd.concat([humans, g]) for t, g in ev[ev.label == 1].groupby("task")}}
    for name, g in subsets.items():
        g = g.dropna(subset=["servidor"])
        yy = g.label.to_numpy()
        print(f"{name:<14}{len(g):>5}{roc_auc_score(yy, -g.score):>12.3f}{roc_auc_score(yy, -g.servidor):>13.3f}"
              f"{tpr_at_fpr(yy, -g.score.to_numpy())[0]:>13.3f}{tpr_at_fpr(yy, -g.servidor.to_numpy())[0]:>14.3f}")


if __name__ == "__main__":
    main()
