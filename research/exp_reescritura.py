"""Experimento: ¿cuánto cambia un texto cuando un modelo lo reescribe? (idea de Raidar, ICLR 2024)

Un modelo apenas modifica un texto que ya está escrito "a su manera" (redactado o parafraseado con
IA) y cambia más un texto humano. Se mide con la distancia de edición entre palabras del texto y su
reescritura, y se comprueba si esa señal añade algo al meta-clasificador actual.

Uso:  python research/exp_reescritura.py [--modelo Qwen/Qwen2.5-0.5B-Instruct] [--n-humanos 200]
Salida: tabla en pantalla y data/exp_reescritura_<modelo>.parquet (caché por texto)
"""

import argparse
import difflib
import json
import re
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

from backend.app.detectors import meta  # noqa: E402

PROMPT = ("Reescribe el siguiente texto mejorando su redacción. Mantén el significado y la extensión. "
          "Responde solo con el texto reescrito.\n\n{texto}")
MAX_WORDS = 200


def recortar(texto: str, n: int = MAX_WORDS) -> str:
    """Primeras ~n palabras, cortando al final de una oración si es posible."""
    palabras = texto.split()
    if len(palabras) <= n:
        return texto
    corto = " ".join(palabras[:n])
    fin = max(corto.rfind(". "), corto.rfind("? "), corto.rfind("! "))
    return corto[: fin + 1] if fin > len(corto) // 2 else corto


def palabras(texto: str) -> list[str]:
    return re.findall(r"\w+", texto.lower())


def cambio(original: str, reescrito: str) -> dict:
    a, b = palabras(original), palabras(reescrito)
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return {
        "cambio": 1 - sm.ratio(),                                  # 0 = idéntico, 1 = todo distinto
        "conservadas": sum(m.size for m in sm.get_matching_blocks()) / max(len(a), 1),
        "ratio_longitud": len(b) / max(len(a), 1),
    }


def muestra(n_humanos: int, n_parafrasis: int, n_redactar: int, seed: int) -> pd.DataFrame:
    test = pd.read_parquet(ROOT / "data" / "corpus" / "test.parquet")
    test = test[test.domain == "academico"]
    hum = test[test.label == 0]
    partes = [hum.sample(min(len(hum), n_humanos), random_state=seed)]
    par = test[test.task == "parafrasear"]
    por_modelo = n_parafrasis // par.model.nunique()
    partes += [g.sample(min(len(g), por_modelo), random_state=seed) for _, g in par.groupby("model")]
    red = test[test.task == "redactar"]
    partes.append(red.sample(min(len(red), n_redactar), random_state=seed))
    return pd.concat(partes, ignore_index=True)


def reescribir(textos: list[str], modelo: str, lote: int) -> list[str]:
    sys.modules.setdefault("torchaudio", None)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(modelo, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(modelo, dtype=torch.bfloat16).to("cuda").eval()
    salida, t0 = [], time.time()
    # Agrupar por longitud para no esperar a la generación más larga de cada lote.
    orden = sorted(range(len(textos)), key=lambda i: len(textos[i]))
    resultado = [None] * len(textos)
    for k in range(0, len(orden), lote):
        idx = orden[k:k + lote]
        prompts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(texto=textos[i])}],
                                           tokenize=False, add_generation_prompt=True) for i in idx]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=int(MAX_WORDS * 2.2), do_sample=False,
                                 repetition_penalty=1.05, pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for i, g in zip(idx, tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            resultado[i] = g.strip()
        hechos = k + len(idx)
        if hechos % (lote * 10) == 0 or hechos == len(orden):
            print(f"  {hechos}/{len(orden)}  ({(time.time() - t0) / hechos:.2f} s/texto)", flush=True)
    return resultado


def tpr_a_fpr(y, p, fpr_obj=0.01):
    fpr, tpr, _ = roc_curve(y, p)
    return float(tpr[np.searchsorted(fpr, fpr_obj, side="right") - 1])


def oof_meta(X, y, grupos):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, grupos):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        lr = LogisticRegression(class_weight="balanced").fit((X[tr] - mu) / sd, y[tr])
        p[te] = lr.predict_proba((X[te] - mu) / sd)[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n-humanos", type=int, default=200)
    ap.add_argument("--n-parafrasis", type=int, default=240)
    ap.add_argument("--n-redactar", type=int, default=100)
    ap.add_argument("--lote", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cache = ROOT / "data" / f"exp_reescritura_{args.modelo.split('/')[-1]}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
        print(f"Reescrituras leídas de {cache}")
    else:
        df = muestra(args.n_humanos, args.n_parafrasis, args.n_redactar, args.seed)
        df["recorte"] = df.text.map(recortar)
        print(f"Reescribiendo {len(df)} textos con {args.modelo}")
        df["reescrito"] = reescribir(df.recorte.tolist(), args.modelo, args.lote)
        df.drop(columns=["text"]).to_parquet(cache)
    df = pd.concat([df.reset_index(drop=True),
                    pd.DataFrame([cambio(a, b) for a, b in zip(df.recorte, df.reescrito)])], axis=1)

    # Menos cambio = más probable IA, así que la puntuación es -cambio.
    humanos = df[df.label == 0]
    print(f"\nCambio medio al reescribir:  humanos {humanos.cambio.mean():.3f}", end="")
    for tarea, g in df[df.label == 1].groupby("task"):
        print(f"  |  {tarea} {g.cambio.mean():.3f}", end="")
    print("\n\nSolo la señal de reescritura (AUROC, humanos frente a cada grupo):")
    grupos = {"parafrasear (todas)": df.task == "parafrasear", "redactar": df.task == "redactar"}
    for modelo, g in df[df.task == "parafrasear"].groupby("model"):
        grupos[f"  parafrasear / {modelo.split('/')[-1]}"] = (df.task == "parafrasear") & (df.model == modelo)
    for nombre, mask in grupos.items():
        sub = df[(df.label == 0) | mask]
        print(f"  {nombre:<45} n={len(sub):>4}  AUROC={roc_auc_score(sub.label, -sub.cambio):.3f}")

    # ¿Aporta algo al meta actual? Mismas señales del caché del meta, con y sin la reescritura.
    feats = pd.read_parquet(ROOT / "data" / "meta_features_clasificador.parquet")
    df = df.merge(feats[["id", "binoculars", "estilo", "clasificador"]], on="id", how="inner")
    base = np.array([meta.features(b, s, c) for b, s, c in zip(df.binoculars, df.estilo, df.clasificador)])
    con = np.column_stack([base, df.cambio, df.ratio_longitud])
    y, g = df.label.to_numpy(), df.source_id.to_numpy()
    print(f"\nMeta con y sin la señal (validación cruzada agrupada por tema, n={len(df)}):")
    for nombre, X in (("meta actual (3 señales)", base), ("meta + reescritura", con)):
        p = oof_meta(X, y, g)
        par = (df.label == 0) | (df.task == "parafrasear")
        print(f"  {nombre:<26} AUROC={roc_auc_score(y, p):.3f}  TPR@1%={tpr_a_fpr(y, p):.3f}  "
              f"| paráfrasis: AUROC={roc_auc_score(y[par], p[par]):.3f}  TPR@1%={tpr_a_fpr(y[par], p[par]):.3f}")

    print("\nEjemplos (humano / paráfrasis):")
    for _, r in pd.concat([humanos.head(1), df[df.task == "parafrasear"].head(1)]).iterrows():
        print(json.dumps({"tarea": r.task, "cambio": round(r.cambio, 3), "original": r.recorte[:160],
                          "reescrito": r.reescrito[:160]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
