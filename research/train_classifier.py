"""Entrena un clasificador supervisado humano/IA (XLM-RoBERTa) para español.

Datos de entrenamiento:
  - AuTexTification 2023 (es, train): textos cortos, modelos de 2022
  - Corpus propio (train, solo dominio académico): textos largos, modelos abiertos actuales,
    incluidas paráfrasis de textos humanos (el punto débil de Binoculars)

Opcional (experimento "v3", octubre 2026): suavizado de etiquetas y elegir el mejor punto según un
15 % de los temas del corpus propio (``--label-smoothing 0.1 --val-frac 0.15``). Quitó la saturación
extrema de las probabilidades, pero no mejoró al meta-clasificador y generalizó peor a AuTexTification
(AUROC 0.765 frente a 0.898), así que por defecto se entrena como la versión en uso.

Uso:  python research/train_classifier.py [--epochs 2] [--out models/clasificador]
Salida: models/clasificador/ (o --out)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.modules.setdefault("torchaudio", None)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "models" / "clasificador"


def load_data(seed: int, autext_n: int, ai_oversample: int, val_frac: float):
    autext = pd.read_parquet(ROOT / "data" / "autextification_es_train.parquet")
    autext = autext[(autext.domain != "tweets") & (autext.text.str.split().str.len() >= 30)]
    autext = autext.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), autext_n // 2), random_state=seed))
    corpus = pd.read_parquet(ROOT / "data" / "corpus" / "train.parquet")
    corpus = corpus[corpus.domain == "academico"]
    sources = corpus.source_id.drop_duplicates().sample(frac=1, random_state=seed)
    val_sources = set(sources[: int(len(sources) * val_frac)])
    val = corpus[corpus.source_id.isin(val_sources)].reset_index(drop=True)
    corpus = corpus[~corpus.source_id.isin(val_sources)]
    corpus = pd.concat([corpus[corpus.label == 0]] + [corpus[corpus.label == 1]] * ai_oversample)
    train = pd.concat([autext[["text", "label"]], corpus[["text", "label"]]]).sample(frac=1, random_state=seed)

    test_corpus = pd.read_parquet(ROOT / "data" / "corpus" / "test.parquet")
    test_corpus = test_corpus[test_corpus.domain == "academico"]
    test_autext = pd.read_parquet(ROOT / "data" / "autextification_es_test.parquet")
    test_autext = test_autext[test_autext.text.str.split().str.len() >= 30].sample(2000, random_state=seed)
    return train, val, {"corpus_test": test_corpus, "autext_test": test_autext}


def validate(model, tok, val, max_len) -> dict:
    p = predict(model, tok, val.text.tolist(), max_len)
    y = val.label.to_numpy()
    eps = 1e-6
    para = (val.task == "parafrasear").to_numpy()
    return {
        "auroc": roc_auc_score(y, p),
        "auroc_parafrasis": roc_auc_score(y[(y == 0) | para], p[(y == 0) | para]) if para.any() else float("nan"),
        "logloss": float(-np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))),
        "humanos_p099": float((p[y == 0] >= 0.99).mean()),
    }


@torch.inference_mode()
def predict(model, tok, texts, max_len, batch_size=32):
    model.eval()
    probs = []
    for i in range(0, len(texts), batch_size):
        enc = tok(list(texts[i:i + batch_size]), truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt").to(model.device)
        with torch.autocast("cuda", enabled=model.device.type == "cuda"):
            logits = model(**enc).logits.float()
        probs.append(torch.softmax(logits, -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="FacebookAI/xlm-roberta-base")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-len", type=int, default=384)
    parser.add_argument("--autext-n", type=int, default=12000)
    parser.add_argument("--ai-oversample", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.0, help="0 = sin validación (se guarda el final)")
    parser.add_argument("--eval-every", type=int, default=400, help="pasos entre validaciones")
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train, val, tests = load_data(args.seed, args.autext_n, args.ai_oversample, args.val_frac)
    print(f"Entrenamiento: {len(train)} textos (IA={train.label.mean():.0%}); validación: {len(val)} textos")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.batch_size)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    texts, labels = train.text.tolist(), torch.tensor(train.label.to_numpy())
    step, t0 = 0, time.time()
    best, best_state = None, None

    def checkpoint():
        nonlocal best, best_state
        m = validate(model, tok, val, args.max_len)
        print(f"  [validación paso {step}] AUROC={m['auroc']:.4f}  paráfrasis={m['auroc_parafrasis']:.4f}  "
              f"logloss={m['logloss']:.3f}  humanos p≥0.99={m['humanos_p099']:.1%}", flush=True)
        # Mejor punto: AUROC en paráfrasis (el punto débil) y, a igualdad, menor pérdida.
        key = (round(m["auroc_parafrasis"], 3), -m["logloss"])
        if best is None or key > best:
            best = key
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        model.train()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(texts))
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = loss_fn(model(**enc).logits.float(), labels[idx].to(device))
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()
            sched.step()
            step += 1
            if step % 100 == 0:
                print(f"  época {epoch + 1} paso {step}/{steps}  pérdida {loss.item():.3f}  "
                      f"({(time.time() - t0) / step:.2f} s/paso)", flush=True)
            if len(val) and step % args.eval_every == 0:
                checkpoint()
        if len(val):
            checkpoint()

    if best_state is not None:
        model.load_state_dict(best_state)

    for name, df in tests.items():
        p = predict(model, tok, df.text.tolist(), args.max_len)
        acc = ((p >= 0.5) == df.label.to_numpy()).mean()
        print(f"[{name}] n={len(df)}  AUROC={roc_auc_score(df.label, p):.3f}  exactitud={acc:.3f}")
        if "task" in df:
            for task, g in df[df.label == 1].groupby("task"):
                print(f"   IA detectada (p≥0.5) en {task}: {(p[g.index.map(df.index.get_loc)] >= 0.5).mean():.0%}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"Modelo guardado en {out}")


if __name__ == "__main__":
    main()
