"""Entrena un clasificador supervisado humano/IA (XLM-RoBERTa) para español.

Datos de entrenamiento:
  - AuTexTification 2023 (es, train): textos cortos, modelos de 2022
  - Corpus propio (train, solo dominio académico): textos largos, modelos abiertos actuales,
    incluidas paráfrasis de textos humanos (el punto débil de Binoculars)

Uso:  python research/train_classifier.py [--epochs 2] [--base FacebookAI/xlm-roberta-base]
Salida: models/clasificador/
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


def load_data(seed: int, autext_n: int, ai_oversample: int):
    autext = pd.read_parquet(ROOT / "data" / "autextification_es_train.parquet")
    autext = autext[(autext.domain != "tweets") & (autext.text.str.split().str.len() >= 30)]
    autext = autext.groupby("label", group_keys=False).apply(
        lambda g: g.sample(min(len(g), autext_n // 2), random_state=seed))
    corpus = pd.read_parquet(ROOT / "data" / "corpus" / "train.parquet")
    corpus = corpus[corpus.domain == "academico"]
    corpus = pd.concat([corpus[corpus.label == 0]] + [corpus[corpus.label == 1]] * ai_oversample)
    train = pd.concat([autext[["text", "label"]], corpus[["text", "label"]]]).sample(frac=1, random_state=seed)

    test_corpus = pd.read_parquet(ROOT / "data" / "corpus" / "test.parquet")
    test_corpus = test_corpus[test_corpus.domain == "academico"]
    test_autext = pd.read_parquet(ROOT / "data" / "autextification_es_test.parquet")
    test_autext = test_autext[test_autext.text.str.split().str.len() >= 30].sample(2000, random_state=seed)
    return train, {"corpus_test": test_corpus, "autext_test": test_autext}


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
    parser.add_argument("--ai-oversample", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    train, tests = load_data(args.seed, args.autext_n, args.ai_oversample)
    print(f"Entrenamiento: {len(train)} textos (IA={train.label.mean():.0%})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * (len(train) // args.batch_size)
    sched = get_linear_schedule_with_warmup(optim, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    texts, labels = train.text.tolist(), torch.tensor(train.label.to_numpy())
    step, t0 = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        order = np.random.default_rng(args.seed + epoch).permutation(len(texts))
        for i in range(0, len(order) - args.batch_size + 1, args.batch_size):
            idx = order[i:i + args.batch_size]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=args.max_len,
                      padding=True, return_tensors="pt").to(device)
            with torch.autocast(device, enabled=device == "cuda"):
                loss = model(**enc, labels=labels[idx].to(device)).loss
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

    for name, df in tests.items():
        p = predict(model, tok, df.text.tolist(), args.max_len)
        acc = ((p >= 0.5) == df.label.to_numpy()).mean()
        print(f"[{name}] n={len(df)}  AUROC={roc_auc_score(df.label, p):.3f}  exactitud={acc:.3f}")
        if "task" in df:
            for task, g in df[df.label == 1].groupby("task"):
                print(f"   IA detectada (p≥0.5) en {task}: {(p[g.index.map(df.index.get_loc)] >= 0.5).mean():.0%}")

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"Modelo guardado en {OUT}")


if __name__ == "__main__":
    main()
