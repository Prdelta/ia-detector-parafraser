"""Genera textos de IA emparejados con los textos humanos del corpus.

Para cada texto humano elegido se pide al modelo una de estas tareas:
  redactar     escribir sobre el mismo tema y con extensión parecida (caso típico)
  parafrasear  reescribir el texto humano "con otras palabras" (contenido humano, estilo IA)
  humanizar    redactar pidiendo explícitamente que no suene a IA (intento de evasión)

Proveedores:
  local      modelo abierto de Hugging Face en tu GPU/CPU (gratis)
  anthropic  API de Claude (requiere ANTHROPIC_API_KEY y `pip install anthropic`)
  openai     cualquier API compatible con OpenAI: OpenAI, Gemini, Groq, OpenRouter, Ollama
             (requiere --base-url y la clave en OPENAI_API_KEY)

Ejemplos:
  python research/corpus/generate_ai.py --provider local --model BSC-LT/salamandra-2b-instruct --domain academico --n 300
  python research/corpus/generate_ai.py --provider anthropic --model claude-opus-5-5 --domain academico --n 300
  python research/corpus/generate_ai.py --provider openai --model gpt-5 --base-url https://api.openai.com/v1 --domain academico --n 300

Es reanudable: si se interrumpe, al relanzarlo continúa donde se quedó.
"""

import argparse
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_human import is_spanish  # noqa: E402

CORPUS = Path(__file__).resolve().parents[2] / "data" / "corpus"
MIN_WORDS = 100
TASK_WEIGHTS = {"redactar": 0.6, "parafrasear": 0.2, "humanizar": 0.2}

# Variantes de la petición de paráfrasis: así se parecen más a cómo la usa un estudiante.
PARAPHRASE_PROMPTS = [
    "Reescribe en castellano el siguiente texto con tus propias palabras, manteniendo todas las ideas "
    "y una extensión similar (unas {words} palabras). Responde solo con el texto reescrito.",
    "Parafrasea el siguiente texto para que no se note que está copiado: cambia el vocabulario y la "
    "estructura de las oraciones pero conserva el contenido (unas {words} palabras). Responde solo con el texto.",
    "Mejora la redacción del siguiente texto para entregarlo como trabajo universitario. Puedes reformular "
    "libremente, pero mantén las ideas y una extensión parecida (unas {words} palabras). Responde solo con el texto.",
    "Reescribe el siguiente texto con otras palabras de forma que suene natural y escrito por una persona, "
    "no por una IA: varía la longitud de las oraciones y evita frases hechas (unas {words} palabras). "
    "Responde solo con el texto.",
]

GENRES_ACADEMIC = [
    "el resumen (abstract) de un artículo científico",
    "la introducción de un trabajo de investigación universitario",
    "un apartado del marco teórico de una tesis",
    "un ensayo académico de un estudiante universitario",
    "la sección de discusión de un artículo de investigación",
]


def build_prompt(item: dict, task: str, rng: random.Random) -> str:
    words = max(150, min(item["words"], 600))
    if task == "parafrasear":
        return (
            "Reescribe en castellano el siguiente texto con tus propias palabras, manteniendo todas las ideas "
            f"y una extensión similar (unas {words} palabras). Responde solo con el texto reescrito.\n\n"
            f"{item['text']}"
        )
    if item["domain"] == "academico":
        genre = rng.choice(GENRES_ACADEMIC)
        base = (f"Escribe en castellano {genre} titulado «{item['title']}» (área: {item['field']}). "
                f"Extensión aproximada: {words} palabras. Redacta en prosa continua, sin títulos ni viñetas.")
    else:
        base = (f"Escribe en castellano un texto expositivo de tipo enciclopédico sobre «{item['title']}». "
                f"Extensión aproximada: {words} palabras, en párrafos, sin títulos ni viñetas.")
    if task == "humanizar":
        base += (" Es muy importante que suene escrito por una persona y no por una IA: varía la longitud "
                 "de las oraciones, evita frases hechas y conectores repetitivos, y usa un estilo natural.")
    return base + " Responde solo con el texto."


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = [l for l in text.strip().splitlines()]
    # Quitar preámbulos tipo "Aquí tienes..." y títulos sueltos al inicio.
    while lines and (re.match(r"^\s*(aquí tienes|claro|por supuesto|a continuación|título\s*:|#)", lines[0], re.I)
                     or (len(lines[0].split()) <= 12 and not lines[0].rstrip().endswith((".", ":")))):
        lines.pop(0)
    text = "\n".join(lines)
    text = re.sub(r"\*\*|__|^#+\s*", "", text, flags=re.M)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.M)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- Proveedores ----------

class LocalHF:
    def __init__(self, model: str, batch_size: int):
        sys.modules.setdefault("torchaudio", None)  # torchaudio incompatible en algunos entornos
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(
            "cuda" if torch.cuda.is_available() else "cpu").eval()

    def generate(self, prompts: list[str], max_words: list[int]) -> list[str]:
        out = []
        for i in range(0, len(prompts), self.batch_size):
            chunk = prompts[i:i + self.batch_size]
            texts = [self.tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                                  add_generation_prompt=True) for p in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            max_new = min(1100, int(max(max_words[i:i + self.batch_size]) * 2.0) + 64)
            with self.torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.8,
                                          top_p=0.95, repetition_penalty=1.05, pad_token_id=self.tok.pad_token_id)
            out += self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


class AnthropicAPI:
    def __init__(self, model: str, workers: int):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model, self.workers = model, workers

    def _one(self, prompt: str) -> str:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            output_config={"effort": "low"},
            # Si el modelo rechaza una petición, la API reintenta con un modelo alternativo.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text")

    def generate(self, prompts, max_words):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._one, prompts))


class OpenAICompatible:
    def __init__(self, model: str, base_url: str, workers: int):
        import httpx

        key = os.environ.get("OPENAI_API_KEY", "")
        self.client = httpx.Client(base_url=base_url.rstrip("/"), timeout=180,
                                   headers={"Authorization": f"Bearer {key}"} if key else {})
        self.model, self.workers = model, workers

    def _one(self, prompt: str) -> str:
        r = self.client.post("/chat/completions", json={
            "model": self.model, "messages": [{"role": "user", "content": prompt}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""

    def generate(self, prompts, max_words):
        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._one, prompts))


# ---------- Programa ----------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["local", "anthropic", "openai"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=300, help="textos a generar con este modelo")
    parser.add_argument("--domain", choices=["academico", "enciclopedico"], required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--tareas", nargs="+", choices=list(TASK_WEIGHTS),
                        help="generar solo estas tareas (se guardan en un archivo aparte)")
    args = parser.parse_args()

    human = [json.loads(l) for l in (CORPUS / "human.jsonl").open(encoding="utf-8")]
    human = [h for h in human if h["domain"] == args.domain and is_spanish(h["text"])]
    slug = re.sub(r"[^a-zA-Z0-9.-]+", "_", args.model.split("/")[-1])
    weights = {t: w for t, w in TASK_WEIGHTS.items() if not args.tareas or t in args.tareas}
    if args.tareas:
        slug += "__" + "-".join(sorted(weights))
    out_path = CORPUS / "ai" / f"{slug}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path = CORPUS / "rechazados.jsonl"
    done = {json.loads(l)["source_id"] for l in out_path.open(encoding="utf-8")} if out_path.exists() else set()

    # Muestra reproducible por modelo; cada texto humano recibe una tarea.
    rng = random.Random(f"{slug}-{args.domain}")
    chosen = rng.sample(human, min(args.n, len(human)))
    jobs = []
    for item in chosen:
        task = rng.choices(list(weights), weights=list(weights.values()))[0]
        prompt = build_prompt(item, task, rng)
        if item["id"] not in done:
            jobs.append((item, task, prompt))
    # Agrupar textos de extensión parecida: cada lote espera a su generación más larga.
    jobs.sort(key=lambda job: job[0]["words"])
    print(f"{args.model}: {len(done)} hechos, {len(jobs)} pendientes")
    if not jobs:
        return

    if args.provider == "local":
        gen = LocalHF(args.model, args.batch_size)
    elif args.provider == "anthropic":
        gen = AnthropicAPI(args.model, args.workers)
    else:
        gen = OpenAICompatible(args.model, args.base_url, args.workers)

    kept = 0
    for start in range(0, len(jobs), args.save_every):
        block = jobs[start:start + args.save_every]
        outputs = gen.generate([p for _, _, p in block], [it["words"] for it, _, _ in block])
        with out_path.open("a", encoding="utf-8") as f:
            for (item, task, prompt), raw in zip(block, outputs):
                text = clean(raw)
                reason = ("corto" if len(text.split()) < MIN_WORDS
                          else "idioma" if not is_spanish(text) else None)
                if reason:  # salida truncada, vacía, rechazada o en otro idioma
                    with rejected_path.open("a", encoding="utf-8") as rf:
                        rf.write(json.dumps({"model": args.model, "task": task, "reason": reason,
                                             "raw": raw}, ensure_ascii=False) + "\n")
                    continue
                kept += 1
                f.write(json.dumps({
                    "id": f"{item['id']}__{slug}", "source_id": item["id"], "text": text,
                    "model": args.model, "task": task, "domain": item["domain"],
                    "words": len(text.split()), "prompt": prompt,
                }, ensure_ascii=False) + "\n")
        print(f"  {min(start + args.save_every, len(jobs))}/{len(jobs)} procesados, {kept} válidos", flush=True)


if __name__ == "__main__":
    main()
