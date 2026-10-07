"""Ayudas para reescribir los párrafos marcados como IA.

- ``guidance``: explica por qué un párrafo suena a IA y sugiere cómo reescribirlo (por defecto).
- ``Paraphraser``: paráfrasis automática con un modelo local (opcional, con aviso de uso académico).
"""

import logging
import re
import statistics
import sys
import threading

from . import config
from .detectors.stylometry import LEXICON
from .segment import Sentence, words

log = logging.getLogger(__name__)

PARAPHRASE_NOTICE = (
    "El texto parafraseado sigue siendo generado por IA aunque el detector ya no lo marque. "
    "Revisa las normas de tu institución: presentarlo como propio puede considerarse falta académica."
)


def guidance(sentences: list[Sentence], phrase_hits: list[list[str]], probability: float,
             lang: str = "es") -> dict:
    """Motivos y sugerencias para un párrafo, a partir de sus oraciones."""
    reasons, suggestions = [], []
    phrases = sorted({h.lower() for hits in phrase_hits for h in hits})
    if phrases:
        reasons.append("Frases hechas típicas de IA: " + ", ".join(f"«{p}»" for p in phrases[:6]) + ".")
        suggestions.append("Sustituye las frases hechas por afirmaciones concretas: ¿qué dato, ejemplo o autor respalda la idea?")

    connector_start = LEXICON[lang][1]
    connectors = [m.group(0) for s in sentences if (m := connector_start.match(s.text))]
    if len(connectors) >= 2:
        reasons.append(f"{len(connectors)} oraciones empiezan con conectores formulaicos ({', '.join(connectors[:4])}).")
        suggestions.append("Quita conectores de relleno o reordena las ideas para que se enlacen solas.")

    lengths = [len(words(s.text)) for s in sentences]
    if len(lengths) >= 3 and statistics.pstdev(lengths) / max(statistics.fmean(lengths), 1) < 0.3:
        reasons.append("Las oraciones tienen casi la misma longitud.")
        suggestions.append("Varía el ritmo: combina oraciones cortas con otras más largas.")

    if probability >= 0.65 and not reasons:
        reasons.append("La redacción es muy predecible: vocabulario y estructuras genéricas.")
    if probability >= 0.35:
        suggestions.append("Añade tu propia voz: un ejemplo, tu postura o una cita concreta de tus fuentes.")
    return {"motivos": reasons, "sugerencias": suggestions}


class Paraphraser:
    def __init__(self, model: str = config.PARAPHRASE_MODEL):
        self.model_name = model
        self._lock = threading.Lock()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        sys.modules.setdefault("torchaudio", None)  # ver detectors/binoculars.py
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        log.info("Cargando parafraseador %s en %s", self.model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype).to(device).eval()
        self._loaded = True

    def paraphrase(self, text: str) -> str:
        n_words = len(words(text))
        prompt = (
            "Parafrasea en castellano el siguiente párrafo de un trabajo académico. Conserva todas las ideas, "
            "datos, nombres, cifras y citas, y una extensión similar. Responde solo con el párrafo reescrito.\n\n"
            + text
        )
        with self._lock:
            self._load()
            chat = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
            enc = self.tokenizer(chat, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            with self.torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=int(n_words * 2.2) + 64, do_sample=True,
                                          temperature=0.7, top_p=0.9, pad_token_id=self.tokenizer.eos_token_id)
        result = self.tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        result = re.sub(r"^\s*(aquí tienes[^:\n]*:|párrafo reescrito\s*:)\s*", "", result.strip(), flags=re.I)
        return re.sub(r"\*\*", "", result).strip()


_paraphraser: Paraphraser | None = None


def get_paraphraser() -> Paraphraser | None:
    global _paraphraser
    if not config.PARAPHRASE_MODEL:
        return None
    if _paraphraser is None:
        _paraphraser = Paraphraser()
    return _paraphraser
