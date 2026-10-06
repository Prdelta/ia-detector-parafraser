"""Clasificador supervisado humano/IA (XLM-RoBERTa ajustado con ``research/train_classifier.py``).

Complementa a Binoculars: aprende rasgos de la paráfrasis con IA, que Binoculars apenas detecta.
Los textos largos se dividen en fragmentos y se promedia la probabilidad ponderando por tokens.
"""

import logging
import sys
import threading

import numpy as np

from .. import config

log = logging.getLogger(__name__)


class Classifier:
    def __init__(self, path=config.CLASSIFIER_DIR, max_len: int = config.CLASSIFIER_MAX_TOKENS):
        self.path = path
        self.max_len = max_len
        self._lock = threading.Lock()
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        sys.modules.setdefault("torchaudio", None)  # ver detectors/binoculars.py
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Cargando clasificador %s en %s", self.path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path, dtype=dtype).to(self.device).eval()
        self._loaded = True

    @property
    def model_id(self) -> str:
        return self.path.name

    def probability(self, text: str) -> float:
        self.load()
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        body = self.max_len - 2
        chunks = [ids[i:i + body] for i in range(0, max(len(ids), 1), body)]
        # Un último fragmento muy corto es ruidoso: se une al anterior recortando por delante.
        if len(chunks) > 1 and len(chunks[-1]) < body // 4:
            chunks[-1] = ids[-body:]
        batch = self.tokenizer.pad(
            {"input_ids": [[self.tokenizer.cls_token_id, *c, self.tokenizer.sep_token_id] for c in chunks]},
            return_tensors="pt").to(self.device)
        with self._lock, self.torch.inference_mode():
            logits = self.model(**batch).logits.float()
        probs = self.torch.softmax(logits, -1)[:, 1].cpu().numpy()
        return float(np.average(probs, weights=[len(c) for c in chunks]))


_instance: Classifier | None = None
_failed = False


def get_classifier() -> Classifier | None:
    """Instancia compartida; ``None`` si está desactivado, no entrenado o no pudo cargarse."""
    global _instance, _failed
    if not config.ENABLE_CLASSIFIER or _failed or not (config.CLASSIFIER_DIR / "config.json").exists():
        return None
    if _instance is None:
        try:
            clf = Classifier()
            clf.load()
            _instance = clf
        except Exception:
            log.exception("No se pudo cargar el clasificador; se usarán las demás señales")
            _failed = True
            return None
    return _instance
