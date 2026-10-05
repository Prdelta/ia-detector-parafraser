"""Detector zero-shot Binoculars (Hans et al., 2024, arXiv:2401.12070).

score = perplejidad(texto | performer) / perplejidad-cruzada(observer, performer)

Puntuaciones bajas indican texto generado por IA. Se calculan valores por token
para poder agregarlos por oración sin pasadas adicionales del modelo.
"""

import logging
import threading
from dataclasses import dataclass

import numpy as np

from .. import config

log = logging.getLogger(__name__)

CONTEXT_OVERLAP = 64


@dataclass
class TokenStats:
    offsets: np.ndarray  # (n, 2) posiciones de carácter de cada token evaluado
    ppl: np.ndarray      # (n,) pérdida log del performer sobre el token real
    xppl: np.ndarray     # (n,) entropía cruzada observer→performer

    def score(self, mask: np.ndarray | None = None) -> float | None:
        ppl, xppl = (self.ppl, self.xppl) if mask is None else (self.ppl[mask], self.xppl[mask])
        if len(ppl) == 0:
            return None
        return float(ppl.mean() / xppl.mean())


class Binoculars:
    def __init__(self, observer: str = config.OBSERVER_MODEL, performer: str = config.PERFORMER_MODEL):
        self.observer_name = observer
        self.performer_name = performer
        self._lock = threading.Lock()
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        import sys

        # No se necesita audio; evita fallos si hay un torchaudio incompatible instalado.
        sys.modules.setdefault("torchaudio", None)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        log.info("Cargando %s y %s en %s", self.observer_name, self.performer_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.observer_name)
        self.observer = AutoModelForCausalLM.from_pretrained(self.observer_name, dtype=dtype).to(self.device).eval()
        self.performer = AutoModelForCausalLM.from_pretrained(self.performer_name, dtype=dtype).to(self.device).eval()
        self._loaded = True

    @property
    def model_id(self) -> str:
        return f"{self.observer_name}|{self.performer_name}"

    def token_stats(self, text: str) -> TokenStats:
        import torch
        import torch.nn.functional as F

        self.load()
        enc = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        chunk = config.MAX_TOKENS_PER_CHUNK

        all_ppl, all_xppl, all_off = [], [], []
        with self._lock, torch.inference_mode():
            # Ventanas de `chunk` tokens; cada una arrastra contexto previo que no se puntúa.
            for start in range(1, len(ids), chunk):
                ctx_start = max(0, start - CONTEXT_OVERLAP)
                end = min(len(ids), start + chunk)
                input_ids = torch.tensor([ids[ctx_start:end]], device=self.device)
                obs_logits = self.observer(input_ids).logits[0, :-1].float()
                perf_logits = self.performer(input_ids).logits[0, :-1].float()
                targets = input_ids[0, 1:]

                ppl = F.cross_entropy(perf_logits, targets, reduction="none")
                xppl = -(F.softmax(obs_logits, -1) * F.log_softmax(perf_logits, -1)).sum(-1)

                keep = slice(start - ctx_start - 1, None)  # solo tokens nuevos de esta ventana
                all_ppl.append(ppl[keep].cpu().numpy())
                all_xppl.append(xppl[keep].cpu().numpy())
                all_off.extend(offsets[start:end])
                del obs_logits, perf_logits

        if not all_ppl:
            return TokenStats(np.zeros((0, 2), int), np.zeros(0), np.zeros(0))
        return TokenStats(np.array(all_off, dtype=int), np.concatenate(all_ppl), np.concatenate(all_xppl))

    def score(self, text: str) -> float | None:
        return self.token_stats(text).score()


_instance: Binoculars | None = None
_failed = False


def get_detector() -> Binoculars | None:
    """Instancia compartida; ``None`` si está desactivado o no pudo cargarse."""
    global _instance, _failed
    if not config.ENABLE_BINOCULARS or _failed:
        return None
    if _instance is None:
        try:
            det = Binoculars()
            det.load()
            _instance = det
        except Exception:
            log.exception("No se pudo cargar Binoculars; se usará solo estilometría")
            _failed = True
            return None
    return _instance
