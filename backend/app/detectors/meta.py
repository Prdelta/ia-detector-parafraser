"""Combina las señales con una regresión logística entrenada (``research/train_meta.py``).

Rasgos: puntuación Binoculars (cruda), probabilidad de estilometría y logit del clasificador.
Si falta el archivo, o fue entrenado con otros modelos, ``load`` devuelve ``None`` y el
analizador usa pesos fijos.
"""

import json
import math
from dataclasses import dataclass

from .. import config

FEATURES = ("binoculars", "estilo", "clasificador")


def logit(p: float, eps: float = 1e-4) -> float:
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1 - p))


def features(bino_score: float, style_prob: float, clf_prob: float) -> list[float]:
    return [bino_score, style_prob, logit(clf_prob)]


@dataclass
class Meta:
    coef: list[float]
    intercept: float
    mean: list[float]
    std: list[float]

    def probability(self, x: list[float]) -> float:
        z = self.intercept + sum(c * (v - m) / s for c, v, m, s in zip(self.coef, x, self.mean, self.std))
        z = max(min(z, 50), -50)
        return 1.0 / (1.0 + math.exp(-z))


def load(binoculars_id: str, classifier_id: str) -> Meta | None:
    path = config.META_FILE
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if (data.get("features") != list(FEATURES) or data.get("binoculars") != binoculars_id
            or data.get("clasificador") != classifier_id):
        return None
    return Meta(data["coef"], data["intercept"], data["mean"], data["std"])
