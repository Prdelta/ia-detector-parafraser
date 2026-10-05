"""Convierte la puntuación Binoculars en una probabilidad calibrada.

Los parámetros se generan con ``research/calibrate.py`` para el par de modelos
en uso. Sin archivo se usan valores por defecto conservadores.
"""

import json
import math
from dataclasses import dataclass

from .. import config


@dataclass
class Calibration:
    # P(IA) = sigmoid(slope * score + intercept); slope es negativo.
    slope: float
    intercept: float
    threshold_low_fpr: float  # puntuación por debajo de la cual se marca IA con FPR ≈ 1 %
    model_id: str = ""
    calibrated: bool = False

    def probability(self, score: float) -> float:
        z = self.slope * score + self.intercept
        z = max(min(z, 50), -50)
        return 1.0 / (1.0 + math.exp(-z))


# Valores por defecto: centro aproximado en 0.90, como en el artículo original.
DEFAULT = Calibration(slope=-30.0, intercept=27.0, threshold_low_fpr=0.85)


def load(model_id: str) -> Calibration:
    path = config.CALIBRATION_FILE
    if not path.exists():
        return DEFAULT
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model_id") != model_id:
        return DEFAULT
    return Calibration(
        slope=data["slope"],
        intercept=data["intercept"],
        threshold_low_fpr=data["threshold_low_fpr"],
        model_id=model_id,
        calibrated=True,
    )
