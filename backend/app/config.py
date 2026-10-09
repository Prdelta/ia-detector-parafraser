"""Configuración central, sobrescribible con variables de entorno."""

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT_DIR / "models"
FRONTEND_DIR = ROOT_DIR / "frontend"

# Par de modelos para Binoculars: deben compartir tokenizador.
# "observer" = modelo base, "performer" = versión instruct del mismo modelo.
# Por defecto 0.5B para que ambos quepan en una GPU de 6 GB (o en CPU).
OBSERVER_MODEL = os.getenv("IADECCION_OBSERVER", "Qwen/Qwen2.5-0.5B")
PERFORMER_MODEL = os.getenv("IADECCION_PERFORMER", "Qwen/Qwen2.5-0.5B-Instruct")

# "0" desactiva Binoculars (útil para pruebas o máquinas sin recursos).
ENABLE_BINOCULARS = os.getenv("IADECCION_BINOCULARS", "1") == "1"

# Clasificador supervisado (research/train_classifier.py); "0" lo desactiva.
ENABLE_CLASSIFIER = os.getenv("IADECCION_CLASIFICADOR", "1") == "1"
CLASSIFIER_DIR = Path(os.getenv("IADECCION_CLASIFICADOR_DIR", MODELS_DIR / "clasificador"))
CLASSIFIER_MAX_TOKENS = 384  # igual que en el entrenamiento

# Pesos aprendidos para combinar las señales (research/train_meta.py).
META_FILE = Path(os.getenv("IADECCION_META", MODELS_DIR / "meta.json"))

# Modelo para la paráfrasis automática opcional; vacío o "0" la desactiva.
PARAPHRASE_MODEL = os.getenv("IADECCION_PARAFRASEADOR", "Qwen/Qwen2.5-1.5B-Instruct")
if PARAPHRASE_MODEL == "0":
    PARAPHRASE_MODEL = ""
MAX_PARAPHRASE_WORDS = 400

# Búsqueda opcional de coincidencias en OpenAlex y Wikipedia (envía palabras clave); "0" la desactiva.
ENABLE_SIMILARITY = os.getenv("IADECCION_SIMILITUD", "1") == "1"

# Tokens por fragmento al evaluar textos largos.
MAX_TOKENS_PER_CHUNK = int(os.getenv("IADECCION_MAX_TOKENS", "512"))

# Límites de entrada.
MIN_WORDS = 80
RECOMMENDED_WORDS = 250
MAX_CHARS = 60_000
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

CALIBRATION_FILE = MODELS_DIR / "calibration.json"


def calibration_file(lang: str = "es") -> Path:
    """Calibración de Binoculars por idioma: calibration.json (es), calibration_en.json (en)."""
    return CALIBRATION_FILE if lang == "es" else MODELS_DIR / f"calibration_{lang}.json"
