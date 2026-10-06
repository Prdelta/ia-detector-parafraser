"""Combina los detectores y construye el informe que ve el estudiante."""

import numpy as np

from . import config, rewrite
from .detectors import binoculars, calibration, stylometry
from .segment import normalize, split_sentences, words

# Peso de cada señal cuando Binoculars está disponible.
W_BINOCULARS = 0.8
W_STYLE = 0.2

# Una oración con pocos tokens es ruidosa: se amplía con sus vecinas.
MIN_SENTENCE_TOKENS = 30

LEVELS = [(0.35, "bajo"), (0.65, "medio"), (1.01, "alto")]


def _level(p: float) -> str:
    return next(name for limit, name in LEVELS if p < limit)


def _verdict(p: float) -> str:
    if p < 0.35:
        return "Probablemente escrito por una persona"
    if p < 0.65:
        return "Resultado incierto o texto mixto"
    return "Probablemente generado con IA"


def analyze_text(raw_text: str) -> dict:
    text = normalize(raw_text)[: config.MAX_CHARS]
    sentences = split_sentences(text)
    n_words = len(words(text))
    warnings = []

    if n_words < config.MIN_WORDS:
        raise ValueError(f"El texto es demasiado corto ({n_words} palabras). Se necesitan al menos {config.MIN_WORDS}.")
    if n_words < config.RECOMMENDED_WORDS:
        warnings.append(f"Con menos de {config.RECOMMENDED_WORDS} palabras el resultado es poco fiable.")
    if len(raw_text) > config.MAX_CHARS:
        warnings.append(f"Solo se analizaron los primeros {config.MAX_CHARS:,} caracteres.")

    style = stylometry.analyze(text, sentences)
    detector = binoculars.get_detector()

    sentence_probs = np.full(len(sentences), style.score)
    signals = {"estilometria": {"probabilidad": round(style.score, 3), "rasgos": style.features}}
    reasons = list(style.reasons)

    if detector is not None:
        calib = calibration.load(detector.model_id)
        if not calib.calibrated:
            warnings.append("El detector aún no está calibrado para este modelo; los porcentajes son aproximados.")
        stats = detector.token_stats(text)
        doc_score = stats.score()
        bino_prob = calib.probability(doc_score)
        doc_prob = W_BINOCULARS * bino_prob + W_STYLE * style.score
        signals["binoculars"] = {
            "puntuacion": round(doc_score, 4),
            "umbral_bajo_fpr": calib.threshold_low_fpr,
            "probabilidad": round(bino_prob, 3),
            "modelo": detector.model_id,
            "calibrado": calib.calibrated,
        }
        if doc_score < calib.threshold_low_fpr:
            reasons.insert(0, "El texto es muy predecible para los modelos de lenguaje (patrón típico de IA).")

        # Asignar cada token a su oración por la posición de inicio.
        starts = np.array([s.start for s in sentences])
        token_sentence = np.searchsorted(starts, stats.offsets[:, 0], side="right") - 1
        for i in range(len(sentences)):
            lo, hi = i, i
            mask = token_sentence == i
            while mask.sum() < MIN_SENTENCE_TOKENS and (lo > 0 or hi < len(sentences) - 1):
                lo, hi = max(lo - 1, 0), min(hi + 1, len(sentences) - 1)
                mask = (token_sentence >= lo) & (token_sentence <= hi)
            s_score = stats.score(mask)
            if s_score is not None:
                # Solo señales locales: el estilo global del documento contaminaría
                # los párrafos humanos de un texto mixto.
                hit_bonus = 0.05 * min(len(style.sentence_hits[i]), 2)
                sentence_probs[i] = min(1.0, calib.probability(s_score) + hit_bonus)
    else:
        doc_prob = style.score
        warnings.append("Solo se usó el análisis de estilo (el modelo principal no está disponible). Confianza reducida.")

    sentence_words = np.array([len(words(s.text)) for s in sentences])
    flagged = sentence_probs >= 0.65
    ai_fraction = float(sentence_words[flagged].sum() / max(sentence_words.sum(), 1))

    paragraphs = []
    for p_idx in sorted({s.paragraph for s in sentences}):
        idx = [i for i, s in enumerate(sentences) if s.paragraph == p_idx]
        weights = np.maximum(sentence_words[idx], 1)
        p_prob = float(np.average(sentence_probs[idx], weights=weights))
        start, end = sentences[idx[0]].start, sentences[idx[-1]].end
        paragraphs.append({
            "indice": p_idx,
            "texto": text[start:end],
            "probabilidad": round(p_prob, 3),
            "nivel": _level(p_prob),
            **rewrite.guidance([sentences[i] for i in idx], [style.sentence_hits[i] for i in idx], p_prob),
        })

    return {
        "texto": text,
        "parrafos": paragraphs,
        "parafrasis_disponible": bool(config.PARAPHRASE_MODEL),
        "probabilidad_ia": round(float(doc_prob), 3),
        "veredicto": _verdict(doc_prob),
        "confianza": _confidence(n_words, detector is not None, doc_prob, style.score),
        "fraccion_texto_marcado": round(ai_fraction, 3),
        "palabras": n_words,
        "oraciones": [
            {
                "texto": s.text,
                "inicio": s.start,
                "fin": s.end,
                "parrafo": s.paragraph,
                "probabilidad": round(float(p), 3),
                "nivel": _level(p),
                "expresiones": hits,
            }
            for s, p, hits in zip(sentences, sentence_probs, style.sentence_hits)
        ],
        "motivos": reasons,
        "senales": signals,
        "avisos": warnings,
        "aviso_legal": (
            "Este resultado es una estimación estadística y puede equivocarse. "
            "No constituye prueba de autoría ni debe usarse como única base para sanciones."
        ),
    }


def _confidence(n_words: int, has_model: bool, doc_prob: float, style_prob: float) -> str:
    if not has_model or n_words < config.RECOMMENDED_WORDS:
        return "baja"
    decisive = abs(doc_prob - 0.5) > 0.3
    agree = (doc_prob >= 0.5) == (style_prob >= 0.5)
    if n_words >= 500 and decisive and agree:
        return "alta"
    return "media"
