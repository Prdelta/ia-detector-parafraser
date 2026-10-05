"""Rasgos estilométricos asociados a texto generado por IA en español.

Es una señal débil y complementaria. Los pesos son provisionales y deben
reemplazarse por los aprendidos con ``research/train_meta.py`` cuando haya datos.
"""

import math
import re
import statistics
from dataclasses import dataclass, field

from ..segment import Sentence, words

# Expresiones sobrerrepresentadas en la prosa de los LLM en español.
AI_PHRASES = [
    r"cabe (?:destacar|mencionar|señalar|resaltar)",
    r"es (?:importante|fundamental|crucial|esencial|vital) (?:destacar|señalar|mencionar|tener en cuenta|considerar|recordar|reconocer)",
    r"en (?:resumen|conclusión|definitiva|síntesis)",
    r"juega(?:n)? un papel (?:crucial|fundamental|clave|importante|esencial|vital)",
    r"desempeña(?:n)? un papel (?:crucial|fundamental|clave|importante|esencial|vital)",
    r"en el (?:ámbito|contexto|marco|panorama) (?:de|actual|educativo|académico)",
    r"en la era (?:digital|actual|moderna)",
    r"en el mundo (?:actual|moderno|de hoy)",
    r"un (?:aspecto|factor|elemento|pilar) (?:clave|fundamental|crucial|esencial)",
    r"no (?:solo|sólo) .{1,60}? sino (?:también|que)",
    r"sin lugar a dudas",
    r"a medida que",
    r"en última instancia",
    r"(?:profundizar|ahondar) en",
    r"(?:fomentar|potenciar|impulsar) (?:la|el|una|un)",
    r"(?:un|una) (?:enfoque|perspectiva|visión) (?:integral|holística|multidimensional|innovador[a]?)",
    r"(?:amplia|vasta|rica) (?:gama|variedad)",
    r"(?:panorama|paisaje) (?:actual|complejo|cambiante)",
    r"(?:navegar|abordar) (?:los|las|el|la) (?:desafíos|retos|complejidades)",
    r"(?:desafíos|retos) y oportunidades",
    r"de manera (?:significativa|efectiva|integral|eficiente)",
    r"(?:transformador|transformadora|revolucionario|revolucionaria)",
    r"(?:sinergia|sinergias)",
    r"(?:en este sentido|por otro lado|asimismo|además|por lo tanto|en consecuencia),",
]
_AI_RE = re.compile("|".join(f"(?:{p})" for p in AI_PHRASES), re.IGNORECASE)

# Conectores al inicio de oración (los LLM los encadenan con regularidad).
CONNECTOR_START = re.compile(
    r"^(?:además|asimismo|por otro lado|por otra parte|en este sentido|por lo tanto|"
    r"en consecuencia|finalmente|en primer lugar|en segundo lugar|por último|en resumen|"
    r"en conclusión|sin embargo|no obstante|de igual manera|del mismo modo)\b",
    re.IGNORECASE,
)


@dataclass
class StyleResult:
    score: float  # 0..1, mayor = más parecido a IA
    features: dict
    reasons: list[str] = field(default_factory=list)
    sentence_hits: list[list[str]] = field(default_factory=list)


def _mattr(tokens: list[str], window: int = 50) -> float:
    """Moving-Average Type-Token Ratio: riqueza léxica independiente de la longitud."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(len(tokens) - window + 1)]
    return sum(ratios) / len(ratios)


def _cv(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return statistics.pstdev(values) / mean if mean else 0.0


def analyze(text: str, sentences: list[Sentence]) -> StyleResult:
    tokens = [w.lower() for w in words(text)]
    n_words = max(len(tokens), 1)
    sent_lengths = [len(words(s.text)) for s in sentences if words(s.text)]

    paragraphs: dict[int, int] = {}
    for s, length in zip(sentences, (len(words(s.text)) for s in sentences)):
        paragraphs[s.paragraph] = paragraphs.get(s.paragraph, 0) + length

    sentence_hits = [[m.group(0) for m in _AI_RE.finditer(s.text)] for s in sentences]
    n_hits = sum(len(h) for h in sentence_hits)
    connector_starts = sum(1 for s in sentences if CONNECTOR_START.match(s.text))

    features = {
        "palabras": len(tokens),
        "oraciones": len(sent_lengths),
        "long_media_oracion": round(statistics.fmean(sent_lengths), 2) if sent_lengths else 0.0,
        "variacion_oraciones": round(_cv(sent_lengths), 3),  # "burstiness"
        "variacion_parrafos": round(_cv(list(paragraphs.values())), 3) if len(paragraphs) >= 3 else None,
        "riqueza_lexica": round(_mattr(tokens), 3),
        "frases_ia_por_100": round(100 * n_hits / n_words, 3),
        "conectores_inicio": round(connector_starts / max(len(sentences), 1), 3),
        "preguntas_exclamaciones": sum(s.text.count("?") + s.text.count("!") for s in sentences),
    }

    # Puntuación logística con pesos provisionales (centrados en valores típicos).
    z = 0.0
    reasons = []
    burst = features["variacion_oraciones"]
    z += -4.0 * (burst - 0.45)
    if burst < 0.35 and len(sent_lengths) >= 5:
        reasons.append("Las oraciones tienen longitudes muy uniformes.")
    rate = features["frases_ia_por_100"]
    z += 1.2 * min(rate, 4.0) - 0.6
    if rate >= 1.0:
        reasons.append(f"Uso frecuente de expresiones típicas de IA ({n_hits} encontradas).")
    conn = features["conectores_inicio"]
    z += 3.0 * (conn - 0.15)
    if conn >= 0.3:
        reasons.append("Muchas oraciones empiezan con conectores formulaicos.")
    para = features["variacion_parrafos"]
    if para is not None:
        z += -2.0 * (para - 0.35)
        if para < 0.2:
            reasons.append("Los párrafos tienen extensiones casi idénticas.")
    if features["preguntas_exclamaciones"] > 0:
        z -= 0.3

    score = 1.0 / (1.0 + math.exp(-z))
    return StyleResult(score=score, features=features, reasons=reasons, sentence_hits=sentence_hits)
