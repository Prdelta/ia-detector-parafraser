"""Segmentación del texto en párrafos y oraciones, conservando posiciones."""

import re
from dataclasses import dataclass

# Abreviaturas frecuentes en español académico que no cierran oración.
_ABBREVIATIONS = {
    "sr", "sra", "srta", "dr", "dra", "lic", "ing", "prof", "etc", "pág", "págs",
    "p", "pp", "vol", "núm", "no", "cap", "ed", "eds", "fig", "figs", "cf", "vs",
    "ej", "aprox", "art", "inc", "et", "al", "op", "cit", "ibid", "ss", "ud", "uds",
    # inglés
    "mr", "mrs", "ms", "jr", "sr", "st", "e.g", "i.e", "approx", "dept", "univ", "ca",
}

_STOP_ES = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o".split())
_STOP_EN = set("the of and to in is that for it with as was on are be by this from or an which have not".split())

_SENTENCE_END = re.compile(r"[.!?…]+[\"'»”)\]]*(?=\s+|$)")


@dataclass
class Sentence:
    text: str
    start: int
    end: int
    paragraph: int


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraph_spans(text: str):
    start = 0
    for match in re.finditer(r"\n\s*\n", text):
        yield start, match.start()
        start = match.end()
    yield start, len(text)


def split_sentences(text: str) -> list[Sentence]:
    """Divide en oraciones. Las posiciones se refieren a ``text`` tal cual."""
    sentences: list[Sentence] = []
    for p_idx, (p_start, p_end) in enumerate(_paragraph_spans(text)):
        cursor = p_start
        for match in _SENTENCE_END.finditer(text, p_start, p_end):
            end = match.end()
            before = text[cursor:match.start()]
            last_word = re.findall(r"(\w+)$", before)
            if match.group().startswith(".") and last_word:
                word = last_word[0].lower()
                # Abreviatura o inicial ("J. Pérez"): no cortar.
                if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                    continue
                # Número decimal o numeración ("3.5", "1."): no cortar si sigue un dígito.
                if word.isdigit() and end < p_end and text[end:end + 2].strip()[:1].isdigit():
                    continue
            _append(sentences, text, cursor, end, p_idx)
            cursor = end
        _append(sentences, text, cursor, p_end, p_idx)
    return sentences


def _append(sentences, text, start, end, paragraph):
    raw = text[start:end]
    stripped = raw.strip()
    if not stripped:
        return
    offset = start + (len(raw) - len(raw.lstrip()))
    sentences.append(Sentence(stripped, offset, offset + len(stripped), paragraph))


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:[-'][^\W\d_]+)*", text, flags=re.UNICODE)


def detect_language(text: str) -> str:
    """'es' o 'en' según la proporción de palabras vacías de cada idioma."""
    toks = [w.lower() for w in words(text)]
    es = sum(t in _STOP_ES for t in toks)
    en = sum(t in _STOP_EN for t in toks)
    return "en" if en > es else "es"
