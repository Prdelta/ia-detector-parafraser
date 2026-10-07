"""Coincidencias con fuentes abiertas (OpenAlex y Wikipedia).

Para cada párrafo se eligen unas pocas palabras clave y se buscan en las fuentes; después se
compara el texto con los candidatos mediante 5-gramas de palabras. Solo salen del servidor las
palabras clave, nunca el texto; el análisis es opcional y la interfaz lo advierte.

No es un detector de plagio completo: solo cubre resúmenes académicos (OpenAlex) y artículos de
Wikipedia, y una coincidencia puede ser una cita correctamente referenciada.
"""

import html
import logging
import os
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import httpx

from .segment import normalize, split_sentences

log = logging.getLogger(__name__)

# Wikipedia rechaza (403) los User-Agent sin forma de contacto; solo admite ASCII.
HEADERS = {"User-Agent": os.getenv("IADECCION_USER_AGENT",
                                   "IAdeccion/0.1 (https://github.com/Prdelta/ia-detector-parafraser)")}
NGRAM = 5
MIN_SENTENCE_WORDS = 8
MATCH_THRESHOLD = 0.4  # fracción de los 5-gramas de la oración presentes en la fuente
MAX_QUERIES = 8
KEYWORDS_PER_QUERY = 6

_STOP = set("""
a al algo algunas algunos ante antes aquel aquella aquellas aquellos aqui asi aun aunque bajo bien cada casi como con
contra cual cuales cualquier cuando cuanto de debe deben del desde donde dos durante e el ella ellas ello ellos en
entre era eran es esa esas ese eso esos esta estaba estado estan estar estas este esto estos fue fueron gran ha haber
habia hace hacen hacia han hasta hay la las le les lo los mas me mediante mientras mismo misma mucho muchos muy ni no
nos o otra otras otro otros para parte pero poco por porque puede pueden pues que quien se sea segun ser si sido sin
sino sobre son su sus tal tambien tanto te tiene tienen todo todos tras tres tu un una unas uno unos y ya
the of and to in is that for it with as was on are be by this from or an which have not their these those
""".split())


def _fold(text: str) -> str:
    """Minúsculas y sin tildes, para comparar con independencia de la ortografía."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9ñ]+", _fold(text))


def _ngrams(tokens: list[str], n: int = NGRAM) -> set[tuple[str, ...]]:
    return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def keywords(paragraph: str, k: int = KEYWORDS_PER_QUERY) -> list[str]:
    """Palabras más informativas del párrafo: frecuentes en él, largas y no vacías."""
    counts: dict[str, int] = {}
    first: dict[str, str] = {}
    for word in re.findall(r"[^\W\d_]{4,}", paragraph):
        key = _fold(word)
        if key in _STOP:
            continue
        counts[key] = counts.get(key, 0) + 1
        first.setdefault(key, word)
    ranked = sorted(counts, key=lambda w: (-counts[w], -len(w)))
    return [first[w].lower() for w in ranked[:k]]


@dataclass
class Source:
    title: str
    url: str
    kind: str  # "OpenAlex" o "Wikipedia"
    text: str
    ngrams: set = field(default_factory=set, repr=False)


# ---------- búsqueda ----------

def _openalex_abstract(inv: dict) -> str:
    pos = {i: w for w, idxs in inv.items() for i in idxs}
    text = " ".join(pos[i] for i in sorted(pos))
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text))).strip()


def search_openalex(client: httpx.Client, query: str, limit: int = 5) -> list[Source]:
    r = client.get("https://api.openalex.org/works", params={
        "search": query, "per-page": limit, "filter": "has_abstract:true",
        "select": "id,doi,title,abstract_inverted_index"})
    r.raise_for_status()
    return [Source(w.get("title") or "(sin título)", w.get("doi") or w["id"], "OpenAlex",
                   _openalex_abstract(w["abstract_inverted_index"]))
            for w in r.json().get("results", []) if w.get("abstract_inverted_index")]


def search_wikipedia(client: httpx.Client, query: str, lang: str = "es", limit: int = 3) -> list[Source]:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    r = client.get(api, params={"action": "query", "list": "search", "srsearch": query,
                                "srlimit": limit, "format": "json"})
    r.raise_for_status()
    sources = []
    for hit in r.json().get("query", {}).get("search", []):
        page = client.get(api, params={"action": "query", "prop": "extracts", "explaintext": 1,
                                       "titles": hit["title"], "format": "json", "formatversion": 2})
        page.raise_for_status()
        pages = page.json().get("query", {}).get("pages", [])
        if pages and pages[0].get("extract"):
            url = f"https://{lang}.wikipedia.org/wiki/{hit['title'].replace(' ', '_')}"
            sources.append(Source(hit["title"], url, "Wikipedia", pages[0]["extract"]))
    return sources


def default_search(query: str, lang: str = "es") -> list[Source]:
    with httpx.Client(headers=HEADERS, timeout=10, follow_redirects=True) as client:
        found = []
        for search in (search_openalex, lambda c, q: search_wikipedia(c, q, lang)):
            try:
                found += search(client, query)
            except (httpx.HTTPError, ValueError, KeyError) as exc:
                log.warning("Búsqueda de similitud fallida (%s): %s", query, exc)
        return found


# ---------- comparación ----------

def compare(text: str, sources: list[Source]) -> dict:
    """Marca las oraciones cuyos 5-gramas aparecen en alguna fuente."""
    for s in sources:
        s.ngrams = _ngrams(_tokens(s.text))
    sentences = split_sentences(text)
    matched, total_words, matched_words = [], 0, 0
    per_source: dict[str, dict] = {}
    for i, sent in enumerate(sentences):
        toks = _tokens(sent.text)
        total_words += len(toks)
        grams = _ngrams(toks)
        if len(toks) < MIN_SENTENCE_WORDS or not grams:
            continue
        best, best_overlap = None, 0.0
        for s in sources:
            overlap = len(grams & s.ngrams) / len(grams)
            if overlap > best_overlap:
                best, best_overlap = s, overlap
        if best and best_overlap >= MATCH_THRESHOLD:
            matched_words += len(toks)
            src = per_source.setdefault(best.url, {"titulo": best.title, "url": best.url, "tipo": best.kind,
                                                   "palabras": 0})
            src["palabras"] += len(toks)
            matched.append({"indice": i, "texto": sent.text, "parrafo": sent.paragraph, "url": best.url,
                            "solapamiento": round(best_overlap, 3)})
    fuentes = sorted(per_source.values(), key=lambda s: -s["palabras"])
    for s in fuentes:
        s["fraccion"] = round(s.pop("palabras") / max(total_words, 1), 4)
    return {
        "fraccion_coincidente": round(matched_words / max(total_words, 1), 4),
        "fuentes": fuentes,
        "oraciones": matched,
    }


def check(raw_text: str, lang: str = "es", search=default_search) -> dict:
    text = normalize(raw_text)
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if len(p.split()) >= 25] or [text]
    # Párrafos largos primero: más probabilidad de coincidir y menos consultas.
    paragraphs = sorted(paragraphs, key=lambda p: -len(p.split()))[:MAX_QUERIES]
    queries = list(dict.fromkeys(" ".join(keywords(p)) for p in paragraphs))
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda q: search(q, lang), queries))
    sources = list({s.url: s for found in results for s in found}.values())
    out = compare(text, sources)
    out["consultas"] = queries
    out["fuentes_revisadas"] = len(sources)
    out["aviso"] = ("Solo se comparó con resúmenes académicos de OpenAlex y artículos de Wikipedia. "
                    "Una coincidencia puede ser una cita bien referenciada; si no hay coincidencias, "
                    "no significa que el texto sea original.")
    return out
