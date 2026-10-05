"""Recolecta textos humanos en español escritos antes de ChatGPT (nov. 2022).

Fuentes:
  - academico:     resúmenes de artículos 2010-2021 (OpenAlex, licencia CC0 de metadatos)
  - enciclopedico: artículos de Wikipedia en español en su revisión del 31-12-2021 (CC BY-SA)

Uso:  python research/corpus/collect_human.py --academico 1200 --enciclopedico 400
Salida: data/corpus/human.jsonl
"""

import argparse
import html
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path

import httpx

OUT = Path(__file__).resolve().parents[2] / "data" / "corpus" / "human.jsonl"
HEADERS = {"User-Agent": "IAdeccion-research/0.1 (detector educativo de texto IA; uso no comercial)"}
CUTOFF = "2021-12-31T23:59:59Z"

_ES = set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o este sí porque esta entre cuando muy sin sobre también fue había ha".split())
_EN = set("the of and to in is that for it with as was on are be by this from or an which".split())


# Marcadores de catalán, portugués y gallego, lenguas que comparten muchas palabras con el castellano.
_OTHER_IBERIAN = re.compile(
    r"\b(?:és|amb|els|aquest|aquesta|dels|perquè|però|não|são|uma|através|então|unha|polo|pola|tamén|ademais)\b"
    r"|ção\b|ões\b|\b[ld]'\w",
    re.IGNORECASE,
)


def is_spanish(text: str, min_tokens: int = 50) -> bool:
    toks = re.findall(r"[a-záéíóúñü]+", text.lower())
    if len(toks) < min_tokens:
        return False
    es = sum(t in _ES for t in toks) / len(toks)
    en = sum(t in _EN for t in toks) / len(toks)
    other = len(_OTHER_IBERIAN.findall(text)) / len(toks)
    return es > 0.2 and en < 0.03 and other < 0.01


def n_words(text: str) -> int:
    return len(text.split())


def get_json(client: httpx.Client, url: str, params: dict, tries: int = 5):
    """GET con reintentos y espera exponencial; ``None`` si sigue fallando."""
    for attempt in range(tries):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(2 ** attempt)
    return None


# ---------- OpenAlex ----------

def _abstract(inv: dict) -> str:
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    text = " ".join(pos[i] for i in sorted(pos))
    text = html.unescape(re.sub(r"<[^>]+>", " ", text))
    # Quitar encabezados y versiones en otros idiomas pegadas al resumen.
    text = re.sub(r"^\s*(resumen|introducci[oó]n)\s*[:.]?\s*", "", text, flags=re.I)
    text = re.split(r"\b(?:Abstract|ABSTRACT|Resumo|Palabras clave|Palabras-clave|Keywords|Descriptores)\b", text)[0]
    return re.sub(r"\s+", " ", text).strip()


def collect_openalex(target: int, client: httpx.Client) -> list[dict]:
    items, seen, seed = [], set(), 0
    while len(items) < target and seed < 200:
        seed += 1
        data = get_json(
            client,
            "https://api.openalex.org/works",
            {
                "filter": "language:es,has_abstract:true,publication_year:2010-2021,type:article",
                "sample": 200, "seed": seed, "per_page": 200,
                "select": "id,title,publication_year,abstract_inverted_index,primary_topic",
            },
        )
        for w in (data or {}).get("results", []):
            if w["id"] in seen or not w.get("abstract_inverted_index") or not w.get("title"):
                continue
            seen.add(w["id"])
            text = _abstract(w["abstract_inverted_index"])
            if not (150 <= n_words(text) <= 600) or not is_spanish(text):
                continue
            topic = w.get("primary_topic") or {}
            items.append({
                "id": "oa_" + w["id"].rsplit("/", 1)[-1],
                "text": text,
                "domain": "academico",
                "title": re.sub(r"<[^>]+>", "", w["title"]).strip(),
                "field": (topic.get("field") or {}).get("display_name", "desconocido"),
                "year": w["publication_year"],
                "source": w["id"],
            })
        print(f"  OpenAlex: {len(items)}/{target}")
        time.sleep(0.2)
    return items[:target]


# ---------- Wikipedia (revisión de 2021) ----------

class _Paragraphs(HTMLParser):
    """Extrae el texto de los <p> de primer nivel, sin tablas, referencias ni notas."""

    SKIP = {"table", "sup", "style", "script", "figure"}

    def __init__(self):
        super().__init__()
        self.paragraphs, self._buf, self._in_p, self._skip = [], [], 0, 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self._skip += 1
        elif tag == "p":
            self._in_p += 1

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self._skip = max(0, self._skip - 1)
        elif tag == "p" and self._in_p:
            self._in_p -= 1
            text = re.sub(r"\s+", " ", "".join(self._buf)).strip()
            if len(text.split()) >= 15:
                self.paragraphs.append(text)
            self._buf = []

    def handle_data(self, data):
        if self._in_p and not self._skip:
            self._buf.append(data)


def collect_wikipedia(target: int, client: httpx.Client) -> list[dict]:
    api = "https://es.wikipedia.org/w/api.php"
    items, seen = [], set()
    while len(items) < target:
        # Los artículos aleatorios suelen ser esbozos: se piden 50 y se filtran por tamaño.
        data = get_json(client, api, {"action": "query", "generator": "random", "grnnamespace": 0,
                                      "grnlimit": 50, "prop": "info", "format": "json"})
        pages_info = (data or {}).get("query", {}).get("pages", {}).values()
        for page in [p for p in pages_info if p.get("length", 0) >= 12000]:
            title = page["title"]
            if title in seen:
                continue
            seen.add(title)
            rev = get_json(client, api, {
                "action": "query", "prop": "revisions", "titles": title, "rvlimit": 1,
                "rvstart": CUTOFF, "rvdir": "older", "rvprop": "ids|timestamp", "format": "json",
            })
            pages = list(((rev or {}).get("query") or {}).get("pages", {}).values())
            if not pages or "revisions" not in pages[0]:
                continue  # el artículo no existía en 2021
            revid = pages[0]["revisions"][0]["revid"]
            parsed = get_json(client, api, {"action": "parse", "oldid": revid, "prop": "text",
                                            "format": "json", "formatversion": 2})
            if not parsed or "parse" not in parsed:
                continue
            parser = _Paragraphs()
            parser.feed(parsed["parse"]["text"])
            text = ""
            for p in parser.paragraphs:  # acumular párrafos hasta ~600 palabras
                if n_words(text) + n_words(p) > 650:
                    break
                text = f"{text}\n\n{p}" if text else p
            text = re.sub(r"\[\d+\]|\[cita requerida\]|​", "", text).strip()
            if n_words(text) < 250 or not is_spanish(text):
                continue
            items.append({
                "id": f"wk_{revid}",
                "text": text,
                "domain": "enciclopedico",
                "title": title,
                "field": "wikipedia",
                "year": int(pages[0]["revisions"][0]["timestamp"][:4]),
                "source": f"https://es.wikipedia.org/w/index.php?oldid={revid}",
            })
            if len(items) >= target:
                break
            time.sleep(0.5)
        print(f"  Wikipedia: {len(items)}/{target}")
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--academico", type=int, default=1200)
    parser.add_argument("--enciclopedico", type=int, default=400)
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sources = {"academico": (collect_openalex, args.academico),
               "enciclopedico": (collect_wikipedia, args.enciclopedico)}
    items = []
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for name, (collect, target) in sources.items():
            # Cada fuente se guarda aparte para no repetir descargas si otra falla.
            part = OUT.with_name(f"human_{name}.jsonl")
            cached = [json.loads(line) for line in part.open(encoding="utf-8")] if part.exists() else []
            if len(cached) >= target:
                batch = cached[:target]
            else:
                batch = collect(target, client)
                with part.open("w", encoding="utf-8") as f:
                    for it in batch:
                        f.write(json.dumps(it, ensure_ascii=False) + "\n")
            items += batch
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            it["words"] = n_words(it["text"])
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} textos humanos -> {OUT}")


if __name__ == "__main__":
    main()
