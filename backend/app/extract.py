"""Extracción de texto desde archivos subidos (PDF, DOCX, TXT)."""

import io
import re
from pathlib import Path


class UnsupportedFile(ValueError):
    pass


SUPPORTED = {".pdf", ".docx", ".txt", ".md"}

# Entrada de bibliografía (APA y similares): "Apellido, A. B., ... (2020). Título".
# No es prosa del estudiante y solo añadiría ruido al análisis.
_REFERENCE = re.compile(
    r"^[^\W\d_][^,()]{1,60}, (?:[^\W\d_]{1,2}\.[\s-]?)+.{0,400}?\((?:1[89]|20)\d{2}[a-z]?(?:, [^)]*)?\)\.")


def _without_references(paragraphs: list[str]) -> str:
    return "\n\n".join(p for p in paragraphs if p and not _REFERENCE.match(p))


def extract_text(filename: str, data: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix == ".pdf":
        return _from_pdf(data)
    if suffix == ".docx":
        return _from_docx(data)
    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8-sig", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
    raise UnsupportedFile(f"Formato no soportado: {suffix or 'desconocido'}. Usa PDF, DOCX o TXT.")


def _from_pdf(data: bytes) -> str:
    import fitz  # PyMuPDF

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PDF dañado o cifrado
        raise UnsupportedFile("No se pudo leer el PDF.") from exc
    pages = []
    with doc:
        for page in doc:
            # Cada bloque suele ser un párrafo; se unen sus líneas internas.
            for block in page.get_text("blocks", sort=True):
                if block[6] != 0:  # 0 = bloque de texto, 1 = imagen
                    continue
                pages.append(" ".join(block[4].split()))
    return _without_references(pages)


def _from_docx(data: bytes) -> str:
    import docx
    from docx.oxml.ns import qn

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise UnsupportedFile("No se pudo leer el DOCX.") from exc
    # Se recorre el XML en lugar de ``document.paragraphs``: esa API omite el texto dentro de
    # controles de contenido (w:sdt), que es donde Zotero y Mendeley ponen las citas.
    # Las tablas y los cuadros de texto se omiten, como antes (fragmentos que no son prosa).
    skip = {qn("w:tbl"), qn("w:txbxContent")}
    paragraphs = []
    for p in document.element.body.iter(qn("w:p")):
        if any(a.tag in skip for a in p.iterancestors()):
            continue
        parts = []
        for el in p.iter(qn("w:t"), qn("w:tab"), qn("w:br")):
            if any(a.tag == qn("w:txbxContent") for a in el.iterancestors()):
                continue
            parts.append(el.text or "" if el.tag == qn("w:t") else " ")
        text = " ".join("".join(parts).split())
        if text:
            paragraphs.append(text)
    return _without_references(paragraphs)
