"""Extracción de texto desde archivos subidos (PDF, DOCX, TXT)."""

import io
from pathlib import Path


class UnsupportedFile(ValueError):
    pass


SUPPORTED = {".pdf", ".docx", ".txt", ".md"}


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
    return "\n\n".join(p for p in pages if p)


def _from_docx(data: bytes) -> str:
    import docx

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise UnsupportedFile("No se pudo leer el DOCX.") from exc
    return "\n\n".join(p.text.strip() for p in document.paragraphs if p.text.strip())
