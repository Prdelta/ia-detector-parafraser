"""API web del detector. Ejecutar con:  uvicorn backend.app.main:app"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, rewrite, similarity
from .analyzer import analyze_text
from .detectors import binoculars
from .extract import UnsupportedFile, extract_text
from .segment import detect_language

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Cargar los modelos al arrancar para que el primer análisis no tarde.
    await run_in_threadpool(binoculars.get_detector)
    yield


app = FastAPI(
    title="IAdección",
    description="Detector gratuito de escritura con IA para estudiantes",
    version="0.1.0",
    lifespan=lifespan,
)


class TextIn(BaseModel):
    texto: str


@app.get("/api/salud")
def health():
    det = binoculars.get_detector()
    return {"estado": "ok", "modelo": det.model_id if det else None}


async def _analyze(text: str) -> dict:
    try:
        return await run_in_threadpool(analyze_text, text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/api/analizar")
async def analyze(body: TextIn):
    # Privacidad: el texto solo vive en memoria durante el análisis; no se guarda.
    return await _analyze(body.texto)


@app.post("/api/analizar-archivo")
async def analyze_file(archivo: UploadFile = File(...)):
    data = await archivo.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera los 10 MB.")
    try:
        text = extract_text(archivo.filename, data)
    except UnsupportedFile as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    result = await _analyze(text)
    result["archivo"] = archivo.filename
    return result


@app.post("/api/parafrasear")
async def paraphrase(body: TextIn):
    paraphraser = rewrite.get_paraphraser()
    if paraphraser is None:
        raise HTTPException(status_code=503, detail="La paráfrasis automática está desactivada en este servidor.")
    n_words = len(body.texto.split())
    if not 5 <= n_words <= config.MAX_PARAPHRASE_WORDS:
        raise HTTPException(status_code=422,
                            detail=f"El párrafo debe tener entre 5 y {config.MAX_PARAPHRASE_WORDS} palabras.")
    try:
        result = await run_in_threadpool(paraphraser.paraphrase, body.texto)
    except Exception:
        logging.exception("Fallo en la paráfrasis")
        raise HTTPException(status_code=500, detail="No se pudo parafrasear el párrafo.")
    return {"texto": result, "aviso": rewrite.PARAPHRASE_NOTICE}


@app.post("/api/similitud")
async def similar_sources(body: TextIn):
    if not config.ENABLE_SIMILARITY:
        raise HTTPException(status_code=503, detail="La búsqueda de coincidencias está desactivada en este servidor.")
    if len(body.texto.split()) < config.MIN_WORDS or len(body.texto) > config.MAX_CHARS:
        raise HTTPException(status_code=422, detail="El texto no tiene una extensión válida.")
    # Solo salen del servidor palabras clave de cada párrafo, no el texto.
    return await run_in_threadpool(similarity.check, body.texto, detect_language(body.texto))


@app.get("/")
def index():
    return FileResponse(config.FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=config.FRONTEND_DIR), name="static")
