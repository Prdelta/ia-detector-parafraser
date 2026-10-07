import os

os.environ["IADECCION_BINOCULARS"] = "0"
os.environ["IADECCION_PARAFRASEADOR"] = "0"
os.environ["IADECCION_CLASIFICADOR"] = "0"

from fastapi.testclient import TestClient  # noqa: E402

from backend.app import similarity  # noqa: E402
from backend.app.main import app  # noqa: E402

FUENTE = (
    "El Teatro María Guerrero es un teatro situado en la calle de Tamayo y Baus de Madrid. "
    "Es la sede del Centro Dramático Nacional desde 1978 y fue inaugurado en el año 1885 con el nombre de Teatro de la Princesa. "
    "El edificio fue diseñado por el arquitecto Agustín Ortiz de Villajos con un estilo neomudéjar."
)
PROPIO = (
    "Ayer por la tarde estuve pensando en cómo organizamos el trabajo del grupo de laboratorio. "
    "Creo que repartimos mal las tareas porque nadie quiso encargarse de revisar las mediciones del sensor, "
    "y al final tuvimos que repetir la práctica entera el jueves, algo que nos costó casi cuatro horas."
)


def fake_search(query, lang="es"):
    return [similarity.Source("Teatro María Guerrero", "https://es.wikipedia.org/wiki/X", "Wikipedia", FUENTE)]


def test_coincidencia_copiada_sin_tildes_ni_mayusculas():
    copia = FUENTE.lower().replace("í", "i").replace("é", "e")
    r = similarity.check(PROPIO + "\n\n" + copia, search=fake_search)
    assert r["fuentes"][0]["url"] == "https://es.wikipedia.org/wiki/X"
    assert {o["parrafo"] for o in r["oraciones"]} == {1}  # solo el párrafo copiado
    assert 0.4 < r["fraccion_coincidente"] < 0.7


def test_texto_propio_sin_coincidencias():
    r = similarity.check(PROPIO, search=fake_search)
    assert r["fraccion_coincidente"] == 0 and r["fuentes"] == [] and r["oraciones"] == []


def test_solo_se_envian_palabras_clave():
    enviadas = []
    similarity.check(PROPIO + "\n\n" + FUENTE, search=lambda q, lang="es": enviadas.append(q) or [])
    for q in enviadas:
        assert len(q.split()) <= similarity.KEYWORDS_PER_QUERY
        assert q not in PROPIO and q not in FUENTE  # nunca una frase literal del texto


def test_palabras_clave_sin_palabras_vacias():
    kw = similarity.keywords("Así que el laboratorio del laboratorio tenía mediciones y más mediciones del sensor.")
    assert kw[:2] == ["laboratorio", "mediciones"]
    assert "así" not in kw and "tenía" in kw


def test_api_similitud(monkeypatch):
    monkeypatch.setattr(similarity, "default_search", fake_search)
    monkeypatch.setattr(similarity.check, "__defaults__", ("es", fake_search))
    client = TestClient(app)
    r = client.post("/api/similitud", json={"texto": (PROPIO + "\n\n" + FUENTE + " ") * 2})
    assert r.status_code == 200 and r.json()["fuentes"]
    assert client.post("/api/similitud", json={"texto": "muy corto"}).status_code == 422
