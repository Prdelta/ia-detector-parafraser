import io
import os

os.environ["IADECCION_BINOCULARS"] = "0"  # pruebas rápidas sin descargar modelos
os.environ["IADECCION_PARAFRASEADOR"] = "0"

import docx  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from backend.app.analyzer import analyze_text  # noqa: E402
from backend.app.detectors import stylometry  # noqa: E402
from backend.app.extract import UnsupportedFile, extract_text  # noqa: E402
from backend.app.main import app  # noqa: E402
from backend.app.rewrite import guidance  # noqa: E402
from backend.app.segment import split_sentences  # noqa: E402

HUMANO = (
    "Ayer fui al mercado. Llovía a cántaros y, claro, se me olvidó el paraguas en el bus, como siempre. "
    "La señora de las frutas me regaló dos mangos porque dice que estoy flaco. ¿Flaco yo? "
    "Bueno, quizá un poco. Después pasé por la biblioteca de la facultad para devolver el libro de Vargas Llosa "
    "que debía desde marzo; la multa fue ridícula, menos de lo que cuesta un café.\n\n"
    "En la tarde intenté escribir el informe de laboratorio, pero el Excel se cerró tres veces y perdí la tabla. "
    "Terminé a las once. No quedó perfecto. Mañana lo reviso con Camila, que entiende mejor la parte de titulación."
)

IA = (
    "En la era digital, la educación desempeña un papel fundamental en el desarrollo de la sociedad. "
    "Además, es importante destacar que la tecnología ofrece una amplia gama de oportunidades para los estudiantes. "
    "Asimismo, cabe destacar que los docentes deben adoptar un enfoque integral para fomentar el aprendizaje. "
    "Por otro lado, la innovación educativa permite abordar los desafíos de manera efectiva y eficiente.\n\n"
    "En este sentido, la colaboración entre instituciones juega un papel crucial en la mejora de la calidad educativa. "
    "Además, es fundamental considerar los desafíos y oportunidades que surgen en el panorama actual. "
    "Por lo tanto, resulta esencial promover estrategias que potencien las competencias digitales de los estudiantes. "
    "En conclusión, la educación en la era digital representa un aspecto clave para el futuro de la sociedad."
)


def test_split_respeta_abreviaturas_y_decimales():
    s = split_sentences("El Dr. Pérez midió 3.5 cm. Luego habló con la Sra. Gómez. ¿Listo?")
    assert [x.text for x in s] == ["El Dr. Pérez midió 3.5 cm.", "Luego habló con la Sra. Gómez.", "¿Listo?"]


def test_posiciones_de_oraciones():
    text = "Primera oración.\n\nSegunda oración aquí."
    for s in split_sentences(text):
        assert text[s.start:s.end] == s.text
    assert [s.paragraph for s in split_sentences(text)] == [0, 1]


def test_estilometria_distingue_ejemplos_claros():
    h = stylometry.analyze(HUMANO, split_sentences(HUMANO))
    a = stylometry.analyze(IA, split_sentences(IA))
    assert a.score > 0.65 > 0.35 > h.score
    assert a.reasons


def test_texto_corto_rechazado():
    with pytest.raises(ValueError):
        analyze_text("Muy corto.")


def test_informe_completo():
    r = analyze_text(IA + "\n\n" + HUMANO)
    assert 0 <= r["probabilidad_ia"] <= 1
    assert r["confianza"] == "baja"  # sin modelo principal
    assert len(r["oraciones"]) > 10
    assert all(o["nivel"] in {"bajo", "medio", "alto"} for o in r["oraciones"])


def test_extraer_docx():
    buf = io.BytesIO()
    d = docx.Document()
    d.add_paragraph("Hola mundo.")
    d.add_paragraph("Segundo párrafo.")
    d.save(buf)
    assert extract_text("a.docx", buf.getvalue()) == "Hola mundo.\n\nSegundo párrafo."
    with pytest.raises(UnsupportedFile):
        extract_text("a.exe", b"x")


def test_api():
    with TestClient(app) as client:
        assert client.get("/api/salud").json()["estado"] == "ok"
        assert client.get("/").status_code == 200
        r = client.post("/api/analizar", json={"texto": IA})
        assert r.status_code == 200 and "veredicto" in r.json()
        assert client.post("/api/analizar", json={"texto": "corto"}).status_code == 422
        r = client.post("/api/analizar-archivo", files={"archivo": ("t.txt", HUMANO.encode(), "text/plain")})
        assert r.status_code == 200 and r.json()["archivo"] == "t.txt"


def test_parrafos_con_guia():
    r = analyze_text(IA + "\n\n" + HUMANO)
    assert r["texto"].startswith("En la era digital")
    assert [p["indice"] for p in r["parrafos"]] == [0, 1, 2, 3]
    assert "\n\n".join(p["texto"] for p in r["parrafos"]) == r["texto"]
    primero = r["parrafos"][0]
    assert any("Frases hechas" in m for m in primero["motivos"])
    assert any("conectores" in m for m in primero["motivos"])
    assert primero["sugerencias"]
    assert r["parafrasis_disponible"] is False


def test_guia_parrafo_humano_sin_motivos():
    sents = split_sentences(HUMANO.split("\n\n")[0])
    g = guidance(sents, [[] for _ in sents], probability=0.1)
    assert g == {"motivos": [], "sugerencias": []}


def test_parafrasear_desactivado():
    with TestClient(app) as client:
        r = client.post("/api/parafrasear", json={"texto": "Un párrafo cualquiera con varias palabras."})
        assert r.status_code == 503
