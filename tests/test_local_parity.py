"""El análisis en el navegador (frontend/local/core.js) debe coincidir con el backend."""

import json
import os
import shutil
import subprocess
from pathlib import Path

os.environ["IADECCION_BINOCULARS"] = "0"
os.environ["IADECCION_PARAFRASEADOR"] = "0"
os.environ["IADECCION_CLASIFICADOR"] = "0"

import pytest  # noqa: E402

from backend.app.analyzer import analyze_text  # noqa: E402

from test_core import HUMANO, IA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

TEXTOS = [
    HUMANO,
    IA,
    HUMANO + "\n\n" + IA,
    # Abreviaturas, iniciales, decimales, numeración, signos de apertura y comillas.
    ("El Dr. Pérez y la Sra. Gómez midieron 3.5 cm en el 2.º ensayo, según J. R. Martínez et al. (2019, pp. 4-9). "
     "¿Es suficiente? ¡Claro que no! «Hace falta más evidencia», dijo. Además, cabe destacar que el método falla. "
     "Por otro lado, en la era digital todo cambia… Sin embargo, no solo importa la técnica sino también la ética.\r\n\r\n"
     "1. Primer punto del análisis con datos de campo.  2. Segundo punto, con 12.000 casos revisados a mano.\n\n\n\n"
     "Asimismo, la colaboración juega un papel clave. En conclusión, los desafíos y oportunidades son muchos y "
     "requieren un enfoque integral que permita abordar los retos de manera efectiva, eficiente y sostenible en el "
     "tiempo, sin lugar a dudas, a medida que la sociedad avanza hacia nuevos horizontes de desarrollo.") * 2,
    "Texto demasiado corto para analizar.",
]


def _corpus_samples(n=15):
    path = ROOT / "data" / "corpus" / "test.parquet"
    if not path.exists():
        return []
    import pandas as pd

    df = pd.read_parquet(path)
    return df.groupby("label", group_keys=False).apply(lambda g: g.sample(n, random_state=0)).text.tolist()


def _run_js(texts):
    proc = subprocess.run(["node", str(ROOT / "tests" / "local_core.mjs")], input=json.dumps(texts),
                          capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(proc.stdout)


def _close(a, b, path=""):
    if isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, abs=1.01e-3), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    else:
        assert a == b, path


@pytest.mark.skipif(shutil.which("node") is None, reason="requiere Node.js")
def test_core_js_igual_que_backend():
    texts = TEXTOS + _corpus_samples()
    for raw, js in zip(texts, _run_js(texts)):
        try:
            py = analyze_text(raw)
        except ValueError as exc:
            assert js == {"error": str(exc)}
            continue
        _close(js, py)
