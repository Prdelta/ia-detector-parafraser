import importlib.util
import random
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "research" / "corpus" / "generate_ai.py"
_spec = importlib.util.spec_from_file_location("generate_ai", _path)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

ITEM = {"id": "oa_1", "text": "Texto humano de prueba.", "domain": "academico",
        "title": "Aprendizaje y memoria", "field": "Psychology", "words": 220}


def test_clean_quita_preambulos_y_markdown():
    raw = ("Aquí tienes el texto solicitado:\n\n**Aprendizaje y memoria**\n\n"
           "El **aprendizaje** es un proceso complejo.\n- Primera idea.\n\n\n\nFin del texto.")
    assert gen.clean(raw) == "El aprendizaje es un proceso complejo.\nPrimera idea.\n\nFin del texto."


def test_clean_quita_razonamiento():
    assert gen.clean("<think>pienso...</think>Texto final.") == "Texto final."


def test_prompts_por_tarea():
    rng = random.Random(0)
    assert "Reescribe" in gen.build_prompt(ITEM, "parafrasear", rng)
    assert ITEM["text"] in gen.build_prompt(ITEM, "parafrasear", rng)
    redactar = gen.build_prompt(ITEM, "redactar", rng)
    assert "«Aprendizaje y memoria»" in redactar and "220 palabras" in redactar
    assert "no por una IA" in gen.build_prompt(ITEM, "humanizar", rng)
    wiki = dict(ITEM, domain="enciclopedico", words=900)
    assert "enciclopédico" in gen.build_prompt(wiki, "redactar", rng)
    assert "600 palabras" in gen.build_prompt(wiki, "redactar", rng)
