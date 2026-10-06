"""Auditoría de falsos positivos: ¿a qué grupos de escritores humanos acusa más el detector?

Grupos (todos textos humanos anteriores a ChatGPT):
  - Aprendices de español como L2 y hablantes de herencia (COWS-L2H, UC Davis, Apache-2.0, 2017-2021)
  - Resúmenes académicos por área: técnica (ingeniería, informática, medicina...) frente a humanidades
  - Wikipedia en español (revisión de 2021)
  - Textos cortos: resúmenes académicos recortados a 80-150 palabras

Usa el análisis completo de la aplicación (``analyze_text``), es decir, lo que ve el estudiante.

Uso:  python research/audit_fp.py [--n 200]
Salida: tabla en pantalla y models/auditoria_fp.json
"""

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.analyzer import analyze_text  # noqa: E402

COWS_ZIP = ROOT / "data" / "externos" / "cowsl2h.zip"
COWS_URL = "https://github.com/ucdaviscl/cowsl2h/archive/refs/heads/master.zip"
TECH_FIELDS = {"Engineering", "Computer Science", "Medicine", "Mathematics", "Physics and Astronomy",
               "Chemistry", "Materials Science", "Chemical Engineering", "Energy", "Earth and Planetary Sciences",
               "Biochemistry, Genetics and Molecular Biology", "Agricultural and Biological Sciences",
               "Environmental Science", "Veterinary", "Nursing", "Health Professions", "Dentistry", "Pharmacology, Toxicology and Pharmaceutics", "Immunology and Microbiology", "Neuroscience"}


def load_cows() -> pd.DataFrame:
    if not COWS_ZIP.exists():
        import httpx

        COWS_ZIP.parent.mkdir(parents=True, exist_ok=True)
        COWS_ZIP.write_bytes(httpx.get(COWS_URL, follow_redirects=True, timeout=120).content)
    z = zipfile.ZipFile(COWS_ZIP)
    frames = [pd.read_csv(io.BytesIO(z.read(n))) for n in z.namelist()
              if "/csv/" in n and n.endswith(".csv")]
    df = pd.concat(frames, ignore_index=True).dropna(subset=["essay"])
    df["text"] = df["essay"].str.replace("​", "", regex=False)
    home = df["language(s) used at home"].fillna("").str.lower()
    l1 = df["l1 language"].fillna("").str.lower()
    df["grupo"] = "L2 (aprendiz de español)"
    df.loc[home.str.contains("spanish|español") | l1.str.contains("spanish|español"), "grupo"] = "Hablante de herencia"
    return df[df.text.str.split().str.len() >= 100]


def build_groups(n: int, seed: int) -> pd.DataFrame:
    parts = []
    cows = load_cows()
    for grupo, g in cows.groupby("grupo"):
        parts.append(g.sample(min(n, len(g)), random_state=seed).assign(grupo=grupo)[["text", "grupo"]])

    human = pd.read_json(ROOT / "data" / "corpus" / "human.jsonl", lines=True)
    acad = human[human.domain == "academico"]
    tech = acad[acad.field.isin(TECH_FIELDS)]
    hum = acad[~acad.field.isin(TECH_FIELDS)]
    parts.append(tech.sample(min(n, len(tech)), random_state=seed).assign(grupo="Académico técnico")[["text", "grupo"]])
    parts.append(hum.sample(min(n, len(hum)), random_state=seed).assign(grupo="Académico humanidades/sociales")[["text", "grupo"]])
    wiki = human[human.domain == "enciclopedico"]
    parts.append(wiki.sample(min(n, len(wiki)), random_state=seed).assign(grupo="Wikipedia")[["text", "grupo"]])
    short = acad.sample(min(n, len(acad)), random_state=seed + 1).copy()
    short["text"] = short.text.map(lambda t: " ".join(t.split()[:120]))
    parts.append(short.assign(grupo="Texto corto (120 palabras)")[["text", "grupo"]])
    return pd.concat(parts, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=200, help="textos por grupo")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = build_groups(args.n, args.seed)
    probs = []
    for i, text in enumerate(df.text):
        try:
            probs.append(analyze_text(text)["probabilidad_ia"])
        except ValueError:
            probs.append(None)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(df)}", flush=True)
    df["p"] = probs
    df = df.dropna(subset=["p"])

    rows = []
    for grupo, g in df.groupby("grupo", sort=False):
        rows.append({
            "grupo": grupo, "n": len(g),
            "marcado_ia_%": round(100 * (g.p >= 0.65).mean(), 1),
            "incierto_o_mas_%": round(100 * (g.p >= 0.35).mean(), 1),
            "prob_media": round(g.p.mean(), 3),
        })
    table = pd.DataFrame(rows)
    print("\nFalsos positivos por grupo (todos los textos son humanos):")
    print(table.to_string(index=False))
    out = ROOT / "models" / "auditoria_fp.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado en {out}")


if __name__ == "__main__":
    main()
