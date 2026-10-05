# IAdección

Detector **gratuito y de código abierto** de escritura generada con IA, pensado para que los
estudiantes revisen sus trabajos antes de entregarlos. Prioriza el español.

> ⚠️ Ningún detector de IA es infalible. El resultado es una estimación estadística, **no una
> prueba de autoría**, y no debe usarse como única base para sancionar a nadie.

## Cómo funciona

| Señal | Qué mide | Peso |
|---|---|---|
| **Binoculars** (zero-shot) | Lo predecible que es el texto para un modelo base (Qwen2.5) comparado con su versión *instruct*. Los textos de IA puntúan bajo. | 80 % |
| **Estilometría** | Uniformidad de oraciones y párrafos, conectores formulaicos y expresiones típicas de los LLM en español. | 20 % |

- La puntuación se calibra con datos etiquetados (`models/calibration.json`) para que el porcentaje sea interpretable.
- Cada oración recibe su propia probabilidad usando los valores por token de una sola pasada del modelo.
- **Privacidad:** el texto solo existe en memoria durante el análisis; no se guarda en ningún sitio.

## Puesta en marcha

```bash
pip install -r requirements.txt
uvicorn backend.app.main:app --port 8000
# abrir http://localhost:8000
```

La primera ejecución descarga los modelos (~2 GB). Con una GPU de 6 GB cada análisis tarda menos
de 1 s; en CPU funciona, pero es más lento.

Variables de entorno opcionales:

| Variable | Por defecto | Uso |
|---|---|---|
| `IADECCION_OBSERVER` | `Qwen/Qwen2.5-0.5B` | Modelo base |
| `IADECCION_PERFORMER` | `Qwen/Qwen2.5-0.5B-Instruct` | Modelo instruct (mismo tokenizador) |
| `IADECCION_BINOCULARS` | `1` | `0` = solo estilometría |
| `IADECCION_MAX_TOKENS` | `512` | Tamaño de fragmento para textos largos |

Si cambias de modelos, recalibra (ver abajo).

## API

- `POST /api/analizar` — `{"texto": "..."}`
- `POST /api/analizar-archivo` — formulario con `archivo` (PDF, DOCX, TXT; máx. 10 MB)
- `GET /api/salud`

La respuesta incluye `probabilidad_ia`, `veredicto`, `confianza`, `fraccion_texto_marcado`,
las oraciones con su `nivel` (bajo/medio/alto), `motivos` y `avisos`.

## Investigación

```bash
python research/download_data.py --lang es        # corpus AuTexTification 2023 (IberLEF)
python research/calibrate.py --data data/autextification_es_train.parquet --n 2000
python research/calibrate.py --data data/autextification_es_test.parquet --n 3000 --eval-only
```

La métrica principal es **TPR con FPR del 1 %**: cuánta IA se detecta cuando solo se acepta
acusar por error a 1 de cada 100 textos humanos.

### Resultados actuales

Qwen2.5-0.5B / 0.5B-Instruct, AuTexTification en español (textos de ~65 palabras):

| Conjunto | n | AUROC | TPR @ 1 % FPR | Exactitud |
|---|---|---|---|---|
| Train (legal, wiki), usado para calibrar | 2 000 | 0.781 | 0.348 | — |
| **Test (noticias, reseñas), dominios no vistos** | 3 000 | **0.882** | **0.459** | **0.804** |

Limitaciones: los textos de IA de este corpus provienen de modelos de 2022 (BLOOM, GPT-3), y los
textos son cortos. Con textos largos el detector suele separar mejor, pero hay que medirlo con un
corpus propio de modelos actuales (siguiente paso de la hoja de ruta).

### Corpus propio (modelos actuales, textos largos)

```bash
# 1. Textos humanos anteriores a ChatGPT: resúmenes académicos 2010-2021 (OpenAlex)
#    y artículos de Wikipedia en su revisión del 31-12-2021
python research/corpus/collect_human.py --academico 1200 --enciclopedico 400

# 2. Textos de IA emparejados (mismo tema y extensión). Tareas: redactar, parafrasear, "humanizar"
python research/corpus/generate_ai.py --provider local --model BSC-LT/salamandra-2b-instruct --domain academico --n 300
python research/corpus/generate_ai.py --provider anthropic --model claude-opus-5-5 --domain academico --n 300
python research/corpus/generate_ai.py --provider openai --model gpt-5 --base-url https://api.openai.com/v1 --domain academico --n 300

# 3. Unir y dividir en train/test (sin compartir temas entre conjuntos)
python research/corpus/build_dataset.py

# 4. Calibrar con train y evaluar con test (con desglose por modelo y tarea)
python research/calibrate.py --data data/corpus/train.parquet --n 100000 --min-words 100
python research/calibrate.py --data data/corpus/test.parquet --n 100000 --min-words 100 --eval-only
```

El proveedor `openai` admite cualquier API compatible (OpenAI, Gemini, Groq, OpenRouter, Ollama)
cambiando `--base-url`; la clave va en `OPENAI_API_KEY`. El proveedor `anthropic` necesita
`pip install anthropic` y `ANTHROPIC_API_KEY`.

#### Evaluación con el corpus propio (octubre 2026)

1 574 textos humanos (1 176 resúmenes académicos 2010-2021 y 398 artículos de Wikipedia de 2021)
frente a 268 textos académicos de IA generados con tres modelos abiertos. Calibración sin cambios
(AuTexTification). Textos de ≥100 palabras.

| Modelo generador | n | AUROC | IA detectada con 1 % FPR |
|---|---|---|---|
| Salamandra-2B (BSC) | 122 | 0.883 | 72 % |
| EuroLLM-1.7B | 72 | 0.958 | 54 % |
| Qwen2.5-1.5B (*misma familia que el detector*) | 74 | 0.967 | 81 % |

| Tarea de la IA | AUROC | IA detectada con 1 % FPR |
|---|---|---|
| Redactar sobre el tema | 0.991 | 88 % |
| Redactar pidiendo "que no suene a IA" | 0.995 | 87 % |
| **Parafrasear un texto humano** | **0.709** | **11 %** |

Falsos positivos (humano marcado como "probablemente IA", umbral 0.65): **0.19 %**. En la zona
"incierta" (≥0.5) cae el 3 %.

Conclusiones:
- Detecta muy bien el texto redactado por IA, incluso cuando se le pide sonar humano.
- **No detecta la paráfrasis con IA de un texto humano** (punto débil principal).
- El umbral de la aplicación es muy conservador: con EuroLLM solo marca como "probablemente IA"
  el 29 % de los textos redactados, aunque el AUROC sea 0.96. Recalibrar con este corpus
  subiría la detección a costa de más falsos positivos (decisión de política, pendiente).
- Falta medir los modelos comerciales (GPT, Claude, Gemini).

## Pruebas

```bash
python -m pytest tests -q
```

## Hoja de ruta

- [x] MVP: Binoculars + estilometría, resaltado por oración, PDF/DOCX, interfaz web
- [ ] Corpus propio con textos de modelos actuales (GPT, Claude, Gemini, Llama) y textos humanizados
- [ ] Clasificador supervisado (XLM-RoBERTa) y meta-clasificador entrenado en lugar de pesos fijos
- [ ] Auditoría de falsos positivos por grupo (hablantes no nativos, textos técnicos)
- [ ] Inferencia en el navegador (transformers.js) para coste cero y privacidad total
- [ ] Inglés, informe PDF descargable, módulo de similitud con fuentes abiertas
