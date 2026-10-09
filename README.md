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
- **Párrafos para revisar:** cada párrafo marcado muestra por qué suena a IA (frases hechas,
  conectores formulaicos, ritmo uniforme) y sugerencias concretas, con un editor para reescribirlo
  y reanalizar el texto completo.
- **Paráfrasis automática (opcional):** reescribe un párrafo con un modelo local. El resultado
  sigue siendo texto de IA y la interfaz lo advierte; se puede desactivar con `IADECCION_PARAFRASEADOR=0`.
- **Privacidad:** el texto solo existe en memoria durante el análisis; no se guarda en ningún sitio.
- **Informe PDF:** el botón "Descargar informe (PDF)" abre la vista de impresión del resultado (fecha,
  veredicto, texto resaltado, párrafos para revisar y aviso legal) para guardarla como PDF. Se genera en el
  navegador, así que también funciona en el modo local.
- **Coincidencias con fuentes abiertas (opcional):** compara el texto con resúmenes académicos de
  [OpenAlex](https://openalex.org) y artículos de Wikipedia mediante 5-gramas de palabras. A esos servicios
  solo se envían unas palabras clave de cada párrafo, nunca el texto. No es un detector de plagio completo.
- **Inglés (experimental):** se detecta el idioma y se usan expresiones y conectores típicos de los LLM en
  inglés y una calibración propia de Binoculars (`models/calibration_en.json`). El clasificador supervisado
  y el meta-clasificador solo se usan en español.
- **Analizar en mi navegador (opcional):** Binoculars y estilometría se ejecutan en el propio equipo con
  [transformers.js](https://huggingface.co/docs/transformers.js) (WebGPU, o WebAssembly si no hay), así que el
  texto no sale del navegador. La primera vez descarga los modelos cuantizados (~1 GB, q4f16) y quedan en caché.
  No incluye el clasificador supervisado, la paráfrasis ni la lectura de PDF/DOCX.

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
| `IADECCION_CLASIFICADOR` | `1` | `0` desactiva el clasificador supervisado (`models/clasificador/`) |
| `IADECCION_PARAFRASEADOR` | `Qwen/Qwen2.5-1.5B-Instruct` | Modelo de paráfrasis; `0` la desactiva |
| `IADECCION_MAX_TOKENS` | `512` | Tamaño de fragmento para textos largos |
| `IADECCION_SIMILITUD` | `1` | `0` desactiva la búsqueda en OpenAlex y Wikipedia |
| `IADECCION_USER_AGENT` | URL del repositorio | User-Agent de esas consultas (Wikipedia exige un contacto) |

Si cambias de modelos, recalibra (ver abajo).

## API

- `POST /api/analizar` — `{"texto": "..."}`
- `POST /api/analizar-archivo` — formulario con `archivo` (PDF, DOCX, TXT; máx. 10 MB)
- `POST /api/parafrasear` — `{"texto": "párrafo"}` (máx. 400 palabras)
- `POST /api/similitud` — `{"texto": "..."}`: fuentes coincidentes, fracción del texto y oraciones
- `GET /api/salud`

La respuesta incluye `probabilidad_ia`, `veredicto`, `confianza`, `fraccion_texto_marcado`,
las oraciones con su `nivel` (bajo/medio/alto), los `parrafos` con sus `motivos` y `sugerencias`,
y `avisos`.

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

### Clasificador supervisado y meta-clasificador

```bash
# XLM-RoBERTa ajustado con AuTexTification + corpus propio (≈2 h en una GPU de 6 GB)
python research/train_classifier.py
# Pesos aprendidos para combinar Binoculars, estilometría y clasificador
python research/train_meta.py
```

El meta-clasificador se entrena con textos que el clasificador no vio (test del corpus propio y de
AuTexTification) y se evalúa con validación cruzada agrupada por tema. Si `models/clasificador/`
existe pero falta `models/meta.json`, la aplicación usa pesos fijos (50 % Binoculars, 30 %
clasificador, 20 % estilo). La puntuación por oración sigue siendo solo de Binoculars.

La regresión se entrena con clases equilibradas, así que después se desplaza su intercepto para que
el umbral de la aplicación (0.65, "probablemente IA") no marque más del 1 % de los textos humanos
fuera de muestra (`--fpr-objetivo 0.01`; `0` lo desactiva).

#### Resultados (octubre 2026, versión 2 con paráfrasis)

El clasificador se reentrenó añadiendo 1 001 paráfrasis de textos humanos generadas con Salamandra-2B,
EuroLLM-1.7B y Qwen2.5-1.5B (`research/pipeline_parafrasis.sh`), con cuatro enunciados distintos
("con tus propias palabras", "que no se note que está copiado", "mejora la redacción", "que suene humano").
Clasificador XLM-R: AUROC 0.933 en el test del corpus propio y 0.912 en AuTexTification test (antes 0.848).
Sus probabilidades siguen infladas (con p≥0.65 marca a más de la mitad de los humanos), por eso no se usa solo.

Meta-clasificador, validación cruzada agrupada por tema (2 333 textos: test del corpus propio, con
780 paráfrasis, + 1 500 de AuTexTification test). Porcentajes con el umbral 0.65:

| Método | AUROC | TPR @ 1 % FPR | Humanos marcados | IA marcada |
|---|---|---|---|---|
| Binoculars | 0.822 | 0.338 | 1.4 % | 36 % |
| Pesos fijos | 0.850 | 0.330 | 6.3 % | 54 % |
| Meta sin ajustar | 0.867 | 0.348 | 11.4 % | 66 % |
| **Meta ajustado (en uso)** | **0.867** | **0.348** | **1.1 %** | **35 %** |

Por tarea, en el corpus propio (meta ajustado: 0.2 % de humanos marcados):

| Tarea de la IA | AUROC | IA marcada (≥0.65) |
|---|---|---|
| Redactar | 0.998 | 78 % |
| Humanizar | 0.999 | 77 % |
| **Parafrasear** | **0.887** (antes 0.664) | **12 %** (antes 0 %) |

- La paráfrasis ya se distingue (AUROC 0.89), pero con el umbral prudente de la aplicación solo se
  marca el 12 %; el 30 % queda en la zona "incierta" y el 58 % pasa como humano. Sigue siendo el punto débil.
- Las probabilidades del clasificador están saturadas (≥0.99 en el 70 % de los textos humanos), así que
  no sirve como señal propia de paráfrasis: solo se cita como motivo cuando el resultado combinado es IA.
- Las cifras globales bajan respecto a la versión 1 porque el conjunto de evaluación ahora tiene muchas
  más paráfrasis (780 frente a ~55), que son lo más difícil; no son comparables directamente.
- Coeficientes estandarizados: Binoculars −1.24, clasificador 1.25, estilo 0.44.
- La versión 1 (clasificador y meta) está guardada en `data/clasificador_v1` y `data/meta_v1.json`.
- Experimento descartado ("v3", `research/pipeline_clasificador_v3.sh`): suavizado de etiquetas y elegir
  el mejor punto con validación. Quitó la saturación extrema, pero el meta quedó igual (paráfrasis:
  AUROC 0.875 frente a 0.887; marcada 14 % frente a 12 %) y el clasificador generalizó peor a
  AuTexTification (0.765 frente a 0.898). Se guarda en `data/clasificador_v3`.

### Modo navegador

El código del navegador está en `frontend/local/`: `core.js` replica la segmentación, la estilometría y el
informe del backend (`tests/test_local_parity.py` comprueba que coinciden) y `binoculars.js` calcula Binoculars
con los modelos ONNX. La cuantización cambia las puntuaciones, así que tiene su propia calibración:

```bash
cd research/browser && npm install && cd ../..
python research/calibrate_local.py      # genera frontend/local/calibration.json (Node, CPU, ~1 h)
```

Comparación con el servidor en el test del corpus propio (correlación de puntuaciones 0.925):

| Tarea de la IA | AUROC navegador | AUROC servidor | TPR @ 1 % FPR navegador | TPR @ 1 % FPR servidor |
|---|---|---|---|---|
| Todo | 0.845 | 0.879 | 0.537 | 0.701 |
| Redactar | 0.940 | 0.980 | 0.700 | 0.850 |
| Humanizar | 0.981 | 0.999 | 0.615 | 1.000 |
| Parafrasear | 0.448 | 0.478 | 0.000 | 0.000 |

La cuantización q4f16 cuesta algo de precisión; el modo navegador es una alternativa por privacidad,
no un sustituto del servidor.

### Auditoría de falsos positivos

```bash
python research/audit_fp.py      # ~20 min en GPU; guarda models/auditoria_fp.json
```

Todos los textos son humanos y anteriores a ChatGPT; se usa el análisis completo de la aplicación
(meta-clasificador ajustado). "Marcado" = p≥0.65; "incierto o más" = p≥0.35.

| Grupo | n | Marcado como IA | Incierto o más | Prob. media |
|---|---|---|---|---|
| Hablante de herencia (COWS-L2H) | 200 | 0.0 % | 1.0 % | 0.12 |
| Aprendiz de español L2 (COWS-L2H) | 200 | 0.0 % | 9.0 % | 0.14 |
| Académico técnico | 185 | 0.0 % | 0.0 % | 0.05 |
| Académico humanidades/sociales | 179 | 0.6 % | 2.8 % | 0.06 |
| Wikipedia | 108 | 0.0 % | 0.0 % | 0.09 |
| Texto corto (120 palabras) | 200 | 0.0 % | 5.5 % | 0.09 |

- Ningún grupo supera el 1 % de acusaciones con el umbral de la aplicación.
- Los aprendices de español L2 son el grupo que más cae en la zona "incierta" (9 %).
  La interfaz debe seguir dejando claro que "incierto" no es una acusación.
- Los textos de COWS-L2H no se usaron en ningún entrenamiento. Los académicos y Wikipedia salen del
  test del corpus propio, con el que se ajustó el meta-clasificador, así que sus cifras pueden ser
  algo optimistas.

### Inglés (experimental)

```bash
python research/download_data.py --lang en
bash research/pipeline_ingles.sh     # genera models/calibration_en.json
```

Binoculars calibrado con AuTexTification en inglés (train: legal y wiki) y evaluado en dominios no
vistos (test: noticias y reseñas, 3 000 textos):

| Conjunto | AUROC | TPR @ 1 % FPR | Exactitud |
|---|---|---|---|
| Train (calibración) | 0.825 | 0.294 | — |
| **Test (noticias, reseñas)** | **0.920** | **0.552** | **0.843** |

Funciona mejor que en español con el mismo corpus (0.882 de AUROC), pero falla con el generador "A"
de AuTexTification (AUROC 0.74). Falta un corpus propio en inglés con modelos actuales.

## Pruebas

```bash
python -m pytest tests -q
```

## Hoja de ruta

- [x] MVP: Binoculars + estilometría, resaltado por oración, PDF/DOCX, interfaz web
- [ ] Corpus propio con textos de modelos actuales (GPT, Claude, Gemini, Llama) y textos humanizados
  (hecho con modelos abiertos; faltan los comerciales)
- [x] Clasificador supervisado (XLM-RoBERTa) y meta-clasificador entrenado en lugar de pesos fijos
- [x] Auditoría de falsos positivos por grupo (hablantes no nativos, textos técnicos)
- [x] Inferencia en el navegador (transformers.js) para coste cero y privacidad total
- [x] Informe PDF descargable
- [x] Módulo de similitud con fuentes abiertas (OpenAlex y Wikipedia)
- [ ] Inglés: detección de idioma, estilometría y calibración propias (hecho, experimental); falta un
  corpus propio en inglés y un clasificador multilingüe
- [ ] Paráfrasis: más ejemplos de paráfrasis en el entrenamiento (`research/pipeline_parafrasis.sh`)
