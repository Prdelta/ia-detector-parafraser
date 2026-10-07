// Análisis sin servidor: traducción fiel de segment.py, detectors/stylometry.py,
// rewrite.guidance y analyzer.py. Funciona en el navegador y en Node (calibración y pruebas).
// tests/test_local_parity.py comprueba que da los mismos resultados que el backend.

export const MIN_WORDS = 80;
export const RECOMMENDED_WORDS = 250;
export const MAX_CHARS = 60000;
const W_BINOCULARS = 0.8;
const W_STYLE = 0.2;
const MIN_SENTENCE_TOKENS = 30;
const LEVELS = [[0.35, "bajo"], [0.65, "medio"], [1.01, "alto"]];

// ---------- segmentación ----------

const ABBREVIATIONS = new Set([
  "sr", "sra", "srta", "dr", "dra", "lic", "ing", "prof", "etc", "pág", "págs",
  "p", "pp", "vol", "núm", "no", "cap", "ed", "eds", "fig", "figs", "cf", "vs",
  "ej", "aprox", "art", "inc", "et", "al", "op", "cit", "ibid", "ss", "ud", "uds",
  "mr", "mrs", "ms", "jr", "st", "e.g", "i.e", "approx", "dept", "univ", "ca",
]);
const SENTENCE_END = /[.!?…]+["'»”)\]]*(?=\s+|$)/gu;
const WORD = /\p{L}[\p{L}\p{M}]*(?:[-']\p{L}[\p{L}\p{M}]*)*/gu;
const LAST_WORD = /[\p{L}\p{M}\p{N}_]+$/u;

export function normalize(text) {
  return text
    .replace(/\r\n/g, "\n").replace(/\r/g, "\n").replace(/ /g, " ")
    .replace(/[ \t]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

export const words = (text) => text.match(WORD) || [];

const STOP_ES = new Set("de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o".split(" "));
const STOP_EN = new Set("the of and to in is that for it with as was on are be by this from or an which have not".split(" "));

/** 'es' o 'en' según la proporción de palabras vacías de cada idioma (segment.detect_language). */
export function detectLanguage(text) {
  let es = 0, en = 0;
  for (const w of words(text)) {
    const t = w.toLowerCase();
    if (STOP_ES.has(t)) es += 1;
    if (STOP_EN.has(t)) en += 1;
  }
  return en > es ? "en" : "es";
}

function* paragraphSpans(text) {
  let start = 0;
  for (const m of text.matchAll(/\n\s*\n/g)) {
    yield [start, m.index];
    start = m.index + m[0].length;
  }
  yield [start, text.length];
}

function append(sentences, text, start, end, paragraph) {
  const raw = text.slice(start, end);
  const stripped = raw.trim();
  if (!stripped) return;
  const offset = start + (raw.length - raw.trimStart().length);
  sentences.push({ text: stripped, start: offset, end: offset + stripped.length, paragraph });
}

export function splitSentences(text) {
  const sentences = [];
  let pIdx = 0;
  for (const [pStart, pEnd] of paragraphSpans(text)) {
    const para = text.slice(pStart, pEnd);
    let cursor = pStart;
    for (const m of para.matchAll(SENTENCE_END)) {
      const mStart = pStart + m.index;
      const end = mStart + m[0].length;
      const lw = text.slice(cursor, mStart).match(LAST_WORD);
      if (m[0].startsWith(".") && lw) {
        const word = lw[0].toLowerCase();
        if (ABBREVIATIONS.has(word) || (word.length === 1 && /^\p{L}$/u.test(word))) continue;
        if (/^\p{N}+$/u.test(word) && end < pEnd && /^\p{N}/u.test(text.slice(end, end + 2).trim())) continue;
      }
      append(sentences, text, cursor, end, pIdx);
      cursor = end;
    }
    append(sentences, text, cursor, pEnd, pIdx);
    pIdx += 1;
  }
  return sentences;
}

// ---------- estilometría ----------

const AI_PHRASES = [
  "cabe (?:destacar|mencionar|señalar|resaltar)",
  "es (?:importante|fundamental|crucial|esencial|vital) (?:destacar|señalar|mencionar|tener en cuenta|considerar|recordar|reconocer)",
  "en (?:resumen|conclusión|definitiva|síntesis)",
  "juega(?:n)? un papel (?:crucial|fundamental|clave|importante|esencial|vital)",
  "desempeña(?:n)? un papel (?:crucial|fundamental|clave|importante|esencial|vital)",
  "en el (?:ámbito|contexto|marco|panorama) (?:de|actual|educativo|académico)",
  "en la era (?:digital|actual|moderna)",
  "en el mundo (?:actual|moderno|de hoy)",
  "un (?:aspecto|factor|elemento|pilar) (?:clave|fundamental|crucial|esencial)",
  "no (?:solo|sólo) .{1,60}? sino (?:también|que)",
  "sin lugar a dudas",
  "a medida que",
  "en última instancia",
  "(?:profundizar|ahondar) en",
  "(?:fomentar|potenciar|impulsar) (?:la|el|una|un)",
  "(?:un|una) (?:enfoque|perspectiva|visión) (?:integral|holística|multidimensional|innovador[a]?)",
  "(?:amplia|vasta|rica) (?:gama|variedad)",
  "(?:panorama|paisaje) (?:actual|complejo|cambiante)",
  "(?:navegar|abordar) (?:los|las|el|la) (?:desafíos|retos|complejidades)",
  "(?:desafíos|retos) y oportunidades",
  "de manera (?:significativa|efectiva|integral|eficiente)",
  "(?:transformador|transformadora|revolucionario|revolucionaria)",
  "(?:sinergia|sinergias)",
  "(?:en este sentido|por otro lado|asimismo|además|por lo tanto|en consecuencia),",
];
const AI_RE = new RegExp(AI_PHRASES.map((p) => `(?:${p})`).join("|"), "giu");
// \b de Python es Unicode; en JS se usa una aserción equivalente.
const CONNECTOR_START = new RegExp(
  "^(?:además|asimismo|por otro lado|por otra parte|en este sentido|por lo tanto|" +
    "en consecuencia|finalmente|en primer lugar|en segundo lugar|por último|en resumen|" +
    "en conclusión|sin embargo|no obstante|de igual manera|del mismo modo)(?![\\p{L}\\p{M}\\p{N}_])",
  "iu",
);

// Inglés (experimental), como AI_PHRASES_EN y CONNECTOR_START_EN de stylometry.py.
const AI_PHRASES_EN = [
  "it is (?:important|worth|crucial|essential|vital) to (?:note|mention|consider|remember|highlight|recognize)",
  "it is worth noting",
  "(?:plays?|playing) an? (?:crucial|pivotal|vital|key|significant|central|essential) role",
  "in (?:today[’']s|the modern|the digital|this) (?:world|age|era|landscape)",
  "delv(?:e|es|ed|ing) (?:into|deeper)",
  "a testament to",
  "(?:rich|intricate|vibrant) tapestry",
  "in (?:conclusion|summary)",
  "to sum up",
  "not only .{1,60}? but also",
  "navigat(?:e|es|ing) (?:the )?(?:complexities|challenges|landscape)",
  "(?:ever-evolving|ever-changing|rapidly evolving) (?:landscape|world|field)",
  "foster(?:s|ing)? (?:a|an|the) ",
  "leverag(?:e|es|ing) (?:the|its|their) ",
  "seamless(?:ly)?",
  "multifaceted",
  "pivotal",
  "paramount",
  "an? (?:wide|broad|vast) (?:range|array|variety) of",
  "(?:challenges|obstacles) and opportunities",
  "the realm of",
  "serv(?:e|es) as an? (?:powerful|valuable|crucial|vital|key)",
  "underscor(?:e|es|ing) the importance",
  "(?:furthermore|moreover|additionally|consequently|therefore|in addition),",
];
const AI_RE_EN = new RegExp(AI_PHRASES_EN.map((p) => `(?:${p})`).join("|"), "giu");
const CONNECTOR_START_EN = new RegExp(
  "^(?:furthermore|moreover|additionally|in addition|on the other hand|in this sense|therefore|" +
    "consequently|finally|firstly|secondly|lastly|in summary|in conclusion|however|nevertheless|" +
    "similarly|likewise|overall)(?![\\p{L}\\p{M}\\p{N}_])",
  "iu",
);
const LEXICON = { es: [AI_RE, CONNECTOR_START], en: [AI_RE_EN, CONNECTOR_START_EN] };

// round() de Python: los empates exactos se redondean al par (toFixed los sube).
function round(x, d) {
  const exact = Math.abs(x).toFixed(80);
  const dot = exact.indexOf(".");
  const rest = exact.slice(dot + 1 + d);
  if (/^50*$/.test(rest)) {
    const kept = exact.slice(0, dot + 1 + d).replace(/\.$/, "");
    const last = Number(kept.at(-1));
    const down = Number(kept);
    const step = 10 ** -d;
    return Math.sign(x) * Number((last % 2 === 0 ? down : down + step).toFixed(d));
  }
  return Number(x.toFixed(d));
}
const fmean = (v) => v.reduce((a, b) => a + b, 0) / v.length;
const pstdev = (v) => {
  const m = fmean(v);
  return Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length);
};
const cv = (v) => {
  if (v.length < 2) return 0;
  const m = fmean(v);
  return m ? pstdev(v) / m : 0;
};

function mattr(tokens, window = 50) {
  if (!tokens.length) return 0;
  if (tokens.length <= window) return new Set(tokens).size / tokens.length;
  let total = 0;
  const n = tokens.length - window + 1;
  for (let i = 0; i < n; i++) total += new Set(tokens.slice(i, i + window)).size / window;
  return total / n;
}

export function stylometry(text, sentences, lang = "es") {
  const [aiRe, connectorStart] = LEXICON[lang];
  const tokens = words(text).map((w) => w.toLowerCase());
  const nWords = Math.max(tokens.length, 1);
  const lengths = sentences.map((s) => words(s.text).length);
  const sentLengths = lengths.filter((n) => n > 0);

  const paragraphs = new Map();
  sentences.forEach((s, i) => paragraphs.set(s.paragraph, (paragraphs.get(s.paragraph) || 0) + lengths[i]));

  const sentenceHits = sentences.map((s) => [...s.text.matchAll(aiRe)].map((m) => m[0]));
  const nHits = sentenceHits.reduce((a, h) => a + h.length, 0);
  const connectorStarts = sentences.filter((s) => connectorStart.test(s.text)).length;

  const features = {
    palabras: tokens.length,
    oraciones: sentLengths.length,
    long_media_oracion: sentLengths.length ? round(fmean(sentLengths), 2) : 0,
    variacion_oraciones: round(cv(sentLengths), 3),
    variacion_parrafos: paragraphs.size >= 3 ? round(cv([...paragraphs.values()]), 3) : null,
    riqueza_lexica: round(mattr(tokens), 3),
    frases_ia_por_100: round((100 * nHits) / nWords, 3),
    conectores_inicio: round(connectorStarts / Math.max(sentences.length, 1), 3),
    preguntas_exclamaciones: sentences.reduce((a, s) => a + (s.text.split("?").length - 1) + (s.text.split("!").length - 1), 0),
  };

  let z = 0;
  const reasons = [];
  const burst = features.variacion_oraciones;
  z += -4.0 * (burst - 0.45);
  if (burst < 0.35 && sentLengths.length >= 5) reasons.push("Las oraciones tienen longitudes muy uniformes.");
  const rate = features.frases_ia_por_100;
  z += 1.2 * Math.min(rate, 4.0) - 0.6;
  if (rate >= 1.0) reasons.push(`Uso frecuente de expresiones típicas de IA (${nHits} encontradas).`);
  const conn = features.conectores_inicio;
  z += 3.0 * (conn - 0.15);
  if (conn >= 0.3) reasons.push("Muchas oraciones empiezan con conectores formulaicos.");
  const para = features.variacion_parrafos;
  if (para !== null) {
    z += -2.0 * (para - 0.35);
    if (para < 0.2) reasons.push("Los párrafos tienen extensiones casi idénticas.");
  }
  if (features.preguntas_exclamaciones > 0) z -= 0.3;

  return { score: 1 / (1 + Math.exp(-z)), features, reasons, sentenceHits };
}

// ---------- guía de reescritura ----------

export function guidance(sentences, phraseHits, probability, lang = "es") {
  const reasons = [];
  const suggestions = [];
  const phrases = [...new Set(phraseHits.flat().map((h) => h.toLowerCase()))].sort(pyCompare);
  if (phrases.length) {
    reasons.push("Frases hechas típicas de IA: " + phrases.slice(0, 6).map((p) => `«${p}»`).join(", ") + ".");
    suggestions.push("Sustituye las frases hechas por afirmaciones concretas: ¿qué dato, ejemplo o autor respalda la idea?");
  }
  const connectorStart = LEXICON[lang][1];
  const connectors = sentences.map((s) => s.text.match(connectorStart)).filter(Boolean).map((m) => m[0]);
  if (connectors.length >= 2) {
    reasons.push(`${connectors.length} oraciones empiezan con conectores formulaicos (${connectors.slice(0, 4).join(", ")}).`);
    suggestions.push("Quita conectores de relleno o reordena las ideas para que se enlacen solas.");
  }
  const lengths = sentences.map((s) => words(s.text).length);
  if (lengths.length >= 3 && pstdev(lengths) / Math.max(fmean(lengths), 1) < 0.3) {
    reasons.push("Las oraciones tienen casi la misma longitud.");
    suggestions.push("Varía el ritmo: combina oraciones cortas con otras más largas.");
  }
  if (probability >= 0.65 && !reasons.length) reasons.push("La redacción es muy predecible: vocabulario y estructuras genéricas.");
  if (probability >= 0.35) suggestions.push("Añade tu propia voz: un ejemplo, tu postura o una cita concreta de tus fuentes.");
  return { motivos: reasons, sugerencias: suggestions };
}

// Orden de sorted() de Python: por punto de código.
function pyCompare(a, b) {
  const x = [...a], y = [...b];
  for (let i = 0; i < Math.min(x.length, y.length); i++) {
    const d = x[i].codePointAt(0) - y[i].codePointAt(0);
    if (d) return d;
  }
  return x.length - y.length;
}

// ---------- Binoculars: estadísticas por token ----------

/** Puntuación Binoculars de los tokens seleccionados (null si no hay). */
export function binocularsScore(stats, keep = () => true) {
  let p = 0, x = 0, n = 0;
  for (let i = 0; i < stats.ppl.length; i++) {
    if (!keep(stats.tokenSentence[i])) continue;
    p += stats.ppl[i];
    x += stats.xppl[i];
    n += 1;
  }
  return n ? p / n / (x / n) : null;
}

export const calibratedProbability = (calib, score) => {
  const z = Math.max(Math.min(calib.slope * score + calib.intercept, 50), -50);
  return 1 / (1 + Math.exp(-z));
};

// ---------- informe ----------

/** Normaliza y recorta como analyzer.analyze_text. */
export function prepare(rawText) {
  const text = normalize(rawText).slice(0, MAX_CHARS);
  return { text, sentences: splitSentences(text) };
}

const level = (p) => LEVELS.find(([limit]) => p < limit)[1];
const verdict = (p) =>
  p < 0.35 ? "Probablemente escrito por una persona" : p < 0.65 ? "Resultado incierto o texto mixto" : "Probablemente generado con IA";

/**
 * Mismo informe que /api/analizar. `stats` = {ppl, xppl, tokenSentence} de binoculars.js
 * (null = solo estilometría); `calib` = calibración del par de modelos del navegador.
 */
export function buildReport(rawText, text, sentences, stats, calib, modelId) {
  const nWords = words(text).length;
  const warnings = [];
  if (nWords < MIN_WORDS) throw new Error(`El texto es demasiado corto (${nWords} palabras). Se necesitan al menos ${MIN_WORDS}.`);
  if (nWords < RECOMMENDED_WORDS) warnings.push(`Con menos de ${RECOMMENDED_WORDS} palabras el resultado es poco fiable.`);
  if (rawText.length > MAX_CHARS) warnings.push(`Solo se analizaron los primeros ${MAX_CHARS.toLocaleString("en")} caracteres.`);

  const lang = detectLanguage(text);
  if (lang === "en")
    warnings.push(
      "Texto en inglés: el análisis en inglés es experimental y no usa el clasificador " +
        "supervisado (entrenado solo en español).",
    );

  const style = stylometry(text, sentences, lang);
  const sentenceProbs = sentences.map(() => style.score);
  const signals = { estilometria: { probabilidad: round(style.score, 3), rasgos: style.features } };
  const reasons = [...style.reasons];
  let docProb;

  if (stats) {
    const docScore = binocularsScore(stats);
    // La calibración del navegador es la del español; en inglés los porcentajes son aproximados.
    const calibrado = lang === "es";
    if (!calibrado) warnings.push("El detector aún no está calibrado para este modelo; los porcentajes son aproximados.");
    const binoProb = calibratedProbability(calib, docScore);
    docProb = W_BINOCULARS * binoProb + W_STYLE * style.score;
    signals.binoculars = {
      puntuacion: round(docScore, 4),
      umbral_bajo_fpr: calib.threshold_low_fpr,
      probabilidad: round(binoProb, 3),
      modelo: modelId,
      calibrado,
      en_navegador: true,
    };
    if (docScore < calib.threshold_low_fpr) reasons.unshift("El texto es muy predecible para los modelos de lenguaje (patrón típico de IA).");

    const perSentence = new Array(sentences.length).fill(0);
    for (const s of stats.tokenSentence) perSentence[s] += 1;
    for (let i = 0; i < sentences.length; i++) {
      let lo = i, hi = i;
      const count = () => perSentence.slice(lo, hi + 1).reduce((a, b) => a + b, 0);
      while (count() < MIN_SENTENCE_TOKENS && (lo > 0 || hi < sentences.length - 1)) {
        lo = Math.max(lo - 1, 0);
        hi = Math.min(hi + 1, sentences.length - 1);
      }
      const sScore = binocularsScore(stats, (s) => s >= lo && s <= hi);
      if (sScore !== null) {
        const hitBonus = 0.05 * Math.min(style.sentenceHits[i].length, 2);
        sentenceProbs[i] = Math.min(1, calibratedProbability(calib, sScore) + hitBonus);
      }
    }
  } else {
    docProb = style.score;
    warnings.push("Solo se usó el análisis de estilo (el modelo principal no está disponible). Confianza reducida.");
  }

  const sentenceWords = sentences.map((s) => words(s.text).length);
  let flaggedWords = 0, totalWords = 0;
  sentences.forEach((_, i) => {
    totalWords += sentenceWords[i];
    if (sentenceProbs[i] >= 0.65) flaggedWords += sentenceWords[i];
  });

  const paragraphs = [...new Set(sentences.map((s) => s.paragraph))].sort((a, b) => a - b).map((pIdx) => {
    const idx = sentences.map((s, i) => (s.paragraph === pIdx ? i : -1)).filter((i) => i >= 0);
    const weights = idx.map((i) => Math.max(sentenceWords[i], 1));
    const pProb = idx.reduce((a, i, k) => a + sentenceProbs[i] * weights[k], 0) / weights.reduce((a, b) => a + b, 0);
    return {
      indice: pIdx,
      texto: text.slice(sentences[idx[0]].start, sentences[idx.at(-1)].end),
      probabilidad: round(pProb, 3),
      nivel: level(pProb),
      ...guidance(idx.map((i) => sentences[i]), idx.map((i) => style.sentenceHits[i]), pProb, lang),
    };
  });

  return {
    texto: text,
    idioma: lang,
    parrafos: paragraphs,
    parafrasis_disponible: false,
    probabilidad_ia: round(docProb, 3),
    veredicto: verdict(docProb),
    confianza: confidence(nWords, Boolean(stats), docProb, style.score),
    fraccion_texto_marcado: round(flaggedWords / Math.max(totalWords, 1), 3),
    palabras: nWords,
    oraciones: sentences.map((s, i) => ({
      texto: s.text,
      inicio: s.start,
      fin: s.end,
      parrafo: s.paragraph,
      probabilidad: round(sentenceProbs[i], 3),
      nivel: level(sentenceProbs[i]),
      expresiones: style.sentenceHits[i],
    })),
    motivos: reasons,
    senales: signals,
    avisos: warnings,
    aviso_legal:
      "Este resultado es una estimación estadística y puede equivocarse. " +
      "No constituye prueba de autoría ni debe usarse como única base para sanciones.",
  };
}

function confidence(nWords, hasModel, docProb, styleProb) {
  if (!hasModel || nWords < RECOMMENDED_WORDS) return "baja";
  const decisive = Math.abs(docProb - 0.5) > 0.3;
  const agree = docProb >= 0.5 === styleProb >= 0.5;
  return nWords >= 500 && decisive && agree ? "alta" : "media";
}
