// Web Worker del modo "Analizar en mi navegador": el texto nunca sale del equipo.
// Mensajes recibidos: {texto}. Enviados: {tipo: "progreso" | "resultado" | "error", ...}.
import * as tf from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";
import { createBinoculars } from "./binoculars.js";
import { buildReport, prepare } from "./core.js";

tf.env.allowLocalModels = false;

let ready = null;

async function device() {
  // q4f16 necesita WebGPU con shader-f16; si no, WebAssembly (mucho más lento).
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter?.features.has("shader-f16")) return "webgpu";
  } catch {}
  return "wasm";
}

function load() {
  ready ??= (async () => {
    const files = new Map();
    const progress = (e) => {
      if (e.status !== "progress" || !e.total) return;
      files.set(e.file + e.name, [e.loaded, e.total]);
      let loaded = 0, total = 0;
      for (const [l, t] of files.values()) { loaded += l; total += t; }
      postMessage({ tipo: "progreso", fase: "descarga", cargado: loaded, total });
    };
    const dev = await device();
    const [det, calib] = await Promise.all([
      createBinoculars(tf, { device: dev, progress }),
      fetch(new URL("./calibration.json", import.meta.url)).then((r) => r.json()),
    ]);
    return { det, calib, dev };
  })();
  ready.catch(() => (ready = null));
  return ready;
}

onmessage = async ({ data }) => {
  try {
    const { text, sentences } = prepare(data.texto);
    // Comprobar la longitud antes de descargar nada.
    buildReport(data.texto, text, sentences, null, null, null);
    const { det, calib, dev } = await load();
    postMessage({ tipo: "progreso", fase: "analisis", hecho: 0, total: 1 });
    const stats = await det.tokenStats(text, sentences, (hecho, total) =>
      postMessage({ tipo: "progreso", fase: "analisis", hecho, total }));
    const report = buildReport(data.texto, text, sentences, stats, calib, det.modelId);
    report.senales.binoculars.dispositivo = dev;
    if (dev === "wasm") report.avisos.push("Tu navegador no tiene WebGPU: el análisis local se hizo en la CPU.");
    postMessage({ tipo: "resultado", datos: report });
  } catch (err) {
    postMessage({ tipo: "error", mensaje: err.message || String(err) });
  }
};
