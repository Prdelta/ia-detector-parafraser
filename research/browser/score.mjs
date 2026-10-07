// Puntúa con el Binoculars del navegador (frontend/local) textos JSONL de stdin: {"id", "text"} por línea.
// Escribe {"id", "score"} por línea en stdout. Uso: node score.mjs < textos.jsonl > puntuaciones.jsonl
import * as tf from "@huggingface/transformers";
import readline from "node:readline";
import { createBinoculars } from "../../frontend/local/binoculars.js";
import { binocularsScore, prepare } from "../../frontend/local/core.js";

const det = await createBinoculars(tf, { device: process.env.DEVICE || "cpu" });
const t0 = Date.now();
let n = 0;
for await (const line of readline.createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  const { id, text } = JSON.parse(line);
  const { text: clean, sentences } = prepare(text);
  const stats = await det.tokenStats(clean, sentences);
  process.stdout.write(JSON.stringify({ id, score: binocularsScore(stats) }) + "\n");
  if (++n % 50 === 0) console.error(`  ${n} textos (${((Date.now() - t0) / 1000 / n).toFixed(2)} s/texto)`);
}
