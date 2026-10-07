// Binoculars con transformers.js (navegador o Node). Equivale a backend/app/detectors/binoculars.py,
// pero con modelos ONNX cuantizados y ventanas más cortas para limitar la memoria.
// Recibe el módulo de transformers.js como parámetro para no depender de cómo se importe.

export const OBSERVER = "onnx-community/Qwen2.5-0.5B";
export const PERFORMER = "onnx-community/Qwen2.5-0.5B-Instruct";
export const DTYPE = "q4f16";
export const MODEL_ID = `${OBSERVER}|${PERFORMER}|${DTYPE}`;
const CHUNK = 256;
const CONTEXT_OVERLAP = 64;

export async function createBinoculars(tf, { device = "webgpu", progress } = {}) {
  const opts = { dtype: DTYPE, device, progress_callback: progress };
  const tokenizer = await tf.AutoTokenizer.from_pretrained(OBSERVER, { progress_callback: progress });
  const observer = await tf.AutoModelForCausalLM.from_pretrained(OBSERVER, opts);
  const performer = await tf.AutoModelForCausalLM.from_pretrained(PERFORMER, opts);

  async function logits(model, ids) {
    const n = ids.length;
    const input_ids = new tf.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]);
    const attention_mask = new tf.Tensor("int64", new BigInt64Array(n).fill(1n), [1, n]);
    const out = await model({ input_ids, attention_mask });
    return toFloat32(out.logits);
  }

  /**
   * Pérdida del performer (ppl) y entropía cruzada observer→performer (xppl) de cada token,
   * con la oración a la que pertenece. Se tokeniza por oraciones para conocer esa pertenencia.
   */
  async function tokenStats(text, sentences, onWindow) {
    const ids = [];
    const owner = [];
    sentences.forEach((s, i) => {
      // El espacio previo va con la oración (el BPE de Qwen une " palabra" en un token).
      const start = i === 0 ? 0 : sentences[i - 1].end;
      const end = i + 1 < sentences.length ? s.end : text.length;
      const piece = text.slice(start, end);
      for (const id of tokenizer.encode(piece, { add_special_tokens: false })) {
        ids.push(id);
        owner.push(i);
      }
    });

    const ppl = [], xppl = [], tokenSentence = [];
    const windows = Math.ceil(Math.max(ids.length - 1, 0) / CHUNK);
    for (let w = 0, start = 1; start < ids.length; w++, start += CHUNK) {
      const ctxStart = Math.max(0, start - CONTEXT_OVERLAP);
      const end = Math.min(ids.length, start + CHUNK);
      const window = ids.slice(ctxStart, end);
      const obs = await logits(observer, window);
      const perf = await logits(performer, window);
      const vocab = obs.length / window.length;
      // La fila r predice el token ctxStart + r + 1; solo se puntúan los tokens nuevos.
      for (let r = start - ctxStart - 1; r < window.length - 1; r++) {
        const o = obs.subarray(r * vocab, (r + 1) * vocab);
        const p = perf.subarray(r * vocab, (r + 1) * vocab);
        const lseO = logSumExp(o);
        const lseP = logSumExp(p);
        let cross = 0;
        for (let v = 0; v < vocab; v++) cross += Math.exp(o[v] - lseO) * p[v];
        const target = ctxStart + r + 1;
        ppl.push(lseP - p[ids[target]]);
        xppl.push(lseP - cross);
        tokenSentence.push(owner[target]);
      }
      onWindow?.(w + 1, windows);
    }
    return { ppl: Float64Array.from(ppl), xppl: Float64Array.from(xppl), tokenSentence: Int32Array.from(tokenSentence) };
  }

  return { tokenStats, modelId: MODEL_ID };
}

function logSumExp(row) {
  let max = -Infinity;
  for (let i = 0; i < row.length; i++) if (row[i] > max) max = row[i];
  let sum = 0;
  for (let i = 0; i < row.length; i++) sum += Math.exp(row[i] - max);
  return max + Math.log(sum);
}

// WebGPU puede devolver float16 como Uint16Array (bits crudos).
function toFloat32(tensor) {
  const d = tensor.data;
  if (d instanceof Float32Array) return d;
  if (tensor.type === "float16" && d instanceof Uint16Array) {
    const out = new Float32Array(d.length);
    for (let i = 0; i < d.length; i++) out[i] = halfToFloat(d[i]);
    return out;
  }
  return Float32Array.from(d);
}

function halfToFloat(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const f = h & 0x3ff;
  if (e === 0) return s * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : s * Infinity;
  return s * 2 ** (e - 15) * (1 + f / 1024);
}
