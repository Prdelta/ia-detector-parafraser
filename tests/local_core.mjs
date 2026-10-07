// Ejecuta frontend/local/core.js sobre los textos recibidos por stdin (JSON) para test_local_parity.py.
import { buildReport, prepare } from "../frontend/local/core.js";

let input = "";
for await (const chunk of process.stdin) input += chunk;
const out = JSON.parse(input).map((raw) => {
  const { text, sentences } = prepare(raw);
  try {
    return buildReport(raw, text, sentences, null, null, null);
  } catch (err) {
    return { error: err.message };
  }
});
process.stdout.write(JSON.stringify(out));
