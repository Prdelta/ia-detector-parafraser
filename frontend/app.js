const $ = (id) => document.getElementById(id);
let modo = "pegar";
let archivo = null;

const contarPalabras = (t) => (t.match(/[\p{L}]+/gu) || []).length;

$("texto").addEventListener("input", () => {
  $("contador").textContent = `${contarPalabras($("texto").value)} palabras`;
});

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    modo = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    $("panel-pegar").classList.toggle("hidden", modo !== "pegar");
    $("panel-archivo").classList.toggle("hidden", modo !== "archivo");
  });
});

function elegirArchivo(f) {
  archivo = f || null;
  $("drop-text").textContent = archivo ? `📄 ${archivo.name}` : "Arrastra un PDF, DOCX o TXT aquí, o haz clic para elegirlo (máx. 10 MB)";
}
$("archivo").addEventListener("change", (e) => elegirArchivo(e.target.files[0]));
const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => elegirArchivo(e.dataTransfer.files[0]));

function mostrarError(msg) {
  $("error").textContent = msg;
  $("error").classList.toggle("hidden", !msg);
}

// ---------- modo local (frontend/local) ----------

const CARGANDO = "Analizando… puede tardar unos segundos.";
let worker = null;

const local = () => $("modo-local").checked;
try {
  $("modo-local").checked = localStorage.getItem("iadeccion-local") === "1";
} catch {}
// La búsqueda de fuentes se hace en el servidor: no está disponible en el modo local.
const actualizarOpciones = () => $("similitud-opcion").classList.toggle("hidden", local());
$("modo-local").addEventListener("change", () => {
  try {
    localStorage.setItem("iadeccion-local", local() ? "1" : "0");
  } catch {}
  actualizarOpciones();
});
actualizarOpciones();

function analizarLocal(texto) {
  worker ??= new Worker("/static/local/worker.js", { type: "module" });
  return new Promise((resolve, reject) => {
    worker.onmessage = ({ data }) => {
      if (data.tipo === "progreso") {
        $("cargando-texto").textContent =
          data.fase === "descarga"
            ? `Descargando modelos: ${Math.round(data.cargado / 1e6)} de ${Math.round(data.total / 1e6)} MB (solo la primera vez)…`
            : `Analizando en tu equipo… fragmento ${data.hecho} de ${data.total}`;
      } else if (data.tipo === "resultado") resolve(data.datos);
      else reject(new Error(data.mensaje));
    };
    worker.onerror = (e) => reject(new Error(e.message || "El análisis local falló en este navegador."));
    worker.postMessage({ texto });
  });
}

async function analizarServidor(texto) {
  const res = await fetch("/api/analizar", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texto }),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || "Error al analizar.");
  return data;
}

let analisisLocal = false;
const analizarTexto = (texto) => {
  analisisLocal = local();
  return analisisLocal ? analizarLocal(texto) : analizarServidor(texto);
};

$("analizar").addEventListener("click", async () => {
  mostrarError("");
  if (local() && modo !== "pegar") return mostrarError("El análisis en el navegador solo admite texto pegado.");
  let req;
  if (modo === "pegar") {
    const texto = $("texto").value.trim();
    if (contarPalabras(texto) < 80) return mostrarError("Pega al menos 80 palabras.");
  } else {
    if (!archivo) return mostrarError("Elige un archivo primero.");
    const fd = new FormData();
    fd.append("archivo", archivo);
    req = { method: "POST", body: fd };
  }

  $("analizar").disabled = true;
  $("cargando-texto").textContent = CARGANDO;
  $("cargando").classList.remove("hidden");
  try {
    if (modo === "pegar") {
      pintar(await analizarTexto($("texto").value.trim()));
    } else {
      const res = await fetch("/api/analizar-archivo", req);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Error al analizar.");
      analisisLocal = false;
      pintar(data);
    }
  } catch (err) {
    mostrarError(err.message);
  } finally {
    $("analizar").disabled = false;
    $("cargando").classList.add("hidden");
  }
});

// El informe es la vista de impresión del resultado: el navegador lo guarda como PDF sin que el
// texto salga del equipo (funciona igual en el modo local).
$("informe").addEventListener("click", () => {
  const ahora = new Date();
  const datos = [
    `Generado el ${ahora.toLocaleString("es")}`,
    ultimo.archivo ? `Archivo: ${ultimo.archivo}` : null,
    analisisLocal ? "Análisis en el navegador (Binoculars y estilometría)" : "Análisis en el servidor",
  ];
  $("informe-datos").textContent = datos.filter(Boolean).join(". ") + ".";
  const titulo = document.title;
  document.title = `Informe IAdeccion ${ahora.toISOString().slice(0, 10)}`;  // nombre del PDF
  window.print();
  document.title = titulo;
});

$("otra").addEventListener("click", () => {
  $("resultado").classList.add("hidden");
  $("input-card").classList.remove("hidden");
  window.scrollTo({ top: 0, behavior: "smooth" });
});

const pct = (x) => `${Math.round(x * 100)}%`;

function escapar(t) {
  const d = document.createElement("div");
  d.textContent = t;
  return d.innerHTML;
}

function resaltarExpresiones(texto, expresiones) {
  let html = escapar(texto);
  for (const e of new Set(expresiones)) {
    html = html.split(escapar(e)).join(`<span class="phrase">${escapar(e)}</span>`);
  }
  return html;
}

let ultimo = null;

function textoConCambios() {
  return ultimo.parrafos
    .map((p) => {
      const editor = document.querySelector(`textarea[data-parrafo="${p.indice}"]`);
      return editor ? editor.value.trim() : p.texto;
    })
    .join("\n\n");
}

function pintarParrafos(d) {
  // Los títulos (menos de 8 palabras) no se proponen para reescribir.
  const marcados = d.parrafos.filter((p) => p.nivel !== "bajo" && contarPalabras(p.texto) >= 8);
  $("parrafos-marcados").innerHTML = marcados.length
    ? marcados
        .map(
          (p) => `
      <article class="parrafo ${p.nivel}">
        <header><strong>Párrafo ${p.indice + 1}</strong><span class="badge ${p.nivel}">${p.nivel} · ${pct(p.probabilidad)}</span></header>
        <blockquote>${escapar(p.texto)}</blockquote>
        ${p.motivos.length ? `<h4>Por qué suena a IA</h4><ul>${p.motivos.map((m) => `<li>${escapar(m)}</li>`).join("")}</ul>` : ""}
        ${p.sugerencias.length ? `<h4>Cómo mejorarlo</h4><ul>${p.sugerencias.map((s) => `<li>${escapar(s)}</li>`).join("")}</ul>` : ""}
        <label class="editor-label">Tu versión</label>
        <textarea class="editor" data-parrafo="${p.indice}">${escapar(p.texto)}</textarea>
        ${
          d.parafrasis_disponible
            ? `<details class="parafrasis">
                 <summary>Parafrasear automáticamente con IA</summary>
                 <p class="aviso-ia">El resultado seguirá siendo texto generado por IA aunque el detector ya no lo marque. Revisa las normas de tu institución antes de usarlo.</p>
                 <button class="boton boton-papel" data-parafrasear="${p.indice}">Generar paráfrasis</button>
                 <div class="resultado-ia" id="parafrasis-${p.indice}"></div>
               </details>`
            : ""
        }
      </article>`
        )
        .join("")
    : "<p>No hay párrafos marcados como posible IA. 🎉</p>";
  $("reanalizar").classList.toggle("hidden", !marcados.length);
  $("copiar").classList.toggle("hidden", !marcados.length);
}

$("parrafos-marcados").addEventListener("click", async (e) => {
  const usar = e.target.closest("[data-usar]");
  if (usar) {
    const i = usar.dataset.usar;
    document.querySelector(`textarea[data-parrafo="${i}"]`).value = $(`parafrasis-${i}`).dataset.texto;
    return;
  }
  const btn = e.target.closest("[data-parafrasear]");
  if (!btn) return;
  const i = btn.dataset.parafrasear;
  const caja = $(`parafrasis-${i}`);
  const texto = document.querySelector(`textarea[data-parrafo="${i}"]`).value;
  btn.disabled = true;
  caja.innerHTML = "<p class='nota'>Generando…</p>";
  try {
    const res = await fetch("/api/parafrasear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "No se pudo parafrasear.");
    caja.dataset.texto = data.texto;
    caja.innerHTML = `<blockquote>${escapar(data.texto)}</blockquote>
      <p class="aviso-ia">${escapar(data.aviso)}</p>
      <button class="boton boton-papel" data-usar="${i}">Usar esta versión en el editor</button>`;
  } catch (err) {
    caja.innerHTML = `<p class="error">${escapar(err.message)}</p>`;
  } finally {
    btn.disabled = false;
  }
});

$("reanalizar").addEventListener("click", async () => {
  const btn = $("reanalizar");
  btn.disabled = true;
  $("error-reescritura").classList.add("hidden");
  $("cargando-texto").textContent = CARGANDO;
  $("cargando").classList.remove("hidden");
  try {
    pintar(await analizarTexto(textoConCambios()));
  } catch (err) {
    $("error-reescritura").textContent = err.message;
    $("error-reescritura").classList.remove("hidden");
  } finally {
    btn.disabled = false;
    $("cargando").classList.add("hidden");
  }
});

$("copiar").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(textoConCambios());
    $("copiar").textContent = "¡Copiado!";
  } catch {
    $("copiar").textContent = "No se pudo copiar";
  }
  setTimeout(() => ($("copiar").textContent = "Copiar texto con mis cambios"), 2000);
});

function pintar(d) {
  ultimo = d;
  pintarParrafos(d);
  $("input-card").classList.add("hidden");
  $("resultado").classList.remove("hidden");

  $("porcentaje").textContent = pct(d.probabilidad_ia);
  // La marca se desliza sobre la escala humano / incierto / IA (umbrales 35 % y 65 %).
  const marca = $("escala-marca");
  marca.style.left = "0%";
  requestAnimationFrame(() => requestAnimationFrame(() => (marca.style.left = `${d.probabilidad_ia * 100}%`)));
  $("veredicto").textContent = d.veredicto;
  $("confianza").textContent = d.confianza;
  $("fraccion").textContent = pct(d.fraccion_texto_marcado);
  $("palabras").textContent = d.palabras.toLocaleString("es");
  $("idioma").textContent = d.idioma === "en" ? "inglés (experimental)" : "español";
  $("avisos").innerHTML = d.avisos.map((a) => `<li>${escapar(a)}</li>`).join("");

  // Reconstruir párrafos con las oraciones resaltadas.
  const parrafos = [];
  for (const o of d.oraciones) {
    (parrafos[o.parrafo] ||= []).push(
      `<span class="s ${o.nivel}" title="Probabilidad de IA: ${pct(o.probabilidad)}">${resaltarExpresiones(o.texto, o.expresiones)}</span>`
    );
  }
  $("texto-marcado").innerHTML = parrafos.filter(Boolean).map((p) => `<p>${p.join(" ")}</p>`).join("");

  const motivos = d.motivos.length ? d.motivos : ["No se encontraron señales destacadas de IA."];
  $("motivos").innerHTML = motivos.map((m) => `<li>${escapar(m)}</li>`).join("");
  buscarFuentes(d);
  $("tecnico").textContent = JSON.stringify(d.senales, null, 2);
  $("legal").textContent = d.aviso_legal;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ---------- coincidencias con fuentes abiertas ----------

async function buscarFuentes(d) {
  const activa = $("buscar-fuentes").checked && !analisisLocal;
  $("similitud-card").classList.toggle("hidden", !activa);
  if (!activa) return;
  const caja = $("similitud");
  caja.innerHTML = "<p class='nota'>Buscando en OpenAlex y Wikipedia…</p>";
  try {
    const res = await fetch("/api/similitud", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ texto: d.parrafos.map((p) => p.texto).join("\n\n") }),
    });
    const r = await res.json();
    if (!res.ok) throw new Error(r.detail || "No se pudo completar la búsqueda.");
    if (d !== ultimo) return;  // llegó tarde: ya se está mostrando otro análisis
    caja.innerHTML = r.fuentes.length
      ? `<p><strong>${pct(r.fraccion_coincidente)}</strong> del texto coincide con alguna de estas fuentes:</p>
         <ul class="fuentes">${r.fuentes
           .map((f) => `<li><a href="${escapar(f.url)}" target="_blank" rel="noopener">${escapar(f.titulo)}</a>
                        <span class="nota">${f.tipo} · ${pct(f.fraccion)} del texto</span></li>`)
           .join("")}</ul>
         <details><summary>Oraciones que coinciden (${r.oraciones.length})</summary>
           <ul>${r.oraciones.map((o) => `<li>${escapar(o.texto)}</li>`).join("")}</ul></details>
         <p class="nota">${escapar(r.aviso)}</p>`
      : `<p>No se encontraron coincidencias en ${r.fuentes_revisadas} fuentes revisadas.</p>
         <p class="nota">${escapar(r.aviso)}</p>`;
  } catch (err) {
    caja.innerHTML = `<p class="error">${escapar(err.message)}</p>`;
  }
}
