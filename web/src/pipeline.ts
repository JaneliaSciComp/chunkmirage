// The pipeline page: one demo of the gallery (cards.ts), its views computed and served by
// the browser engine (engine.ts) as OME-Zarr, and shown in Neuroglancer.
import { CARDS, type CardLayer, type OpControl, type PipelineCard, pythonInput, pythonViews, type Timeline } from "./cards";
import { codeEditor } from "./editor";
import { Engine, TO_SECONDS } from "./engine";
import schema from "./generated/chunkmirage.schema.json";
import type { Control, ViewAxis } from "./types";

const TO_METRES: Record<string, number> = { nanometer: 1e-9, nm: 1e-9, micrometer: 1e-6, um: 1e-6, millimeter: 1e-3, meter: 1, m: 1 };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const engine = new Engine(showCounts);

function status(text: string) { $("state").textContent = text; }
/** The served view whose chunk failed last, if one did (read afresh: a worker sets it). */
function failedView(): string | null { return engine.failure?.view ?? null; }
function showCounts() {
  const c = engine.counts;
  $("counts").textContent = `Chunks computed here: ${c.computed} · computing ${engine.running} · waiting ${engine.waiting}`
    + ` · given up by the viewer before their turn: ${c.dropped}${c.failed ? ` · failed ${c.failed}` : ""}`;
  const f = engine.failure;  // a chunk of the view whose Python is open failed: say why there
  if (f && coding && f.view === served[coding]) showError(f.message);
  // what its Python printed, full resolution first, then each coarser level (and chunk,
  // where a level has several): the code runs on every level the viewer shows
  const now = coding ? served[coding] : undefined;
  const out = (now ? engine.printed.filter((p) => p.view === now) : []).slice(-12)
    .sort((a, b) => a.level - b.level || String(a.at).localeCompare(String(b.at)));
  $("code-output").hidden = !out.length;
  $("code-output").textContent = out.map((p) => `${p.level ? `level ${p.level}` : "full resolution"}`
    + `${p.at.some(Boolean) ? ` (chunk at ${p.at.join(", ")})` : ""}: ${p.print}`).join("\n");
}

function shader(l: CardLayer, range: [number, number]): string {
  const c = l.colour ?? "#ffffff";
  const alpha = l.alpha ?? 1;
  return `#uicontrol invlerp normalized(range=[${range[0]}, ${range[1]}])\n#uicontrol vec3 colour color(default="${c}")\n`
    + (alpha < 1
      ? `void main() { float v = normalized(); emitRGBA(vec4(colour * v, v * ${alpha.toFixed(3)})); }\n`
      : "void main() { emitRGB(colour * normalized()); }\n");
}

/** Neuroglancer's [scale, unit] for an axis: lengths in metres, times in seconds, anything
 * else (degrees of latitude, say) unitless. */
function dimension(a: ViewAxis, voxel: number): [number, string] {
  if (a.unit in TO_METRES) return [voxel * TO_METRES[a.unit], "m"];
  if (a.unit in TO_SECONDS) return [voxel * TO_SECONDS[a.unit], "s"];
  return [voxel, ""];
}

async function viewerState(card: PipelineCard) {
  const first = engine.sources[Object.keys(card.views)[0]], l0 = first.levels[0];
  const order = [2, 1, 0];  // shown x, y, z: the last axis across
  const names = order.map((a) => first.axes[a].name);
  const dims = Object.fromEntries(order.map((a) => [first.axes[a].name, dimension(first.axes[a], l0.voxel[a])]));
  const url = (view: string) => `zarr3://${engine.url(view)}`;
  const layers = await Promise.all(card.layers.map(async (l) => {
    const source = l.view ? url(l.view) : l.url!;
    if (l.type === "mesh") {  // the view's surface: chunkmirage's mesh layout, segment 1
      return {
        type: "segmentation", name: l.name, source: `precomputed://${engine.url(l.view!)}mesh`, segments: ["1"],
        ...(l.colour ? { segmentDefaultColor: l.colour } : {}), ...(l.detail ? { meshRenderScale: l.detail } : {}),
      };
    }
    if (l.type === "segmentation") {
      return { type: "segmentation", name: l.name, source, selectedAlpha: l.alpha ?? 0.9, ...(l.colour ? { segmentDefaultColor: l.colour } : {}) };
    }
    let range = l.range;
    if (!range && l.percentiles && l.view) range = (await engine.sample(l.view, l.percentiles)) as [number, number];
    const glsl = l.shader ?? (range ? shader(l, range) : undefined);
    // a view with a channel axis (its ops add one): the shader's channels, c^, then the
    // view's axes in its own order (Neuroglancer maps them by position)
    const own = l.view ? engine.sources[l.view] : first;
    const channelled = l.channels
      ? { url: source, transform: { outputDimensions: { "c^": [1, ""], ...Object.fromEntries(own.axes.map((a, i) => [a.name, dimension(a, own.levels[0].voxel[i])])) } } }
      : source;
    return {
      type: "image", name: l.name, source: channelled,
      ...(glsl ? { shader: glsl } : {}),
      ...(l.additive ? { blend: "additive" } : {}),
      ...(l.volume ? { volumeRendering: "on" } : {}),
      ...(l.samples ? { volumeRenderingDepthSamples: l.samples } : {}),
      ...(l.gain !== undefined ? { volumeRenderingGain: l.gain } : {}),
    };
  }));
  const main = document.querySelector("main")!;
  const zoom = card.zoom * dims[names[0]][0] / Math.min(...Object.values(dims).map((d) => d[0]));
  return {
    // the viewer's position is physical, in voxels: the source's origin is part of it
    dimensions: dims, position: order.map((a) => card.position[a] + 0.5 + l0.origin[a] / l0.voxel[a]),
    // Neuroglancer counts zoom in the smallest scale among the dimensions, whatever their
    // units (a day in seconds beside 0.01 degrees, say): the card's is in voxels of x
    displayDimensions: names, crossSectionScale: zoom,
    ...(card.layouts?.includes("3d") ? { projectionScale: zoom * 900, ...(card.turn ? { projectionOrientation: card.turn } : {}) } : {}),  // the 3-D panel's height, the same voxels
    ...(card.orientation ? { crossSectionOrientation: card.orientation } : {}),
    ...(card.playback ? { velocity: { [card.playback.axis]: { velocity: card.playback.velocity, atBoundary: "stop", paused: true } } } : {}),
    ...(card.depth ? { projectionDepth: -card.depth } : {}),  // negative: in view heights
    ...(card.boxes === false ? { showDefaultAnnotations: false } : {}),
    crossSectionBackgroundColor: "#000000", showAxisLines: false, layers,
    layout: card.panels.length === 1
      ? { type: "viewer", layers: card.panels[0], layout: card.layouts?.[0] ?? "xy" }
      : { type: "row", children: card.panels.map((names, i) => ({ type: "viewer", layers: names, layout: card.layouts?.[i] ?? "xy" })) },
    selectedLayer: { visible: false }, size: [main.clientWidth, main.clientHeight],
  };
}

/** The date at the viewer's position on a timeline axis, and what happened that day, kept
 * up to date as the viewer moves or plays (the viewer itself shows time only in seconds). */
function showDates(ng: HTMLIFrameElement, t: Timeline, names: string[]) {
  const axis = names.indexOf(t.axis), start = Date.parse(`${t.start}T00:00:00Z`);
  const fmt = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "long", year: "numeric", timeZone: "UTC" });
  let last = "";
  const update = (position: Float32Array) => {
    const day = new Date(start + Math.floor(position[axis]) * 86400e3), iso = day.toISOString().slice(0, 10);
    if (iso === last) return;
    last = iso;
    $("date").textContent = fmt.format(day);
    $("event").textContent = t.events.find((e) => e.from <= iso && iso <= (e.to ?? e.from))?.text ?? "";
  };
  $("when").hidden = false;
  const attach = () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const viewer = (ng.contentWindow as any)?.viewer;
    if (!viewer?.position) return void setTimeout(attach, 200);
    viewer.position.changed.add(() => update(viewer.position.value));
    update(viewer.position.value);
  };
  attach();
}

type NgLayer = { name: string; type: string; source: unknown; [k: string]: unknown };
interface NgState { layers: NgLayer[]; [k: string]: unknown }
interface NgViewer { state: { toJSON(): NgState; restoreState(s: NgState): void } }

type OpSpec = Record<string, unknown>;
const edited: Record<string, OpSpec[]> = {};  // each view's ops as the sliders set them
const python: Record<string, OpSpec> = {};    // the visitor's python op, after a view's ops
const served: Record<string, string> = {};    // the name each edited view is served under
let firstLayers: Record<string, NgLayer> = {};  // the layers as the card made them
let busy: Promise<unknown> = Promise.resolve();  // edits, one at a time (the first after the viewer is up)

const opsOf = (card: PipelineCard, view: string): OpSpec[] =>
  [...(edited[view] ?? card.views[view].ops ?? []), ...(python[view] ? [python[view]] : [])];

/** View `view` served again with its edits under a new name (returned), and the viewer's
 * layers of it switched to that, keeping the camera. One edit at a time. */
function reserve(card: PipelineCard, ng: HTMLIFrameElement, view: string): Promise<string> {
  const run = busy.then(async () => {
    status("Computing again for what is on screen…");
    const name = await engine.edit(view, { ops: structuredClone(opsOf(card, view)) });
    served[view] = name;
    swapLayers(card, await viewerIn(ng), view, name);
    status("Chunks are computed as the viewer asks for them.");
    return name;
  });
  busy = run.catch(() => {});
  return run;
}

/** Neuroglancer in `ng`, once its page has made it: an edit can come while the iframe is
 * still loading (the code panel is live from the start). */
function viewerIn(ng: HTMLIFrameElement): Promise<NgViewer | undefined> {
  const t0 = performance.now();
  return new Promise((resolve) => {
    const look = () => {
      const viewer = (ng.contentWindow as (Window & { viewer?: NgViewer }) | null)?.viewer;
      if (viewer?.state || performance.now() - t0 > 60_000) resolve(viewer);
      else setTimeout(look, 200);
    };
    look();
  });
}

/** The viewer's layers of `view` reading it as served under `name`: each as the card made it
 * (its shader, colours), unless what the ops make now needs another kind of layer (labels: a
 * segmentation; floats: an image) or another value range (another dtype), when it is a
 * plain layer of that kind, for Neuroglancer to set up. */
function swapLayers(card: PipelineCard, viewer: NgViewer | undefined, view: string, name: string) {
  if (!viewer) return;
  const old = new RegExp(`/virtual/${engine.page}/${view}(~\\d+)?/`), to = `/virtual/${engine.page}/${name}/`;
  const url = (src: unknown) => typeof src === "string" ? src : (src as { url?: string } | null)?.url ?? "";
  const moved = (src: unknown) => typeof src === "string" ? src.replace(old, to) : { ...(src as object), url: url(src).replace(old, to) };
  const now = engine.infos[name], was = engine.infos[view];
  const state = viewer.state.toJSON();
  state.layers = state.layers.map((l) => {
    if (!old.test(url(l.source))) return l;
    const made = card.layers.find((c) => c.name === l.name);
    if (!made || made.type === "mesh") return { ...l, source: moved(l.source) };
    // a segmentation the card made of its code's labels, without them (another dtype): an image
    const type = now.kind === "label" ? "segmentation"
      : now.out.startsWith("float") || (made.type === "segmentation" && now.out !== was.out) ? "image" : made.type;
    if (now.out === was.out) {
      if (l.type === type) return { ...l, source: moved(l.source) };  // the same kind of layer: as the visitor has it (hidden, its shader's controls)
      if (type === made.type) return { ...firstLayers[l.name], source: moved(firstLayers[l.name].source) };  // back to the card's
    }
    return { type, name: l.name, source: moved(l.source), ...(l.visible === false ? { visible: false } : {}) };
  });
  viewer.state.restoreState(state);
}

/** Sliders over op parameters: the view is computed again under a new name, and the viewer's
 * layers of it switched to that, keeping the camera. */
function controls(card: PipelineCard, ng: HTMLIFrameElement) {
  const apply = (c: OpControl, value: number) => {
    const list = (edited[c.view] ??= structuredClone(card.views[c.view].ops ?? []));
    list[c.op] = { ...list[c.op], [c.param]: value };
    reserve(card, ng, c.view).catch((e) => status(`Failed: ${(e as Error).message ?? e}`));
  };
  for (const c of card.controls ?? []) {
    const row = document.createElement("label");
    row.className = "control";
    row.innerHTML = `<span></span><output></output><input type="range">`;
    row.querySelector("span")!.textContent = c.label;
    const input = row.querySelector("input")!, out = row.querySelector("output")!;
    const value = Number((card.views[c.view].ops ?? [])[c.op]?.[c.param] ?? c.min);
    Object.assign(input, { min: String(c.min), max: String(c.max), step: String(c.step), value: String(value) });
    const show = () => { out.textContent = `${input.value}${c.unit}`; };
    show();
    input.addEventListener("input", show);
    input.addEventListener("change", () => apply(c, Number(input.value)));
    $("controls").append(row);
  }
  $("controls").hidden = !card.controls?.length;
}

// ---- the visitor's Python: a python op after a view's ops, edited in the page -----------------

const TEMPLATE = (schema.$defs as unknown as Record<string, { properties: { code: { default: string } } }>).PythonOp.properties.code.default;
let coding: string | null = null;  // the view whose Python is open

/** A Python error as the editor shows it: the code's line and what went wrong, not the
 * worker's whole traceback. */
function showError(message: string) {
  const lines = message.trim().split("\n");
  const invalid = lines.find((l) => /^\s*Value error, /.test(l));
  const at = lines.filter((l) => l.includes('"<python op>"')).pop()?.trim().replace(/^File "<python op>", /, "");
  $("code-error").textContent = invalid
    ? invalid.trim().replace(/^Value error, /, "").replace(/ \[type=.*$/, "")
    : [at, lines[lines.length - 1]].filter(Boolean).join(": ");
  $("code-error").hidden = false;
}

/** What "copy as plugin" gives: a module whose op is this code, its sliders its parameters
 * (as the control page's button gives). */
function pluginSource(code: string, name = "my_op"): string {
  const cls = name.split("_").map((w) => w[0].toUpperCase() + w.slice(1)).join("");
  const body = code.replace(/\\/g, "\\\\").replace(/'''/g, "\\'\\'\\'");
  return `# ${name}.py: an op made from code written for chunkmirage's python op. Register it in\n`
    + `# pyproject.toml:  [project.entry-points."chunkmirage.ops"]  ${name} = "${name}:${cls}"\n`
    + `from chunkmirage.ops import python_op\n\n${cls} = python_op("${name}", '''\\\n${body.endsWith("\n") ? body : `${body}\n`}''')\n`;
}

/** `code` with each `#uicontrol` line's default the value its slider has now (`params`), so
 * the code saved runs from the command line as it does here. */
function withParams(code: string, params: Record<string, unknown> = {}): string {
  return code.split("\n").map((line) => {
    const m = /^(\s*#uicontrol\s+\w+\s+(\w+)\s+\w+\()(.*)(\)\s*)$/.exec(line);
    if (!m || !(m[2] in params)) return line;
    const args = m[3].split(",").map((a) => a.trim()).filter((a) => a && !a.startsWith("default="));
    return `${m[1]}${[...args, `default=${String(params[m[2]])}`].join(", ")}${m[4]}`;
  }).join("\n");
}

/** Python a link carries (`#python=`): python ops by view, the card's views only. */
function linked(card: PipelineCard): Record<string, OpSpec> {
  const m = /[#&]python=([^&]*)/.exec(location.hash);
  if (!m) return {};
  try {
    const got = JSON.parse(decodeURIComponent(m[1])) as Record<string, OpSpec>;
    return Object.fromEntries(Object.entries(got)
      .filter(([v, o]) => v in card.views && o?.op === "python" && typeof o.code === "string")
      .map(([v, o]) => [v, { op: "python", code: o.code, ...(o.params ? { params: o.params } : {}) }]));
  } catch { return {}; }
}

/** The code panel: pick a view, add Python to it and edit it; the view is served again a
 * moment after each change, and the code's #uicontrol lines become sliders under it. Python
 * from a link is shown, and runs when the visitor says so. */
function code(card: PipelineCard, ng: HTMLIFrameElement): () => void {
  const views = pythonViews(card);
  if (!views.length) return () => {};
  const pick = $<HTMLSelectElement>("code-view"), text = $<HTMLTextAreaElement>("code-text");
  const editor = codeEditor(text);  // highlighted, over the textarea the panel reads
  Object.assign(window, { editor: editor.view });  // for a console, and the headless checks
  const names = (v: string) => card.layers.filter((l) => l.view === v).map((l) => l.name).join(", ");
  for (const v of views) pick.append(new Option(`${names(v)}: ${pythonInput(card, v)}`, v));
  pick.hidden = views.length === 1;
  const at = (view: string) => String(opsOf(card, view).length - 1);  // the python op's index
  let timer = 0;
  const sliders = (view: string, controls: Control[]) => {
    const box = $("code-sliders");
    box.replaceChildren();
    for (const c of controls) {
      const row = document.createElement("label");
      row.className = "control";
      row.innerHTML = "<span></span><output></output><input>";
      row.querySelector("span")!.textContent = c.name;
      const input = row.querySelector("input")!, out = row.querySelector("output")!;
      const value = (python[view].params as Record<string, number | boolean> | undefined)?.[c.name] ?? c.default;
      if (c.type === "bool") Object.assign(input, { type: "checkbox", checked: Boolean(value) });
      else Object.assign(input, { type: "range", min: String(c.min), max: String(c.max), step: c.step === null ? "any" : String(c.step), value: String(value) });
      const show = () => { out.textContent = c.type === "bool" ? "" : input.value; };
      show();
      input.addEventListener("input", show);
      input.addEventListener("change", () => {
        const v = c.type === "bool" ? input.checked : Number(input.value);
        python[view] = { ...python[view], params: { ...(python[view].params as object | undefined), [c.name]: v } };
        void run(view);
      });
      box.append(row);
    }
  };
  const run = async (view: string) => {
    engine.failure = null;
    if (coding === view) $("code-error").hidden = true;
    try {
      const name = await reserve(card, ng, view);
      // an error of the view as served before, shown meanwhile, is not this code's
      if (coding === view && failedView() !== name) $("code-error").hidden = true;
      if (coding === view && python[view]) sliders(view, engine.infos[name].controls?.[at(view)] ?? []);
    } catch (e) {
      if (coding === view) showError(String((e as Error).message ?? e));
      status("The view's Python failed: see under it.");
    }
  };
  // Python from a link, shown but not in any view's ops (so not run, by an edit of any
  // kind) until the visitor presses "Run it"
  const pending = linked(card);
  const open = (view: string) => {
    coding = view;
    pick.value = view;
    $("code-where").textContent = `Gets this layer's data ${pythonInput(card, view)}; the layer shows what it returns.`;
    const shown = pending[view] ?? python[view];
    $("code-edit").hidden = !shown;
    $("code-add").hidden = Boolean(shown);
    $("code-trust").hidden = !Object.keys(pending).length;
    document.body.classList.toggle("coding", Boolean(shown));
    $("code-error").hidden = true;
    $("code-sliders").replaceChildren();
    if (!shown) return;
    editor.set(String(shown.code));
    if (!pending[view] && served[view]) sliders(view, engine.infos[served[view]]?.controls?.[at(view)] ?? []);
  };
  pick.addEventListener("change", () => open(pick.value));
  $("code-add").addEventListener("click", () => {
    const view = coding!;
    python[view] = { op: "python", code: TEMPLATE };
    open(view);
    void run(view);
  });
  $("code-remove").addEventListener("click", () => {
    const view = coding!;
    delete pending[view];
    delete python[view];
    open(view);
    void run(view);
  });
  text.addEventListener("input", () => {
    const view = coding!;
    if (pending[view]) { pending[view] = { ...pending[view], code: text.value }; return; }
    python[view] = { ...python[view], code: text.value };
    clearTimeout(timer);
    timer = window.setTimeout(() => void run(view), 700);
  });
  $("code-run").addEventListener("click", () => {
    const views = Object.keys(pending);
    for (const v of views) { python[v] = pending[v]; delete pending[v]; }
    open(coding!);
    for (const v of views) void run(v);
  });
  $("code-share").addEventListener("click", () => {
    const code = encodeURIComponent(JSON.stringify({ ...python, ...pending }));
    void navigator.clipboard.writeText(`${location.origin}${location.pathname}?card=${card.id}#python=${code}`);
    status("Copied a link to this demo with your Python: it opens showing the code, run when the visitor chooses.");
  });
  $("code-plugin").addEventListener("click", () => {
    void navigator.clipboard.writeText(pluginSource(text.value));
    status("Copied a module defining the op, for the chunkmirage.ops entry point.");
  });
  // the chat: what the model is told of the layer with each message, and the code it writes,
  // into the editor and run (a link's code it was not shown is dropped, as by "Remove")
  let asked: string | null = null;  // the view a question was about: the code that answers it is its
  const context = () => {
    const view = coding!, src = engine.sources[view];
    if (!src) throw new Error("the data is still opening: ask once the status says Python is ready");
    asked = view;
    const l0 = src.levels[0], chunk = card.views[view].chunk;
    const ops = card.views[view].ops ?? [], op = pending[view] ?? python[view];
    const input = !ops.length ? src.dtype : card.code?.view === view ? "" : engine.infos[view]?.out ?? "";
    const whole = l0.shape.every((n, i) => (chunk.at(i - l0.shape.length) ?? 0) >= n);
    const shown = (id: string) => $(id).hidden ? "" : $(id).textContent ?? "";
    return [
      `Layer ${names(view)}: apply gets its data ${pythonInput(card, view)}${input ? `, ${input}` : ""}${src.channels > 1 ? `, ${src.channels} channels` : ""}.`,
      `Axes ${src.axes.map((a) => a.name).join(", ")}; full resolution ${l0.shape.join(" x ")} voxels of ${l0.voxel.map((v) => +v.toPrecision(4)).join(" x ")} ${src.axes.at(-1)!.unit};`
        + ` ${src.levels.length} levels: ${src.levels.map((l) => l.shape.join(" x ")).join(", ")}.`,
      `Chunks: ${chunk.join(" x ")} voxels${whole ? " (at full resolution the layer is one chunk)" : ""}.`,
      op ? `The code now${pending[view] ? " (from a link, shown but not run yet)" : ""}:\n\`\`\`python\n${String(op.code).trimEnd()}\n\`\`\`` : "No code yet: the layer shows its data as is.",
      op?.params ? `Its sliders: ${Object.entries(op.params as object).map(([k, v]) => `${k}=${v}`).join(", ")}.` : "",
      shown("code-error") ? `It fails: ${shown("code-error")}` : "",
      shown("code-output") ? `It printed:\n${shown("code-output")}` : "",
    ].filter(Boolean).join("\n");
  };
  const put = (code: string) => {
    const view = asked ?? coding!;  // the layer asked about, if the picker moved meanwhile
    delete pending[view];
    python[view] = { ...python[view], op: "python", code };
    open(view);
    void run(view);
  };
  const ask = $<HTMLDetailsElement>("ask");
  ask.addEventListener("toggle", () => {
    void import("./chat").then((m) => m.chatPanel(ask, { context, put }))
      .catch((e) => status(`The chat failed to load: ${(e as Error).message ?? e}`));
  }, { once: true });
  // a link's Python, else the card's, else a layer as read (the data itself), else the first
  open(Object.keys(pending)[0] ?? card.code?.view ?? views.find((v) => !card.views[v].ops?.length) ?? views[0]);
  $("code").hidden = false;
  // a card's own Python runs from the start: its sliders, once Python has read them
  return () => { if (coding && served[coding] && !pending[coding]) open(coding); };
}

async function start() {
  const id = new URLSearchParams(location.search).get("card") ?? "contacts";
  const card = CARDS.find((c): c is PipelineCard => c.kind === "pipeline" && c.id === id);
  if (!card) { status(`No pipeline demo called "${id}".`); return; }
  document.title = card.title;
  $("title").textContent = card.title;
  $("blurb").textContent = card.blurb;
  $("data").textContent = card.data;
  $("command").textContent = card.command;
  const t0 = performance.now();
  if (card.code) {  // the card's own Python, in its view from the start
    python[card.code.view] = { op: "python", code: card.code.code };
    served[card.code.view] = card.code.view;
  }
  const views = Object.fromEntries(Object.keys(card.views).map((v) => [v, { ...card.views[v], ops: opsOf(card, v) }]));
  const ng = $<HTMLIFrameElement>("ng");
  // the code panel at once (the code is the page's), its edits waiting for the data and viewer
  let up = () => {};
  busy = new Promise<void>((r) => { up = r; });
  const planned = code(card, ng);
  await engine.start(views, status);  // the data opened; Python still loading
  const state = await viewerState(card);
  firstLayers = Object.fromEntries(state.layers.map((l) => [l.name, l as NgLayer]));
  if (card.code) {  // the code the command reads, as it is now
    const view = card.code.view, file = `${view}.py`, save = $<HTMLButtonElement>("save-code");
    Object.assign(save, { hidden: false, textContent: `Save ${file}` });
    $("save-note").hidden = false;
    save.addEventListener("click", () => {
      const op = python[view] ?? { code: card.code!.code };
      const a = Object.assign(document.createElement("a"), { download: file,
        href: URL.createObjectURL(new Blob([withParams(String(op.code), op.params as Record<string, unknown>)], { type: "text/x-python" })) });
      a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 0);  // after the download has started
    });
  }
  ng.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`;
  ng.hidden = false;
  if (card.timeline) showDates(ng, card.timeline, state.displayDimensions);
  up();
  controls(card, ng);
  consumers(card);
  $("empty").hidden = true;
  showCounts();
  Object.assign(window, { engine });  // for a console, and the headless checks
  await engine.ready;  // the viewer's requests wait for it
  planned();
  status(`Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s. Chunks are computed as the viewer asks for them.`);
}

/** The "Read by" picker: the same views read by Neuroglancer, a web map or GDAL, each made
 * the first time it is picked. */
function consumers(card: PipelineCard) {
  const c = card.consumers;
  if (!c) return;
  const pick = $<HTMLSelectElement>("consumer"), note = $("consumer-note");
  const notes: Record<string, string> = {
    neuroglancer: "Neuroglancer reads the views as OME-Zarr.",
    map: "OpenLayers, a GIS web map, reads the views as GeoZarr: the same chunks, laid out as map tiles.",
    gdal: "GDAL, the library under most GIS software, compiled to WebAssembly: it opens a view as a zarr and writes a GeoTIFF.",
  };
  for (const [v, t] of [["neuroglancer", "Neuroglancer"], ["map", "OpenLayers (web map)"], ["gdal", "GDAL (WebAssembly)"]]) pick.append(new Option(t, v));
  const made = new Set<string>();
  const show = async () => {
    const which = pick.value;
    note.textContent = `${notes[which]} Desktop readers: ${c.python}`;
    $("ng").hidden = which !== "neuroglancer";
    $("map").hidden = which !== "map";
    $("gdal").hidden = which !== "gdal";
    if (made.has(which) || which === "neuroglancer") return;
    made.add(which);
    const m = await import("./consumers");
    if (which === "map") await m.showMap($("map"), engine, c, card.position.slice(1), card.zoom);
    else m.gdalPanel($("gdal"), engine, c);
  };
  pick.addEventListener("change", () => void show().catch((e) => status(`Failed: ${(e as Error).message ?? e}`)));
  $("consumers").hidden = false;
  void show();
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
