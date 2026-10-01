// The pipeline page: one demo of the gallery (cards.ts), its views computed and served by
// the browser engine (engine.ts) as OME-Zarr, and shown in Neuroglancer.
import { CARDS, type CardLayer, type OpControl, type PipelineCard, type Timeline } from "./cards";
import { Engine, TO_SECONDS } from "./engine";
import type { ViewAxis } from "./types";

const TO_METRES: Record<string, number> = { nanometer: 1e-9, nm: 1e-9, micrometer: 1e-6, um: 1e-6, millimeter: 1e-3, meter: 1, m: 1 };
const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

const engine = new Engine(showCounts);

function status(text: string) { $("state").textContent = text; }
function showCounts() {
  const c = engine.counts;
  $("counts").textContent = `Chunks computed here: ${c.computed} · computing ${engine.running} · waiting ${engine.waiting}`
    + ` · given up by the viewer before their turn: ${c.dropped}${c.failed ? ` · failed ${c.failed}` : ""}`;
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
      return { type: "segmentation", name: l.name, source: `precomputed://${engine.url(l.view!)}mesh`, segments: ["1"], ...(l.colour ? { segmentDefaultColor: l.colour } : {}) };
    }
    if (l.type === "segmentation") {
      return { type: "segmentation", name: l.name, source, selectedAlpha: l.alpha ?? 0.9, ...(l.colour ? { segmentDefaultColor: l.colour } : {}) };
    }
    let range = l.range;
    if (!range && l.percentiles && l.view) range = (await engine.sample(l.view, l.percentiles)) as [number, number];
    const glsl = l.shader ?? (range ? shader(l, range) : undefined);
    return {
      type: "image", name: l.name, source,
      ...(glsl ? { shader: glsl } : {}),
      ...(l.additive ? { blend: "additive" } : {}),
      ...(l.volume ? { volumeRendering: "on" } : {}),
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

interface NgState { layers: { name: string; source: unknown }[]; [k: string]: unknown }
interface NgViewer { state: { toJSON(): NgState; restoreState(s: NgState): void } }

/** Sliders over op parameters: the view is computed again under a new name, and the viewer's
 * layers of it switched to that, keeping the camera. */
function controls(card: PipelineCard, ng: HTMLIFrameElement) {
  const ops: Record<string, Record<string, unknown>[]> = {};
  let busy = Promise.resolve();
  const apply = (c: OpControl, value: number) => {
    const list = (ops[c.view] ??= structuredClone(card.views[c.view].ops ?? []));
    list[c.op] = { ...list[c.op], [c.param]: value };
    busy = busy.then(async () => {
      status("Computing again for what is on screen…");
      const name = await engine.edit(c.view, { ops: structuredClone(list) });
      const viewer = (ng.contentWindow as (Window & { viewer?: NgViewer }) | null)?.viewer;
      if (viewer) {
        const state = viewer.state.toJSON(), old = new RegExp(`/virtual/${engine.page}/${c.view}(~\\d+)?/`);
        for (const l of state.layers) if (typeof l.source === "string" && old.test(l.source)) l.source = l.source.replace(old, `/virtual/${engine.page}/${name}/`);
        viewer.state.restoreState(state);
      }
      status("Chunks are computed as the viewer asks for them.");
    }).catch((e) => status(`Failed: ${(e as Error).message ?? e}`));
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
  await engine.start(card.views, status);  // the data opened; Python still loading
  const ng = $<HTMLIFrameElement>("ng");
  const state = await viewerState(card);
  ng.src = `ng/index.html#!${encodeURIComponent(JSON.stringify(state))}`;
  ng.hidden = false;
  if (card.timeline) showDates(ng, card.timeline, state.displayDimensions);
  controls(card, ng);
  $("empty").hidden = true;
  showCounts();
  Object.assign(window, { engine });  // for a console, and the headless checks
  await engine.ready;  // the viewer's requests wait for it
  status(`Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s. Chunks are computed as the viewer asks for them.`);
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
