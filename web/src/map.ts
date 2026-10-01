// The map page: a demo of the gallery (cards.ts) whose views the browser engine (engine.ts)
// computes and serves as GeoZarr, drawn by OpenLayers, a GIS web map that knows nothing of
// chunkmirage: it reads the page's chunks as it reads any GeoZarr store. Everything on the
// map is computed: it fits the chosen site and goes no further out. Sliders either
// recompute a view with other op parameters (for the tiles on screen) or restyle a layer
// on the GPU; the site picker recomputes the views from another source.
import "ol/ol.css";
import OlMap from "ol/Map";
import View from "ol/View";
import { ScaleLine, defaults as defaultControls } from "ol/control";
import WebGLTileLayer from "ol/layer/WebGLTile";
import { Projection, addProjection } from "ol/proj";
import GeoZarr from "ol/source/GeoZarr";
import { CARDS, type MapCard, type MapControl } from "./cards";
import { Engine } from "./engine";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const engine = new Engine(showCounts);

function status(text: string) { $("state").textContent = text; }
function showCounts() {
  const c = engine.counts;
  $("counts").textContent = `Tiles computed here: ${c.computed} · computing ${engine.running} · waiting ${engine.waiting}`
    + ` · given up by the map before their turn: ${c.dropped}${c.failed ? ` · failed ${c.failed}` : ""}`;
}

const geozarr = (view: string) => new GeoZarr({ url: engine.geoUrl(view), bands: [view], transition: 0 });

/** A slider for `c`; `apply` gets its value when it is let go (op parameters) or as it
 * moves (style variables, which cost nothing to change). */
function slider(c: MapControl, value: number, live: boolean, apply: (v: number) => void) {
  const row = document.createElement("label");
  row.className = "control";
  row.innerHTML = `<span></span><output></output><input type="range">`;
  row.querySelector("span")!.textContent = c.label;
  const input = row.querySelector("input")!, out = row.querySelector("output")!;
  Object.assign(input, { min: String(c.min), max: String(c.max), step: String(c.step), value: String(value) });
  const show = () => { out.textContent = `${input.value}${c.unit}`; };
  show();
  input.addEventListener("input", () => { show(); if (live) apply(Number(input.value)); });
  if (!live) input.addEventListener("change", () => apply(Number(input.value)));
  $("controls").append(row);
}

async function start() {
  const id = new URLSearchParams(location.search).get("card") ?? "moon";
  const card = CARDS.find((c): c is MapCard => c.kind === "map" && c.id === id);
  if (!card) { status(`No map demo called "${id}".`); return; }
  document.title = card.title;
  $("title").textContent = card.title;
  $("blurb").textContent = card.blurb;
  $("data").textContent = card.data;
  $("command").textContent = card.command;
  const t0 = performance.now();
  await engine.start(card.views, status);  // the data opened; Python still loading
  void engine.ready.then(() => status(`Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s. Tiles are computed as the map asks for them.`));

  // the data's own projection, so nothing is reprojected
  const projection = new Projection({ code: card.projection.code, units: "m", extent: card.projection.extent });
  addProjection(projection);
  for (const v of Object.keys(card.views)) engine.proj[v] = card.projection.code;
  const shown: Record<string, string> = Object.fromEntries(Object.keys(card.views).map((v) => [v, v]));
  const layers = card.layers.map((l) => new WebGLTileLayer({ source: geozarr(l.view), style: l.style, visible: l.visible ?? true }));
  const map = new OlMap({ target: "map", layers, controls: defaultControls().extend([new ScaleLine()]) });
  /** A view of the site: all of it on screen, no further out, down to quarter pixels. */
  const fit = () => {
    const bbox = engine.sources[shown[card.sites.views[0]]].geo!.bbox, [w, h] = map.getSize() ?? [800, 600];
    const whole = Math.max((bbox[2] - bbox[0]) / w, (bbox[3] - bbox[1]) / h);
    map.setView(new View({
      projection, extent: bbox, constrainOnlyCenter: false, showFullExtent: true,
      center: [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], resolution: whole, maxResolution: whole, minResolution: 1.25,
    }));
  };
  fit();
  $("empty").hidden = true;

  // the site: every view that reads it, computed again from the chosen source
  const pick = document.createElement("label");
  pick.className = "control";
  pick.innerHTML = `<span></span><select></select>`;
  pick.querySelector("span")!.textContent = card.sites.label;
  const select = pick.querySelector("select")!;
  for (const o of card.sites.options) select.append(new Option(o.name, o.url));
  select.value = card.views[card.sites.views[0]].source;
  let busy = Promise.resolve();
  const recompute = (view: string, changes: Parameters<Engine["edit"]>[1]) => {
    busy = busy.then(async () => {
      status("Computing…");
      shown[view] = await engine.edit(view, changes);
      for (const [i, l] of card.layers.entries()) if (l.view === view) layers[i].setSource(geozarr(shown[view]));
      status("Tiles are computed as the map asks for them.");
    }).catch((e) => status(`Failed: ${(e as Error).message ?? e}`));
    return busy;
  };
  select.addEventListener("change", async () => {
    for (const v of card.sites.views) await recompute(v, { source: select.value });
    fit();
  });
  $("controls").append(pick);

  for (const c of card.controls) {
    if ("variable" in c) {
      const layer = layers[card.layers.findIndex((l) => l.name === c.layer)];
      slider(c, c.value, true, (v) => layer.updateStyleVariables({ [c.variable]: v }));
    } else {
      const ops = structuredClone(card.views[c.view].ops!) as Record<string, number>[];
      slider(c, ops[c.op][c.param], false, (v) => {
        ops[c.op] = { ...ops[c.op], [c.param]: v };
        void recompute(c.view, { ops: structuredClone(ops) });
      });
    }
  }
  for (const [i, l] of card.layers.entries()) {
    const row = document.createElement("label");
    row.innerHTML = `<input type="checkbox"><span></span>`;
    const box = row.querySelector("input")!;
    box.checked = layers[i].getVisible();
    row.querySelector("span")!.textContent = l.name;
    box.addEventListener("change", () => layers[i].setVisible(box.checked));
    $("layers").append(row);
  }
  $("controls").hidden = $("layers").hidden = false;
  showCounts();
  Object.assign(window, { map, engine });  // for a console, and the headless checks
}

$("copy").addEventListener("click", () => void navigator.clipboard.writeText($("command").textContent ?? ""));
start().catch((e) => { console.error(e); status(`Failed: ${(e as Error).message ?? e}`); });
