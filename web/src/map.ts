// The map page: a demo of the gallery (cards.ts) whose views the browser engine (engine.ts)
// computes and serves as GeoZarr, drawn by OpenLayers, a GIS web map that knows nothing of
// chunkmirage: it reads the page's chunks as it reads any GeoZarr store, next to COGs it
// reads itself. Sliders either recompute a view with other op parameters (for the tiles on
// screen) or restyle a layer on the GPU.
import "ol/ol.css";
import OlMap from "ol/Map";
import View from "ol/View";
import { ScaleLine, defaults as defaultControls } from "ol/control";
import WebGLTileLayer from "ol/layer/WebGLTile";
import { Projection, addProjection } from "ol/proj";
import GeoTIFF from "ol/source/GeoTIFF";
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
  await engine.start(card.views, status);
  status(`Python ready in ${((performance.now() - t0) / 1000).toFixed(1)} s. Tiles are computed as the map asks for them.`);

  // the data's own projection, so neither the computed views nor the COGs are reprojected
  const projection = new Projection({ code: card.projection.code, units: "m", extent: card.projection.extent });
  addProjection(projection);
  for (const v of Object.keys(card.views)) engine.proj[v] = card.projection.code;
  const layers = card.layers.map((l) => {
    const source = l.view ? geozarr(l.view) : new GeoTIFF({ sources: [{ url: l.cog! }], projection, transition: 0 });
    return new WebGLTileLayer({ source, style: l.style, visible: l.visible ?? true, properties: { name: l.name } });
  });
  const map = new OlMap({
    target: "map", layers, controls: defaultControls().extend([new ScaleLine()]),
    view: new View({ projection, center: card.center, resolution: card.resolution, maxResolution: 2000, minResolution: 0.5 }),
  });
  $("empty").hidden = true;

  for (const c of card.controls) {
    if ("variable" in c) {
      const layer = layers[card.layers.findIndex((l) => l.name === c.layer)];
      slider(c, c.value, true, (v) => layer.updateStyleVariables({ [c.variable]: v }));
    } else {
      const spec = card.views[c.view], op = spec.ops![c.op] as Record<string, number>;
      let busy = Promise.resolve();
      slider(c, op[c.param], false, (v) => {
        op[c.param] = v;  // the card's spec, edited: the next edit starts from it
        busy = busy.then(async () => {
          const name = await engine.edit(c.view, spec.ops!);
          for (const [i, l] of card.layers.entries()) if (l.view === c.view) layers[i].setSource(geozarr(name));
        }).catch((e) => status(`Failed: ${(e as Error).message ?? e}`));
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
