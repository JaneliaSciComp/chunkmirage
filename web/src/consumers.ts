// Other readers of a pipeline card's views than Neuroglancer, each knowing nothing of
// chunkmirage: it reads the page's chunks as it reads any store. A web map (OpenLayers,
// reading the views as GeoZarr), and GDAL itself (gdal3.js, GDAL compiled to WebAssembly,
// reading the views' zarr v2 layout, its files fetched from the page as they are served and
// written into GDAL's own file system, then translated to a GeoTIFF to download). Loaded
// only when picked.
import type { CardConsumers } from "./cards";
import type { Engine } from "./engine";

/** Where a level's pixel (row, column) is, in the views' map projection. */
function place(engine: Engine, view: string, level: number) {
  const s = engine.sources[view], bbox = s.geo!.bbox, v = s.levels[level].voxel;
  return { bbox, x: (col: number) => bbox[0] + col * v[2], y: (row: number) => bbox[3] - row * v[1], vx: v[2], vy: v[1] };
}

/** An OpenLayers map of `c.map`'s layers in `target`, centred on `centre` (a full
 * resolution row and column) at `zoom` pixels of it per screen pixel. */
export async function showMap(target: HTMLElement, engine: Engine, c: CardConsumers, centre: number[], zoom: number) {
  await import("ol/ol.css");
  const [{ default: OlMap }, { default: View }, { default: WebGLTileLayer }, { default: GeoZarr }, { Projection, addProjection }, { ScaleLine, defaults }] = await Promise.all([
    import("ol/Map"), import("ol/View"), import("ol/layer/WebGLTile"), import("ol/source/GeoZarr"), import("ol/proj"), import("ol/control"),
  ]);
  const first = c.map.layers[0].view, p = place(engine, first, 0);
  const projection = new Projection({ code: c.code, units: "m", extent: p.bbox });
  addProjection(projection);
  for (const l of c.map.layers) engine.proj[l.view] = c.code;
  const layers = c.map.layers.map((l) => new WebGLTileLayer({ source: new GeoZarr({ url: engine.geoUrl(l.view), bands: [l.view], transition: 0 }), style: l.style }));
  const map = new OlMap({ target, layers, controls: defaults().extend([new ScaleLine()]) });
  map.setView(new View({ projection, extent: p.bbox, center: [p.x(centre[1] + 0.5), p.y(centre[0] + 0.5)], resolution: zoom * p.vx, minResolution: p.vx / 4 }));
  return map;
}

const GDAL = "https://cdn.jsdelivr.net/npm/gdal3.js@2.8.1/dist/package/";
interface GdalApi {
  Module: { FS: { mkdirTree(p: string): void; writeFile(p: string, d: string | Uint8Array): void; analyzePath(p: string): { exists: boolean } } };
  open(path: string): Promise<{ datasets: unknown[]; errors: { message: string }[] }>;
  gdalinfo(ds: unknown, options?: string[]): Promise<unknown>;
  gdal_translate(ds: unknown, options: string[]): Promise<{ local: string }>;
  getFileBytes(path: string | { local: string }): Promise<Uint8Array<ArrayBuffer>>;
}
let gdal: Promise<GdalApi> | null = null;

/** gdal3.js, on this page's thread so the page can write into its file system. */
function loadGdal(): Promise<GdalApi> {
  return (gdal ??= new Promise<void>((resolve, reject) => {
    const s = document.createElement("script");
    s.src = `${GDAL}gdal3.js`;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("could not load gdal3.js"));
    document.head.append(s);
  }).then(() => (window as unknown as { initGdalJs: (o: object) => Promise<GdalApi> }).initGdalJs({ path: GDAL, useWorker: false })));
}

/** The GDAL panel: pick an area and a level; the page fetches the zarr v2 files GDAL needs
 * for it (the chunks computed as they are fetched), writes them into GDAL's file system as
 * served, and GDAL reads them: gdalinfo, then gdal_translate to a georeferenced GeoTIFF (to
 * download) and a PNG (shown). */
export function gdalPanel(target: HTMLElement, engine: Engine, c: CardConsumers) {
  const view = c.gdal.view, s = engine.sources[view];
  target.innerHTML = `
    <div class="gdal">
      <p class="muted">GDAL itself, compiled to WebAssembly, reads this page's virtual zarr: the files below are fetched as they are served (each chunk computed when fetched), written into GDAL's file system, and opened by its Zarr driver like any zarr on disk.</p>
      <div class="row"><label>Area <select id="g-area"></select></label><label>Level <select id="g-level"></select></label><button id="g-run" type="button">Run GDAL</button><a id="g-get" hidden download>Download GeoTIFF</a></div>
      <pre id="g-log"></pre><img id="g-png" alt="">
    </div>`;
  const $ = (id: string) => target.querySelector(`#${id}`) as HTMLElement;
  const area = $("g-area") as HTMLSelectElement, level = $("g-level") as HTMLSelectElement, log = $("g-log");
  for (const a of c.gdal.areas) area.append(new Option(a.name, a.name));
  s.levels.forEach((l, i) => level.append(new Option(`${i}: ${Math.round(l.voxel[2])} m pixels`, String(i))));
  level.value = String(Math.min(c.gdal.level, s.levels.length - 1));
  const say = (t: string) => { log.textContent += `${t}\n`; log.scrollTop = log.scrollHeight; };
  $("g-run").addEventListener("click", async () => {
    log.textContent = "";
    ($("g-get") as HTMLAnchorElement).hidden = true;
    try {
      const li = Number(level.value), l = s.levels[li], f = l.voxel[2] / s.levels[0].voxel[2], a = c.gdal.areas.find((x) => x.name === area.value)!;
      const [r0, r1] = (a.rows ?? [0, s.levels[0].shape[1]]).map((r) => Math.floor(r / f)), [c0, c1] = (a.cols ?? [0, s.levels[0].shape[2]]).map((x) => Math.floor(x / f));
      const rows = [r0, Math.min(r1, l.shape[1])], cols = [c0, Math.min(c1, l.shape[2])];
      say("Loading GDAL (gdal3.js, 40 MB the first time)…");
      const G = await loadGdal(), FS = G.Module.FS, base = engine.zarr2Url(view), root = `/output/${view}`;
      const zarray = await (await fetch(`${base}${li}/.zarray`)).json(), C = zarray.chunks.slice(-2);
      const files = [".zgroup", ".zattrs", `${li}/.zarray`, `${li}/.zattrs`];
      for (let i = Math.floor(rows[0] / C[0]); i * C[0] < rows[1]; i++) for (let j = Math.floor(cols[0] / C[1]); j * C[1] < cols[1]; j++) files.push(`${li}/0/${i}/${j}`);
      say(`Fetching ${files.length} files of ${base}: ${files.length - 4} chunks, each computed the first time anything fetches it…`);
      const t0 = performance.now();
      await Promise.all(files.map(async (name) => {
        const r = await fetch(base + name);
        if (!r.ok) throw new Error(`${name}: ${r.status}`);
        const path = `${root}/${name}`;
        FS.mkdirTree(path.slice(0, path.lastIndexOf("/")));
        FS.writeFile(path, new Uint8Array(await r.arrayBuffer()));
      }));
      say(`  in ${((performance.now() - t0) / 1000).toFixed(1)} s`);
      const opened = await G.open(`${root}/${li}`);
      if (!opened.datasets.length) throw new Error(opened.errors.map((e) => e.message).join("; ") || "GDAL could not open it");
      const ds = opened.datasets[0];
      const info = (await G.gdalinfo(ds)) as { driverShortName: string; size: number[]; bands?: { type: string }[] };
      say(`\n$ gdalinfo ${root}/${li}\nDriver: ${info.driverShortName}   Size: ${info.size.join(" x ")}   Type: ${info.bands?.[0]?.type ?? "?"}`);
      const p = place(engine, view, li);
      const win = [cols[0], rows[0], cols[1] - cols[0], rows[1] - rows[0]].map(String);
      const geo = ["-a_srs", c.code, "-a_ullr", ...[p.x(cols[0]), p.y(rows[0]), p.x(cols[1]), p.y(rows[1])].map((v) => v.toFixed(2))];
      const tifArgs = ["-of", "GTiff", "-srcwin", ...win, ...geo, "-co", "COMPRESS=DEFLATE"];
      say(`\n$ gdal_translate ${tifArgs.join(" ")} ${root}/${li} ${view}.tif`);
      const tif = await G.gdal_translate(ds, tifArgs), bytes = await G.getFileBytes(tif);
      say(`  ${(bytes.length / 1e6).toFixed(1)} MB, georeferenced (${c.code})`);
      const get = $("g-get") as HTMLAnchorElement;
      get.href = URL.createObjectURL(new Blob([bytes], { type: "image/tiff" }));
      get.download = `${view}-${a.name.replace(/\W+/g, "-").toLowerCase()}-level${li}.tif`;
      get.hidden = false;
      const pngArgs = ["-of", "PNG", "-srcwin", ...win, "-ot", "Byte", "-scale", ...c.gdal.scale.map(String), "1", "255", "-a_nodata", "0"];
      say(`$ gdal_translate ${pngArgs.join(" ")} ${root}/${li} preview.png`);
      const png = await G.getFileBytes(await G.gdal_translate(ds, pngArgs));
      ($("g-png") as HTMLImageElement).src = URL.createObjectURL(new Blob([png], { type: "image/png" }));
    } catch (e) {
      say(`Failed: ${(e as Error).message ?? e}`);
    }
  });
}
