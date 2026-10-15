// A Vite plugin serving ONNX Runtime Web's WebAssembly runtime beside the page, for a page
// with an onnx stage (src/onnx.ts). The ORT bundle the page imports ("onnxruntime-web/webgpu")
// fetches its WebAssembly binary at run time from env.wasm.wasmPaths, ort/ beside the page;
// this serves the files there from the installed package in development, and copies them
// into the build under the same prefix, so a deployed page carries the version npm
// installed and loads nothing from a CDN. A page registers it in its Vite config:
//
//   import { ortRuntime } from "<chunkmirage>/web/vite-ort";
//   export default defineConfig({ plugins: [ortRuntime()] });
//
// Only the asyncify build is needed: it is the one the webgpu bundle names, and it runs
// the WASM provider too.
/// <reference types="node" />
import { copyFileSync, createReadStream, existsSync, mkdirSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, extname, join, resolve } from "node:path";
import type { Plugin } from "vite";

const RUNTIME_FILES = ["ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.mjs"];
const MIME: Record<string, string> = { ".wasm": "application/wasm", ".mjs": "text/javascript; charset=utf-8" };

/** The plugin; `prefix` is the directory beside the page the runtime is served under. */
export function ortRuntime(prefix = "ort"): Plugin {
  let dist = "", outDir = "dist";
  return {
    name: "chunkmirage:ort-runtime",
    // the ORT bundle is one self-contained module: left out of the dev server's dependency
    // pre-bundling it is served as it is, its own import.meta.url and wasm loading intact
    config: () => ({ optimizeDeps: { exclude: ["onnxruntime-web"] } }),
    configResolved(config) {
      // the package as the page's project installed it, found through a file its exports
      // map names (package.json is not among them, so resolving it throws)
      dist = dirname(createRequire(resolve(config.root, "package.json")).resolve(`onnxruntime-web/${RUNTIME_FILES[0]}`));
      outDir = resolve(config.root, config.build.outDir);
    },
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? "").split("?")[0], file = RUNTIME_FILES.find((f) => url === `/${prefix}/${f}` || url.endsWith(`/${prefix}/${f}`));
        if (!file || !existsSync(join(dist, file))) return next();
        res.setHeader("Content-Type", MIME[extname(file)] ?? "application/octet-stream");
        createReadStream(join(dist, file)).pipe(res);
      });
    },
    generateBundle(_options, bundle) {
      // the ORT bundle also names its binary with new URL(..., import.meta.url), for when
      // wasmPaths is unset, and Vite turns that into a hashed copy under assets/; the page
      // sets wasmPaths, so that copy is never fetched: dropped rather than shipped twice
      for (const name of Object.keys(bundle)) if (/ort-wasm-simd-threaded.*\.wasm$/.test(name)) delete bundle[name];
    },
    closeBundle() {
      mkdirSync(join(outDir, prefix), { recursive: true });
      for (const file of RUNTIME_FILES) if (existsSync(join(dist, file))) copyFileSync(join(dist, file), join(outDir, prefix, file));
    },
  };
}
