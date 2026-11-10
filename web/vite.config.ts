// The pages (the gallery index.html, register.html, pipeline.html), their workers and the
// service worker, built for any path:
// the docs site serves it under /chunkmirage/browser/. The service worker keeps a fixed name,
// sw.js next to the page, since the page registers it by that name and its scope is ./.
// `vite preview` relays the Neuroglancer client from appspot under /ng/, as serve.py does;
// the deployed site carries its own build there instead.
import { build, defineConfig, type Plugin } from "vite";

const NEUROGLANCER = "https://neuroglancer-demo.appspot.com";
const ng = { "/ng": { target: NEUROGLANCER, changeOrigin: true, rewrite: (p: string) => p.replace(/^\/ng/, "") } };

/** sw.js, built on its own once the pages are, into one file. The page registers it as a
 * classic script (no `type: "module"`), and a classic script cannot import: as one more input
 * of the pages' build, the code it shares with them (urls.ts) went to a chunk of its own that
 * sw.js imported, and every page waited for a service worker that never started. */
function serviceWorker(): Plugin {
  let root = ".", outDir = "dist";
  return {
    name: "chunkmirage-service-worker",
    apply: "build",
    configResolved(config) { ({ root } = config); ({ outDir } = config.build); },
    async closeBundle() {
      const built = await build({
        configFile: false,
        root,
        logLevel: "warn",
        build: {
          outDir, emptyOutDir: false, copyPublicDir: false, target: "es2022",
          rolldownOptions: { input: "src/sw.ts", output: { entryFileNames: "sw.js", codeSplitting: false } },
        },
      });
      const chunks = (Array.isArray(built) ? built : [built]).flatMap((b) => ("output" in b ? b.output : []));
      const sw = chunks.find((c) => c.type === "chunk" && c.fileName === "sw.js");
      if (!sw || sw.type !== "chunk") throw new Error("the service worker build wrote no sw.js");
      if (sw.imports.length || sw.dynamicImports.length || chunks.length > 1)
        throw new Error(`sw.js must be one file, with no imports: it imports ${[...sw.imports, ...sw.dynamicImports].join(", ") || "nothing"}, and its build wrote ${chunks.map((c) => c.fileName).join(", ")}`);
    },
  };
}

export default defineConfig({
  base: "./",
  plugins: [serviceWorker()],
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1000,  // the zstd and blosc codecs (WebAssembly), loaded only for images that use them
    rolldownOptions: {
      input: { index: "index.html", register: "register.html", pipeline: "pipeline.html", map: "map.html", stitch: "stitch.html", track: "track.html" },
      output: { entryFileNames: "assets/[name]-[hash].js" },
    },
  },
  worker: { format: "es" },
  // the Pyodide workers import chunkmirage's Python from ../src as text
  server: { proxy: ng, fs: { allow: [".."] } },
  preview: { proxy: ng },
});
