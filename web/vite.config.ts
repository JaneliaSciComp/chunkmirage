// The pages (the gallery index.html, register.html, pipeline.html), their workers and the
// service worker, built for any path:
// the docs site serves it under /chunkmirage/browser/. The service worker keeps a fixed name,
// sw.js next to the page, since the page registers it by that name and its scope is ./.
// `vite preview` relays the Neuroglancer client from appspot under /ng/, as serve.py does;
// the deployed site carries its own build there instead.
import { defineConfig } from "vite";

const NEUROGLANCER = "https://neuroglancer-demo.appspot.com";
const ng = { "/ng": { target: NEUROGLANCER, changeOrigin: true, rewrite: (p: string) => p.replace(/^\/ng/, "") } };

export default defineConfig({
  base: "./",
  build: {
    target: "es2022",
    chunkSizeWarningLimit: 1000,  // the zstd and blosc codecs (WebAssembly), loaded only for images that use them
    rolldownOptions: {
      input: { index: "index.html", register: "register.html", pipeline: "pipeline.html", map: "map.html", sw: "src/sw.ts" },
      output: { entryFileNames: (chunk) => (chunk.name === "sw" ? "sw.js" : "assets/[name]-[hash].js") },
    },
  },
  worker: { format: "es" },
  // the Pyodide workers import chunkmirage's Python from ../src as text
  server: { proxy: ng, fs: { allow: [".."] } },
  preview: { proxy: ng },
});
