// The page (register.html), its chunk workers and its service worker, built for any path:
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
      input: { register: "register.html", sw: "src/sw.ts" },
      output: { entryFileNames: (chunk) => (chunk.name === "sw" ? "sw.js" : "assets/[name]-[hash].js") },
    },
  },
  worker: { format: "es" },
  server: { proxy: ng },
  preview: { proxy: ng },
});
