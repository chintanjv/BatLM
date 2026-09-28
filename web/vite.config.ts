import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Cross-origin isolation unlocks SharedArrayBuffer -> multithreaded WASM inference (~3-4x faster
// on the no-WebGPU fallback). "credentialless" still allows fetching model files from the HF Hub.
const isolation = {
  "Cross-Origin-Opener-Policy": "same-origin",
  "Cross-Origin-Embedder-Policy": "credentialless",
};

export default defineConfig({
  plugins: [react()],
  worker: { format: "es" },
  optimizeDeps: { exclude: ["@huggingface/transformers"] },
  build: { target: "es2022", chunkSizeWarningLimit: 2000 },
  server: { headers: isolation },
  preview: { headers: isolation },
});
