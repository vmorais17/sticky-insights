import { defineConfig } from 'vite';

/**
 * Vite configuration for Sticky Insights.
 *
 * Key settings for @xenova/transformers (ONNX Runtime Web):
 *
 * optimizeDeps.exclude — Vite's dep bundler must NOT touch @xenova/transformers.
 *   The library resolves its WASM binary paths at runtime using its own VERSION
 *   constant and dynamic CDN URL construction.  If Vite pre-bundles it, those
 *   paths get corrupted and every pipeline run fails with "Failed to fetch".
 *
 * server.headers — Cross-Origin-Opener-Policy + Cross-Origin-Embedder-Policy
 *   are required to unlock SharedArrayBuffer, which onnxruntime-web's threaded
 *   WASM variant depends on.  Without them the runtime falls back to the slower
 *   single-threaded WASM — no crash, but inference is noticeably slower.
 *
 * Privacy note: model weights are fetched once from Hugging Face CDN and then
 * stored in the browser's Cache Storage.  No sticky note content ever leaves
 * the tab — all inference runs locally via WASM.
 */
export default defineConfig({
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },
  server: {
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
});
