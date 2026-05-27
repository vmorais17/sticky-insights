import { defineConfig } from 'vite';

/**
 * Vite configuration for Sticky Insights.
 *
 * optimizeDeps.exclude — the only setting required for @xenova/transformers.
 *
 *   Vite 8's dependency optimiser (Rolldown) pre-bundles packages for faster
 *   dev-server startup.  @xenova/transformers resolves its ONNX Runtime WASM
 *   binary URL at runtime using its own VERSION constant:
 *
 *     https://cdn.jsdelivr.net/npm/@xenova/transformers@{VERSION}/dist/
 *
 *   When Vite rewrites the module graph, that URL construction breaks and every
 *   pipeline run fails with "Failed to fetch" before any model weight download
 *   even begins.  Excluding the package tells Vite to serve it as-is via
 *   native ES-module imports, which preserves the internal path resolution.
 *
 * Why no COOP/COEP headers:
 *
 *   Cross-Origin-Embedder-Policy: require-corp blocks any cross-origin resource
 *   that doesn't respond with Cross-Origin-Resource-Policy: cross-origin.  The
 *   jsDelivr CDN (where onnxruntime-web fetches its WASM binaries) does not
 *   send that header, so COEP causes ort-web.min.js to throw during module
 *   initialisation — which also prevents main.js event listeners from attaching,
 *   breaking ALL buttons in the app, not just Find Themes.
 *
 *   Without SharedArrayBuffer, onnxruntime-web falls back to single-threaded
 *   WASM.  Inference is slower but fully functional.
 *
 * Privacy note: model weights are fetched once from Hugging Face CDN and cached
 * in the browser's Cache Storage.  No sticky note content ever leaves the tab.
 */
export default defineConfig({
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },
});
