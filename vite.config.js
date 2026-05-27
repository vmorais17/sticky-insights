import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

/**
 * Vite configuration for Sticky Insights.
 *
 * The core problem: Vite 8 (Rolldown) processes onnxruntime-web's CJS webpack
 * bundle (ort-web.min.js) in a way that breaks its internal __webpack_require__
 * bootstrap — a circular dependency in the bundle initialises in the wrong order,
 * calling registerBackend on an undefined object. This throws at import time,
 * preventing pipeline.js and then main.js from loading, which means NO event
 * listeners are ever attached and every button in the app stops working.
 *
 * Fix — resolve.alias:
 *   Point 'onnxruntime-web' directly at ort-web.es6.min.js, the ES-module build
 *   that ships alongside the CJS bundle. ESM modules initialise top-to-bottom
 *   in dependency order without a webpack bootstrap, so the circular-init issue
 *   does not occur.
 *
 * Fix — optimizeDeps.exclude for @xenova/transformers:
 *   The package resolves its WASM binary URL at runtime using its own VERSION
 *   constant. Rolldown's module rewriting corrupts that constant, causing every
 *   pipeline run to fail with "Failed to fetch" before any model weight download
 *   begins. Excluding it tells Vite to serve the package as-is.
 *
 * Why no COOP/COEP headers:
 *   Cross-Origin-Embedder-Policy: require-corp blocks CDN-served WASM files
 *   (jsDelivr does not set Cross-Origin-Resource-Policy: cross-origin), causing
 *   the same TypeError. Without SharedArrayBuffer, onnxruntime-web falls back to
 *   single-threaded WASM — slower but fully functional for this workload.
 *
 * Privacy: model weights are fetched once from Hugging Face CDN and cached in
 * the browser's Cache Storage. No sticky note content ever leaves the tab.
 */
export default defineConfig({
  resolve: {
    alias: {
      // Use the ESM build of onnxruntime-web instead of the CJS webpack bundle.
      // ort-web.es6.min.js initialises correctly in Vite's native ESM environment.
      'onnxruntime-web': path.resolve(
        __dirname,
        'node_modules/onnxruntime-web/dist/ort-web.es6.min.js'
      ),
    },
  },
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },
  server: {
    port: 5177,
    strictPort: true,
  },
});
