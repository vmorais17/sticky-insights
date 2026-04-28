#!/usr/bin/env node
/**
 * Build-time overlap resolver for the sticky-notes dataset.
 *
 * Mirrors the runtime resolver previously in src/canvas-view.js. Bakes the
 * resolved (x, y) into public/data/sticky_notes.json and ensures every note
 * carries a `z` stacking-order field (default 0, raised by drag-to-front at
 * runtime). The file lives under public/ so Vite copies it into dist/ for
 * production builds — fetching `/data/sticky_notes.json` works in dev and
 * prod alike.
 *
 * Idempotent: re-running on already-resolved data is a no-op.
 *
 * Run:  node scripts/resolve-overlaps.mjs
 */

import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const NOTE_W      = 160;
const NOTE_H      = 160;
const GAP         = 8;
const ITERS       = 60;
const ORIGIN_PULL = 0.08;

const here     = dirname(fileURLToPath(import.meta.url));
const dataPath = resolve(here, '..', 'public', 'data', 'sticky_notes.json');

function resolveOverlaps(notes) {
  const pos    = notes.map((n) => ({ x: n.x, y: n.y }));
  const origin = notes.map((n) => ({ x: n.x, y: n.y }));

  for (let iter = 0; iter < ITERS; iter++) {
    for (let i = 0; i < pos.length; i++) {
      for (let j = i + 1; j < pos.length; j++) {
        const dx       = pos[j].x - pos[i].x;
        const dy       = pos[j].y - pos[i].y;
        const overlapX = NOTE_W + GAP - Math.abs(dx);
        const overlapY = NOTE_H + GAP - Math.abs(dy);

        if (overlapX > 0 && overlapY > 0) {
          if (overlapX < overlapY) {
            const shift = overlapX / 2;
            pos[i].x -= dx > 0 ? shift : -shift;
            pos[j].x += dx > 0 ? shift : -shift;
          } else {
            const shift = overlapY / 2;
            pos[i].y -= dy > 0 ? shift : -shift;
            pos[j].y += dy > 0 ? shift : -shift;
          }
        }
      }

      pos[i].x += (origin[i].x - pos[i].x) * ORIGIN_PULL;
      pos[i].y += (origin[i].y - pos[i].y) * ORIGIN_PULL;
    }
  }

  return pos;
}

const raw    = await readFile(dataPath, 'utf8');
const notes  = JSON.parse(raw);
const before = notes.map((n) => ({ x: n.x, y: n.y }));
const pos    = resolveOverlaps(notes);

let moved   = 0;
let maxDrift = 0;
const next = notes.map((n, i) => {
  const x = Math.round(pos[i].x);
  const y = Math.round(pos[i].y);
  const drift = Math.hypot(x - before[i].x, y - before[i].y);
  if (drift > 0) moved++;
  if (drift > maxDrift) maxDrift = drift;
  return { ...n, x, y, z: typeof n.z === 'number' ? n.z : 0 };
});

await writeFile(dataPath, JSON.stringify(next, null, 2) + '\n', 'utf8');

console.log(`resolved ${notes.length} notes`);
console.log(`  moved:     ${moved}`);
console.log(`  max drift: ${maxDrift.toFixed(1)} px`);
console.log(`  z field:   ensured (default 0)`);
console.log(`  wrote:     ${dataPath}`);
