/**
 * capture-gifs.mjs — Regenerate the demo GIFs used in the README.
 *
 * Recording strategy:
 *   • ONE browser session, ONE continuous WebM.
 *   • The MiniLM + FLAN-T5 download ONCE inside the pipeline-progress window;
 *     every later scene reuses the already-warm in-memory model singletons.
 *   • Each scene declares { start, end } marks against the recording wall clock.
 *     After the context closes we extract each scene segment with ffmpeg -ss/-to
 *     and encode it as a palette-quantised GIF.
 *   • Each scene may declare a `focus` crop rect to zoom into a region of
 *     interest — applied as an ffmpeg crop filter before scaling.
 *
 * Flow (one unbroken session):
 *   1. Load app → click Find Themes → stage messages → canvas returns with
 *      colour-bordered stickies.
 *   2. From that Canvas state: toggle Related Themes (hull overlay).
 *   3. Back on Canvas → click Insights tab → click a cluster tile → side
 *      panel slides in.
 *   4. With side panel open → focus on contributor breakdown (who said what).
 *   5. Back on Canvas → click Semantics tab → hover clusters and notes.
 *
 * Requirements:
 *   npm install --save-dev playwright && npx playwright install chromium
 *   brew install ffmpeg                    # or equivalent
 *
 * Usage:
 *   npm run dev                            # dev server on :5173
 *   node scripts/capture-gifs.mjs          # in another terminal
 *
 * Env overrides:
 *   CAPTURE_URL   default http://localhost:5173
 *   CAPTURE_KEEP  set to "1" to keep the raw WebM + palette PNGs in docs/.capture-tmp
 */

import { chromium } from 'playwright';
import { execFile } from 'node:child_process';
import { mkdir, rm, readdir, stat } from 'node:fs/promises';
import { promisify } from 'node:util';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const execFileAsync = promisify(execFile);

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT      = path.resolve(__dirname, '..');
const OUT_DIR   = path.join(ROOT, 'docs', 'gifs');
const TMP_DIR   = path.join(ROOT, 'docs', '.capture-tmp');

const URL         = process.env.CAPTURE_URL || 'http://localhost:5173';
const KEEP_TMP    = process.env.CAPTURE_KEEP === '1';
const VIEWPORT    = { width: 1280, height: 800 };
const FPS         = 15;
const GIF_WIDTH   = 960;
const PAD_MS      = 250;                 // buffer around each scene's marks
const PIPELINE_TIMEOUT_MS = 180_000;     // first-run HF download + inference

// ─── Timing helper ───────────────────────────────────────────────────────
// Captured as close to Playwright's recordVideo start as possible. Offsets
// are in seconds, relative to the moment the page first navigates.

function createTimer() {
  let origin = null;
  const segments = [];
  return {
    startRecording() { origin = Date.now(); },
    now() { return (Date.now() - origin) / 1000; },
    mark(name, which /* 'start' | 'end' */) {
      if (origin == null) throw new Error('Timer not started');
      let seg = segments.find(s => s.name === name);
      if (!seg) { seg = { name }; segments.push(seg); }
      seg[which] = (Date.now() - origin) / 1000;
    },
    segments() { return segments.filter(s => s.start != null && s.end != null); },
  };
}

// ─── Scene definitions ───────────────────────────────────────────────────
// Each scene: drive the page, call mark(name, 'start' | 'end') to bound the
// segment that will be extracted from the master WebM.
//
// focus: optional { x, y, w, h } crop rect in viewport px — use to zoom into a
//        region (e.g. the contributor breakdown). Null = full viewport.

const scenes = [
  {
    name: 'pipeline-progress',
    description: 'Click Find Themes, record stage-by-stage progress messages',
    focus: null,
    run: async ({ page, mark }) => {
      mark('pipeline-progress', 'start');
      await page.getByRole('button', { name: /find themes/i }).click();
      // Cut the GIF at ~10s so first-run HF download doesn't inflate the loop.
      // The pipeline itself may keep running; the `post-pipeline` wait below
      // (outside this scene) blocks until it's actually done.
      await page.waitForTimeout(10_000);
      mark('pipeline-progress', 'end');

      // Not part of the cut: block until the progress bar is hidden (class
      // `.hidden` sets display:none). `state: 'hidden'` matches any non-visible
      // reason — including the class-applied display:none.
      await page.locator('#progress-section').waitFor({ state: 'hidden', timeout: PIPELINE_TIMEOUT_MS });
      await page.waitForTimeout(500); // let the canvas settle with borders on
    },
  },
  {
    name: 'related-themes',
    description: 'Toggle Related Themes hull overlay on the canvas',
    // Zoom into the top-right (toggle + a chunk of canvas where hulls appear).
    focus: { x: 640, y: 0, w: 640, h: 560 },
    run: async ({ page, mark }) => {
      // Scene assumes we're already on Canvas with colour-bordered stickies.
      // The <input id="hull-toggle"> is visually hidden (opacity:0; width:0),
      // so we click the wrapping <label class="pill-toggle"> instead.
      const toggle = page.locator('#hull-toggle-wrap .pill-toggle');

      mark('related-themes', 'start');
      await page.waitForTimeout(400);
      await toggle.click();                // hulls ON  — show them appear
      await page.waitForTimeout(1800);
      await toggle.click();                // hulls OFF — end on clean canvas
      await page.waitForTimeout(1000);
      mark('related-themes', 'end');
      // End state = hulls OFF. Next scene can assume a clean entry.
    },
  },
  {
    name: 'insights-selection',
    description: 'Insights tab → click cluster tile → side panel slides in',
    focus: null,
    run: async ({ page, mark }) => {
      // Entry guard: ensure hulls OFF and Canvas view active before the cut,
      // so the first frame of the extracted segment can't show leftover hulls.
      const hullInput = page.locator('#hull-toggle');
      if (await hullInput.isChecked().catch(() => false)) {
        await page.locator('#hull-toggle-wrap .pill-toggle').click();
        await page.waitForTimeout(350);
      }
      await page.waitForTimeout(400); // settle past PAD_MS back-reach

      mark('insights-selection', 'start');
      await page.getByRole('button', { name: /^insights$/i }).click();
      await page.waitForTimeout(1400);     // tile grid render + eye tracks it
      // Click the first cluster tile
      await page.locator('.cv-tile').first().click();
      // Side panel slides in — show it settled
      await page.locator('.cv-panel').waitFor({ state: 'visible' }).catch(() => {});
      await page.waitForTimeout(1800);
      mark('insights-selection', 'end');
      // Leave panel open — next scene zooms into it.
    },
  },
  {
    name: 'insights-consensus',
    description: 'Side panel open → zoom to contributor breakdown (voice balance)',
    // Crop to the right-side panel; exact bounds computed at run time below.
    focus: 'dynamic:.cv-panel',
    run: async ({ page, mark }) => {
      const contribHeader = page.locator('.cv-contrib-header');
      // Scroll the breakdown into the panel viewport if it isn't already.
      await contribHeader.scrollIntoViewIfNeeded().catch(() => {});
      await page.waitForTimeout(600);

      mark('insights-consensus', 'start');
      // Sweep the cursor down the contributor list so the highlight pulses.
      const list = page.locator('.cv-contrib-list');
      const box  = await list.boundingBox();
      if (box) {
        await page.mouse.move(box.x + box.width * 0.5, box.y + 20, { steps: 10 });
        await page.waitForTimeout(900);
        await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.6, { steps: 20 });
        await page.waitForTimeout(1400);
      } else {
        await page.waitForTimeout(2200);
      }
      mark('insights-consensus', 'end');

      // Close panel + reset to Canvas for next scene.
      await page.locator('.cv-panel-close').click().catch(() => {});
      await page.waitForTimeout(300);
      await page.getByRole('button', { name: /^canvas$/i }).click();
      await page.waitForTimeout(500);
    },
  },
  {
    name: 'semantic-view',
    description: 'Semantics tab → hover two cluster regions in 2-D PCA',
    focus: null,
    run: async ({ page, mark }) => {
      // Entry guard: the prior scene may leave us on Insights. Explicitly
      // return to Canvas and settle before the cut begins, so the first
      // frame of the extracted segment starts from Canvas view.
      await page.getByRole('button', { name: /^canvas$/i }).click();
      await page.waitForTimeout(700); // render + clear PAD_MS back-reach

      mark('semantic-view', 'start');
      await page.getByRole('button', { name: /^semantics$/i }).click();
      await page.waitForTimeout(1600);
      const box = await page.locator('#semantic-view').boundingBox();
      if (box) {
        await page.mouse.move(box.x + box.width * 0.35, box.y + box.height * 0.40, { steps: 25 });
        await page.waitForTimeout(1300);
        await page.mouse.move(box.x + box.width * 0.65, box.y + box.height * 0.60, { steps: 25 });
        await page.waitForTimeout(1500);
      }
      mark('semantic-view', 'end');
    },
  },
];

// ─── Driver ──────────────────────────────────────────────────────────────

async function drive(page, timer) {
  for (const scene of scenes) {
    console.log(`  ▸ ${scene.name}`);
    // Resolve dynamic focus (e.g. `dynamic:.cv-panel`) just-in-time.
    if (typeof scene.focus === 'string' && scene.focus.startsWith('dynamic:')) {
      const sel = scene.focus.slice('dynamic:'.length);
      const bb  = await page.locator(sel).boundingBox();
      if (bb) {
        const pad = 12;
        scene.focus = {
          x: Math.max(0, Math.floor(bb.x - pad)),
          y: Math.max(0, Math.floor(bb.y - pad)),
          w: Math.min(VIEWPORT.width,  Math.ceil(bb.width  + 2 * pad)),
          h: Math.min(VIEWPORT.height, Math.ceil(bb.height + 2 * pad)),
        };
      } else {
        scene.focus = null;
      }
    }
    await scene.run({ page, mark: timer.mark.bind(timer) });
  }
}

// ─── ffmpeg: extract + encode each scene from the master WebM ────────────

function buildFilter({ fps, width, focus }) {
  const parts = [];
  if (focus) {
    // crop=w:h:x:y — applied in source-pixel space (1280×800)
    parts.push(`crop=${focus.w}:${focus.h}:${focus.x}:${focus.y}`);
  }
  parts.push(`fps=${fps}`);
  parts.push(`scale=${width}:-1:flags=lanczos`);
  return parts.join(',');
}

async function encodeScene(webmPath, scene, tmpDir) {
  const start = Math.max(0, scene.start - PAD_MS / 1000);
  const end   = scene.end + PAD_MS / 1000;

  const baseFilter = buildFilter({ fps: FPS, width: GIF_WIDTH, focus: scene.focus });
  const palette    = path.join(tmpDir, `${scene.name}.palette.png`);
  const gif        = path.join(OUT_DIR, `${scene.name}.gif`);

  // Pass 1 — palette
  await execFileAsync('ffmpeg', [
    '-y', '-ss', String(start), '-to', String(end), '-i', webmPath,
    '-vf', `${baseFilter},palettegen=stats_mode=diff`,
    palette,
  ]);

  // Pass 2 — apply palette
  await execFileAsync('ffmpeg', [
    '-y', '-ss', String(start), '-to', String(end), '-i', webmPath, '-i', palette,
    '-filter_complex',
    `${baseFilter}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle`,
    gif,
  ]);

  const { size } = await stat(gif);
  return { name: scene.name, kb: Math.round(size / 1024), path: gif };
}

// ─── Main ────────────────────────────────────────────────────────────────

async function main() {
  await mkdir(OUT_DIR, { recursive: true });
  await rm(TMP_DIR, { recursive: true, force: true });
  await mkdir(TMP_DIR, { recursive: true });

  console.log(`→ launching chromium against ${URL}`);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({
    viewport: VIEWPORT,
    recordVideo: { dir: TMP_DIR, size: VIEWPORT },
    reducedMotion: 'no-preference',
  });
  const page   = await context.newPage();
  const timer  = createTimer();

  try {
    // Start the timer *immediately* after navigation, not at launch. Playwright's
    // recordVideo begins on page load; anchor our marks to the same moment.
    await page.goto(URL, { waitUntil: 'networkidle' });
    timer.startRecording();

    await drive(page, timer);

    await page.waitForTimeout(300); // hold last frame
  } finally {
    await context.close();   // flushes the WebM
    await browser.close();
  }

  // Locate the single .webm Playwright wrote
  const webms = (await readdir(TMP_DIR)).filter(f => f.endsWith('.webm'));
  if (webms.length === 0) throw new Error('No WebM produced');
  const webmPath = path.join(TMP_DIR, webms[0]);
  console.log(`✓ master recording: ${path.relative(ROOT, webmPath)}`);

  // Resolve marks → scene objects with timings
  const segs = timer.segments();
  const results = [];
  for (const scene of scenes) {
    const seg = segs.find(s => s.name === scene.name);
    if (!seg) {
      console.warn(`  ⚠ ${scene.name}: no timing marks, skipping`);
      continue;
    }
    console.log(`  ▸ encoding ${scene.name}  (${seg.start.toFixed(2)}s → ${seg.end.toFixed(2)}s)`);
    results.push(await encodeScene(webmPath, { ...scene, ...seg }, TMP_DIR));
  }

  if (!KEEP_TMP) await rm(TMP_DIR, { recursive: true, force: true });

  console.log('\n✓ capture complete');
  for (const r of results) {
    console.log(`  ${r.name.padEnd(22)}  ${String(r.kb).padStart(5)} KB  →  ${path.relative(ROOT, r.path)}`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
