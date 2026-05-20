/**
 * main.js — Application entry point.
 * Wires together: import → canvas render → clustering → both views.
 *
 * Privacy: note content stays in the browser. Boards are imported from /
 * exported to local JSON files; nothing is uploaded by this app, and any
 * cross-device sync is delegated entirely to the user (e.g., place the
 * downloaded file in iCloud Drive to share between devices).
 */

import { loadNotes, validateNotes } from './loader.js';
import { downloadBoard, pickBoardFile } from './board-file.js';
import { clusterNotes, getFlanT5 } from './pipeline.js';
import { renderCanvas } from './canvas-view.js';
import { renderClusterView } from './cluster-view.js';
import { renderSemanticView, invalidateSemanticView } from './semantic-view.js';
import { computeInsights } from './insights-agent.js';
import { runReActLoop } from './agent/react.js';

// ─── Element refs ────────────────────────────────────────────────
const canvasView       = document.getElementById('canvas-view');
const clusterView      = document.getElementById('cluster-view');
const semanticView     = document.getElementById('semantic-view');
const canvasContainer  = document.getElementById('canvas-container');
const clusterContainer = document.getElementById('cluster-container');

const btnClusterAction = document.getElementById('btn-cluster-action');
const btnDownload      = document.getElementById('btn-download');
const btnCanvas        = document.getElementById('btn-canvas');
const btnCluster       = document.getElementById('btn-cluster');
const btnSemantics     = document.getElementById('btn-semantics');
const viewToggle       = document.getElementById('view-toggle');
const hullToggleWrap   = document.getElementById('hull-toggle-wrap');
const hullToggle       = document.getElementById('hull-toggle');

const progressSection = document.getElementById('progress-section');
const progressBar     = document.getElementById('progress-bar');
const progressLabel   = document.getElementById('progress-label');
const progressDetail  = document.getElementById('progress-detail');

const boardPicker      = document.getElementById('board-picker');
const btnImport        = document.getElementById('btn-import');
const btnStarter       = document.getElementById('btn-starter');
const boardPickerError = document.getElementById('board-picker-error');

// ─── App state ───────────────────────────────────────────────────
let notes             = [];
let embeddingsReduced = [];
let assignments       = [];
let clusters          = [];
let clusterViewApi    = null;
let currentBoardName  = 'board.json'; // suggested filename for download
let noteIdToIdx       = new Map();    // rebuilt whenever `notes` is replaced
let dirty             = false;        // true between drag and download

// ─── Progress helpers ────────────────────────────────────────────
function showProgress(label = '', pct = 0) {
  progressSection.classList.remove('hidden');
  progressLabel.textContent = label;
  progressDetail.textContent = '';
  progressBar.style.width = `${pct}%`;
}

function hideProgress() {
  progressSection.classList.add('hidden');
}

function updateProgress(label, pct) {
  progressLabel.textContent = label;
  progressBar.style.width = `${pct}%`;
}

// ─── View switching ───────────────────────────────────────────────
function showView(name) {
  canvasView.classList.toggle('hidden', name !== 'canvas');
  clusterView.classList.toggle('hidden', name !== 'cluster');
  semanticView.classList.toggle('hidden', name !== 'semantics');
  btnCanvas.classList.toggle('active', name === 'canvas');
  btnCluster.classList.toggle('active', name === 'cluster');
  btnSemantics.classList.toggle('active', name === 'semantics');
  hullToggleWrap.classList.toggle('hidden', name !== 'canvas');

  if (name === 'semantics') {
    renderSemanticView(semanticView, notes, embeddingsReduced, assignments, clusters, {
      onNoteClick: (colorIdx) => { showView('cluster'); clusterViewApi?.selectCluster(colorIdx); },
    });
  }
}

btnCanvas.addEventListener('click', () => showView('canvas'));
btnCluster.addEventListener('click', () => showView('cluster'));
btnSemantics.addEventListener('click', () => showView('semantics'));

// ─── Hull toggle ──────────────────────────────────────────────────
function applyHullVisibility() {
  const hullLayer = canvasContainer.querySelector('.hull-layer');
  if (!hullLayer) return;
  hullLayer.classList.toggle('hull-hidden', !hullToggle.checked);
}

hullToggle.addEventListener('change', applyHullVisibility);

// ─── Edit & drag → dirty (no auto-save in this app) ──────────────
// All canvas mutations (drag, text edit, color change, add, delete)
// run through these handlers, mark the board `dirty`, and re-render
// when needed. Persistence happens when the user clicks Download.
// The `dirty` flag drives the Download button styling and a
// beforeunload warning so unsaved changes aren't lost to a stray
// tab close.

function setDirty(flag) {
  if (dirty === flag) return;
  dirty = flag;
  btnDownload.classList.toggle('dirty', flag);
}

window.addEventListener('beforeunload', (e) => {
  if (!dirty) return;
  e.preventDefault();
  // Most browsers ignore this string and show their own generic prompt;
  // setting returnValue is required for the warning to appear at all.
  e.returnValue = '';
});

// Single source of truth for renderCanvas options. Pass `extra` to
// override or extend (e.g. autoEditNoteId after a fresh create).
function renderOptions(extra = {}) {
  return {
    onDragEnd:         handleNoteDragEnd,
    onNoteEdit:        handleNoteEdit,
    onNoteDelete:      handleNoteDelete,
    onNoteColorChange: handleNoteColorChange,
    onNoteCreate:      handleNoteCreate,
    ...extra,
  };
}

function reRenderCanvas(extra = {}) {
  if (assignments.length > 0) {
    const labels = clusters.map((c) => c.label);
    renderCanvas(canvasContainer, notes, assignments, labels, renderOptions(extra));
    applyHullVisibility();
  } else {
    renderCanvas(canvasContainer, notes, null, null, renderOptions(extra));
  }
}

// Drops cluster state. Adding/deleting notes shifts indices, which would
// misalign assignments[i] with notes[i] and stale the embeddings cache.
function invalidateClustering() {
  if (assignments.length === 0) return;
  assignments       = [];
  clusters          = [];
  embeddingsReduced = [];
  invalidateSemanticView();
  viewToggle.classList.add('hidden');
  hullToggleWrap.classList.add('hidden');
  showView('canvas');
}

function handleNoteDragEnd(id, { x, y, z }) {
  const i = noteIdToIdx.get(id);
  if (i === undefined) return;
  notes[i].x = x;
  notes[i].y = y;
  notes[i].z = z;
  // If hulls are present, re-render to recompute around the new position.
  if (assignments.length > 0) reRenderCanvas();
  setDirty(true);
}

function handleNoteEdit(id, newText) {
  const i = noteIdToIdx.get(id);
  if (i === undefined) return;
  notes[i].text = newText;
  setDirty(true);
  // Text changes don't shift indices — assignments stay structurally valid.
  // Embeddings are stale though; the user can re-cluster for updated themes.
}

function handleNoteColorChange(id, color) {
  const i = noteIdToIdx.get(id);
  if (i === undefined) return;
  notes[i].color = color;
  setDirty(true);
  reRenderCanvas();
}

function handleNoteDelete(id) {
  const i = noteIdToIdx.get(id);
  if (i === undefined) return;
  notes.splice(i, 1);
  noteIdToIdx = new Map(notes.map((n, idx) => [n.id, idx]));
  invalidateClustering();
  setDirty(true);
  reRenderCanvas();
}

function handleNoteCreate({ x, y }) {
  const newNote = {
    id:     `note_${crypto.randomUUID().slice(0, 8)}`,
    text:   'New note',
    x, y,
    author: 'you',
    color:  'yellow',
    z:      notes.reduce((m, n) => (n.z > m ? n.z : m), 0) + 1,
  };
  notes.push(newNote);
  noteIdToIdx.set(newNote.id, notes.length - 1);
  invalidateClustering();
  setDirty(true);
  reRenderCanvas({ autoEditNoteId: newNote.id });
}

// ─── Cluster action ───────────────────────────────────────────────
btnClusterAction.addEventListener('click', async () => {
  if (notes.length === 0) return;

  btnClusterAction.disabled = true;
  showProgress('Warming up...', 0);

  try {
    // All inference runs in the browser — no note text ever leaves the tab
    const {
      results: pipelineResults,
      embeddingsReduced: reduced,
      perNoteSilhouette,
      centroids,
    } = await clusterNotes(notes, { onProgress: updateProgress });
    embeddingsReduced = reduced;

    const regularResults = pipelineResults.filter(r => r.cluster_id !== -1);
    const noiseResult    = pipelineResults.find(r => r.cluster_id === -1) ?? null;

    const idxOf  = new Map(notes.map((n, i) => [n.id, i]));
    assignments  = new Array(notes.length).fill(0);
    const labels = [];
    clusters     = [];

    regularResults.forEach((r, colorIdx) => {
      labels.push(r.label);
      clusters.push({ label: r.label, keyphrases: r.keyphrases ?? [], silhouette_score: r.silhouette_score, note_ids: r.note_ids });
      for (const id of r.note_ids) {
        const i = idxOf.get(id);
        if (i !== undefined) assignments[i] = colorIdx;
      }
    });

    if (noiseResult) {
      const noiseColorIdx = regularResults.length;
      labels.push(noiseResult.label);
      clusters.push({ label: noiseResult.label, keyphrases: [], silhouette_score: 0, note_ids: noiseResult.note_ids });
      for (const id of noiseResult.note_ids) {
        const i = idxOf.get(id);
        if (i !== undefined) assignments[i] = noiseColorIdx;
      }
    }

    // ── Level 1: Reactive agent (synchronous, runs before rendering) ──────
    const insights = computeInsights(notes, clusters, assignments, perNoteSilhouette, centroids);

    invalidateSemanticView();
    hideProgress();

    renderCanvas(canvasContainer, notes, assignments, labels, renderOptions());
    clusterViewApi = renderClusterView(clusterContainer, notes, assignments, clusters, { insights });

    applyHullVisibility();
    viewToggle.classList.remove('hidden');
    hullToggleWrap.classList.remove('hidden');
    showView('cluster');

    // ── Level 3: ReAct agent (async, streams to Agent Activity panel) ─────
    // FLAN-T5 is already loaded from Stage 4 — getFlanT5() returns immediately.
    const flanT5 = await getFlanT5().catch(() => null);
    runReActLoop(
      insights.rankedClusters,
      notes,
      (step) => clusterViewApi?.addAgentStep(step),
      flanT5,
      insights,
    ).then(({ markdownReport }) => {
      clusterViewApi?.activateDownload(markdownReport, 'sticky-insights-report.md');
    }).catch(err => {
      console.warn('[agent] ReAct loop error:', err);
    });

  } catch (err) {
    hideProgress();
    console.error('Insight discovery failed:', err);
    alert(`Insight discovery failed: ${err.message}`);
  } finally {
    btnClusterAction.disabled = false;
  }
});

// ─── Board picker: import / starter / download ────────────────────
function showPickerError(msg) {
  boardPickerError.textContent = msg;
  boardPickerError.classList.remove('hidden');
}

function clearPickerError() {
  boardPickerError.textContent = '';
  boardPickerError.classList.add('hidden');
}

// Loads validated notes into app state and reveals the canvas.
function loadBoard({ data, name }) {
  let validated;
  try {
    validated = validateNotes(data);
  } catch (err) {
    showPickerError(`Couldn't load that board: ${err.message}`);
    return;
  }
  notes            = validated;
  noteIdToIdx      = new Map(notes.map((n, i) => [n.id, i]));
  currentBoardName = name;
  setDirty(false);
  boardPicker.classList.add('hidden');
  canvasView.classList.remove('hidden');
  btnClusterAction.disabled = false;
  btnDownload.classList.remove('hidden');
  renderCanvas(canvasContainer, notes, null, null, renderOptions());
}

async function handleImport() {
  clearPickerError();
  try {
    const { data, name } = await pickBoardFile();
    loadBoard({ data, name });
  } catch (err) {
    if (err.name === 'AbortError') return; // user cancelled the picker
    showPickerError(`Couldn't import: ${err.message}`);
  }
}

async function handleStartFromStarter() {
  clearPickerError();
  try {
    const data = await loadNotes('/data/sticky_notes.json');
    loadBoard({ data, name: 'sticky-notes.json' });
  } catch (err) {
    showPickerError(`Couldn't load starter: ${err.message}`);
  }
}

function handleDownload() {
  if (notes.length === 0) return;
  downloadBoard(notes, currentBoardName);
  setDirty(false);
}

btnImport.addEventListener('click', handleImport);
btnStarter.addEventListener('click', handleStartFromStarter);
btnDownload.addEventListener('click', handleDownload);

btnClusterAction.disabled = true;
