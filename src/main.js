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
import { clusterNotes } from './pipeline.js';
import { renderCanvas } from './canvas-view.js';
import { renderClusterView } from './cluster-view.js';
import { renderSemanticView, invalidateSemanticView } from './semantic-view.js';

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

// ─── Drag → dirty (no auto-save in this app) ──────────────────────
// Drag updates positions in memory only. Persistence happens when the
// user clicks Download. The `dirty` flag drives the Download button
// styling and a beforeunload warning so unsaved changes aren't lost
// to a stray tab close.

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

function handleNoteDragEnd(id, { x, y, z }) {
  const i = noteIdToIdx.get(id);
  if (i === undefined) return;
  notes[i].x = x;
  notes[i].y = y;
  notes[i].z = z;

  if (assignments.length > 0) {
    const labels = clusters.map((c) => c.label);
    renderCanvas(canvasContainer, notes, assignments, labels, { onDragEnd: handleNoteDragEnd });
    applyHullVisibility();
  }

  setDirty(true);
}

// ─── Cluster action ───────────────────────────────────────────────
btnClusterAction.addEventListener('click', async () => {
  if (notes.length === 0) return;

  btnClusterAction.disabled = true;
  showProgress('Warming up...', 0);

  try {
    // All inference runs in the browser — no note text ever leaves the tab
    const { results: pipelineResults, embeddingsReduced: reduced } =
      await clusterNotes(notes, { onProgress: updateProgress });
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

    invalidateSemanticView();

    hideProgress();

    renderCanvas(canvasContainer, notes, assignments, labels, { onDragEnd: handleNoteDragEnd });
    clusterViewApi = renderClusterView(clusterContainer, notes, assignments, clusters);

    applyHullVisibility();

    viewToggle.classList.remove('hidden');
    hullToggleWrap.classList.remove('hidden');
    showView('canvas');

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
  renderCanvas(canvasContainer, notes, null, null, { onDragEnd: handleNoteDragEnd });
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
