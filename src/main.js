/**
 * main.js — Application entry point.
 * Wires together: board picker → canvas render → clustering → both views.
 *
 * Privacy: note content stays in the browser. With FSA persistence, it also
 * stays on the user's local disk inside whichever folder they chose; nothing
 * is uploaded by this app, and any cross-device sync is delegated entirely
 * to the user's OS-level cloud-drive client.
 */

import { loadNotes, validateNotes } from './loader.js';
import {
  isSupported as fsSupported,
  openBoard, createBoard, readBoard,
  listRecentBoards, forgetBoard,
} from './board-fs.js';
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

const boardPicker            = document.getElementById('board-picker');
const btnContinue            = document.getElementById('btn-continue');
const continueName           = document.getElementById('continue-name');
const btnOpen                = document.getElementById('btn-open');
const btnCreate              = document.getElementById('btn-create');
const boardPickerError       = document.getElementById('board-picker-error');
const boardPickerUnsupported = document.getElementById('board-picker-unsupported');

// ─── App state ───────────────────────────────────────────────────
// Module-scope so showView() can pass current values to renderSemanticView
// regardless of when the Semantics tab is activated relative to clustering.
let notes             = [];
let embeddingsReduced = [];
let assignments       = [];
let clusters          = [];
let clusterViewApi    = null;
let currentBoard      = null; // { id, handle, name } once a board is opened

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

    // Adapt ClusterResult[] → format expected by existing renderers.
    // Regular clusters (cluster_id >= 0) are rendered with sequential colour indices.
    // The noise group (cluster_id === -1) is appended as an extra group.
    const regularResults = pipelineResults.filter(r => r.cluster_id !== -1);
    const noiseResult    = pipelineResults.find(r => r.cluster_id === -1) ?? null;

    const noteIdToIdx = new Map(notes.map((n, i) => [n.id, i]));
    assignments = new Array(notes.length).fill(0);
    const labels = [];
    clusters     = [];

    regularResults.forEach((r, colorIdx) => {
      labels.push(r.label);
      clusters.push({ label: r.label, keyphrases: r.keyphrases ?? [], silhouette_score: r.silhouette_score, note_ids: r.note_ids });
      for (const id of r.note_ids) {
        const i = noteIdToIdx.get(id);
        if (i !== undefined) assignments[i] = colorIdx;
      }
    });

    if (noiseResult) {
      const noiseColorIdx = regularResults.length;
      labels.push(noiseResult.label);
      clusters.push({ label: noiseResult.label, keyphrases: [], silhouette_score: 0, note_ids: noiseResult.note_ids });
      for (const id of noiseResult.note_ids) {
        const i = noteIdToIdx.get(id);
        if (i !== undefined) assignments[i] = noiseColorIdx;
      }
    }

    invalidateSemanticView();

    hideProgress();

    renderCanvas(canvasContainer, notes, assignments, labels);
    clusterViewApi = renderClusterView(clusterContainer, notes, assignments, clusters);

    // Apply hull visibility immediately after render — hull-layer starts hidden
    // because the toggle defaults to OFF. Re-clustering respects current toggle state.
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

// ─── Board picker ─────────────────────────────────────────────────
function showPickerError(msg) {
  boardPickerError.textContent = msg;
  boardPickerError.classList.remove('hidden');
}

function clearPickerError() {
  boardPickerError.textContent = '';
  boardPickerError.classList.add('hidden');
}

// Loads validated notes into app state and reveals the canvas.
// Caller has already confirmed read access; data is the parsed JSON contents.
async function loadBoard({ id, handle, data, name }) {
  let validated;
  try {
    validated = validateNotes(data);
  } catch (err) {
    // Bad schema — drop the recents entry so we don't keep offering it.
    if (id) await forgetBoard(id);
    showPickerError(`Couldn't load that board: ${err.message}`);
    return;
  }
  notes        = validated;
  currentBoard = { id, handle, name };
  boardPicker.classList.add('hidden');
  canvasView.classList.remove('hidden');
  btnClusterAction.disabled = false;
  renderCanvas(canvasContainer, notes);
}

async function handleContinue(record) {
  clearPickerError();
  try {
    const data = await readBoard(record.handle); // ensurePermission inside
    await loadBoard({ id: record.id, handle: record.handle, data, name: record.name });
  } catch (err) {
    if (err.name === 'NotAllowedError') {
      showPickerError('Permission was denied. Pick another board?');
      return;
    }
    if (err.name === 'NotFoundError') {
      await forgetBoard(record.id);
      btnContinue.classList.add('hidden');
      showPickerError(`"${record.name}" wasn't found on disk and was removed from recents.`);
      return;
    }
    showPickerError(`Couldn't open "${record.name}": ${err.message}`);
  }
}

async function handleOpen() {
  clearPickerError();
  try {
    const result = await openBoard();
    await loadBoard({ ...result, name: result.handle.name });
  } catch (err) {
    if (err.name === 'AbortError') return; // user cancelled the picker
    showPickerError(`Couldn't open board: ${err.message}`);
  }
}

async function handleCreate() {
  clearPickerError();
  try {
    // Seed from the bundled starter dataset (already validated by loadNotes).
    const starter = await loadNotes('/data/sticky_notes.json');
    const result  = await createBoard(starter, 'sticky-notes.json');
    await loadBoard({ ...result, name: result.handle.name });
  } catch (err) {
    if (err.name === 'AbortError') return;
    showPickerError(`Couldn't create board: ${err.message}`);
  }
}

btnOpen.addEventListener('click', handleOpen);
btnCreate.addEventListener('click', handleCreate);

async function setupPicker() {
  btnClusterAction.disabled = true;

  if (!fsSupported) {
    boardPickerUnsupported.classList.remove('hidden');
    btnOpen.disabled   = true;
    btnCreate.disabled = true;
    return;
  }

  const recents = await listRecentBoards();
  if (recents.length > 0) {
    const last = recents[0];
    continueName.textContent = last.name;
    btnContinue.classList.remove('hidden');
    btnContinue.onclick = () => handleContinue(last);
  }
}

setupPicker();
