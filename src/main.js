/**
 * main.js — Application entry point.
 * Wires together: data loading → canvas render → clustering → both views.
 */

import { loadNotes } from './loader.js';
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

// ─── App state ───────────────────────────────────────────────────
// Module-scope so showView() can pass current values to renderSemanticView
// regardless of when the Semantics tab is activated relative to clustering.
let notes             = [];
let embeddingsReduced = [];
let assignments       = [];
let clusters          = [];
let clusterViewApi    = null;

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

// ─── Initial load: fetch notes and render canvas immediately ──────
async function init() {
  try {
    notes = await loadNotes('/data/sticky_notes.json');
    renderCanvas(canvasContainer, notes);
  } catch (err) {
    console.error('Failed to load notes:', err);
    canvasContainer.innerHTML = `
      <p style="padding:24px;color:#e11d48;font-size:14px;">
        Error loading sticky notes: ${err.message}
      </p>`;
  }
}

init();
