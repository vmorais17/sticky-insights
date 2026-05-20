/**
 * cluster-view.js — Insights summary with agentic capabilities (Levels 1–3)
 *
 * Layout:
 *   Left column (scrollable):
 *     1. Board summary (Level 1 reactive agent)
 *     2. Merge suggestion cards (Level 1, dismissible)
 *     3. Agent Activity panel (Level 2/3, collapsible, streams steps)
 *     4. Cluster grid (sorted by composite rank, rank badges)
 *     5. Needs Review section (Level 1, collapsible)
 *   Right panel (detail panel, slides in on tile click)
 */

import { clusterColor } from './canvas-view.js';

const STICKY_COLORS = {
  yellow:  '#FFF176',
  pink:    '#F48FB1',
  blue:    '#81D4FA',
  green:   '#C5E1A5',
  purple:  '#CE93D8',
  orange:  '#FFCC80',
  white:   '#FFFFFF',
  default: '#FFF176',
};

const AUTHOR_PALETTE = [
  '#3d2db5', '#c0392b', '#1a5f7a', '#b07d00',
  '#2d6e4e', '#7d2d80', '#b85c00', '#0e7c65',
];

function authorColor(author) {
  let h = 0;
  for (let i = 0; i < author.length; i++) h = (h * 31 + author.charCodeAt(i)) >>> 0;
  return AUTHOR_PALETTE[h % AUTHOR_PALETTE.length];
}

function authorInitials(author) {
  const parts = author.replace(/_/g, ' ').trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return author.replace(/[^a-zA-Z]/g, '').slice(0, 2).toUpperCase() || '?';
}

function authorCounts(noteIdxs, notes) {
  const counts = new Map();
  for (const ni of noteIdxs) {
    const a = notes[ni].author;
    counts.set(a, (counts.get(a) ?? 0) + 1);
  }
  return counts;
}

// ─── Level 1: Board summary banner ───────────────────────────────────────────

function buildBoardSummary(summaryText) {
  const el = document.createElement('div');
  el.className = 'cv-board-summary';
  el.setAttribute('role', 'status');
  el.textContent = summaryText;
  return el;
}

// ─── Level 1: Merge suggestion cards ─────────────────────────────────────────

function buildMergeSuggestions(suggestions, clusters, onDismiss) {
  if (!suggestions || suggestions.length === 0) return null;

  const wrapper = document.createElement('div');
  wrapper.className = 'cv-merge-suggestions';

  suggestions.forEach((suggestion, idx) => {
    if (suggestion.dismissed) return;

    const { labelA, labelB, clusterA, clusterB, similarity } = suggestion;
    const colorA = clusterColor(clusterA);
    const colorB = clusterColor(clusterB);
    const noteCountA = clusters[clusterA]?.note_ids?.length ?? 0;
    const noteCountB = clusters[clusterB]?.note_ids?.length ?? 0;
    const silA = (clusters[clusterA]?.silhouette_score ?? 0).toFixed(2);
    const silB = (clusters[clusterB]?.silhouette_score ?? 0).toFixed(2);

    const card = document.createElement('div');
    card.className = 'cv-merge-card';
    card.setAttribute('role', 'status');

    card.innerHTML = `
      <div class="cv-merge-card-body">
        <span class="cv-merge-icon" aria-hidden="true">~</span>
        <div class="cv-merge-text">
          <span class="cv-merge-label">Clusters may describe the same theme. Consider merging.</span>
          <div class="cv-merge-clusters">
            <span class="cv-merge-cluster-tag" style="border-color:${colorA}; color:${colorA}">
              ${labelA} <span class="cv-merge-meta">${noteCountA} notes · sil ${silA}</span>
            </span>
            <span class="cv-merge-sep">and</span>
            <span class="cv-merge-cluster-tag" style="border-color:${colorB}; color:${colorB}">
              ${labelB} <span class="cv-merge-meta">${noteCountB} notes · sil ${silB}</span>
            </span>
          </div>
          <span class="cv-merge-similarity">Centroid similarity: ${(similarity * 100).toFixed(0)}%</span>
        </div>
        <button class="cv-merge-dismiss" aria-label="Dismiss merge suggestion" data-idx="${idx}">✕</button>
      </div>
    `;

    card.querySelector('.cv-merge-dismiss').addEventListener('click', () => {
      suggestion.dismissed = true;
      card.style.opacity   = '0';
      card.style.transform = 'translateY(-4px)';
      setTimeout(() => card.remove(), 200);
      onDismiss?.(idx);
    });

    wrapper.appendChild(card);
  });

  return wrapper.children.length === 0 ? null : wrapper;
}

// ─── Level 2/3: Agent Activity panel ─────────────────────────────────────────

const TOOL_LABELS = {
  classify_cluster:   'Classify',
  generate_brief:     'Brief',
  create_action_items:'Action items',
  export_summary:     'Export',
};

function buildAgentPanel() {
  const section = document.createElement('div');
  section.className = 'cv-agent-section';

  const header = document.createElement('button');
  header.className = 'cv-agent-header';
  header.setAttribute('aria-expanded', 'false');
  header.setAttribute('aria-controls', 'cv-agent-body');
  header.innerHTML = `
    <span class="cv-agent-header-dot"></span>
    <span class="cv-agent-header-title">Agent Activity</span>
    <span class="cv-agent-header-chevron" aria-hidden="true">›</span>
  `;

  const body = document.createElement('div');
  body.className = 'cv-agent-body';
  body.id        = 'cv-agent-body';
  body.setAttribute('role', 'log');
  body.setAttribute('aria-live', 'polite');

  const goalEl = document.createElement('div');
  goalEl.className = 'cv-agent-goal';
  goalEl.textContent = 'Waiting for pipeline to complete…';

  const stepsEl = document.createElement('div');
  stepsEl.className = 'cv-agent-steps';

  const completeEl = document.createElement('div');
  completeEl.className = 'cv-agent-complete hidden';

  const downloadBtn = document.createElement('button');
  downloadBtn.className = 'btn-primary cv-download-btn';
  downloadBtn.textContent = 'Download Report';
  downloadBtn.disabled = true;
  downloadBtn.title = 'Run the pipeline first to generate a report';

  body.appendChild(goalEl);
  body.appendChild(stepsEl);
  body.appendChild(completeEl);
  body.appendChild(downloadBtn);

  section.appendChild(header);
  section.appendChild(body);

  // Toggle expand/collapse
  header.addEventListener('click', () => {
    const open = body.classList.toggle('cv-agent-body--open');
    header.setAttribute('aria-expanded', open ? 'true' : 'false');
    header.querySelector('.cv-agent-header-chevron').textContent = open ? '⌄' : '›';
  });

  let _downloadContent = null;
  let _downloadFilename = 'report.md';

  downloadBtn.addEventListener('click', () => {
    if (!_downloadContent) return;
    const blob = new Blob([_downloadContent], { type: 'text/markdown' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = _downloadFilename;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Public API exposed to main.js
  function addAgentStep(step) {
    // Auto-open the panel on first meaningful step
    if (!body.classList.contains('cv-agent-body--open') && step.type !== 'progress') {
      body.classList.add('cv-agent-body--open');
      header.setAttribute('aria-expanded', 'true');
      header.querySelector('.cv-agent-header-chevron').textContent = '⌄';
    }

    switch (step.type) {
      case 'goal': {
        goalEl.textContent = step.text;
        break;
      }

      case 'reason': {
        appendStep(stepsEl, 'reason', null, step.text, null);
        break;
      }

      case 'tool': {
        const toolLabel = TOOL_LABELS[step.tool] ?? step.tool;
        const clusterName = step.cluster === 'all' ? 'all clusters' : step.cluster;
        if (step.status === 'running') {
          appendStep(stepsEl, 'running', toolLabel, clusterName, null);
        } else {
          // Update the last running row for this tool+cluster to 'done'
          const last = findLastRunningRow(stepsEl, step.tool, step.cluster);
          if (last) {
            markStepDone(last, step.result);
          } else {
            appendStep(stepsEl, 'done', toolLabel, clusterName, step.result);
          }
        }
        break;
      }

      case 'retry': {
        const toolLabel = TOOL_LABELS[step.tool] ?? step.tool;
        appendStep(stepsEl, 'retry', toolLabel, step.cluster, 'retrying…');
        break;
      }

      case 'progress': {
        const pct = Math.round((step.processed / step.total) * 100);
        // Update or create progress bar
        let progressRow = stepsEl.querySelector('.cv-agent-progress-row');
        if (!progressRow) {
          progressRow = document.createElement('div');
          progressRow.className = 'cv-agent-progress-row';
          progressRow.innerHTML = `<div class="cv-agent-progress-bar-track"><div class="cv-agent-progress-bar-fill"></div></div><span class="cv-agent-progress-label"></span>`;
          stepsEl.appendChild(progressRow);
        }
        progressRow.querySelector('.cv-agent-progress-bar-fill').style.width = `${pct}%`;
        progressRow.querySelector('.cv-agent-progress-label').textContent =
          `${step.processed} / ${step.total} clusters`;
        break;
      }

      case 'warn': {
        appendStep(stepsEl, 'warn', null, null, step.text);
        break;
      }

      case 'complete': {
        completeEl.classList.remove('hidden');
        completeEl.textContent = step.text;
        header.querySelector('.cv-agent-header-dot').classList.add('cv-agent-header-dot--done');
        break;
      }
    }

    // Scroll to bottom
    stepsEl.scrollTop = stepsEl.scrollHeight;
  }

  function activateDownload(content, filename) {
    _downloadContent  = content;
    _downloadFilename = filename ?? 'report.md';
    downloadBtn.disabled = false;
    downloadBtn.title    = 'Download the analysis report as a Markdown file';
  }

  return { section, addAgentStep, activateDownload };
}

function appendStep(container, status, tool, cluster, result) {
  const row = document.createElement('div');
  row.className = `cv-agent-step cv-agent-step--${status}`;
  row.dataset.tool    = tool    ?? '';
  row.dataset.cluster = cluster ?? '';

  const icon  = status === 'running' ? '◌' : status === 'done' ? '✓' : status === 'retry' ? '↻' : status === 'warn' ? '!' : '→';
  const label = tool ? `${tool} — ${cluster}` : (cluster || result || '');

  row.innerHTML = `
    <span class="cv-agent-step-icon" aria-hidden="true">${icon}</span>
    <span class="cv-agent-step-label">${label}</span>
    ${result ? `<span class="cv-agent-step-result">${result}</span>` : ''}
  `;
  container.appendChild(row);

  // Trigger enter animation
  requestAnimationFrame(() => row.classList.add('cv-agent-step--visible'));
  return row;
}

function findLastRunningRow(container, tool, cluster) {
  const toolLabel = TOOL_LABELS[tool] ?? tool;
  const rows = Array.from(container.querySelectorAll('.cv-agent-step--running'));
  return rows.reverse().find(
    r => r.dataset.tool === toolLabel && r.dataset.cluster === cluster
  );
}

function markStepDone(row, result) {
  row.classList.remove('cv-agent-step--running');
  row.classList.add('cv-agent-step--done');
  row.querySelector('.cv-agent-step-icon').textContent = '✓';
  if (result) {
    const existing = row.querySelector('.cv-agent-step-result');
    if (existing) {
      existing.textContent = result;
    } else {
      const span = document.createElement('span');
      span.className   = 'cv-agent-step-result';
      span.textContent = result;
      row.appendChild(span);
    }
  }
}

// ─── Level 1: Needs Review section ───────────────────────────────────────────

function buildNeedsReview(outliers) {
  if (!outliers || outliers.length === 0) return null;

  const section = document.createElement('div');
  section.className = 'cv-needs-review';

  const header = document.createElement('button');
  header.className = 'cv-needs-review-header';
  header.setAttribute('aria-expanded', 'false');
  header.setAttribute('aria-controls', 'cv-needs-review-body');
  header.innerHTML = `
    <span class="cv-needs-review-title">Needs Review (${outliers.length})</span>
    <span class="cv-needs-review-subtitle">The algorithm wasn't confident about these.</span>
    <span class="cv-needs-review-chevron" aria-hidden="true">›</span>
  `;

  const body = document.createElement('div');
  body.className = 'cv-needs-review-body';
  body.id        = 'cv-needs-review-body';

  outliers.forEach(outlier => {
    const card = document.createElement('div');
    card.className = 'cv-outlier-card';

    const textEl = document.createElement('div');
    textEl.className = 'cv-outlier-text';
    textEl.textContent = outlier.text;

    const meta = document.createElement('div');
    meta.className = 'cv-outlier-meta';

    const authorSpan = document.createElement('span');
    authorSpan.className = 'cv-outlier-author';

    const avatar = document.createElement('span');
    avatar.className   = 'cv-avatar cv-avatar--sm';
    avatar.textContent = authorInitials(outlier.author);
    avatar.style.background = authorColor(outlier.author);

    authorSpan.appendChild(avatar);
    authorSpan.appendChild(document.createTextNode(` ${outlier.author}`));

    const reasonSpan = document.createElement('span');
    reasonSpan.className   = 'cv-outlier-reason';
    reasonSpan.textContent = outlier.reason;

    meta.appendChild(authorSpan);
    meta.appendChild(reasonSpan);
    card.appendChild(textEl);
    card.appendChild(meta);
    body.appendChild(card);
  });

  section.appendChild(header);
  section.appendChild(body);

  header.addEventListener('click', () => {
    const open = body.classList.toggle('cv-needs-review-body--open');
    header.setAttribute('aria-expanded', open ? 'true' : 'false');
    header.querySelector('.cv-needs-review-chevron').textContent = open ? '⌄' : '›';
  });

  return section;
}

// ─── Main export ──────────────────────────────────────────────────────────────

/**
 * @param {HTMLElement} container    - #cluster-container
 * @param {Array}       notes        - raw note objects
 * @param {number[]}    assignments  - cluster index per note
 * @param {Array}       clusters     - [{ label, keyphrases, silhouette_score, note_ids }]
 * @param {object}      [options]
 * @param {object}      [options.insights]  - Level 1 insights from computeInsights()
 */
export function renderClusterView(container, notes, assignments, clusters, options = {}) {
  container.innerHTML = '';

  const { insights } = options;

  // Build per-cluster note index lists
  const groups = clusters.map(() => []);
  assignments.forEach((ci, ni) => { if (groups[ci]) groups[ci].push(ni); });

  // ── Left column (scrollable) ──────────────────────────────────────────────
  const leftCol = document.createElement('div');
  leftCol.className = 'cv-left-col';

  // 1. Board summary
  if (insights?.boardSummary) {
    leftCol.appendChild(buildBoardSummary(insights.boardSummary));
  }

  // 2. Merge suggestions
  if (insights?.mergeSuggestions?.length > 0) {
    const mergeEl = buildMergeSuggestions(insights.mergeSuggestions, clusters, null);
    if (mergeEl) leftCol.appendChild(mergeEl);
  }

  // 3. Agent Activity panel (always present, starts collapsed)
  const { section: agentSection, addAgentStep, activateDownload } = buildAgentPanel();
  leftCol.appendChild(agentSection);

  // 4. Cluster grid — sorted by composite rank score (Level 1)
  const sortedClusters = insights?.rankedClusters
    ?? clusters.map((c, ci) => ({ ...c, ci, rank: ci + 1 }));

  const grid = document.createElement('div');
  grid.className = 'cv-grid';

  // ── Detail panel (right side, shared) ──────────────────────────────────────
  const panel = document.createElement('div');
  panel.className = 'cv-panel';
  panel.setAttribute('role', 'complementary');
  panel.setAttribute('aria-label', 'Cluster detail');
  panel.innerHTML = `
    <div class="cv-panel-header">
      <div class="cv-panel-title-wrap">
        <span class="cv-panel-color-dot"></span>
        <span class="cv-panel-title"></span>
        <span class="cv-panel-count"></span>
      </div>
      <button class="cv-panel-close" aria-label="Close panel">✕</button>
    </div>
    <div class="cv-panel-keyphrases"></div>
    <div class="cv-panel-contributors"></div>
    <div class="cv-panel-cards"></div>
  `;

  const panelDot          = panel.querySelector('.cv-panel-color-dot');
  const panelTitle        = panel.querySelector('.cv-panel-title');
  const panelCount        = panel.querySelector('.cv-panel-count');
  const panelKeyphrases   = panel.querySelector('.cv-panel-keyphrases');
  const panelContributors = panel.querySelector('.cv-panel-contributors');
  const panelCards        = panel.querySelector('.cv-panel-cards');
  const panelClose        = panel.querySelector('.cv-panel-close');

  let activeTile = null;
  const tiles    = [];

  function openPanel(ci, tile) {
    const color    = clusterColor(ci);
    const cluster  = clusters[ci];
    const noteIdxs = groups[ci];
    const counts   = authorCounts(noteIdxs, notes);
    const sorted   = [...counts.entries()].sort((a, b) => b[1] - a[1]);

    panelDot.style.background = color;
    panelTitle.textContent    = cluster.label;
    panelCount.textContent    = `${noteIdxs.length} note${noteIdxs.length !== 1 ? 's' : ''}`;

    panelKeyphrases.innerHTML = '';
    (cluster.keyphrases ?? []).slice(0, 3).forEach(kp => {
      const chip = document.createElement('span');
      chip.className   = 'cv-keyphrase-chip';
      chip.textContent = kp;
      chip.style.borderColor = color;
      chip.style.color       = color;
      panelKeyphrases.appendChild(chip);
    });

    panelContributors.innerHTML = '';
    const contribHeader = document.createElement('div');
    contribHeader.className   = 'cv-contrib-header';
    contribHeader.textContent = `${sorted.length} contributor${sorted.length !== 1 ? 's' : ''}`;
    panelContributors.appendChild(contribHeader);

    const contribList = document.createElement('div');
    contribList.className = 'cv-contrib-list';
    sorted.forEach(([author, count]) => {
      const row = document.createElement('div');
      row.className = 'cv-contrib-row';

      const avatar = document.createElement('span');
      avatar.className   = 'cv-avatar';
      avatar.textContent = authorInitials(author);
      avatar.style.background = authorColor(author);
      avatar.setAttribute('role', 'img');
      avatar.setAttribute('aria-label', author);

      const name = document.createElement('span');
      name.className   = 'cv-contrib-name';
      name.textContent = author;

      const bar  = document.createElement('div');
      bar.className  = 'cv-contrib-bar-wrap';
      const fill = document.createElement('div');
      fill.className    = 'cv-contrib-bar-fill';
      fill.style.width  = `${(count / noteIdxs.length) * 100}%`;
      fill.style.background = authorColor(author);
      bar.appendChild(fill);

      const noteCount = document.createElement('span');
      noteCount.className   = 'cv-contrib-note-count';
      noteCount.textContent = count;

      row.appendChild(avatar);
      row.appendChild(name);
      row.appendChild(bar);
      row.appendChild(noteCount);
      contribList.appendChild(row);
    });
    panelContributors.appendChild(contribList);

    panelCards.innerHTML = '';
    noteIdxs.forEach(ni => {
      const note    = notes[ni];
      const bgColor = STICKY_COLORS[note.color] ?? STICKY_COLORS.default;

      const card = document.createElement('div');
      card.className         = 'cv-note-card';
      card.style.background  = bgColor;
      card.style.borderLeft  = `3px solid ${color}`;

      const text = document.createElement('div');
      text.textContent = note.text;

      const footer = document.createElement('div');
      footer.className = 'cv-note-footer';

      const avatar = document.createElement('span');
      avatar.className   = 'cv-avatar cv-avatar--sm';
      avatar.textContent = authorInitials(note.author);
      avatar.style.background = authorColor(note.author);
      avatar.setAttribute('role', 'img');
      avatar.setAttribute('aria-label', note.author);

      const authorName = document.createElement('span');
      authorName.className   = 'cv-note-author';
      authorName.textContent = note.author;

      footer.appendChild(avatar);
      footer.appendChild(authorName);
      card.appendChild(text);
      card.appendChild(footer);
      panelCards.appendChild(card);
    });

    if (activeTile) activeTile.classList.remove('cv-tile--active');
    activeTile = tile;
    tile.classList.add('cv-tile--active');
    tile.style.setProperty('--tile-active-color', color);

    panel.classList.add('cv-panel--open');
    panelClose.focus();
  }

  function closePanel() {
    panel.classList.remove('cv-panel--open');
    if (activeTile) {
      activeTile.classList.remove('cv-tile--active');
      activeTile.focus();
      activeTile = null;
    }
  }

  panelClose.addEventListener('click', closePanel);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && panel.classList.contains('cv-panel--open')) closePanel();
  });

  // ── Build tiles (in ranked order) ─────────────────────────────────────────
  sortedClusters.forEach((rankedCluster) => {
    const ci       = rankedCluster.ci ?? 0;
    const rank     = rankedCluster.rank;
    const cluster  = clusters[ci];
    const color    = clusterColor(ci);
    const noteIdxs = groups[ci];
    const counts   = authorCounts(noteIdxs, notes);
    const authors  = [...counts.keys()];

    const tile = document.createElement('button');
    tile.className = 'cv-tile';
    tile.setAttribute('aria-label', `${cluster.label}, rank ${rank}, ${noteIdxs.length} notes, ${authors.length} contributors`);
    tile.style.setProperty('--tile-color', color);

    // Rank badge
    const rankBadge = document.createElement('span');
    rankBadge.className   = 'cv-rank-badge';
    rankBadge.textContent = rank;
    rankBadge.setAttribute('aria-label', `Rank ${rank}`);

    const tileHeader = document.createElement('div');
    tileHeader.className = 'cv-tile-header';

    const dot = document.createElement('span');
    dot.className        = 'cv-tile-dot';
    dot.style.background = color;

    const label = document.createElement('span');
    label.className   = 'cv-tile-label';
    label.textContent = cluster.label;

    const count = document.createElement('span');
    count.className   = 'cv-tile-count';
    count.textContent = noteIdxs.length;

    tileHeader.appendChild(dot);
    tileHeader.appendChild(label);
    tileHeader.appendChild(count);

    const avatarStrip = document.createElement('div');
    avatarStrip.className = 'cv-tile-avatars';
    const MAX_SHOWN = 5;
    authors.slice(0, MAX_SHOWN).forEach(author => {
      const av = document.createElement('span');
      av.className   = 'cv-avatar cv-avatar--sm';
      av.textContent = authorInitials(author);
      av.style.background = authorColor(author);
      av.setAttribute('role', 'img');
      av.setAttribute('aria-label', `${author}, ${counts.get(author)} note${counts.get(author) !== 1 ? 's' : ''}`);
      avatarStrip.appendChild(av);
    });
    if (authors.length > MAX_SHOWN) {
      const more = document.createElement('span');
      more.className   = 'cv-avatar cv-avatar--sm cv-avatar--more';
      more.textContent = `+${authors.length - MAX_SHOWN}`;
      avatarStrip.appendChild(more);
    }

    const chips = document.createElement('div');
    chips.className = 'cv-tile-chips';
    (cluster.keyphrases ?? []).slice(0, 2).forEach(kp => {
      const chip = document.createElement('span');
      chip.className   = 'cv-tile-chip';
      chip.textContent = kp;
      chips.appendChild(chip);
    });

    tile.appendChild(rankBadge);
    tile.appendChild(tileHeader);
    tile.appendChild(avatarStrip);
    tile.appendChild(chips);
    tile.addEventListener('click', () => openPanel(ci, tile));
    tiles[ci] = tile;
    grid.appendChild(tile);
  });

  leftCol.appendChild(grid);

  // 5. Needs Review section (Level 1)
  const needsReviewEl = buildNeedsReview(insights?.outliers);
  if (needsReviewEl) leftCol.appendChild(needsReviewEl);

  container.appendChild(leftCol);
  container.appendChild(panel);

  return {
    selectCluster(ci) {
      const tile = tiles[ci];
      if (tile) openPanel(ci, tile);
    },
    addAgentStep,
    activateDownload,
  };
}
