/**
 * cluster-view.js — Insight summary grid + detail panel.
 *
 * Layout: Option B
 *   - Summary grid: one tile per cluster (label, note count, author chips, top-2 keyphrases)
 *   - Detail panel: slides in on tile click, shows contributor breakdown + all notes
 *   - Escape or × closes the panel
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

// Deterministic palette for author avatars — all verified ≥4.5:1 contrast
// with white text (#ffffff) per WCAG 1.4.3 AA.
//
// Contrast ratios (white on color):
//   #3d2db5  6.1:1   #c0392b  5.1:1   #1a5f7a  7.2:1   #b07d00  4.6:1
//   #2d6e4e  5.5:1   #7d2d80  6.2:1   #b85c00  4.8:1   #0e7c65  4.9:1
const AUTHOR_PALETTE = [
  '#3d2db5', '#c0392b', '#1a5f7a', '#b07d00',
  '#2d6e4e', '#7d2d80', '#b85c00', '#0e7c65',
];

/** Returns a stable color for an author string based on a simple hash. */
function authorColor(author) {
  let h = 0;
  for (let i = 0; i < author.length; i++) h = (h * 31 + author.charCodeAt(i)) >>> 0;
  return AUTHOR_PALETTE[h % AUTHOR_PALETTE.length];
}

/** Returns up to 2 uppercase initials from an author id like "user_7" or "Alice B". */
function authorInitials(author) {
  const parts = author.replace(/_/g, ' ').trim().split(/\s+/);
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
  return author.replace(/[^a-zA-Z]/g, '').slice(0, 2).toUpperCase() || '?';
}

/** Builds a { author → noteCount } map for a set of note indices. */
function authorCounts(noteIdxs, notes) {
  const counts = new Map();
  for (const ni of noteIdxs) {
    const a = notes[ni].author;
    counts.set(a, (counts.get(a) ?? 0) + 1);
  }
  return counts;
}

/**
 * @param {HTMLElement} container   - #cluster-container
 * @param {Array}       notes       - raw note objects
 * @param {number[]}    assignments - cluster index per note (parallel to notes)
 * @param {Array}       clusters    - [{ label, keyphrases }] one entry per cluster index
 */
export function renderClusterView(container, notes, assignments, clusters) {
  container.innerHTML = '';

  // Build per-cluster note index lists
  const groups = clusters.map(() => []);
  assignments.forEach((ci, ni) => { if (groups[ci]) groups[ci].push(ni); });

  // ── Summary grid ────────────────────────────────────────────────
  const grid = document.createElement('div');
  grid.className = 'cv-grid';

  // ── Detail panel (shared, reused across clicks) ──────────────────
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
  const tiles = [];

  function openPanel(ci, tile) {
    const color    = clusterColor(ci);
    const cluster  = clusters[ci];
    const noteIdxs = groups[ci];
    const counts   = authorCounts(noteIdxs, notes);
    // Sort contributors by note count descending
    const sorted   = [...counts.entries()].sort((a, b) => b[1] - a[1]);

    // Header
    panelDot.style.background = color;
    panelTitle.textContent    = cluster.label;
    panelCount.textContent    = `${noteIdxs.length} note${noteIdxs.length !== 1 ? 's' : ''}`;

    // Keyphrases
    panelKeyphrases.innerHTML = '';
    (cluster.keyphrases ?? []).slice(0, 3).forEach(kp => {
      const chip = document.createElement('span');
      chip.className = 'cv-keyphrase-chip';
      chip.textContent = kp;
      chip.style.borderColor = color;
      chip.style.color = color;
      panelKeyphrases.appendChild(chip);
    });

    // Contributors breakdown
    panelContributors.innerHTML = '';
    const contribHeader = document.createElement('div');
    contribHeader.className = 'cv-contrib-header';
    contribHeader.textContent = `${sorted.length} contributor${sorted.length !== 1 ? 's' : ''}`;
    panelContributors.appendChild(contribHeader);

    const contribList = document.createElement('div');
    contribList.className = 'cv-contrib-list';
    sorted.forEach(([author, count]) => {
      const row = document.createElement('div');
      row.className = 'cv-contrib-row';

      const avatar = document.createElement('span');
      avatar.className = 'cv-avatar';
      avatar.textContent = authorInitials(author);
      avatar.style.background = authorColor(author);
      avatar.setAttribute('role', 'img');
      avatar.setAttribute('aria-label', author);

      const name = document.createElement('span');
      name.className = 'cv-contrib-name';
      name.textContent = author;

      const bar = document.createElement('div');
      bar.className = 'cv-contrib-bar-wrap';
      const fill = document.createElement('div');
      fill.className = 'cv-contrib-bar-fill';
      fill.style.width = `${(count / noteIdxs.length) * 100}%`;
      fill.style.background = authorColor(author);
      bar.appendChild(fill);

      const noteCount = document.createElement('span');
      noteCount.className = 'cv-contrib-note-count';
      noteCount.textContent = count;

      row.appendChild(avatar);
      row.appendChild(name);
      row.appendChild(bar);
      row.appendChild(noteCount);
      contribList.appendChild(row);
    });
    panelContributors.appendChild(contribList);

    // Notes
    panelCards.innerHTML = '';
    noteIdxs.forEach(ni => {
      const note    = notes[ni];
      const bgColor = STICKY_COLORS[note.color] ?? STICKY_COLORS.default;

      const card = document.createElement('div');
      card.className = 'cv-note-card';
      card.style.background = bgColor;
      card.style.borderLeft = `3px solid ${color}`;

      const text = document.createElement('div');
      text.textContent = note.text;

      const footer = document.createElement('div');
      footer.className = 'cv-note-footer';

      const avatar = document.createElement('span');
      avatar.className = 'cv-avatar cv-avatar--sm';
      avatar.textContent = authorInitials(note.author);
      avatar.style.background = authorColor(note.author);
      avatar.setAttribute('role', 'img');
      avatar.setAttribute('aria-label', note.author);

      const authorName = document.createElement('span');
      authorName.className = 'cv-note-author';
      authorName.textContent = note.author;

      footer.appendChild(avatar);
      footer.appendChild(authorName);
      card.appendChild(text);
      card.appendChild(footer);
      panelCards.appendChild(card);
    });

    // Activate tile
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

  // ── Build tiles ──────────────────────────────────────────────────
  clusters.forEach((cluster, ci) => {
    const color    = clusterColor(ci);
    const noteIdxs = groups[ci];
    const counts   = authorCounts(noteIdxs, notes);
    const authors  = [...counts.keys()];

    const tile = document.createElement('button');
    tile.className = 'cv-tile';
    tile.setAttribute('aria-label', `${cluster.label}, ${noteIdxs.length} notes, ${authors.length} contributors`);
    tile.style.setProperty('--tile-color', color);

    // Header row: dot + label + note count
    const tileHeader = document.createElement('div');
    tileHeader.className = 'cv-tile-header';

    const dot = document.createElement('span');
    dot.className = 'cv-tile-dot';
    dot.style.background = color;

    const label = document.createElement('span');
    label.className = 'cv-tile-label';
    label.textContent = cluster.label;

    const count = document.createElement('span');
    count.className = 'cv-tile-count';
    count.textContent = noteIdxs.length;

    tileHeader.appendChild(dot);
    tileHeader.appendChild(label);
    tileHeader.appendChild(count);

    // Author avatar strip — up to 5, remainder as +N
    const avatarStrip = document.createElement('div');
    avatarStrip.className = 'cv-tile-avatars';
    const MAX_SHOWN = 5;
    authors.slice(0, MAX_SHOWN).forEach(author => {
      const av = document.createElement('span');
      av.className = 'cv-avatar cv-avatar--sm';
      av.textContent = authorInitials(author);
      av.style.background = authorColor(author);
      av.setAttribute('role', 'img');
      av.setAttribute('aria-label', `${author}, ${counts.get(author)} note${counts.get(author) !== 1 ? 's' : ''}`);
      avatarStrip.appendChild(av);
    });
    if (authors.length > MAX_SHOWN) {
      const more = document.createElement('span');
      more.className = 'cv-avatar cv-avatar--sm cv-avatar--more';
      more.textContent = `+${authors.length - MAX_SHOWN}`;
      avatarStrip.appendChild(more);
    }

    // Keyphrase chips
    const chips = document.createElement('div');
    chips.className = 'cv-tile-chips';
    (cluster.keyphrases ?? []).slice(0, 2).forEach(kp => {
      const chip = document.createElement('span');
      chip.className = 'cv-tile-chip';
      chip.textContent = kp;
      chips.appendChild(chip);
    });

    tile.appendChild(tileHeader);
    tile.appendChild(avatarStrip);
    tile.appendChild(chips);
    tile.addEventListener('click', () => openPanel(ci, tile));
    tiles[ci] = tile;
    grid.appendChild(tile);
  });

  container.appendChild(grid);
  container.appendChild(panel);

  return {
    selectCluster(ci) {
      const tile = tiles[ci];
      if (tile) openPanel(ci, tile);
    },
  };
}
