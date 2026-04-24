/**
 * canvas-view.js — SVG canvas rendering notes at their original x/y positions.
 *
 * Notes are rendered as square sticky cards at the coordinates from the JSON.
 * Overlapping notes are pushed apart while staying near their original position.
 * After clustering, a cluster-color border + dot identifies each group.
 * Hover reveals the author name.
 */

import * as d3 from 'd3';

const NOTE_W  = 160;
const NOTE_H  = 160; // square stickies
const GAP     = 8;   // minimum gap between notes after collision resolution
const PADDING = 72;  // canvas padding around the note extent

// Sticky-note paper palette (pastel fills tuned for ≥4.5:1 text contrast)
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

/** Returns the hex fill color for a note's color key. */
export function stickyColor(colorKey) {
  return STICKY_COLORS[colorKey] ?? STICKY_COLORS.default;
}

// Cluster color palette — 24 distinct hues covering the full wheel.
// All verified ≥ 4.5:1 contrast ratio on white (WCAG 1.4.3 AA).
// 24 slots means k≤24 never repeats a color (K_MAX = 24).
//
// Contrast ratios (white text on color):
//   #be123c  6.3:1   #c2410c  5.8:1   #a16207  5.0:1   #3f6212  6.6:1
//   #166534  7.1:1   #0f766e  5.5:1   #0e7490  5.4:1   #0369a1  5.6:1
//   #1d4ed8  6.7:1   #4338ca  7.6:1   #6d28d9  6.9:1   #86198f  7.8:1
//   #9d174d  7.6:1   #92400e  7.0:1   #7c2d12  9.5:1   #14532d  8.2:1
//   #1e3a8a  9.1:1   #4a044e 12.4:1   #065f46  8.9:1   #1e40af  9.0:1
//   #6b21a8  8.9:1   #831843  9.1:1   #78350f  8.0:1   #134e4a  8.8:1
const CLUSTER_PALETTE = [
  '#be123c', // crimson       H≈345
  '#c2410c', // burnt orange  H≈20
  '#a16207', // dark amber    H≈43
  '#3f6212', // olive green   H≈87
  '#166534', // forest green  H≈140
  '#0f766e', // teal          H≈175
  '#0e7490', // cerulean      H≈193
  '#0369a1', // ocean blue    H≈205
  '#1d4ed8', // royal blue    H≈225
  '#4338ca', // indigo        H≈247
  '#6d28d9', // violet        H≈271
  '#86198f', // fuchsia       H≈296
  '#9d174d', // raspberry     H≈333
  '#92400e', // amber-brown   H≈30
  '#7c2d12', // deep orange   H≈17
  '#14532d', // emerald       H≈152
  '#1e3a8a', // navy          H≈229
  '#4a044e', // plum          H≈302
  '#065f46', // deep teal     H≈163
  '#1e40af', // sapphire      H≈222
  '#6b21a8', // deep violet   H≈279
  '#831843', // deep rose     H≈338
  '#78350f', // dark sienna   H≈36
  '#134e4a', // dark cyan     H≈178
];

/**
 * Returns a CSS color for a cluster index.
 * @param {number} idx
 */
export function clusterColor(idx) {
  return CLUSTER_PALETTE[idx % CLUSTER_PALETTE.length];
}

/**
 * Resolves overlapping notes by iteratively pushing them apart.
 * Each note is treated as an axis-aligned rectangle (NOTE_W × NOTE_H).
 * A soft pull back toward the original position preserves spatial intent.
 *
 * @param {Array<{x: number, y: number}>} notes
 * @returns {Array<{x: number, y: number}>} resolved positions (same order)
 */
function resolveOverlaps(notes) {
  const ITERS       = 60;
  const ORIGIN_PULL = 0.08; // how strongly each note is pulled back to its origin

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
          // Push apart along the axis of least penetration
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

      // Gently pull back toward original position
      pos[i].x += (origin[i].x - pos[i].x) * ORIGIN_PULL;
      pos[i].y += (origin[i].y - pos[i].y) * ORIGIN_PULL;
    }
  }

  return pos;
}

/**
 * Expands a convex hull polygon outward from its centroid by `pad` pixels.
 * Used to give cluster hulls breathing room beyond the note rectangle edges.
 *
 * @param {[number, number][]} hull  — points from d3.polygonHull (CCW order)
 * @param {number} pad              — pixels to push each vertex outward
 * @returns {[number, number][]}
 */
export function expandHull(hull, pad) {
  const cx = hull.reduce((s, p) => s + p[0], 0) / hull.length;
  const cy = hull.reduce((s, p) => s + p[1], 0) / hull.length;
  return hull.map(([x, y]) => {
    const dx  = x - cx;
    const dy  = y - cy;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    return [x + (dx / len) * pad, y + (dy / len) * pad];
  });
}

/**
 * Renders or re-renders the canvas view.
 *
 * @param {HTMLElement} container      #canvas-container element
 * @param {Array} notes                raw note objects
 * @param {number[]|null} assignments  cluster index per note, or null if not yet clustered
 * @param {string[]|null} labels       cluster label per cluster index
 */
export function renderCanvas(container, notes, assignments = null, labels = null) {
  const pos = resolveOverlaps(notes);

  const xs   = pos.map((p) => p.x);
  const ys   = pos.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const maxX = Math.max(...xs) + NOTE_W;
  const maxY = Math.max(...ys) + NOTE_H;

  const svgW = maxX - minX + PADDING * 2;
  const svgH = maxY - minY + PADDING * 2;

  container.innerHTML = '';

  // Shared tooltip — one element reused across all notes
  const tooltip = d3
    .select(container)
    .append('div')
    .attr('class', 'cluster-tooltip')
    .style('display', 'none');

  const svg = d3
    .select(container)
    .append('svg')
    .attr('width', svgW)
    .attr('height', svgH)
    .attr('role', 'img')
    .attr('aria-label', 'Sticky notes canvas');

  const g = svg.append('g').attr('transform', `translate(${PADDING - minX}, ${PADDING - minY})`);

  const noteGroups = g
    .selectAll('g.sticky')
    .data(notes)
    .enter()
    .append('g')
    .attr('class', 'sticky')
    .attr('transform', (_, i) => `translate(${pos[i].x}, ${pos[i].y})`)
    // Paper-lift shadow: two layered drop-shadows, heavier at bottom
    .style('filter', 'drop-shadow(0 4px 8px rgba(0,0,0,0.15)) drop-shadow(0 1px 3px rgba(0,0,0,0.10))');

  // Background rectangle — no border, 4px radius
  noteGroups
    .append('rect')
    .attr('width', NOTE_W)
    .attr('height', NOTE_H)
    .attr('rx', 4)
    .attr('fill', (d) => STICKY_COLORS[d.color] ?? STICKY_COLORS.default)
    .attr('stroke', (_, i) => assignments ? clusterColor(assignments[i]) : 'none')
    .attr('stroke-width', 2.5);

  // Note text — centered vertically and horizontally within the sticky
  noteGroups
    .append('foreignObject')
    .attr('x', 14)
    .attr('y', 12)
    .attr('width', NOTE_W - 28)
    .attr('height', NOTE_H - 38)
    .append('xhtml:div')
    .style('display', 'flex')
    .style('align-items', 'center')
    .style('justify-content', 'center')
    .style('text-align', 'center')
    .style('width', '100%')
    .style('height', '100%')
    .style('font-size', '14px')
    .style('line-height', '1.45')
    .style('overflow', 'hidden')
    .style('word-break', 'break-word')
    .style('color', '#333333')
    .text((d) => d.text);

  // Author — 12px, revealed on hover, bottom-aligned
  noteGroups
    .append('foreignObject')
    .attr('x', 14)
    .attr('y', NOTE_H - 26)
    .attr('width', NOTE_W - 28)
    .attr('height', 18)
    .append('xhtml:div')
    .attr('class', 'sticky-author')
    .style('font-size', '12px')
    .style('color', 'rgba(0,0,0,0.70)') /* ≥ 5.7:1 on all sticky pastels — WCAG AA */
    .style('font-style', 'italic')
    .style('opacity', 0)
    .style('transition', 'opacity 0.15s')
    .text((d) => d.author);

  // Populated later (after hull paths are created) but captured by reference in
  // the closures below, so handlers see the full map at event time.
  const hullPaths = new Map(); // cluster_id → d3 path selection

  noteGroups
    .on('mouseover', function (_, d) {
      d3.select(this).select('.sticky-author').style('opacity', 1);

      if (assignments) {
        const i = notes.indexOf(d);
        const clusterIdx = assignments[i];

        // Highlight hovered cluster hull; dim all others
        hullPaths.forEach((path, ci) => {
          const active = ci === clusterIdx;
          path.transition().duration(150).ease(d3.easeQuadOut)
            .attr('fill-opacity',   active ? 0.18 : 0.02)
            .attr('stroke-opacity', active ? 0.75 : 0.08)
            .attr('stroke-width',   active ? 2.5  : 1.5);
        });

        if (labels) {
          const color = clusterColor(clusterIdx);
          tooltip
            .style('display', 'block')
            .style('border-color', color)
            .style('color', color)
            .text(labels[clusterIdx]);
        }
      }
    })
    .on('mousemove', function (event) {
      if (!assignments || !labels) return;
      // Position tooltip relative to the canvas container
      const rect = container.getBoundingClientRect();
      tooltip
        .style('left', `${event.clientX - rect.left + 14}px`)
        .style('top',  `${event.clientY - rect.top  - 10}px`);
    })
    .on('mouseout', function () {
      d3.select(this).select('.sticky-author').style('opacity', 0);
      tooltip.style('display', 'none');

      // Restore all hulls to resting state
      hullPaths.forEach(path => {
        path.transition().duration(300).ease(d3.easeQuadOut)
          .attr('fill-opacity',   0.07)
          .attr('stroke-opacity', 0.30)
          .attr('stroke-width',   1.5);
      });
    });

  // Cluster indicator dot (top-right corner)
  if (assignments) {
    noteGroups
      .append('circle')
      .attr('cx', NOTE_W - 10)
      .attr('cy', 10)
      .attr('r', 5)
      .attr('fill', (_, i) => clusterColor(assignments[i]))
      .attr('stroke', 'white')
      .attr('stroke-width', 1.5);
  }

  // ── Cluster convex hulls ─────────────────────────────────────────────────────
  // Appended after all note groups so hulls sit on top in SVG paint order.
  // Hull points come from all four note corners so the polygon wraps the card
  // area rather than just connecting centres. Catmull-Rom smooths the outline.
  // Each hull enters with a scale-from-centroid + fade-in transition, staggered
  // 50 ms per cluster so they reveal sequentially rather than all at once.
  if (assignments) {
    const HULL_PAD = 16; // px to push each vertex outward from the centroid

    const byCluster = new Map();
    notes.forEach((_, i) => {
      const ci = assignments[i];
      if (!byCluster.has(ci)) byCluster.set(ci, []);
      byCluster.get(ci).push(i);
    });

    const lineGen = d3.line().curve(d3.curveCatmullRomClosed.alpha(0.5));
    const hullLayer = g.append('g').attr('class', 'hull-layer');

    byCluster.forEach((noteIdxs, ci) => {
      const color = clusterColor(ci);

      // Collect all four corners of every note in this cluster.
      // A single note produces 4 non-collinear points — always a valid hull.
      const corners = [];
      for (const ni of noteIdxs) {
        const x = pos[ni].x, y = pos[ni].y;
        corners.push([x,          y         ]);
        corners.push([x + NOTE_W, y         ]);
        corners.push([x,          y + NOTE_H]);
        corners.push([x + NOTE_W, y + NOTE_H]);
      }

      const hull = d3.polygonHull(corners);
      if (!hull) return;

      const expanded = expandHull(hull, HULL_PAD);

      // Centroid of the hull vertices — used as the transform-origin for the
      // scale animation so the hull grows outward from its own centre.
      const [cx, cy] = d3.polygonCentroid(hull);
      const scaleIn  = `translate(${cx},${cy}) scale(0.6) translate(${-cx},${-cy})`;
      const scaleOut = `translate(${cx},${cy}) scale(1)   translate(${-cx},${-cy})`;

      const path = hullLayer.append('path')
        .attr('d', lineGen(expanded))
        .attr('fill', color)
        .attr('fill-opacity', 0)          // start invisible
        .attr('stroke', color)
        .attr('stroke-opacity', 0)
        .attr('stroke-width', 1.5)
        .attr('stroke-linejoin', 'round')
        .attr('pointer-events', 'none')   // notes beneath remain hoverable
        .attr('transform', scaleIn);

      // Register in the shared map so hover handlers can target this hull
      hullPaths.set(ci, path);

      path.transition()
          .duration(450)
          .delay(ci * 50)                 // stagger: each cluster 50 ms after the previous
          .ease(d3.easeCubicOut)
          .attr('transform', scaleOut)
          .attr('fill-opacity', 0.07)
          .attr('stroke-opacity', 0.30);
    });
  }
}
