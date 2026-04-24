/**
 * semantic-view.js — 2D semantic space visualisation of clustered notes.
 *
 * Projects per-note reduced embeddings to 2D (raw U·Σ PCA scores, not L2-
 * normalised) and renders an annotated scatter plot:
 *
 *   Layer 1 — Convex hull polygons, opacity encodes silhouette score
 *   Layer 2 — Note dots, radius encodes typicality (radial magnitude)
 *   Layer 3 — Cluster labels (outside zoom-root — constant pixel size at any zoom level)
 *   Layer 4 — Tooltip connector lines (outside zoom-root)
 *
 * Navigation:
 *   Scroll wheel  — zoom in/out centred on cursor
 *   Pinch gesture — zoom in/out (two-finger touch)
 *   (Drag pan is intentionally disabled — scroll/pinch only)
 *
 * Interactions:
 *   Hull hover   — dim non-hovered labels + other hulls/dots; no auto-zoom
 *   Dot hover    — dim non-cluster labels + show note tooltip
 *   Connector    — dashed line from anchor (dot/centroid) to cursor; stretches live
 *   Mode bar     — floating "Note | Cluster" pill above tooltip, active mode highlighted
 *
 * Idempotent: renders once per clustering run. Call invalidateSemanticView()
 * after re-clustering so the next tab activation re-renders with fresh data.
 */

import { clusterColor, expandHull, stickyColor } from './canvas-view.js';
import { pcaProject }               from './pca.js';
import * as d3                      from 'd3';

let _rendered = false;

export function invalidateSemanticView() {
  _rendered = false;
}

export function renderSemanticView(container, notes, embeddingsReduced, assignments, clusters, { onNoteClick } = {}) {
  if (_rendered) return;
  _rendered = true;

  // Respect OS-level reduced-motion preference — skip decorative entrance
  // animations (dot fly-in, hull/label fade-in) and jump to final state.
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  container.innerHTML = '';

  // ── 2D Projection ─────────────────────────────────────────────────────────
  // normalize: false → raw U·Σ scores with natural radial spread.
  const coords2d = pcaProject(embeddingsReduced, 2, { normalize: false });

  const magnitudes = coords2d.map(([x, y]) => Math.sqrt(x * x + y * y));
  const maxMag     = Math.max(...magnitudes) || 1;
  const restingR   = magnitudes.map(m => 4 + 5 * (m / maxMag)); // [4, 9]px

  // ── Scales ────────────────────────────────────────────────────────────────
  // Degenerate domain guard: if all points share one axis value, pad ±1.
  const [x0, x1] = d3.extent(coords2d, d => d[0]);
  const [y0, y1] = d3.extent(coords2d, d => d[1]);
  const xScale = d3.scaleLinear().domain([x0 === x1 ? x0 - 1 : x0, x1 === x0 ? x1 + 1 : x1]).range([64, 896]);
  const yScale = d3.scaleLinear().domain([y0 === y1 ? y0 - 1 : y0, y1 === y0 ? y1 + 1 : y1]).range([64, 476]);

  const scaled      = coords2d.map(([x, y]) => [xScale(x), yScale(y)]);
  const noteIdToIdx = new Map(notes.map((n, i) => [n.id, i]));

  // Scene center used to derive per-cluster outward projection direction.
  const sceneCx = d3.mean(scaled, d => d[0]);
  const sceneCy = d3.mean(scaled, d => d[1]);

  // ── DOM structure ─────────────────────────────────────────────────────────
  const tooltip = d3.select(container).append('div').attr('class', 'sv-tooltip');
  const modebar = d3.select(container).append('div').attr('class', 'sv-modebar');
  modebar.html(
    '<span class="sv-mode-item sv-mode-note">Note</span>' +
    '<span class="sv-mode-sep"> | </span>' +
    '<span class="sv-mode-item sv-mode-cluster">Cluster</span>'
  );

  const svg = d3.select(container)
    .append('svg')
    .attr('viewBox', '0 0 960 540')
    .attr('width', '100%')
    .attr('height', 'auto')
    .attr('role', 'img')
    .attr('aria-label', 'Semantic space — notes projected to 2D by PCA');

  // zoomRoot receives the D3 zoom transform (hulls + dots scale/pan together).
  // labelLayer and connectorLayer sit outside so their pixel sizes stay constant
  // and their coordinates are always in outer SVG space.
  const zoomRoot       = svg.append('g').attr('class', 'zoom-root');
  const hullLayer      = zoomRoot.append('g').attr('class', 'hull-layer');
  const dotLayer       = zoomRoot.append('g').attr('class', 'dot-layer');
  const labelLayer     = svg.append('g').attr('class', 'label-layer');      // outside zoom-root
  const connectorLayer = svg.append('g').attr('class', 'connector-layer'); // outside zoom-root

  const connector = connectorLayer.append('line')
    .attr('stroke-width', 1)
    .attr('stroke-dasharray', '4 3')
    .attr('opacity', 0);

  // ── Scroll / pinch zoom ───────────────────────────────────────────────────
  // Only wheel and touch events are captured — drag panning is disabled.
  // Labels and connector are outside zoomRoot so they don't scale with zoom.
  // The zoom handler re-positions labels by applying the live transform to
  // each label's stored centroid, keeping text anchored to the visual hull.
  let currentZoomTransform = d3.zoomIdentity;
  let activeAnchor         = null; // { lx, ly } in zoomRoot local coords, set on hover

  const zoomBehavior = d3.zoom()
    .scaleExtent([0.5, 8])
    .filter(event => event.type !== 'dblclick') // wheel zoom + drag pan + touch pinch; block dblclick
    .on('zoom', (event) => {
      currentZoomTransform = event.transform;
      zoomRoot.attr('transform', event.transform.toString());

      // Re-position labels to track their cluster centroid through the zoom transform.
      labelLayer.selectAll('text[data-cidx]').each(function () {
        const meta = hullMeta.get(+this.getAttribute('data-cidx'));
        if (!meta) return;
        const [tx, ty] = event.transform.apply([meta.lx, meta.ly]);
        d3.select(this).attr('x', tx).attr('y', ty);
      });

      // Re-anchor connector if a hover is active.
      if (activeAnchor) {
        const [ox, oy] = currentZoomTransform.apply([activeAnchor.lx, activeAnchor.ly]);
        connector.attr('x1', ox).attr('y1', oy);
      }
    });

  svg.call(zoomBehavior);

  // ── SVG ↔ container coordinate conversion ────────────────────────────────
  // These operate in OUTER SVG space (independent of zoomRoot transform).
  function containerToSvg(contX, contY) {
    const svgRect = svg.node().getBoundingClientRect();
    const cRect   = container.getBoundingClientRect();
    return {
      x: (contX - (svgRect.left - cRect.left)) / (svgRect.width  / 960),
      y: (contY - (svgRect.top  - cRect.top))  / (svgRect.height / 540),
    };
  }

  // Map zoomRoot local coords → outer SVG coords using the live zoom transform.
  function localToOuter(lx, ly) {
    const [ox, oy] = currentZoomTransform.apply([lx, ly]);
    return { x: ox, y: oy };
  }

  // ── Silhouette → resting opacity ─────────────────────────────────────────
  function restingOp(s) {
    if (s > 0.30)  return { fill: 0.22, stroke: 0.45 };
    if (s >= 0.10) return { fill: 0.10, stroke: 0.45 };
    return              { fill: 0.04, stroke: 0.45 };
  }

  // ── LAYER 1 — Convex hulls ────────────────────────────────────────────────
  const hullMeta = new Map(); // colorIdx → { path, cx, cy, lx, ly, resting, memberIdxs }
  const lineGen  = d3.line().curve(d3.curveCatmullRomClosed.alpha(0.5));

  // d3.polygonHull needs ≥ 3 non-collinear points.
  // For 1-2 note clusters we inflate each dot into a small circle of synthetic
  // vertices so the hull is a well-formed polygon (pill / circle shape).
  // The same fallback catches collinear 3-note degenerate cases.
  const HULL_SYNTH_R = 24; // px radius for synthetic inflation
  const HULL_SYNTH_N = 10; // vertices per inflated circle

  function buildHull(pts) {
    if (pts.length >= 3) {
      const h = d3.polygonHull(pts);
      if (h) return h;
    }
    const synthetic = [];
    for (const [px, py] of pts) {
      for (let k = 0; k < HULL_SYNTH_N; k++) {
        const a = (2 * Math.PI * k) / HULL_SYNTH_N;
        synthetic.push([px + HULL_SYNTH_R * Math.cos(a), py + HULL_SYNTH_R * Math.sin(a)]);
      }
    }
    return d3.polygonHull(synthetic);
  }

  clusters.forEach((cluster, colorIdx) => {
    const memberIdxs = (cluster.note_ids ?? [])
      .map(id => noteIdToIdx.get(id))
      .filter(i => i !== undefined);

    const memberPts = memberIdxs.map(i => scaled[i]);
    const hull = buildHull(memberPts);
    if (!hull) return; // should not occur after synthetic fallback, guard anyway

    const expanded = expandHull(hull, 20);
    // Mean of member note positions — more robust to outliers than d3.polygonCentroid,
    // which is area-weighted and can be pulled far from the note mass by a single outlier.
    const cx = d3.mean(memberPts, p => p[0]);
    const cy = d3.mean(memberPts, p => p[1]);
    const color    = clusterColor(colorIdx);
    const resting  = restingOp(cluster.silhouette_score ?? 0);

    // Outward direction: from scene center through cluster centroid.
    // Fallback to "up" when the centroid sits exactly on the scene center.
    let dx = cx - sceneCx, dy = cy - sceneCy;
    const dlen = Math.sqrt(dx * dx + dy * dy) || 1;
    dx /= dlen; dy /= dlen;

    // Farthest expanded-hull vertex along the outward direction (relative to centroid).
    const maxProj = Math.max(...expanded.map(([vx, vy]) => (vx - cx) * dx + (vy - cy) * dy));

    // Hull-exit point (perimeter) + label anchor (28 px beyond).
    const LABEL_OFFSET = 56; // doubled — connector anchors here, outside the hull perimeter
    const ex = cx + dx * maxProj;
    const ey = cy + dy * maxProj;
    const lx = ex + dx * LABEL_OFFSET;
    const ly = ey + dy * LABEL_OFFSET;

    const scaleIn  = `translate(${cx},${cy}) scale(0.95) translate(${-cx},${-cy})`;
    const scaleOut = `translate(${cx},${cy}) scale(1.0)  translate(${-cx},${-cy})`;

    const path = hullLayer.append('path')
      .attr('d', lineGen(expanded))
      .attr('fill', color)
      .attr('fill-opacity', 0)
      .attr('stroke', color)
      .attr('stroke-opacity', 0)
      .attr('stroke-width', 1.5)
      .attr('stroke-linejoin', 'round')
      .attr('transform', scaleIn)
      .style('cursor', 'pointer');

    hullMeta.set(colorIdx, { path, cx, cy, lx, ly, ex, ey, dx, resting, memberIdxs });

    if (reducedMotion) {
      path.attr('transform', scaleOut)
          .attr('fill-opacity', resting.fill)
          .attr('stroke-opacity', resting.stroke);
    } else {
      path.transition()
        .delay(1000).duration(800).ease(d3.easeCubicOut)
        .attr('transform', scaleOut)
        .attr('fill-opacity', resting.fill)
        .attr('stroke-opacity', resting.stroke);
    }

    const memberSet = new Set(memberIdxs);

    path
      .on('mouseover', function (event) {
        const s  = cluster.silhouette_score ?? 0;
        const cl = s > 0.30 ? 'high confidence' : s >= 0.10 ? 'moderate' : 'broad theme';
        const kp = (cluster.keyphrases ?? []).slice(0, 3)
          .map(k => `<strong>${escapeHtml(k)}</strong>`).join(' · ');

        tooltip
          .style('display', 'block')
          .style('border-color', color)
          .html(
            `<div class="sv-tt-label" style="color:${color}">${escapeHtml(cluster.label.toUpperCase())}</div>` +
            `<div class="sv-tt-cohesion">cohesion: ${s.toFixed(2)} · ${cl}</div>` +
            `<div class="sv-tt-count">${cluster.note_ids.length} notes</div>` +
            (kp ? `<div class="sv-tt-keyphrases">${kp}</div>` : '')
          );

        setMode('cluster');
        positionTooltip(event);
        initConnector(cx, cy, color); // anchor at hull centroid

        // Dim non-hovered cluster labels + ticks — makes the focused cluster's label
        // the only visible text landmark, reducing visual noise on dense boards.
        labelLayer.selectAll('text').transition().duration(150)
          .attr('opacity', function () {
            return +this.getAttribute('data-cidx') === colorIdx ? 1 : 0.12;
          });

        dots.transition().duration(150)
          .attr('opacity', (_, i) => memberSet.has(i) ? 1.0 : 0.20)
          .attr('r',       (_, i) => memberSet.has(i) ? restingR[i] + 2 : restingR[i]);

        hullMeta.forEach(({ path: p, resting: r }, ci) => {
          p.transition().duration(150)
            .attr('fill-opacity',   ci === colorIdx ? Math.min(r.fill * 2, 0.35) : 0.02)
            .attr('stroke-opacity', ci === colorIdx ? 0.90 : 0.08);
        });
      })
      .on('mousemove', function (event) {
        positionTooltip(event);
      })
      .on('mouseout', function () {
        hideTooltip();
        restoreAll();
      });
  });

  // ── Label visibility classification ──────────────────────────────────────
  // Dense center clusters produce overlapping labels that no-one can read.
  // Strategy: greedy independent set ordered by spatial isolation descending.
  // A cluster earns an always-visible label only if no already-placed centroid
  // is within LABEL_ZONE px — preventing overlap at default zoom (960×540).
  // Remaining clusters show their label on hover only.
  const LABEL_ZONE = 150; // SVG viewBox px; roughly 1–1.5× a typical label width

  const _centroidList = [...hullMeta.entries()].map(([ci, { cx, cy }]) => {
    let minD = Infinity;
    for (const [ci2, { cx: cx2, cy: cy2 }] of hullMeta) {
      if (ci2 === ci) continue;
      const d = Math.hypot(cx - cx2, cy - cy2);
      if (d < minD) minD = d;
    }
    return { ci, cx, cy, minD };
  });
  _centroidList.sort((a, b) => b.minD - a.minD); // most isolated first

  const labelAlwaysVisible = new Map(); // ci → boolean
  const _placed = []; // centroids already claimed by an always-visible label
  for (const { ci, cx, cy } of _centroidList) {
    const clear = _placed.every(p => Math.hypot(cx - p.cx, cy - p.cy) > LABEL_ZONE);
    labelAlwaysVisible.set(ci, clear);
    if (clear) _placed.push({ cx, cy });
  }

  // ── LAYER 2 — Note dots ───────────────────────────────────────────────────
  const dots = dotLayer.selectAll('circle')
    .data(notes)
    .enter()
    .append('circle')
    .attr('cx', 480)
    .attr('cy', 270)
    .attr('r',  (_, i) => restingR[i])
    .attr('fill', (_, i) => stickyColor(notes[i].color))
    .attr('fill-opacity', 0.72)
    .attr('stroke', (_, i) => clusterColor(assignments[i]))
    .attr('stroke-width', 1.5)
    .attr('stroke-opacity', 0.45)
    .attr('opacity', 0)
    .style('cursor', 'pointer');

  if (reducedMotion) {
    dots
      .attr('cx', (_, i) => scaled[i][0])
      .attr('cy', (_, i) => scaled[i][1])
      .attr('opacity', 1);
  } else {
    dots.transition()
      .delay((_, i) => i * 20)
      .duration(1200)
      .ease(d3.easeCubicInOut)
      .attr('cx', (_, i) => scaled[i][0])
      .attr('cy', (_, i) => scaled[i][1])
      .attr('opacity', 1);
  }

  dots
    .on('mouseover', function (event, d) {
      const i        = notes.indexOf(d);
      const colorIdx = assignments[i];
      const cluster  = clusters[colorIdx];
      if (!cluster) return;
      const color = clusterColor(colorIdx);

      tooltip
        .style('display', 'block')
        .style('border-color', color)
        .html(
          `<div class="sv-tt-label" style="color:${color}">${escapeHtml(cluster.label.toUpperCase())}</div>` +
          `<div class="sv-tt-text" style="background:${stickyColor(d.color)}">${escapeHtml(d.text)}</div>` +
          `<div class="sv-tt-author">${escapeHtml(d.author)}</div>`
        );

      setMode('note');
      positionTooltip(event);
      const meta = hullMeta.get(colorIdx);
      initConnector(meta ? meta.cx : scaled[i][0], meta ? meta.cy : scaled[i][1], color); // anchor at hull centroid

      // Same highlight as hull hover: dim other labels, other dots, other hulls.
      labelLayer.selectAll('text').transition().duration(150)
        .attr('opacity', function () {
          return +this.getAttribute('data-cidx') === colorIdx ? 1 : 0.12;
        });

      const memberSet = hullMeta.get(colorIdx)?.memberIdxs
        ? new Set(hullMeta.get(colorIdx).memberIdxs) : new Set();

      dots.transition().duration(150)
        .attr('opacity', (_, j) => memberSet.has(j) ? 1.0 : 0.20)
        .attr('r',       (_, j) => memberSet.has(j) ? restingR[j] + 2 : restingR[j]);

      hullMeta.forEach(({ path: p, resting: r }, ci) => {
        p.transition().duration(150)
          .attr('fill-opacity',   ci === colorIdx ? Math.min(r.fill * 2, 0.35) : 0.02)
          .attr('stroke-opacity', ci === colorIdx ? 0.90 : 0.08);
      });
    })
    .on('mousemove', function (event) {
      positionTooltip(event);
    })
    .on('mouseout', function () {
      hideTooltip();
      restoreAll();
    })
    .on('click', function (_, d) {
      if (!onNoteClick) return;
      const i = notes.indexOf(d);
      onNoteClick(assignments[i]);
    });

  // ── LAYER 3 — Cluster labels (outside zoom-root) ──────────────────────────
  // Labels sit at their cluster centroid (constant pixel size across zoom levels).
  // Visibility is driven by spatial isolation:
  //   always-visible  — isolated clusters (greedy independent set, LABEL_ZONE px)
  //   hover-only      — dense center clusters; revealed on hull/dot hover
  // This prevents the overlapping text mass that forms when many clusters crowd
  // the centre of the embedding space.
  clusters.forEach((cluster, colorIdx) => {
    const meta    = hullMeta.get(colorIdx);
    if (!meta) return;

    const always   = labelAlwaysVisible.get(colorIdx) ?? false;
    const n        = cluster.note_ids.length;
    const fontSize = Math.max(12, Math.min(18, 10 + n));
    const color    = clusterColor(colorIdx);
    const baseOp   = always ? 1 : 0; // dense labels start invisible

    labelLayer.append('text')
      .attr('data-cidx', colorIdx)
      .attr('data-always', always ? '1' : '0')
      .attr('x', meta.lx)
      .attr('y', meta.ly)
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('fill', color)
      .attr('font-size', fontSize)
      .attr('font-weight', 600)
      .attr('letter-spacing', '0.04em')
      .attr('pointer-events', 'none')
      .attr('opacity', reducedMotion ? baseOp : 0)
      .text(cluster.label.toUpperCase())
      .call(sel => {
        if (!reducedMotion && always) {
          sel.transition().delay(1400).duration(400).attr('opacity', 1);
        } else if (!reducedMotion && !always) {
          // hover-only: stay at 0 — no entrance animation
          sel.attr('opacity', 0);
        }
      });
  });

  // ── Utilities ─────────────────────────────────────────────────────────────

  function restoreAll() {
    // Always-visible labels restore to 1; hover-only labels return to 0.
    labelLayer.selectAll('text').transition().duration(300).ease(d3.easeQuadOut)
      .attr('opacity', function () {
        return this.getAttribute('data-always') === '1' ? 1 : 0;
      });
    dots.transition().duration(300).ease(d3.easeQuadOut)
      .attr('opacity', 1)
      .attr('r', (_, i) => restingR[i]);

    hullMeta.forEach(({ path: p, resting: r }) => {
      p.transition().duration(300).ease(d3.easeQuadOut)
        .attr('fill-opacity',   r.fill)
        .attr('stroke-opacity', r.stroke);
    });
  }

  function positionTooltip(event) {
    const cRect  = container.getBoundingClientRect();
    const ttNode = tooltip.node();
    let x = event.clientX - cRect.left + 14;
    let y = event.clientY - cRect.top  - 10;
    x = Math.max(0, Math.min(cRect.width  - ttNode.offsetWidth,  x));
    y = Math.max(0, Math.min(cRect.height - ttNode.offsetHeight, y));
    tooltip.style('left', `${x}px`).style('top', `${y}px`);

    const mw = modebar.node().offsetWidth  || 80;
    const mh = modebar.node().offsetHeight || 24;
    modebar
      .style('left', `${x + ttNode.offsetWidth / 2 - mw / 2}px`)
      .style('top',  `${y - mh - 6}px`);

    updateConnectorEndpoint();
  }

  function setMode(mode) {
    modebar.style('display', 'flex');
    modebar.select('.sv-mode-note').classed('sv-mode-active', mode === 'note');
    modebar.select('.sv-mode-cluster').classed('sv-mode-active', mode === 'cluster');
  }

  function hideTooltip() {
    activeAnchor = null;
    tooltip.style('display', 'none');
    modebar.style('display', 'none');
    connector.transition().duration(150).attr('opacity', 0);
  }

  function initConnector(localX, localY, color) {
    // Store anchor in zoomRoot local coords so the zoom handler can re-project
    // x1/y1 as the user scrolls — keeping the connector anchored to the visual
    // position of the dot/centroid as hulls and dots move under zoom.
    activeAnchor = { lx: localX, ly: localY };
    const outer  = localToOuter(localX, localY);

    // x2/y2 initialised to tooltip center (positionTooltip has already run).
    // updateConnectorEndpoint keeps it in sync as the tooltip follows the cursor.
    const ttRect = tooltip.node().getBoundingClientRect();
    const cRect  = container.getBoundingClientRect();
    const tip    = containerToSvg(
      ttRect.left + ttRect.width  / 2 - cRect.left,
      ttRect.top  + ttRect.height / 2 - cRect.top
    );

    connector
      .attr('x1', outer.x).attr('y1', outer.y)
      .attr('x2', tip.x).attr('y2', tip.y)
      .attr('stroke', color)
      .attr('opacity', 0)
      .transition().duration(200)
      .attr('opacity', 0.6);
  }

  function updateConnectorEndpoint() {
    // Endpoint (x2/y2) is the center of the tooltip div, recomputed each time
    // the tooltip moves. x1/y1 (anchor) is updated by the zoom handler.
    if (!activeAnchor) return;
    const ttRect = tooltip.node().getBoundingClientRect();
    const cRect  = container.getBoundingClientRect();
    const { x, y } = containerToSvg(
      ttRect.left + ttRect.width  / 2 - cRect.left,
      ttRect.top  + ttRect.height / 2 - cRect.top
    );
    connector.attr('x2', x).attr('y2', y);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  // ── Screenshot export ─────────────────────────────────────────────────────
  const btnScreenshot = document.createElement('button');
  btnScreenshot.id        = 'btn-sv-screenshot';
  btnScreenshot.className = 'btn-sv-screenshot';
  btnScreenshot.textContent = 'Save PNG';
  container.appendChild(btnScreenshot);

  btnScreenshot.addEventListener('click', async () => {
    btnScreenshot.style.visibility = 'hidden';
    try {
      const h2c    = await loadHtml2Canvas();
      const canvas = await h2c(container);
      triggerDownload(canvas.toDataURL(), 'semantic-space.png');
    } catch {
      const svgStr = new XMLSerializer().serializeToString(svg.node());
      const blob   = new Blob([svgStr], { type: 'image/svg+xml' });
      const url    = URL.createObjectURL(blob);
      triggerDownload(url, 'semantic-space.svg');
      URL.revokeObjectURL(url);
    } finally {
      btnScreenshot.style.visibility = 'visible';
    }
  });

  function triggerDownload(href, filename) {
    const a = document.createElement('a');
    a.href = href; a.download = filename; a.click();
  }

  async function loadHtml2Canvas() {
    if (window.html2canvas) return window.html2canvas;
    return new Promise((resolve, reject) => {
      const s   = document.createElement('script');
      s.src     = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
      s.onload  = () => resolve(window.html2canvas);
      s.onerror = reject;
      document.head.appendChild(s);
    });
  }
}
