/**
 * insights-agent.js — Level 1 Reactive Agent
 *
 * Post-clustering reasoning layer that acts on pipeline outputs automatically.
 * All computation is local — no note content leaves the browser.
 *
 * Detects: low-confidence note placements (outliers), near-duplicate cluster
 * pairs (merge candidates), and synthesises a plain-language board summary.
 */

const OUTLIER_SILHOUETTE_THRESHOLD = 0.05;
const MERGE_SIMILARITY_THRESHOLD   = 0.75;

/** Cosine similarity on L2-normalised vectors (= dot product). */
function cosineSim(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Returns notes whose per-note silhouette score is below the outlier threshold.
 * These notes did not cluster cleanly — the algorithm placed them with low
 * confidence and they deserve human review.
 *
 * @param {Array}    notes               - raw note objects with id, text, author
 * @param {number[]} assignments         - cluster index per note (parallel to notes[])
 * @param {number[]} perNoteSilhouette   - silhouette score per note (parallel to notes[])
 * @param {Array}    clusters            - cluster objects with .label
 */
export function detectOutliers(notes, assignments, perNoteSilhouette, clusters) {
  const outliers = [];
  for (let i = 0; i < notes.length; i++) {
    const s = perNoteSilhouette[i];
    if (s < OUTLIER_SILHOUETTE_THRESHOLD) {
      const clusterLabel = clusters[assignments[i]]?.label ?? 'Unknown';
      outliers.push({
        noteIdx:        i,
        text:           notes[i].text,
        author:         notes[i].author ?? 'unknown',
        silhouette:     s,
        assignedCluster: clusterLabel,
        reason:         s < 0
          ? 'Fits better in a different cluster than the one it was assigned to.'
          : 'Does not cluster cleanly with others in its group.',
      });
    }
  }
  return outliers;
}

/**
 * Returns pairs of clusters whose centroids have cosine similarity above the
 * merge threshold. High centroid similarity means both clusters describe
 * overlapping semantic territory and may represent the same theme.
 *
 * @param {Array}       clusters  - cluster objects with .label, .note_ids, .silhouette_score
 * @param {number[][]}  centroids - L2-normalised cluster centroids (parallel to clusters[])
 */
export function detectMergeCandidates(clusters, centroids) {
  const suggestions = [];
  for (let i = 0; i < centroids.length; i++) {
    for (let j = i + 1; j < centroids.length; j++) {
      if (!centroids[i] || !centroids[j]) continue;
      const sim = cosineSim(centroids[i], centroids[j]);
      if (sim > MERGE_SIMILARITY_THRESHOLD) {
        suggestions.push({
          clusterA:   i,
          clusterB:   j,
          labelA:     clusters[i]?.label ?? `Cluster ${i}`,
          labelB:     clusters[j]?.label ?? `Cluster ${j}`,
          similarity: sim,
          dismissed:  false,
        });
      }
    }
  }
  return suggestions;
}

/**
 * Generates a single plain-language sentence summarising the board.
 */
export function generateBoardSummary(notes, clusters, outlierCount) {
  const authorSet  = new Set(notes.map(n => n.author).filter(Boolean));
  const topCluster = [...clusters].sort((a, b) => b.note_ids.length - a.note_ids.length)[0];
  const n = clusters.length;
  const x = notes.length;
  const y = authorSet.size;
  return (
    `Your team surfaced ${n} theme${n !== 1 ? 's' : ''} from ${x} note${x !== 1 ? 's' : ''} ` +
    `by ${y} contributor${y !== 1 ? 's' : ''}. ` +
    `The strongest signal is "${topCluster?.label ?? 'Unknown'}". ` +
    `${outlierCount} note${outlierCount !== 1 ? 's' : ''} may need review.`
  );
}

/**
 * Returns clusters sorted by composite score (60% note-count weight,
 * 40% silhouette-score weight). Each entry is augmented with { ci, compositeScore, rank }.
 *
 * Silhouette is in [-1, 1]; we normalise to [0, 1] before weighting so both
 * terms are on the same scale.  Note-count weight is normalised by the largest
 * cluster, so it is also in [0, 1].
 *
 * @param {Array} clusters - cluster objects (index = ci)
 */
export function rankClusters(clusters) {
  const maxNoteCount = Math.max(...clusters.map(c => c.note_ids.length), 1);
  const ranked = clusters.map((cluster, ci) => {
    const noteCountNorm  = cluster.note_ids.length / maxNoteCount;
    const silhouetteNorm = (cluster.silhouette_score + 1) / 2;
    const compositeScore = 0.6 * noteCountNorm + 0.4 * silhouetteNorm;
    return { ...cluster, ci, compositeScore };
  });
  ranked.sort((a, b) => b.compositeScore - a.compositeScore);
  ranked.forEach((c, idx) => { c.rank = idx + 1; });
  return ranked;
}

/**
 * Runs all Level 1 reactive analyses and returns the combined insights object.
 *
 * @param {Array}       notes            - raw note objects
 * @param {Array}       clusters         - pipeline cluster output
 * @param {number[]}    assignments      - per-note cluster assignments
 * @param {number[]}    perNoteSilhouette
 * @param {number[][]}  centroids        - L2-normalised cluster centroids
 */
export function computeInsights(notes, clusters, assignments, perNoteSilhouette, centroids) {
  const outliers        = detectOutliers(notes, assignments, perNoteSilhouette, clusters);
  const mergeSuggestions = detectMergeCandidates(clusters, centroids);
  const boardSummary    = generateBoardSummary(notes, clusters, outliers.length);
  const rankedClusters  = rankClusters(clusters);
  return { outliers, mergeSuggestions, boardSummary, rankedClusters };
}
