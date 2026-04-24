/**
 * test-pipeline.js — Shared test suites for the insight discovery pipeline.
 *
 * Exported for use by test.html (the npm test runner). Not self-executing.
 * Run via: npm test  (starts vite dev server and opens test.html)
 *
 * Browser context required — models use the browser Cache API (WebAssembly).
 * Cannot run in Node.js.
 *
 * Coverage:
 *   Suite 1 — Main (16 notes, 3 clear themes + ambiguous + singleton)
 *     - Output schema: all required fields, correct types
 *     - Stage 3: keyphrases present for clusters with ≥ 3 notes
 *     - Stage 4: label word count, no repeated tokens, unique across clusters
 *     - Output integrity: no duplicate note IDs, all IDs accounted for
 *     - k-selection: result count within valid range for n=16
 *     - Noise group invariants (if produced)
 *   Suite 2 — n < 5 edge case (3 notes)
 *     - Each note becomes its own cluster (pipeline skips clustering below n=5)
 *     - Correct schema on small input
 */

// ── Test data ──────────────────────────────────────────────────────────────

export const NOTES = [
  // Theme A: UX / Design
  { id: 'a1', text: 'Simplify the onboarding flow for new users' },
  { id: 'a2', text: 'Make buttons larger on mobile' },
  { id: 'a3', text: 'Improve contrast ratios for accessibility' },
  { id: 'a4', text: 'Redesign the empty state screens' },
  { id: 'a5', text: 'Add tooltips to every icon' },

  // Theme B: Performance / Engineering
  { id: 'b1', text: 'Reduce initial bundle size' },
  { id: 'b2', text: 'Cache API responses locally' },
  { id: 'b3', text: 'Lazy load images below the fold' },
  { id: 'b4', text: 'Profile and fix render bottlenecks' },
  { id: 'b5', text: 'Too slow' },            // under 4 words — tests short-note handling

  // Theme C: Team Process / Collaboration
  { id: 'c1', text: 'Hold weekly design reviews with engineers' },
  { id: 'c2', text: 'Document decisions in shared Confluence space' },
  { id: 'c3', text: 'Create a team definition of done' },

  // Ambiguous — could fit multiple themes
  { id: 'x1', text: 'Run usability tests with real users before shipping' },
  { id: 'x2', text: 'Set up automated performance monitoring in CI pipeline' },

  // Singleton — semantically isolated, unlikely to cluster with above
  { id: 'z1', text: 'Order more coffee for the kitchen' },
];

// Three notes: triggers the n < 5 individual-labeling path in the pipeline.
export const TINY_NOTES = [
  { id: 't1', text: 'Fix login bug' },
  { id: 't2', text: 'Add dark mode' },
  { id: 't3', text: 'Improve performance' },
];

// ── Shared test runner ─────────────────────────────────────────────────────

/**
 * Runs all test suites against the provided clusterNotes function.
 *
 * @param {Function} clusterNotesFn  - the clusterNotes export from pipeline.js
 * @param {{ log: Function, assert: Function }} callbacks
 *   log(text, hint?)   — emit a line of output; hint is 'pass'|'fail'|'info'|'section'
 *   assert(cond, msg)  — record a pass or failure
 */
export async function runTests(clusterNotesFn, { log, assert }) {

  // ── Suite 1: Main (16 notes) ─────────────────────────────────────────────

  log('═══ Suite 1: Main pipeline (16 notes) ═══', 'section');
  log(`Input: ${NOTES.length} notes\n`, 'info');

  const results = await clusterNotesFn(NOTES, {
    onProgress(label, pct) {
      log(`  [${String(pct).padStart(3)}%] ${label}`, 'info');
    },
  });

  log('\n─── Results ───', 'section');
  for (const r of results) {
    log(`\ncluster_id:    ${r.cluster_id}`);
    log(`label:         "${r.label}"`);
    log(`note_ids:      ${JSON.stringify(r.note_ids)}`);
    log(`keyphrases:    ${JSON.stringify(r.keyphrases)}`);
    log(`fallback_used: ${r.fallback_used}`);
  }

  log('\n─── Assertions ───', 'section');

  // Output schema — every result must have the correct field types.
  // Catches regressions where a field is renamed, removed, or returned as the
  // wrong type (e.g. keyphrases as undefined instead of []).
  for (const r of results) {
    const id = `cluster_id ${r.cluster_id}`;
    assert(Number.isInteger(r.cluster_id),                              `${id}: cluster_id is an integer`);
    assert(r.cluster_id >= -1,                                          `${id}: cluster_id is -1 or non-negative`);
    assert(typeof r.label === 'string' && r.label.trim().length > 0,   `${id}: label is a non-empty string`);
    assert(Array.isArray(r.note_ids) && r.note_ids.length > 0,         `${id}: note_ids is a non-empty array`);
    assert(Array.isArray(r.keyphrases),                                 `${id}: keyphrases is an array`);
    assert(typeof r.fallback_used === 'boolean',                        `${id}: fallback_used is a boolean`);
  }

  // Label word count: cleanLabel() hard-caps at 5 words.
  // A label with 0 words means the fallback chain also failed.
  // A label with > 5 words means the cap is broken.
  for (const r of results) {
    const words = r.label.trim().split(/\s+/);
    assert(
      words.length >= 1 && words.length <= 5,
      `cluster_id ${r.cluster_id}: label "${r.label}" is 1–5 words (got ${words.length})`
    );
  }

  // No repeated tokens in a label — the pipeline's degeneracy gate should
  // catch "User user login" style outputs from FLAN-T5.
  for (const r of results) {
    const words = r.label.toLowerCase().split(/\s+/);
    assert(
      words.length === new Set(words).size,
      `cluster_id ${r.cluster_id}: label "${r.label}" has no repeated tokens`
    );
  }

  // Labels are unique across clusters — the cross-cluster uniqueness gate
  // (cosine sim ≥ 0.85 triggers replacement) should prevent near-duplicates.
  const labels = results.map(r => r.label.toLowerCase().trim());
  assert(
    new Set(labels).size === labels.length,
    `all cluster labels are unique (got: ${JSON.stringify(labels)})`
  );

  // Stage 3: clusters with ≥ 3 notes must have at least 1 keyphrase.
  // TF-IDF + KeyBERT always finds candidates in natural-language notes.
  // A missing keyphrase here means extractCandidates returned [] (stopword
  // collision) or embedCache lookup failed — both are pipeline bugs.
  for (const r of results) {
    if (r.note_ids.length >= 3) {
      assert(
        r.keyphrases.length >= 1,
        `cluster_id ${r.cluster_id} (${r.note_ids.length} notes): has ≥ 1 keyphrase`
      );
    }
  }

  // No duplicate note IDs across clusters — each note must belong to exactly
  // one cluster. A note appearing in two clusters means the assignment array
  // was written to two groups, which is a clustering logic bug.
  const allIds = results.flatMap(r => r.note_ids);
  assert(
    new Set(allIds).size === allIds.length,
    `no note ID appears in more than one cluster`
  );

  // All input note IDs must be present in the output.
  const returnedIds = new Set(allIds);
  for (const note of NOTES) {
    assert(returnedIds.has(note.id), `note id "${note.id}" appears in results`);
  }

  // k-selection bounds: k ∈ [K_MIN=2, K_MAX=14] for both algorithms.
  // k-means also caps at floor(n/2), but agglomerative captures snapshots up
  // to K_MAX regardless of n, so the winning k can be up to min(14, n-1).
  // A result count outside this range means silhouette selection is broken or
  // K_MAX was changed without updating the test.
  const K_MAX = 14;
  const realClusters = results.filter(r => r.cluster_id !== -1);
  const maxK = Math.min(K_MAX, NOTES.length - 1); // 14 for n=16
  assert(
    realClusters.length >= 2 && realClusters.length <= maxK,
    `k is in [2, ${maxK}] for ${NOTES.length} notes (got ${realClusters.length})`
  );

  // Noise group invariants (this pipeline does not currently produce noise, but
  // the contract is: if cluster_id === -1 exists, exactly one, labeled correctly,
  // and sorted last). Asserting the contract now means a future DBSCAN integration
  // won't silently break the UI.
  const noiseGroups = results.filter(r => r.cluster_id === -1);
  if (noiseGroups.length > 0) {
    assert(noiseGroups.length === 1,                              'at most one noise group');
    assert(noiseGroups[0].label === 'Other notes',                'noise group label is "Other notes"');
    assert(results[results.length - 1].cluster_id === -1,        'noise group is last in the array');
  } else {
    log('  (no noise group — all notes clustered)', 'info');
  }

  // ── Suite 2: n < 5 edge case ──────────────────────────────────────────────

  log('\n\n═══ Suite 2: n < 5 edge case (3 notes) ═══', 'section');
  log(`Input: ${TINY_NOTES.length} notes\n`, 'info');

  const tinyResults = await clusterNotesFn(TINY_NOTES, { onProgress() {} });

  log('\n─── Results ───', 'section');
  for (const r of tinyResults) {
    log(`\ncluster_id: ${r.cluster_id}  label: "${r.label}"  note_ids: ${JSON.stringify(r.note_ids)}`);
  }

  log('\n─── Assertions ───', 'section');

  // The pipeline skips clustering for n < 5 and labels each note individually.
  assert(
    tinyResults.length === TINY_NOTES.length,
    `n<5 path: returns exactly ${TINY_NOTES.length} clusters for ${TINY_NOTES.length} notes`
  );

  // Each cluster holds exactly one note in the individual-labeling path.
  assert(
    tinyResults.every(r => r.note_ids.length === 1),
    'n<5 path: each cluster contains exactly 1 note'
  );

  // All tiny note IDs are accounted for.
  const tinyReturnedIds = new Set(tinyResults.flatMap(r => r.note_ids));
  for (const note of TINY_NOTES) {
    assert(tinyReturnedIds.has(note.id), `n<5 path: note id "${note.id}" appears in results`);
  }

  // Schema check on tiny results — same contract as the main suite.
  for (const r of tinyResults) {
    assert(typeof r.label === 'string' && r.label.trim().length > 0, `n<5 cluster_id ${r.cluster_id}: non-empty label`);
    assert(Array.isArray(r.keyphrases),                               `n<5 cluster_id ${r.cluster_id}: keyphrases is an array`);
    assert(typeof r.fallback_used === 'boolean',                      `n<5 cluster_id ${r.cluster_id}: fallback_used is a boolean`);
  }
}
