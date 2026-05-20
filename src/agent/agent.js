/**
 * agent.js — Level 2 Tool-Calling Agent
 *
 * Reads cluster outputs and calls tools based on classification results,
 * streaming reasoning steps to the Agent Activity panel in real time.
 *
 * Privacy: all computation runs in the browser — no data leaves the tab.
 *
 * Tool dispatch logic:
 *   1. classify_cluster    → every cluster
 *   2. generate_brief      → bug_report and feature_request clusters
 *   3. create_action_items → process_issue clusters
 *   4. export_summary      → final step, all clusters
 */

import {
  classify_cluster,
  generate_brief,
  create_action_items,
  export_summary,
} from './tools.js';

/**
 * Runs the tool-calling agent loop over all ranked clusters.
 *
 * @param {Array}         rankedClusters   - clusters with { ci, label, keyphrases, note_ids, rank }
 * @param {Array}         notes            - raw note objects
 * @param {Function}      onStep           - (step: object) => void — UI streaming callback
 * @param {Function|null} flanT5           - FLAN-T5 singleton (may be null)
 * @param {object}        levelOneInsights - { boardSummary, outliers, mergeSuggestions }
 * @returns {Promise<{ classifications, briefs, actionItems, markdownReport }>}
 */
export async function runAgentLoop(rankedClusters, notes, onStep, flanT5, levelOneInsights) {
  const classifications = {};
  const briefs          = {};
  const actionItems     = {};

  onStep({
    type: 'goal',
    text: `Processing ${rankedClusters.length} cluster${rankedClusters.length !== 1 ? 's' : ''} — classifying, generating briefs for actionable items, and preparing export.`,
  });

  const noteById = new Map(notes.map(n => [n.id, n]));

  for (const cluster of rankedClusters) {
    const { ci, label, keyphrases, note_ids } = cluster;
    const memberNotes = (note_ids ?? [])
      .map(id => noteById.get(id))
      .filter(Boolean)
      .map(n => n.text);

    // ── classify ──────────────────────────────────────────────────────────
    onStep({ type: 'tool', tool: 'classify_cluster', cluster: label, status: 'running' });
    const classification = classify_cluster({ cluster_label: label, keyphrases, notes: memberNotes });
    classifications[ci]  = classification;
    onStep({ type: 'tool', tool: 'classify_cluster', cluster: label, status: 'done', result: classification.replace(/_/g, ' ') });

    // ── generate brief for bug reports and feature requests ───────────────
    if (classification === 'bug_report' || classification === 'feature_request') {
      onStep({ type: 'tool', tool: 'generate_brief', cluster: label, status: 'running' });
      const brief = await generate_brief(
        { cluster_label: label, classification, notes: memberNotes, keyphrases },
        flanT5,
      );
      briefs[ci] = brief;
      onStep({ type: 'tool', tool: 'generate_brief', cluster: label, status: 'done', result: brief.slice(0, 70) + (brief.length > 70 ? '…' : '') });
    }

    // ── create action items for process issues ────────────────────────────
    if (classification === 'process_issue') {
      onStep({ type: 'tool', tool: 'create_action_items', cluster: label, status: 'running' });
      const items    = create_action_items({ cluster_label: label, notes: memberNotes, keyphrases });
      actionItems[ci] = items;
      onStep({ type: 'tool', tool: 'create_action_items', cluster: label, status: 'done', result: `${items.length} items` });
    }
  }

  // ── assemble report ───────────────────────────────────────────────────────
  onStep({ type: 'tool', tool: 'export_summary', cluster: 'all', status: 'running' });
  const markdownReport = export_summary({
    clusters:          rankedClusters,
    classifications,
    briefs,
    actionItems,
    boardSummary:      levelOneInsights?.boardSummary,
    outliers:          levelOneInsights?.outliers,
    mergeSuggestions:  levelOneInsights?.mergeSuggestions,
  });
  onStep({ type: 'tool', tool: 'export_summary', cluster: 'all', status: 'done', result: 'Report ready' });

  const briefCount      = Object.keys(briefs).length;
  const actionListCount = Object.keys(actionItems).length;
  onStep({
    type: 'complete',
    text: `Done. Processed ${rankedClusters.length} cluster${rankedClusters.length !== 1 ? 's' : ''}. ` +
          `Created ${briefCount} brief${briefCount !== 1 ? 's' : ''}, ` +
          `${actionListCount} action item list${actionListCount !== 1 ? 's' : ''}. Ready to export.`,
  });

  return { classifications, briefs, actionItems, markdownReport };
}
