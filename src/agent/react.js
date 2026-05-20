/**
 * react.js — Level 3 ReAct Loop Agent
 *
 * Implements the Reason → Act → Observe → repeat pattern for autonomous
 * multi-step cluster processing.  Uses a rule-based reasoner (no additional
 * LLM inference overhead for reasoning steps) with FLAN-T5 available inside
 * tool calls where content generation is needed.
 *
 * Working memory schema:
 *   { processed: string[], pending: string[], results: Record<label, any>, step: number }
 *
 * Termination conditions:
 *   1. memory.pending is empty (all clusters processed)
 *   2. step count reaches MAX_STEPS (safety ceiling — prevents runaway loops)
 *
 * Privacy: all computation runs in the browser — no data leaves the tab.
 */

import {
  classify_cluster,
  generate_brief,
  create_action_items,
  export_summary,
} from './tools.js';

// Each cluster requires classify + finalize (2 steps) + optional brief/action-items (1 step).
// 15 clusters × 3 steps + 5 overhead = 50.  Set ceiling at 100 for boards up to ~30 clusters.
const MAX_STEPS = 100;

// ─── Working memory ───────────────────────────────────────────────────────────

function initMemory(clusters) {
  return {
    processed: [],
    pending:   clusters.map(c => c.label),
    results:   {},
    step:      0,
  };
}

// ─── Rule-based reasoner ──────────────────────────────────────────────────────
//
// Decides the next action based on memory state alone.  Returns one of:
//   { action: 'classify', cluster }
//   { action: 'generate_brief', cluster }
//   { action: 'create_action_items', cluster }
//   { action: 'finalize', cluster }   — marks cluster complete, advances pending
//   { action: 'skip',    label }      — cluster not found, advance pending
//   { action: 'done' }                — all clusters processed

function reason(memory, clusters) {
  if (memory.pending.length === 0) return { action: 'done' };

  const nextLabel = memory.pending[0];
  const cluster   = clusters.find(c => c.label === nextLabel);

  if (!cluster) return { action: 'skip', label: nextLabel };

  const r = memory.results[nextLabel];

  // Phase 1: classify
  if (!r || r.classification === undefined) return { action: 'classify', cluster };

  // Phase 2a: brief for actionable categories
  const needsBrief = r.classification === 'bug_report' || r.classification === 'feature_request';
  if (needsBrief && !r.brief) return { action: 'generate_brief', cluster };

  // Phase 2b: action items for process issues
  if (r.classification === 'process_issue' && !r.actionItems) {
    return { action: 'create_action_items', cluster };
  }

  // Phase 3: all work for this cluster done — advance the queue
  return { action: 'finalize', cluster };
}

// ─── Main ReAct loop ──────────────────────────────────────────────────────────

/**
 * Runs the full ReAct loop over all clusters.
 *
 * @param {Array}         clusters         - ranked cluster objects
 * @param {Array}         notes            - raw note objects
 * @param {Function}      onStep           - (step: object) => void — UI streaming callback
 * @param {Function|null} flanT5           - FLAN-T5 singleton (may be null)
 * @param {object}        levelOneInsights - { boardSummary, outliers, mergeSuggestions }
 */
export async function runReActLoop(clusters, notes, onStep, flanT5, levelOneInsights) {
  const memory         = initMemory(clusters);
  const noteById       = new Map(notes.map(n => [n.id, n]));
  let briefCount       = 0;
  let actionItemCount  = 0;

  onStep({
    type: 'goal',
    text: `Processing ${clusters.length} cluster${clusters.length !== 1 ? 's' : ''} — classifying, generating briefs for actionable items, and preparing export.`,
  });

  // ── ReAct loop ─────────────────────────────────────────────────────────────
  while (memory.step < MAX_STEPS) {
    memory.step++;

    // REASON
    const decision = reason(memory, clusters);

    if (decision.action === 'done') {
      onStep({ type: 'reason', text: 'All clusters processed. Assembling export.' });
      break;
    }

    if (decision.action === 'skip') {
      memory.pending.shift();
      continue;
    }

    const { cluster }   = decision;
    const { label, keyphrases, note_ids, ci } = cluster;
    const memberNotes   = (note_ids ?? [])
      .map(id => noteById.get(id))
      .filter(Boolean)
      .map(n => n.text);

    if (!memory.results[label]) memory.results[label] = { retries: 0 };
    const r = memory.results[label];

    // ACT + OBSERVE
    switch (decision.action) {

      case 'classify': {
        onStep({ type: 'tool', tool: 'classify_cluster', cluster: label, status: 'running' });
        let classification;
        try {
          classification = classify_cluster({ cluster_label: label, keyphrases, notes: memberNotes });
          if (!classification) throw new Error('empty');
        } catch (_) {
          // Retry once with a minimal fallback prompt (simulates modified-prompt retry)
          if (r.retries < 1) {
            r.retries++;
            onStep({ type: 'retry', cluster: label, tool: 'classify_cluster' });
            // Re-attempt immediately (next loop iteration re-enters 'classify')
            break;
          }
          classification = 'unclear';
        }
        r.classification = classification;
        r.retries        = 0; // reset for next phase
        onStep({
          type: 'tool', tool: 'classify_cluster', cluster: label,
          status: 'done', result: classification.replace(/_/g, ' '),
        });
        break;
      }

      case 'generate_brief': {
        onStep({ type: 'tool', tool: 'generate_brief', cluster: label, status: 'running' });
        let brief;
        try {
          brief = await generate_brief(
            { cluster_label: label, classification: r.classification, notes: memberNotes, keyphrases },
            flanT5,
          );
          if (!brief || brief.length < 20) throw new Error('empty');
        } catch (_) {
          if (r.retries < 1) {
            r.retries++;
            onStep({ type: 'retry', cluster: label, tool: 'generate_brief' });
            break;
          }
          // Mark as unclear and continue — this cluster's brief is omitted from report
          brief = `The "${label}" theme requires follow-up. Review the ${memberNotes.length} notes in this cluster.`;
        }
        r.brief   = brief;
        r.retries = 0;
        briefCount++;
        onStep({
          type: 'tool', tool: 'generate_brief', cluster: label,
          status: 'done', result: brief.slice(0, 65) + (brief.length > 65 ? '…' : ''),
        });
        break;
      }

      case 'create_action_items': {
        onStep({ type: 'tool', tool: 'create_action_items', cluster: label, status: 'running' });
        let items;
        try {
          items = create_action_items({ cluster_label: label, notes: memberNotes, keyphrases });
          if (!items || items.length === 0) throw new Error('empty');
        } catch (_) {
          if (r.retries < 1) {
            r.retries++;
            onStep({ type: 'retry', cluster: label, tool: 'create_action_items' });
            break;
          }
          items = [
            `Review all notes in the "${label}" cluster`,
            'Assign ownership to a team member',
            'Define next steps and timeline',
          ];
        }
        r.actionItems = items;
        r.retries     = 0;
        actionItemCount++;
        onStep({
          type: 'tool', tool: 'create_action_items', cluster: label,
          status: 'done', result: `${items.length} item${items.length !== 1 ? 's' : ''}`,
        });
        break;
      }

      case 'finalize': {
        memory.processed.push(label);
        memory.pending.shift();
        onStep({
          type:      'progress',
          processed: memory.processed.length,
          total:     clusters.length,
        });
        break;
      }
    }
  }

  if (memory.step >= MAX_STEPS) {
    onStep({ type: 'warn', text: `Reached step limit (${MAX_STEPS}). Some clusters may be incomplete.` });
  }

  // ── Assemble final results from working memory ──────────────────────────────
  const classifications = {};
  const briefs          = {};
  const actionItems     = {};

  for (const cluster of clusters) {
    const r = memory.results[cluster.label] ?? {};
    if (r.classification) classifications[cluster.ci] = r.classification;
    if (r.brief)          briefs[cluster.ci]          = r.brief;
    if (r.actionItems)    actionItems[cluster.ci]     = r.actionItems;
  }

  onStep({ type: 'tool', tool: 'export_summary', cluster: 'all', status: 'running' });
  const markdownReport = export_summary({
    clusters:         clusters,
    classifications,
    briefs,
    actionItems,
    boardSummary:     levelOneInsights?.boardSummary,
    outliers:         levelOneInsights?.outliers,
    mergeSuggestions: levelOneInsights?.mergeSuggestions,
  });
  onStep({ type: 'tool', tool: 'export_summary', cluster: 'all', status: 'done', result: 'Report ready' });

  const processedCount = memory.processed.length;
  onStep({
    type: 'complete',
    text: `Done. Processed ${processedCount} cluster${processedCount !== 1 ? 's' : ''}. ` +
          `Created ${briefCount} brief${briefCount !== 1 ? 's' : ''}, ` +
          `${actionItemCount} action item list${actionItemCount !== 1 ? 's' : ''}. Ready to export.`,
  });

  return { classifications, briefs, actionItems, markdownReport };
}
