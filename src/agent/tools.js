/**
 * tools.js — Level 2 Agent Tool Registry (MCP pattern)
 *
 * Implements the Model Context Protocol tool-definition + handler pattern for
 * fully local, in-browser execution.  No note content ever leaves the tab.
 *
 * Tools use deterministic heuristics and template generation — FLAN-T5 is
 * accepted as an optional parameter for generate_brief but is not required.
 * This keeps the agent fast even after the pipeline has already exhausted the
 * model's per-session budget.
 */

// ─── Tool definitions (MCP schema) ───────────────────────────────────────────

export const TOOL_DEFINITIONS = [
  {
    name:        'classify_cluster',
    description: 'Classify a cluster into a product feedback category.',
    parameters:  { cluster_label: 'string', keyphrases: 'string[]', notes: 'string[]' },
    returns:     'bug_report | feature_request | process_issue | positive_feedback | unclear',
  },
  {
    name:        'generate_brief',
    description: 'Generate a 3-5 sentence brief for product/engineering handoff.',
    parameters:  { cluster_label: 'string', classification: 'string', notes: 'string[]', keyphrases: 'string[]' },
    returns:     'string',
  },
  {
    name:        'create_action_items',
    description: 'Generate a structured action item list from a cluster.',
    parameters:  { cluster_label: 'string', notes: 'string[]', keyphrases: 'string[]' },
    returns:     'string[]',
  },
  {
    name:        'export_summary',
    description: 'Assemble a full exportable markdown report from all cluster results.',
    parameters:  { clusters: 'object[]', classifications: 'object', briefs: 'object', actionItems: 'object' },
    returns:     'string (markdown)',
  },
];

// ─── Classification signals ───────────────────────────────────────────────────

const CLASSIFICATION_SIGNALS = {
  bug_report: [
    'bug', 'broken', 'crash', 'error', 'fail', 'freeze', 'stuck', 'wrong',
    'issue', 'problem', 'glitch', 'not working', 'not load', 'exception',
    'missing', 'disappeared', 'corrupt', 'lag', 'hang', 'stutter', 'blank',
    'null', 'incorrect', 'broken', 'misbehav', 'unexpected',
  ],
  feature_request: [
    'add', 'need', 'want', 'wish', 'should', 'would like', 'request',
    'option', 'allow', 'support', 'enable', 'improve', 'enhancement',
    'suggest', 'please', 'ability to', 'way to', 'could have', 'would be',
    'missing feature', 'it would', 'allow us',
  ],
  process_issue: [
    'workflow', 'process', 'onboard', 'unclear', 'confusing', 'hard to find',
    'difficult', 'complex', 'navigation', 'understand', 'documentation',
    'help', 'tutorial', 'guide', 'steps', 'instructions', 'team',
    'collaboration', 'permission', 'access', 'policy', 'friction',
  ],
  positive_feedback: [
    'love', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'awesome',
    'fantastic', 'perfect', 'good', 'nice', 'helpful', 'thank', 'appreciate',
    'happy', 'enjoy', 'like', 'useful', 'works well', 'easy to',
  ],
};

// ─── Tool implementations ─────────────────────────────────────────────────────

/**
 * classify_cluster — keyword-signal scoring over label + keyphrases + notes.
 * Fast, deterministic, no model required.
 */
export function classify_cluster({ cluster_label, keyphrases, notes }) {
  const text = [cluster_label, ...(keyphrases ?? []), ...(notes ?? [])]
    .join(' ')
    .toLowerCase();

  const scores = {};
  for (const [category, signals] of Object.entries(CLASSIFICATION_SIGNALS)) {
    scores[category] = signals.filter(s => text.includes(s)).length;
  }

  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  if (ranked[0][1] === 0)              return 'unclear';
  if (ranked[0][1] === ranked[1]?.[1]) return 'unclear';
  return ranked[0][0];
}

/**
 * generate_brief — template-based brief with optional FLAN-T5 enrichment.
 * Template is always the fallback; FLAN-T5 is tried first if a model is passed.
 *
 * @param {object}        params
 * @param {Function|null} flanT5  - loaded text2text-generation pipeline (may be null)
 */
export async function generate_brief({ cluster_label, classification, notes, keyphrases }, flanT5) {
  if (flanT5) {
    try {
      const sampleNotes = (notes ?? [])
        .slice(0, 3)
        .map(t => t.slice(0, 80))
        .join(' | ');
      const prompt =
        `Summarize this user feedback theme for a product team. ` +
        `Theme: "${cluster_label}". Category: ${(classification ?? 'unclear').replace(/_/g, ' ')}. ` +
        `Notes: ${sampleNotes}. Keywords: ${(keyphrases ?? []).join(', ')}. Summary:`;
      const result = await flanT5(prompt, {
        do_sample: true,
        temperature: 0.6,
        max_new_tokens: 80,
        repetition_penalty: 1.2,
      });
      const raw = result[0]?.generated_text ?? '';
      if (raw.length > 40) return raw.trim();
    } catch (_) { /* fall through to template */ }
  }
  return buildTemplateBrief(cluster_label, classification, notes, keyphrases);
}

function buildTemplateBrief(cluster_label, classification, notes, keyphrases) {
  const categoryDesc = {
    bug_report:       'reports a recurring bug or malfunction',
    feature_request:  'requests a new capability or enhancement',
    process_issue:    'highlights a confusing or friction-heavy workflow',
    positive_feedback:'captures positive user sentiment worth reinforcing',
    unclear:          'covers a mixed set of feedback requiring further triage',
  };
  const desc      = categoryDesc[classification] ?? 'covers user feedback';
  const kpStr     = (keyphrases ?? []).slice(0, 2).join(' and ') || cluster_label.toLowerCase();
  const noteCount = (notes ?? []).length;
  const priority  = classification === 'bug_report'      ? 'This warrants a prompt engineering investigation.'
                  : classification === 'feature_request' ? 'Recommend scoping this into the next planning cycle.'
                  : classification === 'process_issue'   ? 'A UX audit or documentation update may resolve this.'
                  :                                        'Review and route to the appropriate team.';
  return (
    `The "${cluster_label}" theme ${desc}. ` +
    `${noteCount} note${noteCount !== 1 ? 's' : ''} contributed to this cluster, with central themes around ${kpStr}. ` +
    `${priority} ` +
    `Key signals: ${(keyphrases ?? []).join(', ') || cluster_label.toLowerCase()}.`
  );
}

/**
 * create_action_items — deterministic, template-driven action list.
 */
export function create_action_items({ cluster_label, notes, keyphrases }) {
  const kp        = (keyphrases ?? []).slice(0, 3);
  const noteCount = (notes ?? []).length;
  const items     = [
    `Review all ${noteCount} note${noteCount !== 1 ? 's' : ''} in the "${cluster_label}" cluster and identify root causes`,
    kp[0] ? `Investigate key signal: "${kp[0]}"` : `Define the scope of "${cluster_label}"`,
    kp[1] ? `Investigate key signal: "${kp[1]}"` : 'Identify the team or system responsible',
    'Assign a DRI (Directly Responsible Individual) for follow-up',
    'Define done criteria before shipping any changes',
    'Schedule a 30-minute review with the affected stakeholders',
  ];
  return items;
}

/**
 * export_summary — assembles a full markdown report from all agent results.
 *
 * @param {object} params
 * @param {Array}  params.clusters         - ranked cluster objects
 * @param {object} params.classifications  - Record<ci, classification>
 * @param {object} params.briefs           - Record<ci, brief>
 * @param {object} params.actionItems      - Record<ci, string[]>
 * @param {string} params.boardSummary
 * @param {Array}  params.outliers
 * @param {Array}  params.mergeSuggestions
 */
export function export_summary({
  clusters, classifications, briefs, actionItems,
  boardSummary, outliers, mergeSuggestions,
}) {
  const now = new Date().toLocaleDateString('en-US', {
    year: 'numeric', month: 'long', day: 'numeric',
  });

  const CATEGORY_LABEL = {
    bug_report:        'Bug Report',
    feature_request:   'Feature Request',
    process_issue:     'Process Issue',
    positive_feedback: 'Positive Feedback',
    unclear:           'Unclear',
  };

  const lines = [
    `# Sticky Insights — Board Analysis Report`,
    ``,
    `*Generated on ${now}*`,
    ``,
    `---`,
    ``,
    `## Board Summary`,
    ``,
    boardSummary || 'No summary available.',
    ``,
    `---`,
    ``,
    `## Themes`,
    ``,
  ];

  for (const cluster of (clusters ?? [])) {
    const ci             = cluster.ci;
    const classification = (classifications ?? {})[ci] ?? 'unclear';
    const brief          = (briefs ?? {})[ci];
    const actions        = (actionItems ?? {})[ci] ?? [];
    const catLabel       = CATEGORY_LABEL[classification] ?? 'Unclear';
    const rankPrefix     = cluster.rank ? `#${cluster.rank} ` : '';

    lines.push(`### ${rankPrefix}${cluster.label}`);
    lines.push(``);
    lines.push(`**Category:** ${catLabel}`);
    lines.push(`**Notes:** ${cluster.note_ids?.length ?? 0}`);
    lines.push(`**Silhouette Score:** ${(cluster.silhouette_score ?? 0).toFixed(3)}`);
    lines.push(`**Keyphrases:** ${(cluster.keyphrases ?? []).join(', ')}`);
    lines.push(``);

    if (brief) {
      lines.push(`**Brief:**`);
      lines.push(``);
      lines.push(brief);
      lines.push(``);
    }

    if (actions.length > 0) {
      lines.push(`**Action Items:**`);
      lines.push(``);
      for (const item of actions) lines.push(`- [ ] ${item}`);
      lines.push(``);
    }

    lines.push(`---`);
    lines.push(``);
  }

  if (outliers?.length > 0) {
    lines.push(`## Notes Needing Review (${outliers.length})`);
    lines.push(``);
    for (const o of outliers) {
      lines.push(`- **"${o.text.slice(0, 80)}"** (by ${o.author}) — ${o.reason}`);
    }
    lines.push(``);
    lines.push(`---`);
    lines.push(``);
  }

  if (mergeSuggestions?.some(s => !s.dismissed)) {
    lines.push(`## Merge Suggestions`);
    lines.push(``);
    for (const s of mergeSuggestions) {
      if (!s.dismissed) {
        lines.push(
          `- Clusters **${s.labelA}** and **${s.labelB}** may describe the same theme ` +
          `(centroid similarity: ${(s.similarity * 100).toFixed(0)}%)`
        );
      }
    }
    lines.push(``);
  }

  return lines.join('\n');
}
