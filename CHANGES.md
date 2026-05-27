# CHANGES — Agentic Capabilities (Levels 1–3)

## What Was Built

Three layers of agentic intelligence were added on top of the existing clustering pipeline, all running entirely in the browser with no external API calls or data egress.

### Level 1 — Reactive Agent (`src/insights-agent.js`)

Runs synchronously after clustering completes, before the UI renders. Produces four artefacts:

- **Outlier detection** — notes with per-note silhouette score < 0.05 are flagged and surfaced in a collapsible "Needs Review" section. The threshold is derived from the silhouette scale: scores below 0.05 indicate the note fits its assigned cluster no better than any other.
- **Merge suggestions** — cluster pairs whose L2-normalised centroid cosine similarity exceeds 0.75 are flagged as likely duplicates. Cards are dismissible and show centroid similarity alongside note counts and silhouette scores.
- **Board summary** — a single sentence generated from cluster stats: "Your team surfaced N themes from X notes by Y contributors. The strongest signal is [top cluster]. N notes may need review."
- **Cluster ranking** — composite score: 60 % normalised note count + 40 % normalised silhouette score. Rank badge shown on each tile; tiles rendered in rank order.

### Level 2 — Tool-Calling Agent (`src/agent/tools.js`, `src/agent/agent.js`)

An MCP-pattern tool registry with four tools, each implemented as a pure in-browser function:

| Tool | Implementation |
|---|---|
| `classify_cluster` | Keyword-signal scoring over 5 categories (bug_report, feature_request, process_issue, positive_feedback, unclear) |
| `generate_brief` | FLAN-T5 (LaMini-Flan-T5-248M via @xenova/transformers) with template fallback if model unavailable or output < 40 chars |
| `create_action_items` | Template-based, 6 items derived from keyphrases and note count |
| `export_summary` | Markdown report: board summary, cluster table, briefs, action items, outliers, merge suggestions |

The sequential agent loop in `agent.js` iterates rankedClusters: classify → brief (for bug_report/feature_request) → action items (for process_issue) → export.

### Level 3 — ReAct Loop Agent (`src/agent/react.js`)

Implements Reason → Act → Observe with working memory and retry logic:

- **Working memory**: `{ processed: string[], pending: string[], results: Record<label, any>, step: number }`
- **Rule-based reasoner**: stateless function over memory snapshot; no additional LLM inference cost per reasoning step
- **Retry logic**: each tool phase retries once on empty/failed output, then falls back to a template result (never silently drops a cluster)
- **Step ceiling**: `MAX_STEPS = 100` (15 clusters × ~3 steps + overhead = ~50; ceiling set at 2× for safety)
- **Streaming UI**: each reasoning/tool step emits to the Agent Activity panel via `onStep` callback; panel auto-opens on first step

---

## Enhancements Delivered

- **Per-note silhouette scores** added to `pipeline.js` (`silhouettePerNote()`). The existing pipeline only computed per-cluster averages; per-note scores are required for outlier detection.
- **Centroid exposure** — `clusterCentroids` array previously local to Stage 3 is now returned from `clusterNotes()` so the insights agent can compute merge candidates.
- **FLAN-T5 reuse** — `getFlanT5` exported as a singleton from `pipeline.js`. The model is already loaded during Stage 4; the agent retrieves it without re-loading.
- **Agent Activity panel** in `cluster-view.js` — collapsible, streams reasoning steps with enter animations, progress bar, completion message, and Download Report button.
- **`flex-shrink: 0`** on all Level 1/2/3 UI sections to prevent the flex column in `.cv-left-col` from compressing them to near-zero height.

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| FLAN-T5 unavailable (first load, WASM not ready) | `generate_brief` falls back to `buildTemplateBrief`; agent continues without blocking |
| Step limit hit before all clusters processed | `MAX_STEPS = 100` covers boards up to ~30 clusters; `warn` step emitted if ceiling is reached |
| Centroid similarity threshold too aggressive (many false merge suggestions) | Threshold set at 0.75 (high end); users can dismiss cards individually |
| Outlier threshold misses edge cases | Per-note silhouette < 0.05 is conservative; only notes with near-zero cluster affinity are flagged |
| `classify_cluster` misclassifies ambiguous notes | Falls back to `'unclear'` category; no brief or action items generated, cluster still exported |

---

## Relevant Signals

- **Silhouette distribution**: MiniLM on short sticky notes produces a compressed distance range (typical scores 0.10–0.55). The 0.05 outlier threshold captures notes at the bottom ~5 % of this range.
- **Cosine similarity for merge**: With 384-dimensional L2-normalised embeddings, centroids above 0.75 cosine similarity are genuinely close. This is roughly equivalent to two clusters whose centroids are within 41° of each other in embedding space.
- **MAX_STEPS derivation**: classify (1) + optional brief or action_items (1) + finalize (1) = 3 steps per cluster. 15 clusters × 3 = 45. Each retry adds 1 step. Ceiling at 100 gives 2× headroom without risk of infinite loops.
- **Template fallback quality**: The `buildTemplateBrief` templates were designed to still be actionable even without model output — they name the cluster, state the note count, and direct to the appropriate team action.

---

## Recommended Next Steps

1. **Replace keyword classifier with a zero-shot model** — `classify_cluster` uses simple keyword matching, which will misclassify nuanced notes. Replacing with a zero-shot BART or DeBERTa model (via @xenova/transformers) would improve accuracy. Prerequisite: confirm latency is acceptable in the browser.

2. **Tune silhouette thresholds per board** — A fixed 0.05 outlier threshold is not adaptive to board size or note density. A better approach: flag the bottom 5th percentile of per-note scores per run. This adjusts automatically to the data distribution.

3. **Add user feedback loop to classifier** — Currently there is no signal on whether classifications are correct. A simple thumbs-up/down on each cluster card would generate labelled data for future fine-tuning.

4. **Persist dismissed merge suggestions** — Dismissed cards currently survive only for the session. Storing dismissed pairs in `localStorage` (keyed by cluster label pair) would prevent re-surfacing them on re-cluster.

5. **Export to structured JSON, not just Markdown** — The current `export_summary` produces a Markdown file. A parallel JSON export would enable downstream tooling (Jira ticket creation, Confluence page sync) without parsing Markdown.
