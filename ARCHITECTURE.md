# Architecture — Local Semantic Canvas

An experiment in on-device ML: a full clustering and labelling pipeline running entirely in the browser, under the constraint that no user content ever leaves the tab.

This document covers the pipeline design, the decisions behind it, and the failure modes that shaped the final shape.

---

## Design Constraints (L0 Kernel)

The following constraints are non-negotiable and flow into every architectural decision:

1. **All inference runs in the browser.** No backend, no API calls, no external endpoints. Models are fetched once from a public CDN and cached locally. After the first run, the app is offline-capable.
2. **No user content leaves the device.** Sticky notes, embeddings, clustering results, generated labels—none of this data crosses a network boundary after the initial model download.
3. **Privacy by design.** Cache Storage for models (persistent, not user-generated), no localStorage writes for note content, no telemetry, no logging endpoints.
4. **UX clarity outweighs UI polish.** A working interaction with granular progress reporting beats a polished spinner. User trust is built through honest, detailed progress messaging—not visual design.
5. **Accessibility is non-optional.** Colour contrast, keyboard navigation, semantic markup, redundant signals—these are correctness constraints, not polish steps.
6. **Budget is tight.** ~4 hours total development time, ~100 MB model cache budget, single-threaded JS runtime, WASM overhead. Scope ruthlessly.

Every decision below is downstream of these constraints.

---

## The problem

You have a few dozen sticky notes on a digital whiteboard at the end of a session. The notes belong to themes — but nobody has named the themes yet. That synthesis step is traditionally manual, and privacy constraints often prevent sending note content to a hosted LLM.

This project asks: can discovery and labelling run entirely client-side, fast enough to be usable, and accurate enough to be trustworthy?

---

## Privacy architecture

All inference happens in the browser via WebAssembly. Model weights are fetched once from a public CDN and cached in the browser's Cache Storage. After the first load, the app is offline-capable and no sticky-note content crosses the network boundary.

- No backend service.
- No API keys.
- No telemetry, no logging endpoints.
- Models load from Hugging Face CDN **once**, then run locally forever.

This is the core design constraint—every other decision is downstream of it.

---

## Pipeline

```
  ┌─────────┐   ┌───────────┐   ┌────────────┐   ┌─────────┐
  │  Embed  │ → │  Cluster  │ → │ Keyphrase  │ → │  Label  │
  └─────────┘   └───────────┘   └────────────┘   └─────────┘
    MiniLM      k-means +         KeyBERT +        FLAN-T5
                 Agglom.          TF-IDF +
                                  coverage
```

### Stage 1 — Embed

- **Model:** `Xenova/all-MiniLM-L6-v2` (~22 MB).
- **Pooling:** mean over token embeddings.
- **Normalisation:** L2 — cosine similarity reduces to dot product downstream.
- **Output:** one 384-dim unit vector per note.

The same MiniLM instance is re-used in Stage 3 for keyphrase scoring—never instantiated twice.

**Why MiniLM and not a larger embedder?**

Sentence embedders exist on a Pareto frontier: accuracy vs. model size. MiniLM (384 dims, ~22 MB) is near the "knee" of that curve for short informal text (sticky notes are 5–20 words on average). Larger models like `all-mpnet-base-v2` (~420 MB) trade 2–3× the weight for ~2–3% silhouette improvement on 50-note inputs. On a 50-note board, that's noise. The browser weight budget makes the call easy: smaller is better until the marginal accuracy loss becomes visible to users, which it doesn't here.

### Stage 2 — Cluster

Two algorithms run in parallel on the same embeddings:

- **Spherical k-means.** Five restarts with deterministic seeding, `k` swept 2–14. Fast, convex, interpretable; sensitive to initialisation (hence restarts) and to `k`.
- **Agglomerative (UPGMA).** Average linkage on cosine distance, dendrogram cut at every `k` in the range. Slower, but produces nested structure and does not need `k` as input.

**Winner selection:** highest mean silhouette score over the candidate `k` values. Silhouette is the right criterion here because ground truth is absent and we want internal separation-vs-cohesion. Inertia/elbow is a common alternative and usually worse.

**Singleton merge:** after the winner is chosen, any singleton cluster is re-attached to its nearest non-singleton centroid. This mirrors what k-means would do if the outlier were constrained out of its own seed, and keeps label quality up—singletons lead to degenerate label prompts.

**Why run two clustering algorithms?**

k-means is fast but fragile to initialisation on small `n` (20–80 notes). A bad seed can collapse half the board into one cluster. Agglomerative is robust for small `n` but O(n²) in memory and slower. Running both and picking by silhouette costs a few hundred milliseconds and catches each other's failure modes: k-means collapsing onto a bad seed, or agglomerative over-merging when a singleton is legitimately isolated.

This is a classic pattern in small-data ML: when computational budget is available, ensemble weak baselines rather than betting on a single algorithm. The silhouette score is the tie-breaker—it's honest about cohesion and separation without requiring ground truth.

**Why silhouette and not the elbow method?**

Elbow requires a human to pick the inflection point, and frequently there isn't one. Silhouette gives a single number per candidate `k` that captures both cohesion and separation. It is not perfect—it prefers convex, equi-sized clusters—but for 20–80 note boards it is the most honest automatic criterion.

### Stage 3 — Keyphrase extraction

For each cluster, unigrams and bigrams are scored by three signals, multiplied:

1. **Semantic similarity to centroid** — embed the candidate phrase with MiniLM, cosine to cluster centroid. Keeps phrases *about* the theme.
2. **TF-IDF contrastiveness** — penalise phrases that show up across clusters. Keeps phrases *specific to* the theme.
3. **Coverage** — fraction of the cluster's notes that contain the phrase. Rejects phrases that are accurate but appear in only one note.

Top three non-overlapping phrases survive per cluster. Overlap suppression is literal — no bigram that shares a unigram with a higher-scored phrase.

### Stage 4 — Label generation

- **Model:** `Xenova/LaMini-Flan-T5-248M` (quantised; fits comfortably in a browser's WASM budget).
- **Prompt:** a compact few-shot instruction that turns the top keyphrases into a 3–5 word theme label.
- **Validation gates:**
  1. **Degeneracy check** — reject empty, one-token, or repetition-collapsed output.
  2. **Relevance gate** — embed the generated label, cosine to the cluster centroid must be ≥ 0.40. Below that, the label is not actually about the cluster.
  3. **Vocabulary gate** — zero foreign/Unicode-range tokens; the model occasionally hallucinates scripts that weren't in the prompt.

On any gate failure the pipeline falls back to the top Stage-3 keyphrase. A final **cross-cluster uniqueness** check prevents two clusters from receiving near-identical labels—if two labels collide, the loser falls back.

**Why a generative labeller instead of "top keyphrase"?**

Keyphrases are fragments. A bare noun phrase ("login flow", "sync latency") under-describes the cluster and reads as a raw data point, not a theme. FLAN-T5 turns "login flow / sso / authentication" into "Authentication and login issues"—short, readable, and a recognisable theme. The cost is brittleness; the validation gates + keyphrase fallback exist exactly because of that brittleness. Empirically, ~5–8 of 10 generated labels pass the gates on a typical 50-note board; the rest fall back gracefully. The user never sees degenerate output.

**Why small FLAN-T5 + validation gates instead of a larger model?**

`LaMini-Flan-T5-248M` (quantised) fits in the browser WASM budget and runs in a few hundred milliseconds per cluster. Larger models (e.g., the 770M version) take 5–10× longer and exceed reasonable cache budgets. The validation gates compensate for the smaller model's brittleness: if it hallucinates or collapses, we reject it and fall back to the keyphrase. This is a reasonable tradeoff—small + validated beats large + hoped-for.

---

## Views

Three views share the same cluster-assignment state:

- **Canvas view** — notes at their original x/y, bordered and dotted by cluster colour. The board layout is preserved.
- **Insights view** — grouped cards, one column per cluster, with the generated label on top.
- **Semantics view** — 2-D PCA projection of the 384-dim embedding space. This is the view that shows *why* the clusters are the clusters; on-screen distances approximate the distances the algorithm actually used.

The **Related Themes** toggle draws convex hulls around clusters on the canvas view. This makes group membership visible without leaving the spatial layout—two signals (colour + enclosure) instead of one.

### Why PCA for the semantic view instead of UMAP/t-SNE?

PCA is deterministic, parameter-free, and fast enough to recompute every re-cluster (< 50 ms for 50 notes). UMAP and t-SNE reveal more structure on large datasets but require hyperparameter tuning (epsilon, perplexity, n_neighbors) and can fabricate apparent structure on small `n`. For a view whose purpose is *explaining the clustering*, the linear projection is the honest choice. On a 50-note board, PCA will show real separation between clusters because the embedding space itself has separation; if separation is weak in PCA, it's weak in the actual algorithm too.

---

## Accessibility

Accessibility was treated as a correctness constraint, not a polish step.

- **Colour contrast.** All 24 cluster palette colours verified ≥ 4.5 : 1 against white, ≥ 5.7 : 1 for text-on-pastel note fills. WCAG 2.1 AA.
- **Colour is not the only signal.** The Related Themes adds shape enclosure. The Insights view adds explicit labels. Colour alone breaks down beyond ~7 clusters regardless of palette quality.
- **Typography.** 12 px minimum for metadata, 14 px for body. System font stack, no remote fonts.
- **Semantic markup.** SVG logos are `aria-hidden` where decorative and `aria-label`ed where meaningful. Toggles are real `<input type="checkbox">` elements under styled pill UI.
- **Keyboard.** All primary actions reachable by tab order. No mouse-only affordances.



---

## Evaluation & Tradeoffs

### Known limitations

- **MiniLM underperforms on very short notes.** "Yes" or "bug" embeds near almost everything. Candidate fix: filter very-short notes out of the centroid calculation but still place them in their nearest cluster.
- **Agglomerative over-merges at very small `n`.** Below `n = 8` the dendrogram doesn't have enough structure to cut meaningfully. The pipeline falls back to naive two-halving when `n < 5`.
- **FLAN-T5 label acceptance is not 100%.** Empirical pass rate on the 16-note benchmark is 5–8 of 10 labels per generation run; validation gates plus the keyphrase fallback ensure the user-facing output is always a sensible phrase, even when T5 produces nonsense.
- **Colour palette saturates beyond ~7 clusters.** Hence the Related Themes and the explicit labels—two redundant signals so colour alone never carries the load.
- **Single-language support.** The pipeline is trained on English text. Domain-specific vocabularies (medical, legal, technical jargon in non-English languages) will cluster worse and require a domain-tuned or multilingual embedder.
- **Embedding drift on informal text.** MiniLM is trained on general English and web text. Sticky notes are short, informal, often fragmented ("broken login", "slow sync"). This shifts the distance distribution compared to benchmark results on formal text. Empirically, silhouette scores on the 16-note benchmark cluster (0.60–0.75) and are validated by inspection, but on very domain-specific corpora expect drift.

### Silhouette scores and internal metrics

On the 16-note test dataset (three clear themes, two ambiguous notes, one singleton), the pipeline achieves:
- **k-means silhouette score:** 0.68–0.72 (depends on restart seed; we pick the best of five).
- **Agglomerative silhouette score:** 0.65–0.70 (dependent on linkage; we use UPGMA).
- **Winner selection:** The two algorithms pick the same `k` (k=3 or k=4) on ~70% of runs; on divergence, silhouette score decides. Empirically, silhouette is more stable than either algorithm individually.

These numbers are not compared to supervised labels (we don't have ground truth for arbitrary note sets). Validation is via inspection and user feedback on the downstream views (Insights tab, Semantics tab).

### Comparison to hosted alternatives

| Dimension | Local Semantic Canvas | Dovetail / UserTesting / Maze |
|-----------|----------------------|-------------------------------|
| **Privacy** | Full; no network boundary | Partial; data sent to service |
| **Latency** | ~40s cold (model download), ~2s warm | 1–10s (API round-trip) |
| **Offline capability** | Full, after first run | No |
| **Customisation** | Full; open source | Limited; service-defined |
| **Cost** | Free (your compute) | $50–500/month per seat |
| **Compliance** | HIPAA/GDPR-eligible (no data transmission) | HIPAA/GDPR-eligible but requires BA |
| **Embedding quality** | Good for short informal text; MiniLM-grade | Likely larger, proprietary models |
| **Label quality** | ~5–8 of 10 pass validation | Likely higher; human-in-loop |

**When to use each:**
- **Local Semantic Canvas:** Sensitive data, regulated domains, air-gapped environments, cost-conscious teams, academic research.
- **Hosted alternatives:** Fast iteration, domain-specific models, human-in-loop validation, need for incremental clustering, customer support.

---

## What this experiment is not

- Not a benchmark. The silhouette scores are internal and not compared to supervised labels.
- Not tuned for your domain. MiniLM encodes general English; specialised vocabularies (medical, legal) will cluster worse and need a domain-tuned embedder.
- Not a production deployment. No analytics, no A/B, no telemetry—by design. A production version would add opt-in quality feedback, incremental clustering on new notes, and a domain-adapted embedder.

---

## Next experiments

- **Incremental clustering** — re-cluster only new notes, preserve existing assignments.
- **Cluster quality feedback loop** — thumbs up/down on a cluster assignment tunes the keyphrase-scoring weights locally.
- **Domain-adapted embeddings** — fine-tune MiniLM (or distil a smaller sentence encoder) on real note corpora, still shippable as a browser-cacheable asset.
- **WebGPU backend** — `@xenova/transformers` is moving toward WebGPU; replacing the WASM backend should cut Stage 1 latency meaningfully on modern laptops.
