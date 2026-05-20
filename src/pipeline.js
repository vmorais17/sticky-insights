/**
 * pipeline.js — Sticky note insight discovery pipeline
 *
 * ╔══════════════════════════════════════════════════════════════════════╗
 * ║  PRIVACY ARCHITECTURE — WHY THIS IS SAFE FOR ENTERPRISE USE         ║
 * ║                                                                      ║
 * ║  All inference runs entirely in the browser via WebAssembly (WASM). ║
 * ║  No sticky note content is ever sent to a remote server.            ║
 * ║  Model weights are fetched once from Hugging Face CDN and cached    ║
 * ║  in the browser's Cache Storage. All text stays in the tab.         ║
 * ║                                                                      ║
 * ║  Models used:                                                        ║
 * ║    Stage 1+3: Xenova/all-MiniLM-L6-v2       (~22 MB, shared singleton)   ║
 * ║    Stage 4:   Xenova/LaMini-Flan-T5-248M (~250 MB, singleton)          ║
 * ╚══════════════════════════════════════════════════════════════════════╝
 *
 * Pipeline overview:
 *   Stage 1 — Embed each note with MiniLM (mean pool, L2 normalize)
 *   Stage 2 — Cluster with two algorithms, pick winner by silhouette score
 *               a) Spherical k-means  (ml-kmeans on L2-normalised vectors)
 *               b) Agglomerative      (average linkage, cosine distance)
 *   Stage 3 — Extract contrastive keyphrases (KeyBERT analogue + TF-IDF)
 *   Stage 4 — Generate insight labels with LaMini-Flan-T5-248M
 */

import { pipeline, env } from '@xenova/transformers';
import { kmeans } from 'ml-kmeans';
import { pcaProject } from './pca.js';

// All model weights are fetched from HuggingFace CDN and cached in the
// browser Cache Storage — no sticky note text ever leaves this tab.
env.allowLocalModels = false;

// HuggingFace CDN uses chunked transfer encoding and omits Content-Length
// headers. @xenova/transformers logs a warning when it can't read that header
// and falls back to a dynamic buffer — behaviour is correct, warning is noise.
const _warn = console.warn;
console.warn = (...args) => {
  if (typeof args[0] === 'string' && args[0].includes('content-length')) return;
  _warn.apply(console, args);
};

// ─── Model singletons ──────────────────────────────────────────────────────
// Lazy-loaded on first call, then reused for the entire session.
// MiniLM is shared between Stage 1 (note embedding) and Stage 3 (keyphrase
// scoring) — it is never instantiated more than once per session.

let _miniLM = null;
let _flanT5 = null;

async function getMiniLM(progressCb) {
  if (_miniLM) return _miniLM;
  _miniLM = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    progress_callback: progressCb,
  });
  return _miniLM;
}

export async function getFlanT5(progressCb) {
  if (_flanT5) return _flanT5;
  _flanT5 = await pipeline('text2text-generation', 'Xenova/LaMini-Flan-T5-248M', {
    progress_callback: progressCb,
  });
  return _flanT5;
}

// ─── Math helpers ──────────────────────────────────────────────────────────

function dotProduct(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Cosine similarity on L2-normalised vectors (= dot product). */
function cosineSim(a, b) {
  return dotProduct(a, b);
}

/** Cosine distance on L2-normalised vectors (= 1 − dot product). */
function cosineDistance(a, b) {
  return 1 - dotProduct(a, b);
}

/** Embed a single string using the shared MiniLM singleton. */
async function embedText(text) {
  const extractor = await getMiniLM();
  const out = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(out.data);
}

// ─── Note text normalisation ───────────────────────────────────────────────
//
// Applied before embedding (Stage 1 only). MiniLM's WordPiece tokeniser splits
// contractions into subword tokens ("can", "##'", "##t") that add noise without
// contributing semantic signal — "can't" and "cannot" should embed identically
// but do not. Expanding contractions before the forward pass produces cleaner,
// more consistent embeddings and tightens intra-cluster cosine distances.
// Not applied to raw note text displayed in the UI or used for keyphrase extraction.

const CONTRACTIONS = [
  [/\bcan't\b/gi,    'cannot'],
  [/\bwon't\b/gi,    'will not'],
  [/\bdon't\b/gi,    'do not'],
  [/\bdoesn't\b/gi,  'does not'],
  [/\bdidn't\b/gi,   'did not'],
  [/\bisn't\b/gi,    'is not'],
  [/\baren't\b/gi,   'are not'],
  [/\bwasn't\b/gi,   'was not'],
  [/\bweren't\b/gi,  'were not'],
  [/\bhadn't\b/gi,   'had not'],
  [/\bhasn't\b/gi,   'has not'],
  [/\bhaven't\b/gi,  'have not'],
  [/\bcouldn't\b/gi, 'could not'],
  [/\bwouldn't\b/gi, 'would not'],
  [/\bshouldn't\b/gi,'should not'],
  [/\bmightn't\b/gi, 'might not'],
  [/\bmustn't\b/gi,  'must not'],
  [/\bit's\b/gi,     'it is'],
  [/\bthat's\b/gi,   'that is'],
  [/\bthere's\b/gi,  'there is'],
  [/\bI'm\b/g,       'I am'],
  [/\byou're\b/gi,   'you are'],
  [/\bwe're\b/gi,    'we are'],
  [/\bthey're\b/gi,  'they are'],
  [/\bI've\b/g,      'I have'],
  [/\bwe've\b/gi,    'we have'],
  [/\bI'll\b/g,      'I will'],
  [/\bwe'll\b/gi,    'we will'],
];

function normalizeForEmbedding(text) {
  let s = text;
  for (const [pattern, replacement] of CONTRACTIONS) s = s.replace(pattern, replacement);
  return s.replace(/\s{2,}/g, ' ').trim();
}

// ─── Stopwords ─────────────────────────────────────────────────────────────

const STOPWORDS = new Set([
  'a','an','the','and','or','but','if','in','on','at','to','for','of','with',
  'is','are','was','were','be','been','have','has','had','do','does','did',
  'will','would','could','should','may','might','can','cannot','not','no',
  'i','me','my','we','our','you','your','it','its','this','that','when',
  'where','how','what','which','who','so','just','too','very','also','even',
  'still','only','get','got','need','make','feel','seem','want','use','used',
  'after','before','about','from','into','by','as','up','out','all','any',
  'there','here','more','some','than','then','both','each','few','same',
  // Contraction fragments: "don't" → "don t" after apostrophe strip — "don"
  // is never a meaningful standalone content word and pollutes bigrams.
  'don','doesn','didn','isn','aren','wasn','weren','won','can',
  // Truncated phrase fragments that survive bigram extraction but are semantically incomplete.
  'something','anything','everything','nothing',
]);

/** Extract deduplicated unigram + bigram candidates, filtering stopwords. */
function extractCandidates(text) {
  const tokens = text
    .toLowerCase()
    .replace(/[^a-z\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !STOPWORDS.has(w));

  const seen = new Set();
  const out = [];

  for (let i = 0; i < tokens.length; i++) {
    if (!seen.has(tokens[i])) { seen.add(tokens[i]); out.push(tokens[i]); }
    if (i < tokens.length - 1) {
      const bi = `${tokens[i]} ${tokens[i + 1]}`;
      if (!seen.has(bi)) { seen.add(bi); out.push(bi); }
    }
  }

  return out;
}

// ─── Stage 3: TF-IDF (inline, no library) ─────────────────────────────────

/**
 * Returns a scoring function tfidf(candidate, docIdx).
 * Each cluster's pooled text is one document in a corpus of N documents.
 *
 * @param {string[]} documents - pooled text per cluster
 */
function buildTfIdf(documents) {
  const N = documents.length;

  // Pre-tokenise for TF counting
  const tokenised = documents.map(d =>
    d.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean)
  );

  function tf(candidate, docIdx) {
    const tokens = tokenised[docIdx];
    if (tokens.length === 0) return 0;
    const text = tokens.join(' ');
    let count = 0;
    let pos = 0;
    while ((pos = text.indexOf(candidate, pos)) !== -1) { count++; pos += candidate.length; }
    return count / tokens.length;
  }

  function idf(candidate) {
    let df = 0;
    for (const tokens of tokenised) {
      if (tokens.join(' ').includes(candidate)) df++;
    }
    return Math.log((N + 1) / (df + 1));
  }

  return (candidate, docIdx) => tf(candidate, docIdx) * idf(candidate);
}

// ─── Stage 3: Contrastive keyphrases ──────────────────────────────────────

/**
 * Embeds every unique candidate across all clusters exactly once.
 *
 * Naive implementation embeds per-cluster, causing ~2,500 sequential MiniLM
 * calls for a 50-note board (10 clusters × ~250 candidates each). Common
 * tokens ("team", "user", "design") appear in many clusters and would be
 * embedded repeatedly. This function deduplicates first, embeds once per
 * unique string, and returns a Map for O(1) lookup in extractKeyphrases.
 *
 * @param {string[][]} candidatesPerCluster
 * @returns {Promise<Map<string, number[]>>}
 */
async function buildCandidateEmbeddings(candidatesPerCluster) {
  const unique = [...new Set(candidatesPerCluster.flat())];
  const cache = new Map();
  for (const candidate of unique) {
    cache.set(candidate, await embedText(candidate));
  }
  return cache;
}

/**
 * Extracts top-3 keyphrases for a cluster using KeyBERT + TF-IDF.
 *
 * For clusters with ≥3 notes: scores by KeyBERT cosine similarity × TF-IDF
 * (contrastive — rewards phrases specific to this cluster over others).
 *
 * For clusters with 1–2 notes: TF-IDF is meaningless with a 1-doc corpus,
 * so score by KeyBERT cosine similarity only. Still produces keyphrases,
 * which feeds the KEYPHRASE_PROMPT_TEMPLATE and avoids the raw-text path
 * that causes FLAN-T5-small to generate sentence fragments.
 *
 * @param {string[]}              memberTexts  - note texts in this cluster
 * @param {number[]}              clusterEmbed - L2-normalised centroid
 * @param {function|null}         tfidfScorer  - pre-built scorer (null for small clusters)
 * @param {number}                docIdx       - this cluster's corpus index
 * @param {Map<string, number[]>} embedCache   - pre-built candidate embeddings
 */
function extractKeyphrases(memberTexts, clusterEmbed, tfidfScorer, docIdx, embedCache) {
  const pooled = memberTexts.join(' ');
  const candidates = extractCandidates(pooled);
  if (candidates.length === 0) return [];

  const scored = [];
  const useContrastive = tfidfScorer !== null && memberTexts.length >= 3;

  for (const candidate of candidates) {
    const vec = embedCache.get(candidate);
    if (!vec) continue;
    const keyBertScore = cosineSim(vec, clusterEmbed);

    // Within-cluster note coverage: fraction of member notes containing at
    // least one word of this candidate. Uses the max per-word coverage so that
    // a bigram like "canvas freezes" inherits "canvas"'s 2/4 coverage rather
    // than the bigram's own 1/4 exact-match coverage.
    //
    // This corrects a systematic TF-IDF failure: when a central term like
    // "canvas" appears in multiple clusters, its IDF drops and niche bigrams
    // ("scrolling stutters", unique to one cluster) outscore it despite being
    // less representative of the cluster as a whole. Coverage rewards breadth
    // within the cluster independently of cross-cluster contrastiveness.
    //
    // Scaled to [0.5, 1.0] via (0.5 + 0.5 × cov) to avoid collapsing scores
    // to near-zero for low-coverage candidates that are still semantically valid.
    const words = candidate.split(' ');
    let maxCov = 0;
    for (const word of words) {
      let hits = 0;
      for (const text of memberTexts) {
        if (text.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/).includes(word)) hits++;
      }
      maxCov = Math.max(maxCov, hits / memberTexts.length);
    }
    const coverageFactor = 0.5 + 0.5 * maxCov;

    const compositeScore = (useContrastive
      ? keyBertScore * tfidfScorer(candidate, docIdx)
      : keyBertScore) * coverageFactor;
    scored.push({ candidate, keyBertScore, compositeScore });
  }

  // Bigrams get a +0.05 composite score boost before sorting.
  // Without this, a high-scoring unigram ("canvas") blocks the more informative
  // bigram that contains it ("canvas freezes") during deduplication, and
  // FLAN-T5's prompt receives a weaker unigram as the top keyword.
  for (const s of scored) {
    if (s.candidate.includes(' ')) s.compositeScore += 0.05;
  }
  scored.sort((a, b) => b.compositeScore - a.compositeScore);

  // Keep top-3 by composite score (KeyBERT × TF-IDF + bigram boost),
  // with token-overlap deduplication. FLAN-T5-small degrades when keywords
  // share tokens — "template" and "template search" in the same list anchors
  // the model on "template" and produces filler. Keep a phrase only if none
  // of its words appear in any already-selected phrase.
  //
  // keyphrases[0] anchors FLAN-T5's prompt and is the fallback label —
  // the composite score (not pure KeyBERT) is the correct ordering criterion
  // here because TF-IDF's contrastiveness signal is what lifts cluster-specific
  // phrases ("canvas freezes") above generic ones ("scrolling stutters").
  const picked = [];
  const usedWords = new Set();
  for (const item of scored) {
    const words = item.candidate.split(' ');
    if (words.some(w => usedWords.has(w))) continue;
    picked.push(item);
    words.forEach(w => usedWords.add(w));
    if (picked.length === 3) break;
  }

  return picked.map(p => p.candidate);
}

// ─── Stage 4: Post-processing ─────────────────────────────────────────────

/**
 * Cleans and normalises a raw FLAN-T5 output into a display-ready insight label.
 *
 * Steps (order matters):
 *  1. Take only the first line — the model sometimes continues after a newline
 *  2. Strip leading non-alpha chars (e.g. "- ", ": ")
 *  3. Strip trailing punctuation
 *  4. Hard cap at 5 words — enforces the 3-5 word contract at the string level
 *  5. Sentence case: first letter uppercase, remainder lowercase
 */
function cleanLabel(raw) {
  const s = raw
    .trim()
    .split('\n')[0]
    // Strip model-added role prefixes: "Topic:", "Title:", "Subject:", "Label:",
    // "Category:", "Theme:", "Topic label:" — LaMini wraps its answer in these.
    // Must run before the leading non-alpha strip so "Topic: Canvas" → "Canvas".
    .replace(/^(?:possible (?:short )?label|(?:short )?label|common meanings?|themes?|topic label|topic|title|subject|categori[zs]ation|category|theme|important|words?|phrases?|answer|note|output|result|summary|urgent|critical|alert)\s*[:\-]\s*/i, '')
    // Strip leading non-alpha chars (quotes, dashes, asterisks, etc.)
    .replace(/^[^a-zA-Z]+/, '')
    // Strip internal commas (e.g. "file names, Embedded" → "file names Embedded")
    .replace(/,/g, '')
    // Collapse mid-label colons — "Sign-In Errors: Authentication Failures" → "Sign-In Errors Authentication Failures".
    // The prefix-strip above already handles leading "Topic: " patterns; this handles
    // any colon the model inserts mid-phrase as a separator or list marker.
    .replace(/\s*:\s*/g, ' ')
    // Strip interior special characters — "@", "#", "(", ")", etc. — that the model
    // occasionally embeds in output (e.g. "Push Notifications (@) Aren't Sent").
    // Hyphens are preserved so "two-factor" survives.
    .replace(/[^a-zA-Z0-9'\-\s]/g, ' ')
    .replace(/\s{2,}/g, ' ')           // collapse any runs of spaces left behind
    // Strip trailing punctuation and wrapping quotes the model sometimes adds
    .replace(/[.;:!?"'–—]+$/, '')
    // Strip list-separator continuations the model adds after the main phrase.
    // " - " and " / " are always separators; compound hyphens ("two-factor") have no spaces.
    // ("Activity history issues - slow search performance" → "Activity history issues")
    // ("Live tracking drops for followers / Offline mode" → "...followers")
    .replace(/\s+[-\/]\s+.*/s, '')
    .trim();

  const words = s.split(/\s+/).filter(Boolean).slice(0, 6); // 6-word cap — consistent with fallback label limit
  if (words.length === 0) return '';

  // Trim trailing function words produced by the 6-word cap landing mid-phrase.
  // "Search Results For Workout Templates And"      → pop "And"
  // "Activity Privacy Settings Are Confusing And"   → pop "And"
  // "Map Freezes Scrolling Stutters On"             → pop "On"
  // TRAILING_FUNCTION_WORDS is defined below; cleanLabel is a function declaration
  // so referencing it here is safe — it is only called after module initialisation.
  while (words.length > 1 && TRAILING_FUNCTION_WORDS.has(words[words.length - 1].toLowerCase())) {
    words.pop();
  }
  if (words.length === 0) return '';

  // Drop a trailing single-character word — it's a truncated token from hitting
  // the max_new_tokens limit mid-word (e.g. "scrolling s" where "s" started "stutters").
  if (words.length > 1 && words[words.length - 1].length === 1) {
    words.pop();
  }
  if (words.length === 0) return '';

  // Title-case every word, preserving acronyms in uppercase.
  // Heuristic: ≤4 chars that are either already all-uppercase OR all-lowercase
  // (keyphrases arrive lowercased, so "cpu" → "CPU", "pdf" → "PDF").
  // Longer words are never treated as acronyms to avoid false positives.
  const KNOWN_ACRONYMS = new Set(['sso','cpu','pdf','pdfs','api','url','ui','ux','id','ai','ml']);
  return words
    .map(w => (w.length <= 4 && (w === w.toUpperCase() || KNOWN_ACRONYMS.has(w.toLowerCase())))
      ? w.toUpperCase()
      : w[0].toUpperCase() + w.slice(1).toLowerCase())
    .join(' ');
}

// Function words that signal a truncated sentence when they appear at the end.
const TRAILING_FUNCTION_WORDS = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be',
  'and', 'or', 'but', 'of', 'for', 'in', 'on', 'to', 'with',
  'when', 'where', 'that', 'which', 'who', 'how',
  'never', 'always', 'often', 'sometimes', // adverbs that signal a truncated predicate
  'full', 'large', 'new', 'other', 'same',  // adjectives that signal a truncated noun phrase ("Full Activity...")
  'few', 'many', 'some', 'several', 'any',  // quantifiers always preceding a noun ("for a few [seconds]")
  'for', 'by', 'at', 'from', 'into',        // prepositions — always signal a truncated prepositional phrase
  'common', 'clear', 'good', 'bad', 'correct', 'wrong', 'right', // predicate adjectives ("...are common", "...is clear")
  'more', 'less', 'better', 'worse', // comparatives that always precede a noun ("...with better [controls]")
  'not', // negation that signals a truncated predicate ("...are not", "...is not")
  // Transitive verbs that always signal a missing object when they end a label
  'include', 'includes', 'contain', 'contains', 'involve', 'involves',
  'require', 'requires', 'cause', 'causes', 'affect', 'affects', 'lead', 'leads',
]);

/**
 * Returns true if the label is structurally degenerate:
 *  - Any token repeats (repetition loop)
 *  - Starts with an article or conjunction — signals a generated sentence
 *  - Starts with a modal/infinitive verb phrase ("need to", "try to") —
 *    signals a propositional statement, not a noun phrase; also catches
 *    semantic inversions where the model negates the cluster's intent
 *    ("need to avoid bulk renames" → cluster actually wants bulk rename controls)
 *  - Ends with a function word or predicate adverb — signals truncation
 *  - Contains a 3rd-person singular verb in non-final position — Subject-Verb-Object
 *    sentence pattern, not a noun phrase. Checked on interior words only so
 *    "issues" / "notes" at the END are treated as nouns (correct).
 */
// Personal pronouns — if any appear inside a label the model has quoted a note
// verbatim rather than synthesising a theme name ("Template Search Results I See...").
const PERSONAL_PRONOUNS = new Set(['i', 'we', 'my', 'our', 'me', 'you', 'your', 'they', 'their', 'he', 'she', 'his', 'her']);

function isDegenerate(label) {
  const words = label.toLowerCase().split(/\s+/);
  // Single-word labels are never useful theme names — the fallback keyphrase
  // pair is always more informative than a lone unigram.
  if (words.length < 2) return true;
  if (words.length !== new Set(words).size) return true;
  if (['a', 'an', 'the', 'i', 'we', 'it', 'although', 'despite', 'however', 'while', 'whereas', 'since', 'because', 'though',
       // Generic meta-framing words the model uses when it doesn't have a specific label
       'introduction', 'guide', 'tips', 'overview', 'summary', 'examples', 'understanding',
      ].includes(words[0])) return true;
  // Personal pronouns anywhere in the label signal the model quoted note text
  // verbatim rather than synthesising a theme ("Results I See Random'toast").
  if (words.some(w => PERSONAL_PRONOUNS.has(w))) return true;
  // Infinitive verb phrases: "need to X", "try to X", "how to X" etc.
  // These are propositional statements that can invert the cluster's meaning.
  if (words.length >= 2 && words[1] === 'to' &&
      ['need', 'needs', 'try', 'tries', 'want', 'wants', 'have', 'has', 'avoid', 'able', 'how'].includes(words[0])) return true;
  if (TRAILING_FUNCTION_WORDS.has(words[words.length - 1])) return true;
  const interior = words.slice(0, -1);
  if (interior.some(w => w.length > 5 && /(?:izes|ises|ates|utes|ects|nizes)$/.test(w))) return true;
  return false;
}

// Strips common English suffixes to a comparable root before vocabulary
// matching. Prevents morphological false positives where "stuttering" in a
// label is rejected because notes contain "stutters" — same word, different
// inflection. Not a full stemmer (Porter/Snowball) — only the suffixes
// FLAN-T5-small commonly adds during generation.
function stem(w) {
  // Two-pass design: first strip plural/verb suffix so compound suffixes
  // ("-tions" → "-tion" → base) collapse to the same root as their singular
  // form. Without the second /tion$/ pass, stem("notifications") → "notification"
  // while stem("notification") → "notificat" — same word, different roots,
  // causing false vocab-foreign rejections.
  return w
    .replace(/ings?$/, '')   // running → run, stutterings → stutter
    .replace(/tion$/, '')    // authentication → authenticat
    .replace(/ed$/, '')      // freezed → freez
    .replace(/er$/, '')      // faster → fast
    .replace(/ly$/, '')      // quickly → quick
    .replace(/s$/, '')       // notifications → notification, stutters → stutter
    .replace(/tion$/, '');   // second pass: notification → notificat
}

/**
 * Returns true if the label contains words that are foreign to the cluster.
 *
 * FLAN-T5-small hallucinations weave cluster vocabulary into otherwise
 * nonsensical phrases ("Adolescent stutters and blurry tumbling"). The
 * cluster words make the embedding similar enough to the centroid to pass a
 * cosine threshold, so cosine alone cannot catch this.
 *
 * A vocabulary check is the correct gate: every non-stopword content word in
 * the label must exist (after stemming) somewhere in the member note texts.
 * Stemming resolves morphological variants ("stuttering" → "stutter" matches
 * "stutters" → "stutter") while still catching true hallucinations like
 * "adolescent" which has no root in any note text.
 */
function isVocabForeign(label, memberTexts) {
  const noteVocab = new Set(
    memberTexts.join(' ')
      .toLowerCase()
      .replace(/[^a-z\s]/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 1 && !STOPWORDS.has(w))
      .map(stem)
  );

  const labelWords = label
    .toLowerCase()
    .replace(/[^a-z\s]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 1 && !STOPWORDS.has(w));

  // Check both the stemmed form and stem+'e' to handle CVC words where removing
  // '-ing' leaves a consonant cluster (e.g. "losing" → "los"; "loses" → "lose";
  // neither equals the other without the +e fallback). This is the CVC rule from
  // the Porter stemmer, applied only at the lookup step — no rewrite of the stem.
  const foreignCount = labelWords.filter(w => {
    const s = stem(w);
    return !noteVocab.has(s) && !noteVocab.has(s + 'e');
  }).length;
  // Allow exactly one foreign word if it is a recognised bridging noun —
  // words the model correctly adds to connect grounded keyphrases even though
  // they don't appear literally in the notes ("Map issues with scrolling").
  // Any other single foreign word (e.g. "censorship", "adolescent") is a
  // hallucination and must be rejected.  Two or more foreign words always fail.
  if (foreignCount === 0) return false;
  if (foreignCount >= 2) return true;

  // foreignCount === 1: accept only if the foreign word is a bridging noun
  const BRIDGING_NOUNS = new Set([
    'issue', 'issues', 'problem', 'problems', 'error', 'errors',
    'bug', 'bugs', 'challenge', 'challenges', 'difficulty', 'difficulties',
    'concern', 'concerns', 'limitation', 'limitations', 'improvement', 'improvements',
    'behavior', 'behaviour', 'performance', 'functionality', 'experience', 'experiences',
    'barrier', 'barriers', 'friction', 'gap', 'gaps',
    'failure', 'failures', 'confusion', 'frustration', 'complaint', 'complaints',
    // Problem-descriptive adjectives the model synthesises correctly but that
    // may not appear verbatim in short note text (e.g. "unclear" when notes say
    // "confusing" or "hard to find"). Relevance gate (≥0.40) already filters
    // hallucinations; allowing one such adjective is low-risk.
    'unclear', 'confusing', 'difficult', 'unreliable', 'incorrect',
    'unexpected', 'unresponsive', 'inconsistent', 'unavailable', 'missing', 'unable',
    // Positive-improvement adjectives the model synthesises when the cluster
    // is about requesting or lacking a feature. Notes say "better" / "need";
    // the model outputs "improved" / "enhanced" — same intent, different word.
    'improved', 'enhanced', 'proper', 'efficient', 'effective',
  ]);
  const foreignWord = labelWords.find(w => {
    const s = stem(w);
    return !noteVocab.has(s) && !noteVocab.has(s + 'e');
  });
  return !BRIDGING_NOUNS.has(foreignWord?.toLowerCase());
}

// ─── Stage 4: LaMini-Flan-T5-248M label generation ────────────────────────
//
// LaMini-Flan-T5-248M is an encoder-decoder model built on FLAN-T5-base.
// Despite LaMini's Alpaca fine-tuning, the Xenova ONNX export follows the
// underlying FLAN-T5 prefix-prompt convention — "### Instruction:" headers
// cause the model to treat the content as a meta-prompt and emit refusals.
// Direct natural-language instructions work correctly.
// FLAN-T5-small (80M) was tested and confirmed unusable: outputs incoherent
// hallucinations ("i am having nightmares", "a template for slambling") with
// 0/12 valid labels — below the ~250M instruction-following threshold.
//
// Note texts are included in the prompt as context alongside keyphrases.
// Without them the model hallucinates on ambiguous keyphrases ("drops syncs"
// → "Changes in Routine"). The note texts ground the generation in the
// cluster's actual vocabulary, which then also satisfies the vocab gate.
//
// Notes are truncated at 300 chars total to stay within T5's 512-token context
// window. Up to 3 notes are sampled (first, middle, last) for coverage.
//
// Decoding: do_sample=true, temperature=0.7 — stochastic so retries draw
// different samples. max_new_tokens=15 caps output at ~5 words.

/**
 * Build a prompt that grounds the label generation in actual note vocabulary.
 * @param {string[]} keyphrases
 * @param {string[]} memberTexts
 */
function buildPrompt(keyphrases, memberTexts) {
  // Sample up to 3 notes (first, middle, last) to stay within token budget.
  // Total snippet capped at 300 chars — well within T5's 512-token context
  // window even after adding the Alpaca wrapper (~40 tokens overhead).
  const sample = memberTexts.length <= 3
    ? memberTexts
    : [memberTexts[0], memberTexts[Math.floor(memberTexts.length / 2)], memberTexts[memberTexts.length - 1]];

  const noteSnippet = sample
    .map(t => t.trim().slice(0, 80))
    .join(' | ')
    .slice(0, 300);

  return `Given these shared themes, generate a short label that captures the common meaning:

Notes: ${noteSnippet}
Keywords: ${keyphrases.join(', ')}
Label:`;
}

const TEXT_PROMPT_TEMPLATE = (text) =>
  `Given these shared themes, generate a short label that captures the common meaning: ${text}`;

/**
 * Generates an insight label via FLAN-T5-small with greedy decoding.
 *
 * Validation gates (applied in order):
 *  1. Structural: non-empty, non-degenerate (no repeated tokens)
 *  2. Semantic relevance: cosine sim to cluster centroid ≥ RELEVANCE_THRESHOLD
 *  3. Vocab: all content words present in member note texts (foreignCount > 0)
 *
 * Returns labelVec alongside label so the caller can run the cross-cluster
 * uniqueness gate without re-embedding.
 *
 * @param {string[]} keyphrases    - top-3 contrastive keyphrases (may be empty)
 * @param {string[]} memberTexts   - raw note texts
 * @param {number[]} clusterEmbed  - L2-normalised cluster centroid
 * @returns {Promise<{label: string, labelVec: number[], fallback_used: boolean}>}
 */
const RELEVANCE_THRESHOLD = 0.40;
const MAX_LABEL_ATTEMPTS = 2; // retry-on-failure: only failing clusters pay extra latency

async function generateLabel(keyphrases, memberTexts, clusterEmbed) {
  const prompt = keyphrases.length > 0
    ? buildPrompt(keyphrases, memberTexts)
    : TEXT_PROMPT_TEMPLATE(memberTexts.join(' ').slice(0, 80));

  const kw_str = keyphrases.join(', ') || '(none)';
  const t5 = await getFlanT5();

  for (let attempt = 1; attempt <= MAX_LABEL_ATTEMPTS; attempt++) {
    let raw = '';

    try {
      const result = await t5(prompt, {
        do_sample: true,
        temperature: 0.7,      // stochastic: each retry draws a different sample
        max_new_tokens: 15,    // raised from 10: prefix tokens consume budget before the label
        repetition_penalty: 1.2,
        no_repeat_ngram_size: 2,
      });
      raw = result[0]?.generated_text ?? '';
      console.log(`[t5] attempt=${attempt} keyphrases=[${kw_str}] → raw="${raw}"`);
    } catch (err) {
      console.warn(`[t5] inference error attempt=${attempt} keyphrases=[${kw_str}]:`, err);
      break; // inference failure is not retryable
    }

    if (!raw) continue;

    const label = cleanLabel(raw);
    if (label.length === 0) {
      console.log(`[t5] attempt=${attempt} REJECT → cleanLabel empty`);
      continue;
    }
    if (isDegenerate(label)) {
      console.log(`[t5] attempt=${attempt} REJECT "${label}" → isDegenerate`);
      continue;
    }
    // Reject if the model echoed back a keyphrase verbatim — it means the model
    // didn't synthesise anything; it just repeated its own input. A label that
    // is identical to a keyphrase adds no information over the fallback.
    if (keyphrases.some(kp => cleanLabel(kp).toLowerCase() === label.toLowerCase())) {
      console.log(`[t5] attempt=${attempt} REJECT "${label}" → echo of keyphrase`);
      continue;
    }

    const labelVec = await embedText(label);
    const relevance = cosineSim(labelVec, clusterEmbed);
    const foreign = isVocabForeign(label, memberTexts);
    console.log(`[t5] attempt=${attempt} label="${label}" relevance=${relevance.toFixed(3)} foreign=${foreign}`);

    if (relevance < RELEVANCE_THRESHOLD) {
      console.log(`[t5] attempt=${attempt} REJECT "${label}" → relevance ${relevance.toFixed(3)} < ${RELEVANCE_THRESHOLD}`);
      continue;
    }
    if (foreign) {
      console.log(`[t5] attempt=${attempt} REJECT "${label}" → vocab foreign`);
      continue;
    }

    // All gates passed
    console.log(`[t5] ACCEPT "${label}" on attempt ${attempt}`);
    return { label, labelVec, fallback_used: false };
  }

  // All attempts exhausted — build a deterministic label from the top-2
  // keyphrases. Both are guaranteed vocabulary-grounded (Stage 3 sources them
  // directly from note text) and non-overlapping (Stage 3 deduplication).
  // "{kp1} and {kp2}" is always more informative than kp1 alone:
  //   keyphrases[0] = "map freezes"  → "Map freezes"
  //   + keyphrases[1] = "scrolling stutters" → "Map freezes and scrolling stutters"
  // If only one keyphrase exists, fall back to it alone. If none, use the
  // first note's opening text. "Insight" is the last-resort sentinel.
  console.log(`[t5] all ${MAX_LABEL_ATTEMPTS} attempts failed for [${keyphrases.join(', ')}] → using keyphrase fallback`);

  // Pick the first two non-overlapping keyphrases for the fallback label.
  // Stage 3 deduplication prevents token overlap between phrases at selection
  // time, but unigrams can still share a root with a later bigram (e.g.
  // "export" and "board exports" both stem "export"). Re-check here so the
  // fallback label never reads "Export and Board Exports".
  const kpTokens = keyphrases.map(kp =>
    new Set(kp.toLowerCase().replace(/[^a-z\s]/g, ' ').split(/\s+/).filter(Boolean).map(stem))
  );

  let kp1 = '', kp2 = '';
  for (let i = 0; i < keyphrases.length && (!kp1 || !kp2); i++) {
    const label = cleanLabel(keyphrases[i]);
    if (!label) continue;
    if (!kp1) { kp1 = label; continue; }
    // Skip kp2 candidate if any of its stems overlap with kp1's stems
    const overlap = [...kpTokens[i]].some(t => kpTokens[0].has(t));
    if (!overlap) kp2 = label;
  }

  let fallbackLabel;
  if (kp1 && kp2) {
    fallbackLabel = `${kp1} and ${kp2}`;
  } else if (kp1 && kp1.split(/\s+/).length === 1) {
    // kp1 is a single generic word (e.g. "Export") — all keyphrases share the
    // same stem so kp2 was never found. Use first member note text instead;
    // it's always grounded and more informative than a lone unigram.
    fallbackLabel = cleanLabel(memberTexts[0]?.slice(0, 40) ?? '') || kp1;
  } else {
    fallbackLabel = kp1 || cleanLabel(memberTexts[0]?.slice(0, 40) ?? '');
  }
  if (!fallbackLabel) fallbackLabel = 'Insight';

  // Hard word cap — the two-keyphrase template can exceed 6 words.
  // Allow up to 6 words so "Map freezes and scrolling stutters" isn't truncated.
  const words = fallbackLabel.split(/\s+/);
  if (words.length > 6) fallbackLabel = words.slice(0, 6).join(' ');

  // Strip trailing function words left by the 6-word cap on memberText slices.
  // The fallback bypasses isDegenerate, so we clean the tail explicitly.
  // ("Exported GPX Files Are Inconsistent And" → "Exported GPX Files Are Inconsistent")
  {
    const fw = fallbackLabel.split(/\s+/);
    while (fw.length > 1 && TRAILING_FUNCTION_WORDS.has(fw[fw.length - 1].toLowerCase())) fw.pop();
    fallbackLabel = fw.join(' ');
  }

  console.log(`[t5] fallback label="${fallbackLabel}"`);
  const fallbackVec = await embedText(fallbackLabel);
  return { label: fallbackLabel, labelVec: fallbackVec, fallback_used: true };
}

// ─── Silhouette score (cosine distance) ───────────────────────────────────
//
// Correct metric for L2-normalised MiniLM vectors: cosine distance = 1 − dot.
// Using Euclidean distance in silhouette on the unit sphere produces the same
// ranking (d²_euclidean = 2·cosine_distance), but cosine is more interpretable
// and avoids the sqrt overhead in the O(n²) loop.
//
// Returns mean silhouette ∈ [-1, 1]. Higher = better separated clusters.

function silhouette(embeddings, assignments, k) {
  const n = embeddings.length;
  if (k <= 1 || n <= k) return 0;

  const groups = Array.from({ length: k }, () => []);
  assignments.forEach((c, i) => groups[c].push(i));

  let total = 0;
  let counted = 0;

  for (let i = 0; i < n; i++) {
    const ci = assignments[i];
    const same = groups[ci];

    // Singleton clusters are excluded from the per-point mean (a(i)=0 makes
    // s(i)=1 unconditionally, which would inflate high-k scores).
    // Instead, the singleton cost is applied at the return site by scaling
    // the mean by the fraction of non-singleton points. This assigns
    // effective s=0 to singletons without distorting the non-singleton mean.
    if (same.length === 1) continue;

    // a(i): mean cosine distance to all other members of the same cluster
    let a = 0;
    for (const j of same) if (j !== i) a += cosineDistance(embeddings[i], embeddings[j]);
    a /= same.length - 1;

    // b(i): mean cosine distance to the nearest other cluster
    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci || groups[c].length === 0) continue;
      let mean = 0;
      for (const j of groups[c]) mean += cosineDistance(embeddings[i], embeddings[j]);
      mean /= groups[c].length;
      if (mean < b) b = mean;
    }

    // Guard: b stays Infinity if all other clusters were empty (unreachable in
    // practice but NaN silhouette would poison the winner comparison silently.
    if (!isFinite(b)) continue;

    total += (b - a) / Math.max(a, b);
    counted++;
  }

  return counted === 0 ? 0 : total / counted;
}

/**
 * Per-note silhouette — same math as silhouette() but returns one score
 * per note instead of per cluster or the overall mean.  Use this to surface
 * individual notes that the algorithm placed ambiguously.
 *
 * @param {number[][]} embeddings
 * @param {number[]} assignments
 * @param {number} k
 * @returns {number[]} per-note silhouette, index = note index
 */
export function silhouettePerNote(embeddings, assignments, k) {
  const n = embeddings.length;
  if (k <= 1 || n <= k) return new Array(n).fill(0);

  const groups = Array.from({ length: k }, () => []);
  assignments.forEach((c, i) => groups[c].push(i));

  const scores = new Array(n).fill(0);

  for (let i = 0; i < n; i++) {
    const ci = assignments[i];
    const same = groups[ci];
    if (same.length === 1) continue;

    let a = 0;
    for (const j of same) if (j !== i) a += cosineDistance(embeddings[i], embeddings[j]);
    a /= same.length - 1;

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci || groups[c].length === 0) continue;
      let mean = 0;
      for (const j of groups[c]) mean += cosineDistance(embeddings[i], embeddings[j]);
      mean /= groups[c].length;
      if (mean < b) b = mean;
    }

    if (!isFinite(b)) continue;
    scores[i] = (b - a) / Math.max(a, b);
  }

  return scores;
}

/**
 * Per-cluster silhouette — same math as silhouette() but returns one score
 * per cluster instead of the overall mean.  Use this to diagnose which
 * clusters are poorly separated without recomputing distances.
 *
 * Interpretation:
 *   s > 0.50  well-separated, notes belong clearly to this cluster
 *   0.25–0.50 reasonable structure, some overlap with neighbors
 *   0.00–0.25 weak structure, cluster boundary is fuzzy
 *   s < 0.00  notes are closer to a neighboring cluster — may be mis-assigned
 *
 * @param {number[][]} embeddings
 * @param {number[]} assignments
 * @param {number} k
 * @returns {number[]} per-cluster mean silhouette, index = cluster id
 */
function silhouettePerCluster(embeddings, assignments, k) {
  const n = embeddings.length;
  if (k <= 1 || n <= k) return new Array(k).fill(0);

  const groups = Array.from({ length: k }, () => []);
  assignments.forEach((c, i) => groups[c].push(i));

  const clusterTotals  = new Array(k).fill(0);
  const clusterCounted = new Array(k).fill(0);

  for (let i = 0; i < n; i++) {
    const ci = assignments[i];
    const same = groups[ci];
    if (same.length === 1) continue;

    let a = 0;
    for (const j of same) if (j !== i) a += cosineDistance(embeddings[i], embeddings[j]);
    a /= same.length - 1;

    let b = Infinity;
    for (let c = 0; c < k; c++) {
      if (c === ci || groups[c].length === 0) continue;
      let mean = 0;
      for (const j of groups[c]) mean += cosineDistance(embeddings[i], embeddings[j]);
      mean /= groups[c].length;
      if (mean < b) b = mean;
    }
    if (!isFinite(b)) continue;

    clusterTotals[ci]  += (b - a) / Math.max(a, b);
    clusterCounted[ci] += 1;
  }

  return clusterTotals.map((t, ci) => (clusterCounted[ci] === 0 ? 0 : t / clusterCounted[ci]));
}

// ─── Algorithm A: Spherical k-means with silhouette k-selection ───────────
//
// ml-kmeans on L2-normalised vectors approximates spherical k-means because
// on the unit hypersphere: d²_euclidean = 2(1 − cosθ), so minimising
// Euclidean inertia is equivalent to maximising cosine similarity. Centroid
// drift off the sphere is negligible at n < 200 and doesn't affect assignment.
//
// k is selected by maximising mean silhouette over k = K_MIN..K_MAX.
// Silhouette measures separation-to-cohesion ratio — the correct criterion
// when no ground truth labels exist.

const K_MIN = 2;
const K_MAX = 16  ;  // raised from 18; PCA sharpens distances enough that agglomerative resolves finer structure
const KMEANS_RUNS = 30;  // multiple inits to escape bad local minima

// Agglomerative is O(n³) in merge steps and O(n²) in memory.
// Above this threshold, skip it and run spherical k-means only.
// 150 notes → ~3.4M distance pairs, ~10s on a mid-range laptop.
// 300 notes → ~27M pairs, would freeze the main thread for >60s.
const AGGLOMERATIVE_MAX_N = 150;

// Deterministic PRNG (LCG, Park-Miller parameters) used to seed Math.random
// before each k-means call. Without seeding, kmeans++ draws from the live
// Math.random stream, so cluster membership changes on every page interaction.
// We derive a unique seed per (k, run) pair so each run explores a different
// initialisation while the full sequence is reproducible across page loads.
function seededRandom(seed) {
  let s = seed >>> 0;
  return () => {
    s = Math.imul(s, 1664525) + 1013904223 >>> 0;
    return s / 0x100000000;
  };
}

function bestKMeans(embeddings, k) {
  let best = null;
  for (let run = 0; run < KMEANS_RUNS; run++) {
    // Inject deterministic random for this (k, run) pair, then restore.
    const rng = seededRandom(0xdeadbeef ^ (k * 2654435761) ^ run);
    const origRandom = Math.random;
    Math.random = rng;
    let result;
    try {
      result = kmeans(embeddings, k, { maxIterations: 150, initialization: 'kmeans++' });
    } finally {
      Math.random = origRandom;
    }
    // Inertia: used only to pick the best run, not for k-selection
    const inertia = embeddings.reduce(
      (s, v, i) => s + cosineDistance(v, result.centroids[result.clusters[i]]), 0
    );
    if (!best || inertia < best.inertia) best = { result, inertia };
  }
  return best.result;
}

/**
 * Merges singleton clusters (size = 1) into their nearest non-singleton cluster
 * by centroid cosine distance.  Singletons form when outlier notes have high
 * cosine distance to all others — keeping them as k=1 clusters produces s=0,
 * confuses label generation, and wastes a palette slot.  Nearest-centroid
 * assignment is the correct fix: it mirrors what k-means would do if the
 * singleton were initialised as a separate cluster.
 *
 * Cluster IDs are compacted after merging so the returned array is a valid
 * 0..k-1 assignment for the new (lower) k.
 *
 * @param {number[][]} embeddings
 * @param {number[]}   assignments  flat cluster assignment per note
 * @param {number}     k
 * @returns {{ assignments: number[], k: number }}
 */
function mergeSingletons(embeddings, assignments, k) {
  const groups = Array.from({ length: k }, () => []);
  assignments.forEach((c, i) => groups[c].push(i));

  const singletonCIs = groups.map((g, ci) => (g.length === 1 ? ci : -1)).filter(ci => ci >= 0);
  if (singletonCIs.length === 0) return { assignments, k };

  // Pre-compute centroids for non-singleton clusters only
  const centroids = new Map();
  groups.forEach((members, ci) => {
    if (members.length <= 1) return;
    const dim = embeddings[members[0]].length;
    const c = new Array(dim).fill(0);
    for (const i of members) for (let d = 0; d < dim; d++) c[d] += embeddings[i][d];
    for (let d = 0; d < dim; d++) c[d] /= members.length;
    centroids.set(ci, c);
  });

  const newAssignments = assignments.slice();
  for (const ci of singletonCIs) {
    const noteIdx = groups[ci][0];
    let nearest = -1, nearestDist = Infinity;
    for (const [nci, centroid] of centroids) {
      const d = cosineDistance(embeddings[noteIdx], centroid);
      if (d < nearestDist) { nearestDist = d; nearest = nci; }
    }
    if (nearest >= 0) newAssignments[noteIdx] = nearest;
  }

  // Compact IDs: removes gaps left by dissolved singleton clusters
  const usedIds = [...new Set(newAssignments)].sort((a, b) => a - b);
  const remap = new Map(usedIds.map((id, idx) => [id, idx]));
  return { assignments: newAssignments.map(a => remap.get(a)), k: usedIds.length };
}

// ─── PCA dimensionality reduction ──────────────────────────────────────────
// pcaProject is defined in pca.js (shared with semantic-view.js).
// Clustering uses the default normalize:true path — L2-normalised output
// preserves cosine distance as the correct metric for silhouette and k-means.
// The n > 1 guard (inside pcaProject) handles the degenerate single-note case.

const PCA_TARGET_DIM = 28;

function runSphericalKMeans(embeddings, kMax = K_MAX) {
  const maxK = Math.min(kMax, Math.floor(embeddings.length / 2));
  if (maxK < K_MIN) {
    const result = bestKMeans(embeddings, K_MIN);
    return { assignments: result.clusters, k: K_MIN, score: silhouette(embeddings, result.clusters, K_MIN) };
  }

  let best = null;
  for (let k = K_MIN; k <= maxK; k++) {
    const result = bestKMeans(embeddings, k);
    const score = silhouette(embeddings, result.clusters, k);
    if (!best || score > best.score) best = { assignments: result.clusters, k, score };
  }
  console.log(`[pipeline] spherical-kmeans  k=${best.k} silhouette=${best.score.toFixed(4)}`);
  return best;
}

// ─── Algorithm B: Agglomerative clustering, average linkage ───────────────
//
// Average linkage (UPGMA) defines inter-cluster distance as the mean pairwise
// cosine distance across all cross-cluster pairs. This is the correct linkage
// for approximately spherical clusters — single linkage chains, complete
// linkage over-splits, average linkage balances both.
//
// Implementation: O(n²) space for the distance matrix (precomputed once),
// O(n³) merges — acceptable for n < 200. No external dependency needed.
//
// k is selected by cutting the dendrogram at the k that maximises silhouette,
// same criterion as the k-means path.

function precomputeDistances(embeddings) {
  const n = embeddings.length;
  // Flat upper-triangle matrix, index (i,j) i<j → i*n - i*(i+1)/2 + j - i - 1
  const dist = new Float32Array((n * (n - 1)) / 2);
  let idx = 0;
  for (let i = 0; i < n - 1; i++) {
    for (let j = i + 1; j < n; j++) {
      dist[idx++] = cosineDistance(embeddings[i], embeddings[j]);
    }
  }
  return dist;
}

function getDist(dist, n, i, j) {
  if (i === j) return 0;
  if (i > j) [i, j] = [j, i];
  return dist[i * n - (i * (i + 1)) / 2 + j - i - 1];
}

function runAgglomerative(embeddings, kMax = K_MAX) {
  const n = embeddings.length;
  const dist = precomputeDistances(embeddings);

  // Each point starts as its own cluster; track membership as a flat array
  // for fast silhouette evaluation at each cut.
  let clusters = Array.from({ length: n }, (_, i) => new Set([i]));

  // Store the merge sequence so we can replay any cut
  const merges = []; // { a, b, distance } at each step

  while (clusters.length > 1) {
    // Find the closest pair of clusters by average linkage
    let minDist = Infinity, mergeA = 0, mergeB = 1;
    for (let a = 0; a < clusters.length - 1; a++) {
      for (let b = a + 1; b < clusters.length; b++) {
        let sum = 0;
        for (const i of clusters[a]) {
          for (const j of clusters[b]) sum += getDist(dist, n, i, j);
        }
        const avg = sum / (clusters[a].size * clusters[b].size);
        if (avg < minDist) { minDist = avg; mergeA = a; mergeB = b; }
      }
    }

    // Record merge and collapse
    merges.push({ a: mergeA, b: mergeB, distance: minDist, snapshot: null });
    clusters[mergeA] = new Set([...clusters[mergeA], ...clusters[mergeB]]);
    clusters.splice(mergeB, 1);

    // Snapshot the assignment array at this cut (k = clusters.length)
    if (clusters.length >= K_MIN && clusters.length <= kMax) {
      const assignments = new Array(n);
      clusters.forEach((members, ci) => members.forEach(i => { assignments[i] = ci; }));
      merges[merges.length - 1].snapshot = { assignments: assignments.slice(), k: clusters.length };
    }
  }

  // Evaluate silhouette for every captured snapshot, pick the best cut.
  let best = null;
  for (const { snapshot } of merges) {
    if (!snapshot) continue;
    const score = silhouette(embeddings, snapshot.assignments, snapshot.k);
    if (!best || score > best.score) best = { ...snapshot, score };
  }

  // Fallback: if n is too small to capture any snapshot, put everything in k=2
  if (!best) {
    const assignments = new Array(n).fill(0);
    for (let i = Math.floor(n / 2); i < n; i++) assignments[i] = 1;
    best = { assignments, k: 2, score: silhouette(embeddings, assignments, 2) };
  }

  console.log(`[pipeline] agglomerative     k=${best.k} silhouette=${best.score.toFixed(4)}`);
  return best;
}

// ─── Main export ───────────────────────────────────────────────────────────

/**
 * Runs the full insight discovery pipeline on a set of sticky notes.
 *
 * All inference runs client-side via WebAssembly — no note text leaves the tab.
 *
 * @param {{ id: string; text: string }[]} notes
 * @param {{ onProgress?: (label: string, pct: number) => void }} [options]
 * @returns {Promise<{
 *   results: Array<{
 *     cluster_id:       number,
 *     label:            string,
 *     note_ids:         string[],
 *     keyphrases:       string[],
 *     silhouette_score: number,
 *     fallback_used:    boolean,
 *   }>,
 *   embeddingsReduced: number[][],
 * }>}
 */
/** Yield a macrotask so the browser can flush a render frame before continuing. */
const frame = () => new Promise(resolve => setTimeout(resolve, 0));

export async function clusterNotes(notes, options = {}) {
  const { onProgress } = options;
  // report() sets the progress state then yields so the browser can repaint.
  // Without the yield, WASM holds the main thread continuously and the bar
  // never updates until the entire pipeline finishes.
  const report = async (label, pct) => { onProgress?.(label, pct); await frame(); };

  // ── Warm up models ─────────────────────────────────────────────────────
  const alreadyCached = !!_miniLM && !!_flanT5;

  if (!alreadyCached) {
    await report('Warming up...', 0);
    // Load both models upfront so Stage 4 doesn't stall after clustering.
    // Progress split is proportional to model weight on disk:
    // MiniLM ~22MB → 3 points (0–3%), FLAN-T5 ~80MB → 12 points (3–15%).
    // Note: @xenova/transformers emits status='download' (not 'downloading').
    await getMiniLM((info) => {
      if (info.status === 'download' && info.total > 0)
        report('Warming up...', Math.round((info.loaded / info.total) * 3));
    });
    await getFlanT5((info) => {
      if (info.status === 'download' && info.total > 0)
        report('Warming up...', 3 + Math.round((info.loaded / info.total) * 12));
    });
    await report('Warming up...', 15);
    await report('Reading your notes...', 20);
  } else {
    await report('Reading your notes...', 0);
  }

  // ── Stage 1: Embed notes ───────────────────────────────────────────────
  const extractor = await getMiniLM();
  const embeddings = [];

  for (let ni = 0; ni < notes.length; ni++) {
    // Prefix each note with a domain signal before embedding.
    // all-MiniLM-L6-v2 was trained on web text; product feedback notes are
    // short, informal, and fragmented — far from the training distribution.
    // The prefix shifts the embedding toward the feedback/complaint region of
    // the model's representation space, improving inter-cluster separation
    // (higher silhouette) without changing any downstream API or model weights.
    // Only Stage 1 embeddings use the prefix — keyphrase candidates (Stage 3)
    // and label embeddings (Stage 4) are short noun phrases and do not benefit
    // from it. Centroid computation averages prefixed embeddings so centroids
    // are consistent with the note vectors they represent.
    const prefixed = `User feedback: ${normalizeForEmbedding(notes[ni].text)}`;
    const out = await extractor(prefixed, { pooling: 'mean', normalize: true });
    embeddings.push(Array.from(out.data));
    // Report per-note progress so the bar moves continuously from 20% → 40%.
    const pct = 20 + Math.round((ni + 1) / notes.length * 20);
    await report('Reading your notes...', pct);
  }

  await report('Reading your notes...', 40);

  // ── Stage 2: Dual-algorithm clustering, winner by silhouette ─────────────
  await report('Finding connections...', 45);

  // PCA projection: cluster in reduced space (30d) instead of full 384d.
  // Original embeddings are kept for Stage 3 keyphrase scoring and Stage 4
  // label relevance — semantic distances there are computed per-note or per-
  // keyphrase against cluster centroids, not across the full n×n matrix, so
  // dimensionality is not the bottleneck. Only inter-note distances (used by
  // silhouette and clustering) benefit from the reduced representation.
  // n > 1 guard is inside pcaProject — returns embeddings unchanged for n ≤ 1.
  // Previously guarded at n >= 5; now always runs so embeddingsReduced is
  // available for the semantic view even on the n < 5 early-return path.
  const embeddingsReduced = notes.length > 1 ? pcaProject(embeddings, PCA_TARGET_DIM) : embeddings;

  const results = [];

  // Special case: < 5 notes → label each individually, skip clustering
  if (notes.length < 5) {
    await report('Finding connections...', 55);
    await report('Identifying themes...', 60);
    await report('Identifying themes...', 75);
    await report('Naming your insights...', 80);

    for (let i = 0; i < notes.length; i++) {
      // Pass the note's own embedding as the cluster centroid — it is the best
      // possible centroid for a 1-note cluster and satisfies generateLabel's
      // relevance gate without a divide-by-zero or undefined dereference.
      const { label, fallback_used } = await generateLabel([], [notes[i].text], embeddings[i]);
      results.push({
        cluster_id: i,
        label: label || notes[i].text.slice(0, 30) || 'Note',
        note_ids: [notes[i].id],
        keyphrases: [],
        silhouette_score: 0, // single-note clusters have no meaningful cohesion score
        fallback_used,
      });
    }

    await report('Naming your insights...', 100);
    // Small-board path: each note is its own cluster, silhouette is undefined
    return {
      results,
      embeddingsReduced,
      perNoteSilhouette: new Array(notes.length).fill(0),
      centroids: embeddings,
    };
  }

  // For small boards (n ≤ AGGLOMERATIVE_MAX_N): run both algorithms and pick
  // the winner by silhouette score. Agglomerative is O(n³) so it is skipped
  // above the threshold — a 300-note board would need ~27M distance pairs and
  // freeze the main thread for over a minute.
  // Dynamic k ceiling: floor(n/3) scales with board size and prevents the
  // cosine-on-sphere silhouette bias from pushing k to K_MAX unconditionally.
  // After PCA + L2-normalisation all notes live on a (dim-1)-sphere where
  // smaller clusters are geometrically "tighter" regardless of semantic content,
  // so silhouette is monotonically non-decreasing with k. Capping at n/3
  // bounds the effective cluster count to a range that remains interpretable
  // (avg ≥ 3 notes/cluster before singleton merging).
  const effectiveKMax = Math.max(K_MIN + 1, Math.min(K_MAX, Math.floor(notes.length / 3)));
  console.log(`[pipeline] effectiveKMax=${effectiveKMax} (n=${notes.length})`);

  const kmResult = runSphericalKMeans(embeddingsReduced, effectiveKMax);
  let winner;

  if (notes.length <= AGGLOMERATIVE_MAX_N) {
    const aggResult = runAgglomerative(embeddingsReduced, effectiveKMax);
    winner = aggResult.score >= kmResult.score ? aggResult : kmResult;
    console.log(`[pipeline] winner=${aggResult.score >= kmResult.score ? 'agglomerative' : 'spherical-kmeans'} score=${winner.score.toFixed(4)}`);
  } else {
    winner = kmResult;
    console.log(`[pipeline] n=${notes.length} > ${AGGLOMERATIVE_MAX_N} → spherical-kmeans only, score=${winner.score.toFixed(4)}`);
  }

  // Singleton post-processing: notes that the algorithm couldn't place cleanly
  // are assigned to their nearest cluster centroid. Keeps them as singletons
  // wastes a palette slot, produces s=0, and confuses label generation.
  const beforeMerge = winner.k;
  const merged = mergeSingletons(embeddingsReduced, winner.assignments, winner.k);
  if (merged.k < beforeMerge) {
    console.log(`[pipeline] merged ${beforeMerge - merged.k} singleton(s) into nearest cluster → k=${merged.k}`);
    winner = { ...winner, assignments: merged.assignments, k: merged.k };
  }

  // Per-cluster silhouette diagnostic — tells you which clusters are poorly
  // separated without requiring a second pass over the data.
  const perCluster = silhouettePerCluster(embeddingsReduced, winner.assignments, winner.k);
  // Per-note silhouette — used by the reactive agent to identify outlier notes
  // whose placement was ambiguous (s < 0.05 signals low algorithmic confidence).
  const perNoteSilhouette = silhouettePerNote(embeddingsReduced, winner.assignments, winner.k);
  console.log('[pipeline] per-cluster silhouette (id: score, size):',
    perCluster.map((s, ci) => {
      const size = winner.assignments.filter(a => a === ci).length;
      return `c${ci}=${s.toFixed(3)}(n=${size})`;
    }).join('  ')
  );

  // Convert flat assignment array → array-of-index-arrays for downstream stages
  const clusterArrays = Array.from({ length: winner.k }, () => []);
  winner.assignments.forEach((ci, i) => clusterArrays[ci].push(i));

  await report('Finding connections...', 55);

  // ── Stage 3: Contrastive keyphrases ───────────────────────────────────
  await report('Identifying themes...', 60);

  // Build TF-IDF corpus — one pooled-text document per real cluster
  const allPooledTexts = clusterArrays.map(indices =>
    indices.map(i => notes[i].text).join(' ')
  );

  // Pre-extract candidates for all clusters, then embed each unique string once.
  // This avoids repeated MiniLM calls for tokens that appear across multiple clusters.
  // Extract candidates for all clusters — including 1–2 note clusters.
  // Small clusters skip TF-IDF but still use KeyBERT, feeding the keyphrase
  // prompt path instead of falling back to raw text in Stage 4.
  const candidatesPerCluster = clusterArrays.map(indices => {
    const memberTexts = indices.map(i => notes[i].text);
    return extractCandidates(memberTexts.join(' '));
  });
  const embedCache = await buildCandidateEmbeddings(candidatesPerCluster);

  // Build TF-IDF scorer once for the whole corpus — not once per cluster.
  const tfidfScorer = buildTfIdf(allPooledTexts);

  const clusterKeyphrases = [];
  const clusterCentroids = []; // reused in Stage 4 for the relevance gate

  for (let ci = 0; ci < clusterArrays.length; ci++) {
    const indices = clusterArrays[ci];
    const memberTexts = indices.map(i => notes[i].text);

    // Compute cluster centroid as mean of member embeddings, then L2-normalise.
    // Stored in clusterCentroids so Stage 4 can use it for the relevance gate
    // without recomputing.
    const dim = embeddings[0].length;
    const clusterEmbed = new Array(dim).fill(0);
    for (const idx of indices) {
      for (let d = 0; d < dim; d++) clusterEmbed[d] += embeddings[idx][d];
    }
    const norm = Math.sqrt(clusterEmbed.reduce((s, v) => s + v * v, 0));
    if (norm > 0) for (let d = 0; d < dim; d++) clusterEmbed[d] /= norm;
    clusterCentroids.push(clusterEmbed);

    // Pass tfidfScorer=null for 1–2 note clusters: TF-IDF requires multiple
    // documents to be contrastive. KeyBERT-only scoring still produces good
    // keyphrases that feed the keyphrase prompt path in Stage 4.
    const scorer = memberTexts.length >= 3 ? tfidfScorer : null;
    const keyphrases = extractKeyphrases(memberTexts, clusterEmbed, scorer, ci, embedCache);
    clusterKeyphrases.push(keyphrases);
  }

  await report('Identifying themes...', 75);

  // ── Stage 4: Generate insight labels ──────────────────────────────────
  // Report per-cluster progress so the bar advances continuously from 80→99%
  // instead of freezing for the entire T5 inference phase. Each cluster takes
  // 2-8 s in WASM; without granular updates the UI appears hung.
  await report('Naming your insights...', 80);

  // Collect label vectors alongside results for the uniqueness gate below.
  const labelVecs = [];
  const total = clusterArrays.length;

  for (let ci = 0; ci < total; ci++) {
    const pct = 80 + Math.round(((ci + 1) / total) * 19); // 80 → 99, leave 100 for post-processing
    await report(`Naming your insights... (${ci + 1}/${total})`, pct);

    const indices = clusterArrays[ci];
    const memberTexts = indices.map(i => notes[i].text);
    const keyphrases = clusterKeyphrases[ci];

    const { label, labelVec, fallback_used } = await generateLabel(keyphrases, memberTexts, clusterCentroids[ci]);

    results.push({
      cluster_id: ci,
      label: label || keyphrases[0] || memberTexts[0].slice(0, 30) || 'Insight',
      note_ids: indices.map(i => notes[i].id),
      keyphrases,
      silhouette_score: perCluster[ci],
      fallback_used,
    });
    labelVecs.push(labelVec);
  }

  await report('Naming your insights...', 100);

  // clusterCentroids: full-dim L2-normalised centroids built in Stage 3.
  // Exposed in the return so the reactive agent can compute centroid cosine
  // similarity for merge suggestions without re-embedding.

  // ── Cross-cluster uniqueness gate (E3) ─────────────────────────────────
  // Two clusters with cosine similarity ≥ 0.85 between their labels would
  // appear nearly identical in the UI. When a collision is detected, the
  // cluster with fewer member notes (less context for a distinctive label)
  // falls back to its second keyphrase as the label, which is guaranteed to
  // be non-overlapping with the first by the token-deduplication in Stage 3.
  const UNIQUENESS_THRESHOLD = 0.85;
  for (let i = 0; i < results.length; i++) {
    for (let j = i + 1; j < results.length; j++) {
      if (!labelVecs[i] || !labelVecs[j]) continue;
      const sim = cosineSim(labelVecs[i], labelVecs[j]);
      if (sim >= UNIQUENESS_THRESHOLD) {
        // The smaller cluster gets its label replaced by the next keyphrase.
        const smallerIdx = results[i].note_ids.length <= results[j].note_ids.length ? i : j;
        const alt = results[smallerIdx].keyphrases[1] || results[smallerIdx].keyphrases[0] || 'Insight';
        console.log(`[pipeline] uniqueness collision (${sim.toFixed(3)}): "${results[i].label}" vs "${results[j].label}" → replace cluster ${smallerIdx} with "${alt}"`);
        results[smallerIdx].label = cleanLabel(alt) || results[smallerIdx].label;
        results[smallerIdx].fallback_used = true;
      }
    }
  }

  // Hard invariant: no result may have an empty label
  for (const r of results) {
    if (!r.label || r.label.trim() === '') {
      r.label = r.keyphrases[0] || 'Insight';
    }
  }

  return { results, embeddingsReduced, perNoteSilhouette, centroids: clusterCentroids };
}
