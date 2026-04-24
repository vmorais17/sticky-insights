/**
 * pca.js — PCA dimensionality reduction (gram-matrix approach).
 *
 * Problem: in 384 dimensions, pairwise cosine distances between n=50 L2-
 * normalised vectors are statistically concentrated near the mean (central
 * limit theorem). The contrast ratio max_dist/min_dist collapses toward 1 as
 * dimensionality grows, which is why per-cluster silhouette scores sit in the
 * 0.07–0.29 range even when clusters are semantically real. The signal lives
 * in the top 20–40 principal components; the remaining ~350 dimensions add
 * only noise that uniformly compresses inter-cluster distance ratios.
 *
 * Fix: project to targetDim components before downstream use. This sharpens
 * the distance contrast without changing what is being measured — semantics
 * are preserved, noise is removed.
 *
 * Implementation uses the gram-matrix SVD (n×n, not d×d): since n < d for
 * typical boards (n ≤ 300, d = 384), the n×n gram matrix G = X X^T is cheaper
 * to compute and eigen-decompose than the d×d covariance. Eigenvectors of G
 * are the left singular vectors of X; the principal component scores are U·Σ.
 *
 * Two normalisation modes:
 *   normalize: true  (default) — L2-normalise each output row so cosine distance
 *                                remains the correct metric. Used for clustering.
 *   normalize: false           — return raw U·Σ scores; points spread in 2D
 *                                according to their distance from the dataset mean.
 *                                Used for semantic space visualisation.
 */

/**
 * Projects embeddings onto their top `targetDim` principal components.
 *
 * Guard: if embeddings.length <= 1 the gram matrix has rank 0 and power
 * iteration produces empty output. Returns embeddings unchanged in that case.
 *
 * @param {number[][]} embeddings  L2-normalised note vectors (n × d)
 * @param {number}     targetDim  Number of components to keep
 * @param {{ normalize?: boolean }} [opts]
 * @returns {number[][]}  Projected vectors (n × min(targetDim, n-1))
 */
export function pcaProject(embeddings, targetDim, { normalize = true } = {}) {
  const n = embeddings.length;
  if (n <= 1) return embeddings; // gram matrix rank = 0; nothing to decompose

  const d = embeddings[0].length;
  const k = Math.min(targetDim, n - 1); // rank of mean-centred gram matrix ≤ n-1

  // 1. Mean-centre
  const mean = new Array(d).fill(0);
  for (const e of embeddings) for (let i = 0; i < d; i++) mean[i] += e[i];
  for (let i = 0; i < d; i++) mean[i] /= n;
  const X = embeddings.map(e => e.map((v, i) => v - mean[i]));

  // 2. Gram matrix G = X X^T  (n×n, symmetric)
  const G = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      let dot = 0;
      for (let l = 0; l < d; l++) dot += X[i][l] * X[j][l];
      G[i][j] = G[j][i] = dot;
    }
  }

  // 3. Top-k eigenvectors via power iteration + deflation.
  //    Each iteration v ← Gd·v / |Gd·v| converges to the dominant eigenvector.
  //    Deflation Gd ← Gd − λ vvᵀ removes each found component so the next
  //    iteration converges to the next-largest.
  const POWER_ITER = 200;
  const eigvecs = [];
  const eigvals = [];
  const totalVariance = G.reduce((s, row, i) => s + row[i], 0); // trace = sum of variances

  const Gd = G.map(row => Float64Array.from(row)); // working copy for deflation

  for (let ei = 0; ei < k; ei++) {
    // Deterministic seed varies per component (avoids correlated starts)
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = (i === ei % n) ? 1.0 : 0.1 * ((i + ei) % 3 - 1);

    for (let iter = 0; iter < POWER_ITER; iter++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += Gd[i][j] * v[j];
        w[i] = s;
      }
      let norm = 0;
      for (const x of w) norm += x * x;
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      for (let i = 0; i < n; i++) v[i] = w[i] / norm;
    }

    // Rayleigh quotient: λ = vᵀ Gd v
    let lambda = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      for (let j = 0; j < n; j++) s += Gd[i][j] * v[j];
      lambda += v[i] * s;
    }
    if (lambda < 1e-10) break; // remaining eigenvalues are numerically zero

    eigvecs.push(v);
    eigvals.push(lambda);

    // Deflate: Gd ← Gd − λ vvᵀ
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) Gd[i][j] -= lambda * v[i] * v[j];
  }

  // 4. Principal component scores: P[i][j] = U[i][j] * √λ_j  (equivalent to U·Σ)
  const actualK = eigvecs.length;
  const varianceCaptured = eigvals.reduce((s, v) => s + v, 0);
  console.log(
    `[pca] ${d}d → ${actualK}d | variance explained: ${(varianceCaptured / totalVariance * 100).toFixed(1)}%`
  );

  const projected = Array.from({ length: n }, (_, i) =>
    Array.from({ length: actualK }, (_, j) => eigvecs[j][i] * Math.sqrt(eigvals[j]))
  );

  if (!normalize) return projected;

  // 5. L2-normalise rows — preserves cosine distance as the correct metric
  //    for clustering (silhouette, k-means, agglomerative).
  //    Skip when normalize: false (visualisation path wants natural radial spread).
  return projected.map(row => {
    let norm = 0;
    for (const v of row) norm += v * v;
    norm = Math.sqrt(norm);
    if (norm < 1e-12) return row;
    return row.map(v => v / norm);
  });
}
