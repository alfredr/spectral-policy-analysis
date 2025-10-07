#!/usr/bin/env python3
"""
Spectral rounding library for block-structured matrices.

Core functions:
- stamp_blocks: Generate random block-structured matrices
- degree_normalize: Normalize by row/column degrees
- top_svd: Compute top-k singular vectors
- spherical_kmeans: Cluster in unit sphere
- build_rectangle_cover: Build rectangle cover from clusters
- cover_stats: Evaluate cover quality
- cohesion: Measure cluster cohesion
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def stamp_blocks(
    m: int,
    n: int,
    num_blocks: int = 4,
    min_hw: Tuple[int, int] = (8, 8),
    max_hw: Tuple[int, int] = (20, 20),
    density: float = 1.0,
    noise: float = 0.0,
    seed: Optional[int] = 42,
) -> Tuple[np.ndarray, List[Dict[str, Any]], int]:
    """
    Create an m x n binary matrix P by stamping 'num_blocks' random rectangles of 1s.
    Each rectangle has height in [min_hw[0], max_hw[0]] and width in [min_hw[1], max_hw[1]].
    If density < 1, we flip each candidate 1 to 0 independently with prob (1-density).
    If noise > 0, we XOR the matrix with a noise matrix containing 1s at the given rate.
    Returns:
      P : (m x n) uint8 array
      blocks : list of dicts with {'r0','c0','h','w'} for each planted block
      noise_count : number of bit flips applied (base exception rate)
    """
    rng = np.random.default_rng(seed)
    P = np.zeros((m, n), dtype=np.uint8)
    blocks = []
    for _ in range(num_blocks):
        h = rng.integers(min_hw[0], max_hw[0] + 1)
        w = rng.integers(min_hw[1], max_hw[1] + 1)
        r0 = rng.integers(0, max(1, m - h + 1))
        c0 = rng.integers(0, max(1, n - w + 1))
        # Stamp a full-1 block, then thin by density
        block = np.ones((h, w), dtype=np.uint8)
        if density < 1.0:
            mask = rng.random((h, w)) < density
            block = (block & mask.astype(np.uint8)).astype(np.uint8)
        P[r0 : r0 + h, c0 : c0 + w] |= block  # union with existing 1s
        blocks.append({"r0": int(r0), "c0": int(c0), "h": int(h), "w": int(w)})

    noise_count = 0
    if noise > 0.0:
        noise_matrix = (rng.random((m, n)) < noise).astype(np.uint8)
        noise_count = int(noise_matrix.sum())
        P = (P ^ noise_matrix).astype(np.uint8)

    return P, blocks, noise_count


def raw_T(P: np.ndarray, S: np.ndarray) -> float:
    """T(S) = (1/|S|^2) * 1_S^T (P P^T) 1_S  (raw overlap average)."""
    if S.sum() == 0:
        return 0.0
    oneS = S.astype(float)
    K = P @ P.T
    num = float(oneS @ K @ oneS)
    den = float((S.sum()) ** 2)
    return num / den


def tilde_T(P: np.ndarray, S: np.ndarray) -> float:
    """T~(S) = (1_S^T K 1_S) / (1_S^T D 1_S), where K=P P^T, D=diag(K)."""
    if S.sum() == 0:
        return 0.0
    oneS = S.astype(float)
    K = P @ P.T
    D = np.diag(K)  # Extract diagonal as 1D array
    num = float(oneS @ K @ oneS)
    den = float(oneS @ (D * oneS))  # Element-wise multiplication for diagonal
    if den == 0:
        return 0.0
    return num / den


def cosine_avg(P: np.ndarray, S: np.ndarray) -> float:
    """
    Average pairwise cosine similarity between rows in S.
    Equals (1/|S|^2) * 1_S^T S_kernel 1_S, S_kernel= D^{-1/2} K D^{-1/2}.
    """
    idx = np.where(S > 0)[0]
    if idx.size == 0:
        return 0.0
    R = P[idx]  # rows in S
    norms = np.linalg.norm(R, axis=1)
    norms[norms == 0] = 1.0
    Rn = R / norms[:, None]
    Sker = Rn @ Rn.T
    return float(Sker.sum()) / (idx.size**2)


def cohesion(
    P: np.ndarray, clusters: List[np.ndarray], measure: str = "cosine"
) -> List[float]:
    """Compute chosen cohesion for each user-cluster indicator in clusters."""
    out = []
    for S in clusters:
        if measure == "raw":
            out.append(raw_T(P, S))
        elif measure == "tilde":
            out.append(tilde_T(P, S))
        else:
            out.append(cosine_avg(P, S))
    return out


def _safe_inv_sqrt(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return 1.0 / np.sqrt(np.maximum(x, eps))


def degree_normalize(
    P: np.ndarray, tau: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Degree-regularized normalization M = Du^{-1/2} P Dr^{-1/2}, with floors via tau.
    """
    du = P.sum(axis=1).astype(float)
    dr = P.sum(axis=0).astype(float)
    if tau is None:
        # a gentle regularizer: average degree across both sides
        tau = (du.sum() + dr.sum()) / (P.shape[0] + P.shape[1] + 1e-9) * 0.5
    Du_is = _safe_inv_sqrt(du + tau)
    Dr_is = _safe_inv_sqrt(dr + tau)
    M = (Du_is[:, None]) * P * (Dr_is[None, :])
    return M, Du_is, Dr_is


def build_kernel(P: np.ndarray, measure: str = "degree") -> np.ndarray:
    """
    Build a kernel matrix for spectral analysis.

    Args:
        P: Binary matrix (m x n)
        measure: One of "degree", "cosine", "raw"
            - "degree": Du^{-1/2} P Dr^{-1/2} (bipartite normalization)
            - "cosine": Row-normalized P (cosine similarity when computing P P^T)
            - "raw": P itself (no normalization)

    Returns:
        M: Kernel matrix for SVD
    """
    if measure == "cosine":
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return P / norms
    elif measure == "raw":
        return P.astype(float)
    else:  # degree (default)
        M, _, _ = degree_normalize(P)
        return M


def top_svd(M: np.ndarray, k: int, method: str = "auto", seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get top-k SVD factors.

    Args:
        M: Input matrix
        k: Number of components
        method: "auto" (chooses based on size), "exact" (np.linalg.svd), "randomized" (sklearn)
        seed: Random seed for randomized SVD

    Returns:
        U, s, Vt: Top-k SVD factors
    """
    k = min(k, min(M.shape) - 1)

    # Auto-select method based on matrix size and k
    if method == "auto":
        # Use randomized for large matrices when k << min(m,n)
        if min(M.shape) > 200 and k < min(M.shape) // 3:
            method = "randomized"
        else:
            method = "exact"

    if method == "randomized":
        try:
            from sklearn.utils.extmath import randomized_svd
            U, s, Vt = randomized_svd(M, n_components=k, random_state=seed if seed is not None else 0)
            return U, s, Vt
        except ImportError:
            # Fallback to exact if sklearn not available
            pass

    # Exact SVD (default)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]


def svd_reconstruct(
    P: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    j: int,
    measure: str = "degree",
) -> np.ndarray:
    """
    Reconstruct P from rank-j SVD approximation.

    Args:
        P: Original binary matrix
        U, s, Vt: SVD factors of kernel matrix M
        j: Rank of approximation
        measure: Kernel type used ("degree", "cosine", "raw")

    Returns:
        P_svd: Rank-j approximation of P, clipped to [0, inf)
    """
    Uj = U[:, :j]
    Vjt = Vt[:j, :]
    M_reconstructed = Uj @ np.diag(s[:j]) @ Vjt

    if measure == "degree":
        # Denormalize: P_svd = Du^{1/2} @ M @ Dr^{1/2}
        _, Du_is, Dr_is = degree_normalize(P)
        P_svd = (1 / Du_is[:, None]) * M_reconstructed * (1 / Dr_is[None, :])
    elif measure == "cosine":
        # Denormalize by row norms
        norms = np.linalg.norm(P, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        P_svd = M_reconstructed * norms
    else:  # raw
        P_svd = M_reconstructed

    return np.clip(P_svd, 0, None)


def spherical_kmeans(
    X: np.ndarray, k: int, iters: int = 50, seed: int = 0
) -> np.ndarray:
    """
    Simple spherical k-means (unit-normalized rows, cosine-based).
    Returns labels in [0..k-1].
    """
    rng = np.random.default_rng(seed)
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # init: random rows
    idx = rng.choice(Xn.shape[0], size=k, replace=False)
    C = Xn[idx].copy()
    for _ in range(iters):
        sims = Xn @ C.T
        labels = np.argmax(sims, axis=1)
        # recompute centroids
        C = np.zeros_like(C)
        for j in range(k):
            mask = labels == j
            if not mask.any():
                # re-seed if empty
                C[j] = Xn[rng.integers(0, Xn.shape[0])]
            else:
                v = Xn[mask].mean(axis=0)
                n = np.linalg.norm(v) + 1e-12
                C[j] = v / n
    return labels


def build_rectangle_cover(  # noqa: C901
    P: np.ndarray,
    user_labels: np.ndarray,
    resource_labels: Optional[np.ndarray],
    tau: float = 0.6,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Given user and (optionally) resource cluster labels, construct a Boolean rectangle cover:
      - If resource_labels is provided: rectangles are (U_c, R_d) for dense blocks (density >= tau).
      - Else: for each user cluster, take majority resource set S_c (>= tau within cluster).
    Returns list of (user_indicator, resource_indicator).
    """
    m, n = P.shape
    cover = []
    if resource_labels is None:
        # Majority resources per user cluster
        for c in np.unique(user_labels):
            Uc = user_labels == c
            if Uc.sum() == 0:
                continue
            col_frac = P[Uc].mean(
                axis=0
            )  # fraction of users in cluster with the resource
            Rc = col_frac >= tau
            if Rc.sum() == 0:
                continue
            cover.append((Uc.astype(np.uint8), Rc.astype(np.uint8)))
    else:
        for c in np.unique(user_labels):
            Uc = user_labels == c
            if Uc.sum() == 0:
                continue
            for d in np.unique(resource_labels):
                Rd = resource_labels == d
                if Rd.sum() == 0:
                    continue
                density = P[Uc][:, Rd].mean()
                if density >= tau:
                    cover.append((Uc.astype(np.uint8), Rd.astype(np.uint8)))
    return cover


def cover_stats(
    P: np.ndarray, cover: List[Tuple[np.ndarray, np.ndarray]]
) -> Dict[str, Any]:
    """
    Evaluate a rectangle cover: how many ones covered, rectangles used, precision/recall on ones.
    """
    if not cover:
        return {
            "rectangles": 0,
            "ones": int(P.sum()),
            "covered_ones": 0,
            "precision": 0.0,
            "recall": 0.0,
        }
    m, n = P.shape
    C = np.zeros_like(P, dtype=np.uint8)
    for Uc, Rc in cover:
        C |= (Uc[:, None] & Rc[None, :]).astype(np.uint8)
    tp = int((C & P).sum())
    fp = int((C & (1 - P)).sum())
    fn = int(((1 - C) & P).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "rectangles": len(cover),
        "ones": int(P.sum()),
        "covered_ones": tp,
        "precision": prec,
        "recall": rec,
    }


# ---------- NEW: kernel & spectral helpers ----------


def build_user_kernel(P: np.ndarray, measure: str = "degree") -> np.ndarray:
    """
    Return a symmetric user-user kernel K_u to compare with left singular space.
    - 'raw'    : K_u = P P^T
    - 'cosine' : K_u = D^{-1/2} P P^T D^{-1/2}
    - 'degree' : K_u = (Du^{-1/2} P Dr^{-1/2}) (Du^{-1/2} P Dr^{-1/2})^T
                = Du^{-1/2} P Dr^{-1} P^T Du^{-1/2}
    """
    if measure == "raw":
        return (P @ P.T).astype(float)
    elif measure == "cosine":
        du = np.linalg.norm(P, axis=1)
        du[du == 0] = 1.0
        Q = P / du[:, None]
        return Q @ Q.T
    else:  # degree
        M, Du_is, Dr_is = degree_normalize(P)
        # K_u = M M^T
        return M @ M.T


def build_resource_kernel(P: np.ndarray, measure: str = "degree") -> np.ndarray:
    """
    Symmetric resource-resource kernel K_r aligned with right singular space.
    """
    if measure == "raw":
        return (P.T @ P).astype(float)
    elif measure == "cosine":
        # column cosine: normalize columns
        dr = np.linalg.norm(P, axis=0)
        dr[dr == 0] = 1.0
        Q = P / dr[None, :]
        return Q.T @ Q
    else:  # degree
        M, Du_is, Dr_is = degree_normalize(P)
        return M.T @ M


def eig_top(K: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Top-k eigenpairs of symmetric PSD matrix K (descending order).
    """
    w, V = np.linalg.eigh(K)
    idx = np.argsort(w)[::-1]
    w = w[idx]
    V = V[:, idx]
    k = min(k, V.shape[1])
    return w[:k], V[:, :k]


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Principal angles between subspaces span(U) and span(V).
    Returns angles in radians, length = min(dimU, dimV).
    """
    # Orthonormalize (safety)
    Uq, _ = np.linalg.qr(U)
    Vq, _ = np.linalg.qr(V)
    s = np.linalg.svd(Uq.T @ Vq, full_matrices=False)[1]
    s = np.clip(s, 0.0, 1.0)
    return np.arccos(s)


def svd_kernel(
    P: np.ndarray, measure: str, k: int, seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience: build kernel M for SVD based on measure and return top-k SVD.
    """
    M = build_kernel(P, measure=measure)
    U, s, Vt = top_svd(M, k=k, method="randomized", seed=seed)
    return U, s, Vt


def svd_vs_eig_alignment(P: np.ndarray, measure: str, k: int, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Quantify the relationship:
      - Left singular vectors U_k vs eigenvectors of K_u
      - Right singular vectors V_k vs eigenvectors of K_r
      - Compare s^2 to eigvals
      - Principal angles between subspaces
    """
    U, s, Vt = svd_kernel(P, measure, k, seed=seed)
    Ku = build_user_kernel(P, measure)
    Kr = build_resource_kernel(P, measure)

    wu, Vu = eig_top(Ku, k)
    wr, Vr = eig_top(Kr, k)

    # principal angles
    ang_u = principal_angles(U[:, :k], Vu[:, :k])
    ang_r = principal_angles(Vt[:k, :].T, Vr[:, :k])

    # match spectra (s^2 vs eigenvalues)
    s2 = s[:k] ** 2
    # normalize for reporting stability
    s2_norm = s2 / (s2.max() + 1e-12)
    wu_norm = wu / (wu.max() + 1e-12)
    wr_norm = wr / (wr.max() + 1e-12)

    # correlation between s^2 and eigenvalues
    def safe_corr(a, b):
        a = a.reshape(-1)
        b = b.reshape(-1)
        if a.size < 2:
            return 1.0
        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)
        return float(np.clip((a * b).mean(), -1, 1))

    return {
        "U": U[:, :k],
        "s": s[:k],
        "V": Vt[:k, :].T,
        "eig_user_vals": wu,
        "eig_user_vecs": Vu,
        "eig_res_vals": wr,
        "eig_res_vecs": Vr,
        "angles_user": ang_u,  # radians
        "angles_res": ang_r,  # radians
        "corr_s2_vs_eig_user": safe_corr(s2, wu),
        "corr_s2_vs_eig_res": safe_corr(s2, wr),
        "s2_norm": s2_norm,
        "wu_norm": wu_norm,
        "wr_norm": wr_norm,
    }


def pr_curve_from_scores(
    P_true: np.ndarray, P_scores: np.ndarray, taus: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Compute precision-recall curve over thresholds.
    Returns dict with taus, precision, recall, f1, best_tau, best_f1, auprc.
    """
    if taus is None:
        taus = np.linspace(P_scores.min(), P_scores.max(), 101)
    precisions, recalls, f1s = [], [], []
    for t in taus:
        P_bin = (P_scores >= t).astype(np.uint8)
        tp = int(((P_bin == 1) & (P_true == 1)).sum())
        fp = int(((P_bin == 1) & (P_true == 0)).sum())
        fn = int(((P_bin == 0) & (P_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
    # AUPRC via trapezoid over recall
    order = np.argsort(recalls)
    R = np.array(recalls)[order]
    Pp = np.array(precisions)[order]
    auprc = np.trapz(Pp, R)
    best_idx = int(np.argmax(f1s))
    return {
        "taus": np.array(taus),
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1s),
        "best_tau": float(taus[best_idx]),
        "best_f1": float(f1s[best_idx]),
        "auprc": float(auprc),
    }


def scree_elbow(svals_sq: np.ndarray) -> int:
    """
    Simple elbow detector: first index with drop > median drop.
    """
    drops = svals_sq[:-1] - svals_sq[1:]
    if len(drops) == 0:
        return 1
    thr = np.median(drops)
    idx = np.argmax(drops > thr)
    return int(idx + 1) if drops[idx] > thr else 1


def median_principal_angle_degrees(angles_rad: np.ndarray) -> float:
    """
    Convert principal angles from radians to degrees and return median.
    """
    return float(np.median(angles_rad) * 180 / np.pi)


def permute_matrix(
    P: np.ndarray,
    planted: List[Dict[str, Any]],
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]], np.ndarray, np.ndarray]:
    """
    Randomly permute rows and columns of P, and update planted block coordinates.

    Args:
        P: Binary matrix (m x n)
        planted: List of planted block dicts with 'r0', 'c0', 'h', 'w'
        seed: Random seed for reproducible permutation

    Returns:
        P_perm: Permuted matrix
        planted_perm: Updated block coordinates (note: blocks may no longer be rectangles in visual space)
        row_perm: Row permutation indices
        col_perm: Column permutation indices
    """
    rng = np.random.default_rng(seed)
    m, n = P.shape

    # Generate random permutations
    row_perm = rng.permutation(m)
    col_perm = rng.permutation(n)

    # Permute matrix
    P_perm = P[row_perm, :][:, col_perm]

    # Note: After permutation, the planted blocks are scattered across the matrix
    # and no longer appear as visual rectangles. We return empty list for planted
    # since drawing rectangles would be misleading.
    planted_perm = []

    return P_perm, planted_perm, row_perm, col_perm


def optimize_svd_params(
    P: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    j_range: Tuple[int, int] = (2, 20),
    tau_range: Tuple[float, float] = (0.1, 0.9),
) -> Dict[str, Any]:
    """
    Find optimal (j, tau) that minimizes total policy complexity using coordinate descent.

    Args:
        P: Binary access matrix
        U, s, Vt: SVD factors
        j_range: (min_j, max_j) range to search
        tau_range: (min_tau, max_tau) range to search

    Returns:
        Dict with best_j, best_tau, min_complexity
    """
    max_j = min(j_range[1], min(U.shape[1], s.shape[0]) - 1)

    # Start with middle values
    current_j = (j_range[0] + max_j) // 2
    current_tau = (tau_range[0] + tau_range[1]) / 2

    best_complexity = float('inf')
    best_j = current_j
    best_tau = current_tau

    # Coordinate descent: alternate optimizing j and tau
    for iteration in range(10):  # Max 10 iterations
        improved = False

        # Optimize tau with fixed j (grid search over tau values)
        tau_values = np.linspace(tau_range[0], tau_range[1], 25)
        for tau in tau_values:
            decomp = svd_based_group_decomposition(P, U, s, Vt, current_j, threshold=float(tau))
            if decomp['complexity'] < best_complexity:
                best_complexity = decomp['complexity']
                current_tau = float(tau)
                best_tau = current_tau
                improved = True

        # Optimize j with fixed tau (discrete search, prefer smaller ranks)
        for j in range(j_range[0], max_j + 1):
            decomp = svd_based_group_decomposition(P, U, s, Vt, j, threshold=current_tau)
            # Prefer smaller j if complexity is equal or better
            if decomp['complexity'] < best_complexity or (decomp['complexity'] == best_complexity and j < best_j):
                best_complexity = decomp['complexity']
                current_j = j
                best_j = j
                improved = True

        # Stop if no improvement
        if not improved:
            break

    return {
        'best_j': int(best_j),
        'best_tau': float(best_tau),
        'min_complexity': int(best_complexity),
    }


def optimize_group_params(
    P: np.ndarray,
    U: np.ndarray,
    j_range: Tuple[int, int] = (2, 20),
    density_thresholds: Optional[np.ndarray] = None,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Grid search to find optimal (j, density_threshold) that minimizes exception count.

    Args:
        P: Binary access matrix
        U: Left singular vectors from SVD
        j_range: (min_j, max_j) range to search
        density_thresholds: Array of density thresholds to try (default: [0.3, 0.4, ..., 0.9])
        seed: Random seed for clustering

    Returns:
        Dict with best_j, best_density, min_exceptions, and search results
    """
    if density_thresholds is None:
        density_thresholds = np.arange(0.3, 1.0, 0.1)

    best_j = j_range[0]
    best_density = density_thresholds[0]
    min_exceptions = float('inf')
    results = []

    num_blocks = min(j_range[1], 10)  # estimate cluster count

    for j in range(j_range[0], min(j_range[1] + 1, U.shape[1])):
        num_clusters = min(num_blocks, j)
        user_labels = spherical_kmeans(U[:, :j], k=num_clusters, seed=seed)

        for density in density_thresholds:
            decomp = group_decomposition(P, user_labels, density_threshold=density)
            num_exc = decomp['num_exceptions']

            results.append({
                'j': j,
                'density': float(density),
                'exceptions': num_exc,
                'complexity': decomp['complexity'],
            })

            if num_exc < min_exceptions:
                min_exceptions = num_exc
                best_j = j
                best_density = float(density)

    return {
        'best_j': best_j,
        'best_density': best_density,
        'min_exceptions': int(min_exceptions),
        'results': results,
    }


def svd_based_group_decomposition(
    P: np.ndarray,
    U: np.ndarray,
    s: np.ndarray,
    Vt: np.ndarray,
    j: int,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Decompose access matrix using SVD components directly (no clustering).
    Each SVD component k defines a latent group with overlapping membership.

    Args:
        P: Binary access matrix (m x n)
        U, s, Vt: SVD factors
        j: Number of components to use
        threshold: Threshold on weighted component values (U*s and Vt*s)

    Returns:
        Dict with:
            - user_to_group: List of (user, group) tuples
            - resource_to_group: List of (resource, group) tuples
            - exceptions: List of ('Permit'/'Deny', user, resource) tuples
            - complexity: Total count of all tuples
    """
    m, n = P.shape
    Uj = U[:, :j]
    sj = s[:j]
    Vtj = Vt[:j, :]

    # Weight by singular values for each component
    U_weighted = Uj * sj[None, :]  # (m, j)
    Vt_weighted = Vtj * sj[:, None]  # (j, n)

    # Normalize to [0, 1] for thresholding
    U_norm = U_weighted / (np.abs(U_weighted).max(axis=0, keepdims=True) + 1e-12)
    Vt_norm = Vt_weighted / (np.abs(Vt_weighted).max(axis=1, keepdims=True) + 1e-12)

    # Build tuples: user->group and resource->group
    user_to_group = []  # List of (user, group) tuples
    resource_to_group = []  # List of (resource, group) tuples
    active_groups = set()

    for k in range(j):
        users_k = np.where(U_norm[:, k] > threshold)[0]
        resources_k = np.where(Vt_norm[k, :] > threshold)[0]

        if len(users_k) > 0 and len(resources_k) > 0:
            active_groups.add(k)
            for u in users_k:
                user_to_group.append((int(u), k))
            for r in resources_k:
                resource_to_group.append((int(r), k))

    # Ensure every user/resource is assigned to at least one group (their top component)
    assigned_users = set(u for u, g in user_to_group)
    assigned_resources = set(r for r, g in resource_to_group)

    for u in range(m):
        if u not in assigned_users:
            # Assign to component with highest absolute loading
            best_k = int(np.argmax(np.abs(U_norm[u, :])))
            user_to_group.append((u, best_k))
            active_groups.add(best_k)

    for r in range(n):
        if r not in assigned_resources:
            # Assign to component with highest absolute loading
            best_k = int(np.argmax(np.abs(Vt_norm[:, r])))
            resource_to_group.append((r, best_k))
            active_groups.add(best_k)

    # Build predicted access from group membership
    P_predicted = np.zeros_like(P, dtype=np.uint8)

    # Group users and resources by their groups
    user_groups = {}
    for u, g in user_to_group:
        if u not in user_groups:
            user_groups[u] = set()
        user_groups[u].add(g)

    resource_groups = {}
    for r, g in resource_to_group:
        if r not in resource_groups:
            resource_groups[r] = set()
        resource_groups[r].add(g)

    # User u can access resource r if they share at least one group
    for u in range(m):
        for r in range(n):
            if u in user_groups and r in resource_groups:
                if user_groups[u] & resource_groups[r]:  # Intersection
                    P_predicted[u, r] = 1

    # Exceptions = symmetric difference between SVD reconstruction and original
    # Groups represent the SVD reconstruction; exceptions patch to match original
    P_svd_recon = svd_reconstruct(P, U, s, Vt, j, measure='degree')
    P_svd_thresh = (P_svd_recon >= threshold).astype(np.uint8)

    exceptions = []
    for u in range(m):
        for r in range(n):
            if P[u, r] == 1 and P_svd_thresh[u, r] == 0:
                exceptions.append(('Permit', u, r))
            elif P[u, r] == 0 and P_svd_thresh[u, r] == 1:
                exceptions.append(('Deny', u, r))

    # Complexity = total number of tuples
    complexity = len(user_to_group) + len(resource_to_group) + len(exceptions)

    return {
        'user_to_group': user_to_group,  # List of (user, group) tuples
        'resource_to_group': resource_to_group,  # List of (resource, group) tuples
        'exceptions': exceptions,
        'complexity': complexity,
        'num_groups': len(active_groups),
        'num_exceptions': len(exceptions),
        'num_user_group_tuples': len(user_to_group),
        'num_resource_group_tuples': len(resource_to_group),
        'method': 'svd_direct',
    }


def group_decomposition(
    P: np.ndarray,
    user_labels: np.ndarray,
    density_threshold: float = 0.6,
) -> Dict[str, Any]:
    """
    Decompose access matrix into group-based rules plus exceptions.

    Args:
        P: Binary access matrix (m users x n resources)
        user_labels: Cluster labels for users (length m)
        density_threshold: Minimum density to create a group->resource rule

    Returns:
        Dict with:
            - user_to_group: List of (user_idx, group_id) tuples
            - group_to_resources: List of (group_id, resource_idxs) tuples
            - exceptions: List of ('Permit', user, resource) or ('Deny', user, resource)
            - complexity: Total number of items across all lists
    """
    m, n = P.shape
    num_groups = len(np.unique(user_labels))

    # 1. User-to-group mappings
    user_to_group = [(u, int(user_labels[u])) for u in range(m)]

    # 2. Group-to-resource mappings
    group_to_resources = []
    for g in range(num_groups):
        users_in_g = np.where(user_labels == g)[0]
        if len(users_in_g) == 0:
            continue

        # Find resources accessed by >= density_threshold of users in this group
        resource_coverage = P[users_in_g].mean(axis=0)  # fraction of group accessing each resource
        resources_for_g = np.where(resource_coverage >= density_threshold)[0]

        if len(resources_for_g) > 0:
            group_to_resources.append((g, resources_for_g.tolist()))

    # 3. Build predicted access matrix from group rules
    P_predicted = np.zeros_like(P, dtype=np.uint8)
    for g, resources in group_to_resources:
        users_in_g = np.where(user_labels == g)[0]
        for u in users_in_g:
            for r in resources:
                P_predicted[u, r] = 1

    # 4. Find exceptions (only count differences, the symmetric difference)
    exceptions = []
    for u in range(m):
        for r in range(n):
            if P[u, r] == 1 and P_predicted[u, r] == 0:
                exceptions.append(('Permit', u, r))
            elif P[u, r] == 0 and P_predicted[u, r] == 1:
                exceptions.append(('Deny', u, r))

    # Complexity: user assignments + group-resource rules + exceptions
    # Note: user_to_group always has m entries, so we count groups + group rules + exceptions
    num_group_rules = sum(len(resources) for _, resources in group_to_resources)
    complexity = num_groups + num_group_rules + len(exceptions)

    return {
        'user_to_group': user_to_group,
        'group_to_resources': group_to_resources,
        'exceptions': exceptions,
        'complexity': complexity,
        'num_groups': num_groups,
        'num_exceptions': len(exceptions),
    }
