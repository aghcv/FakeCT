"""Exploratory PCA and clustering of baseline-relative CT feature vectors.

Feature construction supplies physical and baseline normalization. This module
centres those features but does not standardize individual columns or normalize
individual siblings: both choices would change the meaning of change extent.
PCA fits only the rows supplied by the caller. Anatomical identity, recipe
parameters and training labels are deliberately absent from this API.

K-means partitions a continuous space even when discrete groups do not exist.
Consequently ``candidate_best_k`` is an exploratory partition, not a population
class count. ``suggested_k`` falls back to one when silhouette, restart stability
or minimum size checks do not support that partition. These heuristic checks
are not a hypothesis test or proof that the selected count is correct.
"""

from itertools import combinations

import numpy as np
from scipy.cluster.vq import ClusterError, kmeans2
from scipy.spatial.distance import cdist


DEFAULT_SETTINGS = {
    "seed": 42,
    "variance_target": .90,
    "max_components": 20,
    "pca_oversamples": 10,
    "pca_power_iterations": 2,
    "k_min": 2,
    "k_max": 10,
    "n_init": 4,
    "kmeans_iterations": 100,
    "silhouette_sample_size": 1000,
    "silhouette_threshold": .25,
    "stability_threshold": .75,
    "min_cluster_fraction": .01,
}


def _settings(value):
    supplied = dict(value or {})
    unknown = set(supplied) - set(DEFAULT_SETTINGS)
    if unknown:
        raise ValueError("Unknown clustering settings: " + ", ".join(sorted(unknown)))
    result = dict(DEFAULT_SETTINGS, **supplied)
    for name in ("seed", "max_components", "pca_oversamples",
                 "pca_power_iterations", "k_min", "k_max", "n_init",
                 "kmeans_iterations", "silhouette_sample_size"):
        if isinstance(result[name], bool) or int(result[name]) != result[name]:
            raise ValueError(name + " must be an integer")
        result[name] = int(result[name])
    if result["seed"] < 0 or min(result["pca_oversamples"], result["pca_power_iterations"]) < 0:
        raise ValueError("Seed, oversamples and power iterations must be nonnegative")
    if result["max_components"] < 2 or result["k_min"] < 2 or result["k_max"] < result["k_min"]:
        raise ValueError("Need at least two PCA components and valid K limits")
    if min(result["n_init"], result["kmeans_iterations"], result["silhouette_sample_size"]) < 1:
        raise ValueError("Restart, iteration and silhouette budgets must be positive")
    for name in ("variance_target", "silhouette_threshold", "stability_threshold", "min_cluster_fraction"):
        result[name] = float(result[name])
        if not np.isfinite(result[name]) or not 0 <= result[name] <= 1:
            raise ValueError(name + " must be finite and within [0,1]")
    if result["variance_target"] <= 0 or result["min_cluster_fraction"] >= .5:
        raise ValueError("Variance target must be positive; minimum fraction must be less than 0.5")
    return result


def _features(value):
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not result.shape[0] or not result.shape[1]:
        raise ValueError("Features must be a nonempty rows-by-features matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError("Features contain nonfinite values")
    return result


def adjusted_rand_index(first, second):
    """Adjusted Rand agreement, invariant to arbitrary cluster names."""
    first, second = np.asarray(first), np.asarray(second)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("Partitions must be one-dimensional with equal lengths")
    n = first.size
    if n < 2:
        return 1.
    _, a = np.unique(first, return_inverse=True)
    _, b = np.unique(second, return_inverse=True)
    table = np.zeros((int(a.max()) + 1, int(b.max()) + 1), dtype=np.int64)
    np.add.at(table, (a, b), 1)
    choose_two = lambda counts: float(np.sum(counts * (counts - 1) / 2))
    matches = choose_two(table)
    a_pairs, b_pairs = choose_two(table.sum(axis=1)), choose_two(table.sum(axis=0))
    expected = a_pairs * b_pairs / (n * (n - 1) / 2)
    maximum = (a_pairs + b_pairs) / 2
    denominator = maximum - expected
    if abs(denominator) <= np.finfo(float).eps:
        return 1.
    return float((matches - expected) / denominator)


def silhouette(scores, labels, sample_indices=None):
    """Silhouette averaged over selected rows using distances to all rows.

    The sample is chosen independently of labels. A singleton has silhouette
    zero. Sampling queries, rather than the reference population, avoids losing
    a small cluster from the inter-cluster distance calculation.
    """
    scores = _features(scores)
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size != scores.shape[0]:
        raise ValueError("One cluster label is required per row")
    unique, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    if unique.size < 2 or unique.size >= labels.size:
        return 0.
    indices = (np.arange(labels.size) if sample_indices is None
               else np.asarray(sample_indices, dtype=np.int64))
    if indices.ndim != 1 or indices.size == 0 or np.any(indices < 0) or np.any(indices >= labels.size):
        raise ValueError("Silhouette sample indices are invalid")
    distances = cdist(scores[indices], scores, metric="euclidean")
    means = np.empty((indices.size, unique.size), dtype=np.float64)
    for label_index in range(unique.size):
        means[:, label_index] = distances[:, inverse == label_index].sum(axis=1) / counts[label_index]
    rows = np.arange(indices.size)
    own = inverse[indices]
    own_counts = counts[own]
    a = means[rows, own] * own_counts / np.maximum(own_counts - 1, 1)
    means[rows, own] = np.inf
    b = means.min(axis=1)
    denominator = np.maximum(a, b)
    values = np.divide(b - a, denominator, out=np.zeros_like(a), where=denominator > 0)
    values[own_counts == 1] = 0.
    return float(values.mean())


def _pca(features, settings):
    mean = features.mean(axis=0)
    centred = features - mean
    total = float(np.einsum("ij,ij->", centred, centred))
    if not np.isfinite(total):
        raise ValueError("Feature magnitude overflows PCA; use finite physical normalization")
    maximum = min(settings["max_components"], centred.shape[0], centred.shape[1])
    if total == 0 or centred.shape[0] == 1:
        return mean, np.zeros((2, features.shape[1])), np.zeros((features.shape[0], 2)), {
            "solver": "degenerate", "rank": 0, "retained_components": 2,
            "explained_variance_ratio": [0., 0.], "retained_variance_ratio": 0.,
            "target_variance_ratio": settings["variance_target"], "target_reached": True,
            "total_centered_sum_squares": total,
        }
    width = min(maximum + settings["pca_oversamples"], min(centred.shape))
    if min(centred.shape) <= width or max(centred.shape) <= 128:
        _, singular, vectors = np.linalg.svd(centred, full_matrices=False)
        solver = "exact_svd"
    else:
        # Feature-space random projection makes the random draw independent of
        # row ordering. Re-orthogonalization keeps power iterations stable.
        rng = np.random.default_rng(settings["seed"])
        basis, _ = np.linalg.qr(centred @ rng.normal(size=(centred.shape[1], width)), mode="reduced")
        for _ in range(settings["pca_power_iterations"]):
            feature_basis, _ = np.linalg.qr(centred.T @ basis, mode="reduced")
            basis, _ = np.linalg.qr(centred @ feature_basis, mode="reduced")
        _, singular, vectors = np.linalg.svd(basis.T @ centred, full_matrices=False)
        solver = "randomized_svd"
    singular, vectors = singular[:maximum], vectors[:maximum]
    rank = int(np.count_nonzero(singular > singular[0] * max(centred.shape) * np.finfo(float).eps))
    ratios = singular ** 2 / total
    target_count = int(np.searchsorted(np.cumsum(ratios), settings["variance_target"]) + 1)
    retained = min(maximum, max(2, target_count))
    vectors = vectors[:retained].copy()
    for row in vectors:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    scores = centred @ vectors.T
    actual_ratios = np.einsum("ij,ij->j", scores, scores) / total
    # Always expose PC1 and PC2, including a zero PC2 for a one-feature input.
    if scores.shape[1] < 2:
        vectors = np.vstack((vectors, np.zeros(features.shape[1])))
        scores = np.column_stack((scores, np.zeros(features.shape[0])))
        actual_ratios = np.append(actual_ratios, 0.)
    retained_ratio = min(1., float(actual_ratios.sum()))
    return mean, vectors, scores, {
        "solver": solver, "rank_within_computed_components": rank,
        "retained_components": int(scores.shape[1]),
        "explained_variance_ratio": actual_ratios.tolist(),
        "retained_variance_ratio": retained_ratio,
        "target_variance_ratio": settings["variance_target"],
        "target_reached": bool(retained_ratio >= settings["variance_target"] - 1e-10),
        "total_centered_sum_squares": total,
    }


def _ordered_partition(scores, labels):
    unique = np.unique(labels)
    centres = np.asarray([scores[labels == value].mean(axis=0) for value in unique])
    order = np.lexsort(centres.T[::-1])
    renamed = np.empty(labels.shape, dtype=np.int32)
    for new, old in enumerate(order):
        renamed[labels == unique[old]] = new
    return renamed, centres[order]


def transform_features(result, features):
    """Project baseline controls or other rows without refitting the PCA."""
    features = _features(features)
    mean, components = np.asarray(result["mean"]), np.asarray(result["components"])
    if features.shape[1] != mean.size:
        raise ValueError("Feature dimension differs from fitted PCA")
    return (features - mean) @ components.T


def analyse(features, settings=None):
    """Fit exploratory PCA/K-means and return arrays plus JSON-safe metadata.

    Return keys: ``analysis``, ``scores``, ``labels`` (suggested partition),
    ``candidate_labels`` (highest-silhouette candidate), ``components``,
    ``mean``, ``cluster_centers``, and ``candidate_centroid_features``. Fit
    baseline controls separately with :func:`transform_features`; do not append
    them as repeated training rows.
    """
    features, settings = _features(features), _settings(settings)
    mean, components, scores, pca = _pca(features, settings)
    # Canonical feature-score row ordering makes restart sampling and silhouette
    # queries invariant to input row order, apart from numerical degeneracies.
    canonical = np.lexsort(scores.T[::-1])
    ordered = np.ascontiguousarray(scores[canonical])
    rng = np.random.default_rng(settings["seed"])
    queries = np.sort(rng.choice(scores.shape[0], min(settings["silhouette_sample_size"], scores.shape[0]), replace=False))
    candidates, partitions = [], {}
    distinct = 1 + int(np.count_nonzero(np.any(np.diff(ordered, axis=0) != 0, axis=1)))
    if pca["total_centered_sum_squares"] > 0 and scores.shape[0] > 2:
        for k in range(settings["k_min"], min(settings["k_max"], scores.shape[0] - 1, distinct) + 1):
            runs = []
            for restart in range(settings["n_init"]):
                seed = settings["seed"] + 1009 * restart + 9176 * k
                try:
                    _, labels = kmeans2(ordered, k, iter=settings["kmeans_iterations"],
                                        minit="++", missing="raise", seed=seed)
                except (ClusterError, ValueError):
                    continue
                if np.unique(labels).size != k:
                    continue
                labels, centres = _ordered_partition(ordered, labels)
                inertia = float(np.sum((ordered - centres[labels]) ** 2))
                runs.append((inertia, labels, centres, seed))
            if not runs:
                candidates.append({"k": k, "status": "no_valid_partition", "valid_restarts": 0})
                continue
            runs.sort(key=lambda value: value[0])
            inertia, labels, centres, seed = runs[0]
            agreement = [adjusted_rand_index(a[1], b[1]) for a, b in combinations(runs, 2)]
            stability = float(np.mean(agreement)) if agreement else 1.
            counts = np.bincount(labels, minlength=k)
            score = silhouette(ordered, labels, queries)
            item = {
                "k": k, "status": "complete", "inertia": inertia,
                "silhouette": score, "silhouette_query_rows": int(queries.size),
                "valid_restarts": len(runs), "best_seed": seed,
                "mean_pairwise_adjusted_rand": stability,
                "minimum_pairwise_adjusted_rand": float(min(agreement)) if agreement else 1.,
                "cluster_sizes": counts.tolist(), "minimum_cluster_size": int(counts.min()),
            }
            candidates.append(item)
            restored = np.empty(labels.shape, dtype=np.int32)
            restored[canonical] = labels
            partitions[k] = (restored, centres)
    valid = [value for value in candidates if value["status"] == "complete"]
    reason = []
    if valid:
        best = max(valid, key=lambda value: (value["silhouette"], -value["k"]))
        candidate_k = best["k"]
        candidate_labels, centres = partitions[candidate_k]
        if best["silhouette"] < settings["silhouette_threshold"]:
            reason.append("silhouette_below_threshold")
        if best["mean_pairwise_adjusted_rand"] < settings["stability_threshold"]:
            reason.append("restart_stability_below_threshold")
        if best["valid_restarts"] < min(2, settings["n_init"]):
            reason.append("too_few_valid_restarts")
        if best["minimum_cluster_size"] < max(2, int(np.ceil(settings["min_cluster_fraction"] * features.shape[0]))):
            reason.append("candidate_contains_small_cluster")
    else:
        candidate_k = 1
        candidate_labels = np.zeros(features.shape[0], dtype=np.int32)
        centres = scores.mean(axis=0, keepdims=True)
        reason.append("no_nontrivial_partition")
    suggested_k = candidate_k if not reason else 1
    labels = candidate_labels.copy() if suggested_k > 1 else np.zeros(features.shape[0], dtype=np.int32)
    centroid_features = np.asarray([features[candidate_labels == k].mean(axis=0) for k in range(candidate_k)])
    analysis = {
        "schema_version": "fakect.ct-delta-clustering/1",
        "rows": int(features.shape[0]), "features": int(features.shape[1]),
        "settings": settings,
        "normalization": "mean centering only; input physical/baseline normalization retained",
        "pca": pca,
        "clustering": {
            "space": "retained principal components, without whitening",
            "candidate_best_k": candidate_k, "suggested_k": suggested_k,
            "evidence_passed": not reason, "fallback_reasons": reason,
            "candidates": candidates,
            "candidate_centers_pc": centres.tolist(),
            "candidate_cluster_sizes": np.bincount(candidate_labels, minlength=candidate_k).tolist(),
            "interpretation": "Exploratory K-means partition; not proof of discrete morphological types or a true cluster count.",
        },
    }
    return {"analysis": analysis, "scores": scores, "labels": labels,
            "candidate_labels": candidate_labels, "components": components,
            "mean": mean, "cluster_centers": centres,
            "candidate_centroid_features": centroid_features}
