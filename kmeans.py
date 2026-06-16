#!/usr/bin/env python3
# Lightweight K-Means implementation used for MPPCA initialization

import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X):
        """Fit KMeans and return labels. Centroids are stored in self.centroids."""
        if self.random_state is not None:
            np.random.seed(self.random_state)

        indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[indices].astype(float)

        for _ in range(self.max_iter):
            # Efficient squared-distance computation
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            C_sq = np.sum(self.centroids**2, axis=1)
            dot_product = X @ self.centroids.T
            dists = np.sqrt(np.maximum(X_sq + C_sq - 2 * dot_product, 0))
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                mask = (labels == k)
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    new_centroids[k] = X[np.random.choice(X.shape[0])]

            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids

        return labels
