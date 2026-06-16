# Mixture of Probabilistic PCA (MPPCA) implementation (module-only)

import numpy as np
from scipy.special import logsumexp
from kmeans import KMeans

class MPPCA:
    def __init__(self, n_components, latent_dim, max_iter=50, tol=1e-4, random_state=None):
        self.M = n_components
        self.q = latent_dim
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.log_likelihoods = []

        self.mus = None
        self.Ws = None
        self.sigmas2 = None
        self.pis = None

    def _log_pdf(self, X, mu, W, sigma2):
        d = X.shape[1]
        C = W @ W.T + sigma2 * np.eye(d) + 1e-6 * np.eye(d)
        L = np.linalg.cholesky(C)
        log_det = 2 * np.sum(np.log(np.diag(L)))
        diff = (X - mu).T
        y = np.linalg.solve(L, diff)
        mahalanobis = np.sum(y**2, axis=0)
        const = d * np.log(2 * np.pi)
        return -0.5 * (const + log_det + mahalanobis)

    def _compute_log_rho(self, X):
        N = X.shape[0]
        log_rho = np.zeros((N, self.M))
        for i in range(self.M):
            log_prob = self._log_pdf(X, self.mus[i], self.Ws[i], self.sigmas2[i])
            log_rho[:, i] = np.log(self.pis[i] + 1e-10) + log_prob
        return log_rho

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        N, d = X.shape
        M, q = self.M, self.q

        # Initialization using KMeans
        kmeans = KMeans(n_clusters=M, max_iter=50, random_state=self.random_state)
        labels = kmeans.fit_predict(X)

        self.mus = kmeans.centroids.copy()
        self.pis = np.bincount(labels, minlength=M) / N
        self.Ws = [np.random.randn(d, q) * 0.1 for _ in range(M)]
        self.sigmas2 = np.ones(M) * 0.1

        # Local PCA Init
        for i in range(M):
            X_k = X[labels == i]
            if len(X_k) > d:
                cov = np.cov(X_k.T)
                vals, vecs = np.linalg.eigh(cov)
                idx = np.argsort(vals)[::-1]
                vals = vals[idx]
                vecs = vecs[:, idx]
                sigma2 = np.mean(vals[q:]) if d > q else 1e-3
                self.sigmas2[i] = max(sigma2, 1e-6)
                U_q = vecs[:, :q]
                Lambda_q = np.diag(vals[:q])
                diff = Lambda_q - self.sigmas2[i] * np.eye(q)
                self.Ws[i] = U_q @ np.sqrt(np.maximum(diff, 1e-10))

        # EM Loop
        for it in range(self.max_iter):
            log_rho = self._compute_log_rho(X)
            log_prob_norm = logsumexp(log_rho, axis=1, keepdims=True)
            log_likelihood = np.sum(log_prob_norm)
            self.log_likelihoods.append(log_likelihood)

            R = np.exp(log_rho - log_prob_norm)

            if it > 0 and abs(log_likelihood - self.log_likelihoods[-2]) < self.tol:
                break

            Nk = np.sum(R, axis=0) + 1e-10
            self.pis = Nk / N

            for i in range(M):
                mu_new = np.sum(R[:, i:i+1] * X, axis=0) / Nk[i]
                self.mus[i] = mu_new

                X_centered = X - mu_new
                sqrt_R = np.sqrt(R[:, i:i+1])
                X_weighted = X_centered * sqrt_R
                S = X_weighted.T @ X_weighted / Nk[i]

                vals, vecs = np.linalg.eigh(S)
                idx = np.argsort(vals)[::-1]
                vals = vals[idx]
                vecs = vecs[:, idx]

                sigma2_new = np.mean(vals[q:]) if d > q else 1e-3
                self.sigmas2[i] = max(sigma2_new, 1e-6)

                U_q = vecs[:, :q]
                Lambda_q = np.diag(vals[:q])
                diff = Lambda_q - self.sigmas2[i] * np.eye(q)
                self.Ws[i] = U_q @ np.sqrt(np.maximum(diff, 1e-10))

    def predict(self, X):
        log_rho = self._compute_log_rho(X)
        return np.argmax(log_rho, axis=1)

    def score(self, X):
        log_rho = self._compute_log_rho(X)
        return np.sum(logsumexp(log_rho, axis=1))

    def get_bic(self, X):
        n_samples, d = X.shape
        n_params = self.M * d + (self.M - 1) + self.M + self.M * (d * self.q - 0.5 * self.q * (self.q - 1))
        log_lik = self.score(X)
        return n_params * np.log(n_samples) - 2 * log_lik