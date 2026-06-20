#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 18:31:24 2026

@author: kanishksugotra
"""
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp

# ==========================================
# 1. K-Means Implementation
# ==========================================
class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=None):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[indices]

        for i in range(self.max_iter):
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

# ==========================================
# 2. MPPCA Implementation
# ==========================================
class MPPCA:
    def __init__(self, n_components, latent_dim, max_iter=50, tol=1e-4):
        self.M = n_components
        self.q = latent_dim
        self.max_iter = max_iter
        self.tol = tol
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

    def fit(self, X):
        N, d = X.shape
        M = self.M
        q = self.q
        
        # Initialization
        kmeans = KMeans(n_clusters=M, max_iter=50) # Removed seed for randomness in n_init
        labels = kmeans.fit_predict(X)
        
        self.mus = kmeans.centroids
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
            log_rho = np.zeros((N, M))
            for i in range(M):
                log_rho[:, i] = np.log(self.pis[i] + 1e-10) + self._log_pdf(X, self.mus[i], self.Ws[i], self.sigmas2[i])
            
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

    def score(self, X):
        N = X.shape[0]
        log_rho = np.zeros((N, self.M))
        for i in range(self.M):
            log_rho[:, i] = np.log(self.pis[i] + 1e-10) + self._log_pdf(X, self.mus[i], self.Ws[i], self.sigmas2[i])
        return np.sum(logsumexp(log_rho, axis=1))

    def get_bic(self, X):
        n_samples, d = X.shape
        # Degrees of Freedom formula for Isotropic MPPCA
        n_params = self.M * d + self.M * (d * self.q - 0.5 * self.q * (self.q - 1)) + self.M + (self.M - 1)
        log_lik = self.score(X)
        return n_params * np.log(n_samples) - 2 * log_lik

# ==========================================
# 3. Tuning Execution
# ==========================================
if __name__ == "__main__":
    try:
        # REPLACE WITH YOUR PATH
        df = pd.read_csv("/to/file/path/")
    except FileNotFoundError:
        print("Error: magnetic.csv not found.")
        exit()

    features = ['latitude', 'longitude', 'altitude', 'magnetics']
    data = df[features].values

    # 1. Scale Full Dataset
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_scaled = (data - mean) / std
    
    # Apply Magnetic Weight
    MAG_WEIGHT = 5.0
    data_scaled[:, 3] *= MAG_WEIGHT

    # 2. Subsample 5000 points for Tuning
    print("Subsampling 5000 points for 5-Fold Cross-Validation...")
    np.random.seed(42)
    subset_idx = np.random.choice(len(data_scaled), 5000, replace=False)
    X_subset = data_scaled[subset_idx]

    # 3. Setup Cross Validation
    K_FOLDS = 5
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    # Grid Search Parameters
    M_values = [i for i in range(3, 20)]
    q_values = [1, 2, 3]
    results = []

    print(f"Starting Grid Search with {K_FOLDS}-Fold CV...")
    print(f"Total fits to run: {len(M_values) * len(q_values) * K_FOLDS}")
    
    for M in M_values:
        for q in q_values:
            fold_test_lls = []
            fold_bics = []
            
            # --- Cross Validation Loop ---
            for train_index, val_index in kf.split(X_subset):
                X_train_fold = X_subset[train_index]
                X_val_fold = X_subset[val_index]
                
                # Fit Model
                model = MPPCA(n_components=M, latent_dim=q, max_iter=30) # 30 iter is enough for tuning
                try:
                    model.fit(X_train_fold)
                    
                    # 1. Validation Log-Likelihood (Generalization)
                    val_ll = model.score(X_val_fold) / len(X_val_fold)
                    fold_test_lls.append(val_ll)
                    
                    # 2. Training BIC (Model Selection)
                    bic = model.get_bic(X_train_fold)
                    fold_bics.append(bic)
                    
                except np.linalg.LinAlgError:
                    # Capture failures (rare singular matrix)
                    fold_test_lls.append(-np.inf)
                    fold_bics.append(np.inf)

            # Aggregate Results
            avg_test_ll = np.mean(fold_test_lls)
            std_test_ll = np.std(fold_test_lls)
            avg_bic = np.mean(fold_bics)
            
            print(f"M={M}, q={q} | Avg Val LL: {avg_test_ll:.2f} (+/- {std_test_ll:.2f}) | Avg BIC: {avg_bic:.0f}")
            results.append({
                'M': M, 
                'q': q, 
                'Avg Val LL': avg_test_ll, 
                'Std Val LL': std_test_ll,
                'Avg BIC': avg_bic
            })

    # ==========================================
    # 4. Plotting with Error Bars
    # ==========================================
    res_df = pd.DataFrame(results)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: BIC (Lower is Better)
    for q in q_values:
        subset = res_df[res_df['q'] == q]
        ax1.plot(subset['M'], subset['Avg BIC'], marker='o', label=f'q={q}')
    ax1.set_title(f'Model Selection: Average BIC ({K_FOLDS}-Fold CV)')
    ax1.set_xlabel('Number of Clusters (M)')
    ax1.set_ylabel('Average BIC (Lower is Better)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Cross-Validated Log-Likelihood (Higher is Better)
    # Shaded Regions for Standard Deviation
    for q in q_values:
        subset = res_df[res_df['q'] == q]
        ax2.plot(subset['M'], subset['Avg Val LL'], marker='s', linestyle='--', label=f'q={q}')
        ax2.fill_between(subset['M'], 
                         subset['Avg Val LL'] - subset['Std Val LL'], 
                         subset['Avg Val LL'] + subset['Std Val LL'], 
                         alpha=0.2)
        
    ax2.set_title(f'Generalization: {K_FOLDS}-Fold CV Log-Likelihood')
    ax2.set_xlabel('Number of Clusters (M)')
    ax2.set_ylabel('Avg Log-Likelihood per Sample')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    # Best Model Selection (based on Highest Validation LL)
    best_cfg = res_df.loc[res_df['Avg Val LL'].idxmax()]
    print("\nOptimal Configuration (Best Generalization):")
    print(best_cfg)