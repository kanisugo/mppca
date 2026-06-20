import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp

# ==========================================
# 1. K-Means Implementation
# ==========================================
class KMeans:
    """K-Means clustering from scratch."""
    def __init__(self, n_clusters, max_iter=100, tol=1e-4, random_state=42):
        self.K = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        # Random initialization
        indices = np.random.choice(X.shape[0], self.K, replace=False)
        self.centroids = X[indices]

        for i in range(self.max_iter):
            # Scalable Distance Calculation: ||x - c||^2
            X_sq = np.sum(X**2, axis=1, keepdims=True)
            C_sq = np.sum(self.centroids**2, axis=1)
            dot_product = X @ self.centroids.T
            
            # Avoid negative zeros due to float precision
            dists = np.sqrt(np.maximum(X_sq + C_sq - 2 * dot_product, 0))
            labels = np.argmin(dists, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.K):
                mask = (labels == k)
                if np.any(mask):
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # Re-initialize empty cluster to random point
                    new_centroids[k] = X[np.random.choice(X.shape[0])]
            
            # Convergence check
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
            
        return labels

# ==========================================
# 2. MPPCA Implementation
# ==========================================
class MPPCA:
    """Mixture of Probabilistic PCA."""
    def __init__(self, n_components, latent_dim, max_iter=50, tol=1e-4, random_state=42):
        self.M = n_components
        self.q = latent_dim
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.log_likelihoods = []
        
        # Model Parameters
        self.mus = None      # Means (M, d)
        self.Ws = None       # Weights (M, d, q)
        self.sigmas2 = None  # Noise variances (M,)
        self.pis = None      # Mixing coeffs (M,)
        self.noise_ratios = np.zeros(n_components) # For analysis

    def _log_pdf(self, X, mu, W, sigma2):
        """Calculates log-probability density for a single cluster."""
        d = X.shape[1]
        # Full Covariance Reconstruction: C = WW^T + sigma^2*I
        C = W @ W.T + sigma2 * np.eye(d) + 1e-6 * np.eye(d) # Jitter for stability
        
        # Cholesky Decomposition for stable inverse/determinant
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            # Fallback for rare singular matrices
            L = np.linalg.cholesky(C + 1e-5 * np.eye(d))

        log_det = 2 * np.sum(np.log(np.diag(L)))
        diff = (X - mu).T
        y = np.linalg.solve(L, diff)
        mahalanobis = np.sum(y**2, axis=0)
        
        const = d * np.log(2 * np.pi)
        return -0.5 * (const + log_det + mahalanobis)

    def _compute_log_rho(self, X):
        """Helper: Computes unnormalized log-responsibilities (N x M)."""
        N = X.shape[0]
        log_rho = np.zeros((N, self.M))
        for i in range(self.M):
            log_prob = self._log_pdf(X, self.mus[i], self.Ws[i], self.sigmas2[i])
            log_rho[:, i] = np.log(self.pis[i] + 1e-10) + log_prob
        return log_rho

    def fit(self, X):
        np.random.seed(self.random_state)
        N, d = X.shape
        M, q = self.M, self.q
        
        # --- 1. Initialization (Using KMeans) ---
        print("Initializing with K-Means...")
        kmeans = KMeans(n_clusters=M, max_iter=50, random_state=self.random_state)
        labels = kmeans.fit_predict(X)
        
        self.mus = kmeans.centroids
        self.pis = np.bincount(labels, minlength=M) / N
        self.Ws = [np.random.randn(d, q) * 0.1 for _ in range(M)]
        self.sigmas2 = np.ones(M) * 0.1
        
        # Local PCA Initialization
        for i in range(M):
            X_k = X[labels == i]
            # Fallback if cluster is too small
            if len(X_k) <= d:
                continue
                
            cov = np.cov(X_k.T)
            vals, vecs = np.linalg.eigh(cov)
            # Sort descending
            idx = np.argsort(vals)[::-1]
            vals, vecs = vals[idx], vecs[:, idx]
            
            sigma2 = np.mean(vals[q:]) if d > q else 1e-3
            self.sigmas2[i] = max(sigma2, 1e-6)
            
            U_q = vecs[:, :q]
            Lambda_q = np.diag(vals[:q])
            diff = Lambda_q - self.sigmas2[i] * np.eye(q)
            self.Ws[i] = U_q @ np.sqrt(np.maximum(diff, 1e-10))

        # --- 2. EM Loop ---
        for it in range(self.max_iter):
            # E-Step: Calculate Responsibilities
            log_rho = self._compute_log_rho(X)
            log_prob_norm = logsumexp(log_rho, axis=1, keepdims=True)
            log_likelihood = np.sum(log_prob_norm)
            self.log_likelihoods.append(log_likelihood)
            
            # Normalized Responsibilities (N, M)
            R = np.exp(log_rho - log_prob_norm)
            
            # Convergence Check
            if it > 0 and abs(log_likelihood - self.log_likelihoods[-2]) < self.tol:
                print(f"Converged at iteration {it}")
                break
            
            # M-Step: Update Parameters
            Nk = np.sum(R, axis=0) + 1e-10
            self.pis = Nk / N
            
            for i in range(M):
                # Update Mean
                self.mus[i] = np.sum(R[:, i:i+1] * X, axis=0) / Nk[i]
                
                # Update Weighted Covariance S
                X_centered = X - self.mus[i]
                X_weighted = X_centered * np.sqrt(R[:, i:i+1])
                S = X_weighted.T @ X_weighted / Nk[i]
                
                # Eigen Decomposition of S
                vals, vecs = np.linalg.eigh(S)
                idx = np.argsort(vals)[::-1]
                vals, vecs = vals[idx], vecs[:, idx]

                # Update Noise & Weights (Tipping & Bishop)
                sigma2_new = np.mean(vals[q:]) if d > q else 1e-3
                self.sigmas2[i] = max(sigma2_new, 1e-6)
                
                U_q = vecs[:, :q]
                Lambda_q = np.diag(vals[:q])
                diff = Lambda_q - self.sigmas2[i] * np.eye(q)
                self.Ws[i] = U_q @ np.sqrt(np.maximum(diff, 1e-10))

                # Track Noise Ratio for visualization
                self.noise_ratios[i] = (np.sum(vals[q:]) / np.sum(vals)) if np.sum(vals) > 1e-9 else 0

    def predict(self, X):
        """Returns the cluster index for each sample."""
        log_rho = self._compute_log_rho(X)
        return np.argmax(log_rho, axis=1)

    def score(self, X):
        """Returns the total log-likelihood of the data."""
        log_rho = self._compute_log_rho(X)
        return np.sum(logsumexp(log_rho, axis=1))

    def get_bic(self, X):
        """Bayesian Information Criterion (Lower is better)."""
        N, d = X.shape
        # Params: Means + Mixing + Weights + Noise
        # Weights params = M * (d*q - 0.5*q*(q-1)) due to rotation invariance
        n_params = self.M * d + (self.M - 1) + self.M + self.M * (d * self.q - 0.5 * self.q * (self.q - 1))
        return n_params * np.log(N) - 2 * self.score(X)

# ==========================================
# 3. Visualization Plots
# ==========================================
def plot_results(df, model, k):
    """Encapsulates all plotting logic."""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Clusters Map
    cmap = plt.get_cmap('tab20', k)
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                          c=df['cluster'], cmap=cmap, s=2, alpha=0.6,
                          vmin=-0.5, vmax=k-0.5)
    cbar = plt.colorbar(scatter, ticks=range(k))
    cbar.set_label('Cluster ID')
    plt.title(f'MPPCA Cluster Map (M={k})')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.show()

    # 2. Noise Ratios
    plt.figure(figsize=(8, 5))
    plt.bar(range(model.M), model.noise_ratios, color='crimson', alpha=0.7, edgecolor='black')
    plt.title('Noise Variance Ratio per Cluster')
    plt.xlabel('Cluster ID')
    plt.ylabel('Ratio (Noise / Total Variance)')
    plt.xticks(range(model.M))
    plt.show()

    # 3. Log-Likelihood
    plt.figure(figsize=(8, 5))
    plt.plot(model.log_likelihoods, marker='o', color='darkgreen')
    plt.title('Log-Likelihood Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('Log-Likelihood')
    plt.show()

    # Plot 4: Magnetic Intensity
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(df['longitude'], df['latitude'], c=df['magnetics_final_microlevelled'], cmap='jet', s=2, alpha=1.0)
    plt.colorbar(sc, label='Magnetic Intensity (nT)')
    plt.title('Magnetic Intensity Map', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    # Setting aspect ratio to 'equal' ensures the map isn't distorted
    plt.axis('equal') 
    plt.grid(True, linestyle='--', alpha=0.5)
    
    
    # Plot 5: Altitude Contour
    plt.figure(figsize=(12, 10))
    cntr = plt.tricontourf(df['longitude'], df['latitude'], df['laser_alt'], levels=20, cmap='terrain')
    plt.colorbar(cntr, label='Laser Altitude (m)')
    plt.title('Laser Altitude Contour Map', fontsize=16)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.figure(figsize=(10, 6))
# ==========================================
# 4. Main Execution
# ==========================================
if __name__ == "__main__":
    # Settings
    CSV_PATH = "/Users/kanishksugotra/Downloads/magnetic.csv"
    MAG_WEIGHT = 5.0
    M_CLUSTERS = 12
    Q_LATENT = 3
    
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"Error: File not found at {CSV_PATH}")
        exit()

    # Preprocessing
    features = ['latitude', 'longitude', 'laser_alt', 'magnetics_final_microlevelled']
    data = df[features].values
    
    # Split
    X_train_raw, X_test_raw = train_test_split(data, test_size=0.2, random_state=42)

    # Scale (Fit on Train, Transform on Test)
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    
    X_train = (X_train_raw - mean) / std
    X_test = (X_test_raw - mean) / std
    X_full_scaled = (data - mean) / std
    
    # Apply Feature Weighting
    X_train[:, 3] *= MAG_WEIGHT
    X_test[:, 3]  *= MAG_WEIGHT
    X_full_scaled[:, 3] *= MAG_WEIGHT

    # Train
    print(f"Training MPPCA (M={M_CLUSTERS}, q={Q_LATENT})...")
    model = MPPCA(n_components=M_CLUSTERS, latent_dim=Q_LATENT, max_iter=50)
    model.fit(X_train)

    # Evaluate
    train_ll = model.score(X_train) / len(X_train)
    test_ll = model.score(X_test) / len(X_test)
    bic = model.get_bic(X_test)

    print("-" * 30)
    print("RESULTS")
    print("-" * 30)
    print(f"Train Log-Likelihood: {train_ll:.4f}")
    print(f"Test Log-Likelihood:  {test_ll:.4f}")
    print(f"Test BIC Score:       {bic:.0f}")
    print("-" * 30)

    # Visualize
    df['cluster'] = model.predict(X_full_scaled)
    plot_results(df, model, M_CLUSTERS)