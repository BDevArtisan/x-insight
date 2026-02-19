"""
Advanced clustering algorithms.
Includes Deep Clustering (DEC, IDEC, DCN), Gaussian Mixture Models, Spectral Clustering,
HDBSCAN (auto-k, density-based), and GMM with automatic BIC-based k selection.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from clustpy.deep import DEC, IDEC, DCN


class DECClustering:
    """Deep Embedded Clustering (DEC) wrapper."""
    
    def __init__(self, n_clusters=3, batch_size=256, pretrain_epochs=50, clustering_epochs=100, random_state=42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        
    def fit(self, X):
        self.model = DEC(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            pretrain_epochs=self.pretrain_epochs,
            clustering_epochs=self.clustering_epochs,
            random_state=self.random_state
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)


class IDECClustering:
    """Improved Deep Embedded Clustering (IDEC) wrapper."""
    
    def __init__(self, n_clusters=3, batch_size=256, pretrain_epochs=50, clustering_epochs=100, random_state=42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        
    def fit(self, X):
        self.model = IDEC(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            pretrain_epochs=self.pretrain_epochs,
            clustering_epochs=self.clustering_epochs,
            random_state=self.random_state
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)


class DCNClustering:
    """Deep Clustering Network (DCN) wrapper."""
    
    def __init__(self, n_clusters=3, batch_size=256, pretrain_epochs=50, clustering_epochs=100, random_state=42):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.pretrain_epochs = pretrain_epochs
        self.clustering_epochs = clustering_epochs
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        
    def fit(self, X):
        self.model = DCN(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            pretrain_epochs=self.pretrain_epochs,
            clustering_epochs=self.clustering_epochs,
            random_state=self.random_state
        )
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("Model not fitted yet")
        return self.model.predict(X)


class GMMClustering:
    """Gaussian Mixture Model clustering wrapper."""
    
    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        """
        Initialize GMM.
        
        Parameters:
        -----------
        n_components : int
            Number of mixture components
        covariance_type : str
            Covariance type (full, tied, diag, spherical)
        random_state : int
            Random seed
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=random_state
        )
        self.labels_ = None
        self.probabilities_ = None
        
    def fit(self, X):
        """Fit GMM model."""
        self.model.fit(X)
        self.labels_ = self.model.predict(X)
        self.probabilities_ = self.model.predict_proba(X)
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict cluster probabilities for new data."""
        return self.model.predict_proba(X)
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_
    
    def get_probabilities(self):
        """Get cluster membership probabilities."""
        return self.probabilities_
    
    def get_means(self):
        """Get cluster means."""
        return self.model.means_
    
    def get_covariances(self):
        """Get cluster covariances."""
        return self.model.covariances_


class SpectralClusteringWrapper:
    """Spectral Clustering wrapper."""
    
    def __init__(self, n_clusters=3, affinity='rbf', random_state=42):
        """
        Initialize Spectral Clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        affinity : str
            Affinity matrix type (rbf, nearest_neighbors, precomputed)
        random_state : int
            Random seed
        """
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.random_state = random_state
        self.model = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            random_state=random_state
        )
        self.labels_ = None
        
    def fit(self, X):
        """Fit Spectral Clustering model."""
        self.labels_ = self.model.fit_predict(X)
        return self
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_


class HDBSCANClustering:
    """
    HDBSCAN clustering wrapper — truly automatic k selection.

    Does NOT require specifying the number of clusters.
    Determines cluster structure from data density.
    Noise points are labelled -1.
    """

    def __init__(self, min_cluster_size=5, min_samples=None, metric='euclidean'):
        """
        Parameters
        ----------
        min_cluster_size : int
            Minimum number of patients to form a cluster.
            Smaller values → more (smaller) clusters.
        min_samples : int or None
            Controls how conservative the clustering is.
            Defaults to min_cluster_size when None.
        metric : str
            Distance metric (euclidean recommended for scaled data).
        """
        import hdbscan
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            prediction_data=True
        )
        self.labels_ = None
        self.n_clusters_ = None

    def fit(self, X):
        """Fit HDBSCAN model."""
        self.labels_ = self.model.fit_predict(X)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        return self

    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """Approximate prediction for new points (soft membership)."""
        import hdbscan
        labels, _ = hdbscan.approximate_predict(self.model, X)
        return labels

    def get_cluster_centers(self):
        """Return None — HDBSCAN does not have centroids."""
        return None


class GMMAutoClustering:
    """
    Gaussian Mixture Model with automatic k selection via BIC.

    Tries k = 2 .. max_k and picks the k that minimises the
    Bayesian Information Criterion (BIC). No k needs to be
    specified by the user.
    """

    def __init__(self, max_k=10, covariance_type='full', random_state=42):
        """
        Parameters
        ----------
        max_k : int
            Maximum number of components to evaluate.
        covariance_type : str
            GMM covariance type (full, tied, diag, spherical).
        random_state : int
            Random seed.
        """
        self.max_k = max_k
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.best_k_ = None
        self.bic_scores_ = None
        self.model = None
        self.labels_ = None
        self.probabilities_ = None

    def fit(self, X):
        """Fit GMMs for k=2..max_k and select the best by BIC."""
        bic_scores = []
        models = []
        k_values = range(2, self.max_k + 1)

        for k in k_values:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                random_state=self.random_state
            )
            gmm.fit(X)
            bic_scores.append(gmm.bic(X))
            models.append(gmm)

        best_idx = int(np.argmin(bic_scores))
        self.best_k_ = list(k_values)[best_idx]
        self.bic_scores_ = dict(zip(k_values, bic_scores))
        self.model = models[best_idx]
        self.labels_ = self.model.predict(X)
        self.probabilities_ = self.model.predict_proba(X)
        return self

    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_

    def predict(self, X):
        """Predict cluster labels for new data."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict cluster probabilities for new data."""
        return self.model.predict_proba(X)

    def get_probabilities(self):
        """Get cluster membership probabilities."""
        return self.probabilities_
