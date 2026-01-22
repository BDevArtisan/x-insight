"""
Advanced clustering algorithms.
Includes HDBSCAN, Gaussian Mixture Models, and Spectral Clustering.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


class HDBSCANClustering:
    """HDBSCAN clustering wrapper."""
    
    def __init__(self, min_cluster_size=15, min_samples=5, metric='euclidean'):
        """
        Initialize HDBSCAN.
        
        Parameters:
        -----------
        min_cluster_size : int
            Minimum cluster size
        min_samples : int
            Minimum samples
        metric : str
            Distance metric
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
        self.labels_ = None
        self.probabilities_ = None
        
    def fit(self, X):
        """Fit HDBSCAN model."""
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.probabilities_ = self.model.probabilities_
        return self
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_
    
    def get_probabilities(self):
        """Get cluster membership probabilities."""
        return self.probabilities_


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
