"""
Traditional clustering algorithms.
Includes K-Means, Hierarchical, and K-Medoids.
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist


class KMeansClustering:
    """K-Means clustering wrapper."""
    
    def __init__(self, n_clusters=3, random_state=42):
        """
        Initialize K-Means.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        random_state : int
            Random seed
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.labels_ = None
        self.cluster_centers_ = None
        
    def fit(self, X):
        """Fit K-Means model."""
        self.model.fit(X)
        self.labels_ = self.model.labels_
        self.cluster_centers_ = self.model.cluster_centers_
        return self
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        return self.model.predict(X)
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_
    
    def get_cluster_centers(self):
        """Get cluster centroids."""
        return self.cluster_centers_


class HierarchicalClustering:
    """Hierarchical clustering wrapper."""
    
    def __init__(self, n_clusters=3, linkage_method='ward'):
        """
        Initialize Hierarchical clustering.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        linkage_method : str
            Linkage method (ward, complete, average, single)
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
        self.labels_ = None
        self.linkage_matrix_ = None
        
    def fit(self, X):
        """Fit hierarchical clustering."""
        self.labels_ = self.model.fit_predict(X)
        # Compute linkage matrix for dendrogram
        self.linkage_matrix_ = linkage(X, method=self.linkage_method)
        return self
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_
    
    def get_dendrogram_data(self):
        """Get linkage matrix for dendrogram visualization."""
        return self.linkage_matrix_


class KMedoidsClustering:
    """K-Medoids clustering implementation."""
    
    def __init__(self, n_clusters=3, max_iter=100, random_state=42):
        """
        Initialize K-Medoids.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum iterations
        random_state : int
            Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.labels_ = None
        self.medoid_indices_ = None
        self.medoids_ = None
        
    def fit(self, X):
        """Fit K-Medoids model."""
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]

        medoid_indices = rng.choice(n_samples, self.n_clusters, replace=False)
        
        for _ in range(self.max_iter):
            # Assign labels based on distance to medoids
            distances = cdist(X, X[medoid_indices], metric='euclidean')
            labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoid_indices = np.zeros(self.n_clusters, dtype=int)
            for k in range(self.n_clusters):
                cluster_indices = np.where(labels == k)[0]
                if len(cluster_indices) > 0:
                    # Find point that minimizes total distance within cluster
                    cluster_distances = cdist(X[cluster_indices], X[cluster_indices])
                    total_distances = cluster_distances.sum(axis=1)
                    new_medoid_indices[k] = cluster_indices[np.argmin(total_distances)]
                else:
                    new_medoid_indices[k] = medoid_indices[k]
            
            # Check for convergence
            if np.array_equal(medoid_indices, new_medoid_indices):
                break
            medoid_indices = new_medoid_indices
        
        self.medoid_indices_ = medoid_indices
        self.medoids_ = X[medoid_indices]
        self.labels_ = labels
        return self
    
    def fit_predict(self, X):
        """Fit and predict in one step."""
        self.fit(X)
        return self.labels_
    
    def predict(self, X):
        """Predict cluster labels for new data."""
        if self.medoids_ is None:
            raise ValueError("Model must be fitted before predict")
        distances = cdist(X, self.medoids_, metric='euclidean')
        return np.argmin(distances, axis=1)
    
    def get_medoids(self):
        """Get medoid points."""
        return self.medoids_
    
    def get_medoid_indices(self):
        """Get indices of medoid points."""
        return self.medoid_indices_


def determine_optimal_k(X, k_range=range(2, 11), method='kmeans', metric='multi'):
    """
    Determine optimal number of clusters with multi-metric voting.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    k_range : range
        Range of k values to test
    method : str
        Clustering method (kmeans, hierarchical, kmedoids, gmm, spectral)
    metric : str
        Metric to use:
        - 'multi': Combined score (silhouette + inverse DB + CH normalized)
        - 'silhouette': Silhouette coefficient (higher is better)
        - 'davies_bouldin': Davies-Bouldin index (lower is better)
        - 'calinski_harabasz': Calinski-Harabasz index (higher is better)
        - 'bic': BIC for GMM only (lower is better)
        
    Returns:
    --------
    results : dict
        Dictionary with:
        - 'k': list of k values
        - 'scores': list of metric scores
        - 'silhouette': list of silhouette scores (if multi)
        - 'davies_bouldin': list of DB scores (if multi)
        - 'calinski_harabasz': list of CH scores (if multi)
        - 'quality': quality level ('excellent', 'good', 'moderate', 'weak')
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    results = {'k': [], 'scores': []}
    
    # For multi-metric, store individual metrics
    if metric == 'multi':
        results['silhouette'] = []
        results['davies_bouldin'] = []
        results['calinski_harabasz'] = []
    
    for k in k_range:
        # Import here to avoid circular dependency
        from .advanced import GMMClustering, SpectralClusteringWrapper
        
        if method == 'kmeans':
            model = KMeansClustering(n_clusters=k)
        elif method == 'hierarchical':
            model = HierarchicalClustering(n_clusters=k)
        elif method == 'kmedoids':
            model = KMedoidsClustering(n_clusters=k)
        elif method == 'gmm':
            model = GMMClustering(n_components=k)
        elif method == 'spectral':
            model = SpectralClusteringWrapper(n_clusters=k)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        labels = model.fit_predict(X)
        
        # Special case: BIC for GMM
        if metric == 'bic' and method == 'gmm':
            score = -model.model.bic(X)  # Negative because we want to maximize
            results['k'].append(k)
            results['scores'].append(score)
            continue
        
        # Compute metrics
        if metric == 'silhouette':
            score = silhouette_score(X, labels)
        elif metric == 'davies_bouldin':
            score = -davies_bouldin_score(X, labels)  # Negative to maximize
        elif metric == 'calinski_harabasz':
            score = calinski_harabasz_score(X, labels)
        elif metric == 'multi':
            # Compute all three metrics
            sil = silhouette_score(X, labels)
            db = davies_bouldin_score(X, labels)
            ch = calinski_harabasz_score(X, labels)
            
            results['silhouette'].append(sil)
            results['davies_bouldin'].append(db)
            results['calinski_harabasz'].append(ch)
            
            # Combined score (will normalize later)
            score = 0  # Placeholder
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        results['k'].append(k)
        results['scores'].append(score)
    
    # For multi-metric, compute combined normalized score
    if metric == 'multi':
        sil_scores = np.array(results['silhouette'])
        db_scores = np.array(results['davies_bouldin'])
        ch_scores = np.array(results['calinski_harabasz'])
        
        # Normalize to [0, 1]
        def normalize(arr):
            min_val, max_val = arr.min(), arr.max()
            if max_val - min_val == 0:
                return np.ones_like(arr)
            return (arr - min_val) / (max_val - min_val)
        
        sil_norm = normalize(sil_scores)
        db_norm = normalize(-db_scores)  # Invert DB (lower is better)
        ch_norm = normalize(ch_scores)
        
        # Combined score: weighted average
        combined = 0.5 * sil_norm + 0.25 * db_norm + 0.25 * ch_norm
        results['scores'] = combined.tolist()
        
        # Assess quality based on best silhouette
        best_sil = max(sil_scores)
        if best_sil >= 0.7:
            results['quality'] = 'excellent'
        elif best_sil >= 0.5:
            results['quality'] = 'good'
        elif best_sil >= 0.25:
            results['quality'] = 'moderate'
        else:
            results['quality'] = 'weak'
    
    return results
