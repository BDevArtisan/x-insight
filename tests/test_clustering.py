"""
Unit tests for clustering modules
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from clustering.traditional import KMeansClustering, HierarchicalClustering, KMedoidsClustering
from clustering.advanced import GMMClustering, SpectralClusteringWrapper
from clustering.validation import (
    compute_internal_metrics, compute_external_metrics,
    compare_clustering_methods, profile_clusters
)


@pytest.fixture
def sample_data():
    """Create sample clustered data"""
    np.random.seed(42)
    
    # Create 3 well-separated clusters
    n_samples = 300
    n_features = 5
    
    # Cluster 1
    cluster1 = np.random.randn(100, n_features) + np.array([0, 0, 0, 0, 0])
    # Cluster 2
    cluster2 = np.random.randn(100, n_features) + np.array([5, 5, 5, 5, 5])
    # Cluster 3
    cluster3 = np.random.randn(100, n_features) + np.array([-5, -5, -5, -5, -5])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    true_labels = np.array([0]*100 + [1]*100 + [2]*100)
    
    return X, true_labels


class TestKMeansClustering:
    """Test suite for K-Means clustering"""
    
    def test_initialization(self):
        """Test K-Means initialization"""
        model = KMeansClustering(n_clusters=3)
        assert model.n_clusters == 3
        assert model.model is not None
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict method"""
        X, _ = sample_data
        model = KMeansClustering(n_clusters=3)
        
        labels = model.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 3
        assert model.model is not None
    
    def test_predict_consistency(self, sample_data):
        """Test predict consistency"""
        X, _ = sample_data
        model = KMeansClustering(n_clusters=3)
        
        labels1 = model.fit_predict(X)
        labels2 = model.predict(X)
        
        np.testing.assert_array_equal(labels1, labels2)
    
    def test_get_cluster_centers(self, sample_data):
        """Test getting cluster centers"""
        X, _ = sample_data
        model = KMeansClustering(n_clusters=3)
        model.fit_predict(X)
        
        centers = model.get_cluster_centers()
        
        assert centers.shape == (3, X.shape[1])


class TestHierarchicalClustering:
    """Test suite for Hierarchical clustering"""
    
    def test_initialization(self):
        """Test Hierarchical initialization"""
        model = HierarchicalClustering(n_clusters=3, linkage_method='ward')
        assert model.n_clusters == 3
        assert model.linkage_method == 'ward'
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict method"""
        X, _ = sample_data
        model = HierarchicalClustering(n_clusters=3)
        
        labels = model.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 3
    
    def test_different_linkages(self, sample_data):
        """Test different linkage methods"""
        X, _ = sample_data
        linkages = ['ward', 'complete', 'average']
        
        for linkage in linkages:
            model = HierarchicalClustering(n_clusters=3, linkage_method=linkage)
            labels = model.fit_predict(X)
            assert len(np.unique(labels)) == 3


class TestKMedoidsClustering:
    """Test suite for K-Medoids clustering"""
    
    def test_initialization(self):
        """Test K-Medoids initialization"""
        model = KMedoidsClustering(n_clusters=3, max_iter=100)
        assert model.n_clusters == 3
        assert model.max_iter == 100
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict method"""
        X, _ = sample_data
        model = KMedoidsClustering(n_clusters=3)
        
        labels = model.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 3
    
    def test_medoid_indices(self, sample_data):
        """Test that medoids are actual data points"""
        X, _ = sample_data
        model = KMedoidsClustering(n_clusters=3)
        model.fit_predict(X)
        
        medoid_indices = model.medoid_indices_
        
        assert len(medoid_indices) == 3
        assert all(0 <= idx < len(X) for idx in medoid_indices)
        
        # Medoids should be unique
        assert len(set(medoid_indices)) == 3


class TestGMMClustering:
    """Test suite for Gaussian Mixture Model"""
    
    def test_initialization(self):
        """Test GMM initialization"""
        model = GMMClustering(n_components=3)
        assert model.n_components == 3
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict method"""
        X, _ = sample_data
        model = GMMClustering(n_components=3)
        
        labels = model.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 3  # May be fewer if some components empty
    
    def test_predict_proba(self, sample_data):
        """Test probability predictions"""
        X, _ = sample_data
        model = GMMClustering(n_components=3)
        model.fit_predict(X)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0)  # Probabilities sum to 1


class TestSpectralClustering:
    """Test suite for Spectral clustering"""
    
    def test_initialization(self):
        """Test Spectral initialization"""
        model = SpectralClusteringWrapper(n_clusters=3)
        assert model.n_clusters == 3
    
    def test_fit_predict(self, sample_data):
        """Test fit_predict method"""
        X, _ = sample_data
        model = SpectralClusteringWrapper(n_clusters=3)
        
        labels = model.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) == 3


class TestClusteringValidation:
    """Test suite for clustering validation functions"""
    
    def test_compute_internal_metrics(self, sample_data):
        """Test internal metrics computation"""
        X, true_labels = sample_data
        
        # Use true labels to test metrics
        metrics = compute_internal_metrics(X, true_labels)
        
        assert 'silhouette' in metrics
        assert 'davies_bouldin' in metrics
        assert 'calinski_harabasz' in metrics
        
        # Silhouette should be positive for well-separated clusters
        assert metrics['silhouette'] > 0
    
    def test_compute_external_metrics(self, sample_data):
        """Test external metrics computation"""
        X, true_labels = sample_data
        
        # Run K-Means
        model = KMeansClustering(n_clusters=3)
        pred_labels = model.fit_predict(X)
        
        metrics = compute_external_metrics(true_labels, pred_labels)
        
        assert 'ari' in metrics
        assert 'nmi' in metrics
        
        # ARI and NMI should be high for well-separated clusters
        assert metrics['ari'] > 0.5
        assert metrics['nmi'] > 0.5
    
    def test_compare_clustering_methods(self, sample_data):
        """Test comparison of clustering methods"""
        X, true_labels = sample_data
        
        methods = {
            'K-Means': KMeansClustering(n_clusters=3),
            'Hierarchical': HierarchicalClustering(n_clusters=3)
        }
        
        # Get labels from each method
        methods_labels = {name: model.fit_predict(X) for name, model in methods.items()}
        
        results = compare_clustering_methods(X, true_labels, methods_labels)
        
        assert len(results) == 2
        assert 'K-Means' in results['Method'].values
        assert 'Hierarchical' in results['Method'].values
        
        # Check all metrics present
        expected_cols = ['silhouette', 'davies_bouldin', 'calinski_harabasz', 'ari', 'nmi']
        for col in expected_cols:
            assert col in results.columns
    
    def test_profile_clusters(self, sample_data):
        """Test cluster profiling"""
        import pandas as pd
        
        X, true_labels = sample_data
        
        # Create DataFrame
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=feature_names)
        
        profiles = profile_clusters(df, true_labels, feature_names)
        
        assert len(profiles) == 3  # 3 clusters
        assert len(profiles.columns) == len(feature_names)
        
        # Check cluster 0 mean is close to 0
        assert np.abs(profiles.loc[0].mean()) < 2


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_cluster(self, sample_data):
        """Test with n_clusters=1"""
        X, _ = sample_data
        model = KMeansClustering(n_clusters=1)
        
        labels = model.fit_predict(X)
        
        assert len(np.unique(labels)) == 1
        assert all(labels == 0)
    
    def test_more_clusters_than_samples(self):
        """Test with more clusters than samples"""
        X = np.random.randn(5, 3)
        
        model = KMeansClustering(n_clusters=10)
        
        # Should raise ValueError
        with pytest.raises(ValueError):
            labels = model.fit_predict(X)
    
    def test_high_dimensional_data(self):
        """Test with high-dimensional data"""
        np.random.seed(42)
        X = np.random.randn(100, 50)  # 50 dimensions
        
        model = KMeansClustering(n_clusters=3)
        labels = model.fit_predict(X)
        
        assert len(labels) == 100
        assert len(np.unique(labels)) <= 3


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
