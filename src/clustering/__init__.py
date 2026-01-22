"""
Clustering module initialization.
"""

from .traditional import KMeansClustering, HierarchicalClustering, KMedoidsClustering, determine_optimal_k
from .advanced import GMMClustering, SpectralClusteringWrapper
from .validation import (
    compute_internal_metrics,
    compute_external_metrics,
    compute_silhouette_per_cluster,
    create_confusion_matrix,
    compare_clustering_methods,
    profile_clusters
)

try:
    from .advanced import HDBSCANClustering
    __all__ = [
        'KMeansClustering', 'HierarchicalClustering', 'KMedoidsClustering',
        'GMMClustering', 'SpectralClusteringWrapper', 'HDBSCANClustering',
        'determine_optimal_k',
        'compute_internal_metrics', 'compute_external_metrics',
        'compute_silhouette_per_cluster', 'create_confusion_matrix',
        'compare_clustering_methods', 'profile_clusters'
    ]
except ImportError:
    __all__ = [
        'KMeansClustering', 'HierarchicalClustering', 'KMedoidsClustering',
        'GMMClustering', 'SpectralClusteringWrapper',
        'determine_optimal_k',
        'compute_internal_metrics', 'compute_external_metrics',
        'compute_silhouette_per_cluster', 'create_confusion_matrix',
        'compare_clustering_methods', 'profile_clusters'
    ]
