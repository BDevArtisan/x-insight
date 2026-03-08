"""
Visualization module initialization.
"""

from .dimensionality_reduction import (
    DimensionalityReducer,
    apply_pca,
    apply_tsne,
    compare_reduction_methods
)
from .plots import (
    plot_clusters_2d,
    plot_clusters_interactive,
    plot_dendrogram,
    plot_silhouette_analysis,
    plot_elbow_curve,
    plot_optimal_k_analysis,
    plot_cluster_comparison,
    plot_cluster_profiles,
    plot_radar_chart
)

try:
    from .dimensionality_reduction import apply_umap
    __all__ = [
        'DimensionalityReducer', 'apply_pca', 'apply_tsne', 'apply_umap',
        'compare_reduction_methods',
        'plot_clusters_2d', 'plot_clusters_interactive', 'plot_dendrogram',
        'plot_silhouette_analysis', 'plot_elbow_curve', 'plot_optimal_k_analysis',
        'plot_cluster_comparison', 'plot_cluster_profiles', 'plot_radar_chart'
    ]
except ImportError:
    __all__ = [
        'DimensionalityReducer', 'apply_pca', 'apply_tsne',
        'compare_reduction_methods',
        'plot_clusters_2d', 'plot_clusters_interactive', 'plot_dendrogram',
        'plot_silhouette_analysis', 'plot_elbow_curve', 'plot_optimal_k_analysis',
        'plot_cluster_comparison', 'plot_cluster_profiles', 'plot_radar_chart'
    ]
