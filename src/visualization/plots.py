"""
Visualization plots for clustering results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import silhouette_samples


def plot_clusters_2d(X_reduced, labels, title='Cluster Visualization',
                     centroids=None, method_name='PCA', figsize=(10, 8)):
    """
    Plot clusters in 2D.
    
    Parameters:
    -----------
    X_reduced : np.ndarray
        2D reduced data
    labels : np.ndarray
        Cluster labels
    title : str
        Plot title
    centroids : np.ndarray, optional
        Cluster centroids in reduced space
    method_name : str
        Reduction method name
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    scatter = ax.scatter(X_reduced[:, 0], X_reduced[:, 1], 
                        c=labels, cmap='Set2', alpha=0.7, s=50)
    
    if centroids is not None:
        ax.scatter(centroids[:, 0], centroids[:, 1], 
                  c='red', marker='X', s=200, edgecolors='black', linewidths=2)
    
    ax.set_xlabel(f'{method_name} 1')
    ax.set_ylabel(f'{method_name} 2')
    ax.set_title(title)
    plt.colorbar(scatter, ax=ax, label='Cluster')
    plt.tight_layout()
    
    return fig, ax


def plot_clusters_interactive(X_reduced, labels, patient_ids=None, 
                              method_name='PCA', title='Interactive Cluster Plot'):
    """
    Create interactive 2D cluster plot using Plotly.
    
    Parameters:
    -----------
    X_reduced : np.ndarray
        2D reduced data
    labels : np.ndarray
        Cluster labels
    patient_ids : list, optional
        Patient IDs for hover info
    method_name : str
        Reduction method name
    title : str
        Plot title
        
    Returns:
    --------
    fig : plotly figure
    """
    df_plot = pd.DataFrame({
        f'{method_name}1': X_reduced[:, 0],
        f'{method_name}2': X_reduced[:, 1],
        'Cluster': labels.astype(str)
    })
    
    if patient_ids is not None:
        df_plot['Patient_ID'] = patient_ids
    
    fig = px.scatter(df_plot, 
                     x=f'{method_name}1', 
                     y=f'{method_name}2',
                     color='Cluster',
                     hover_data=['Patient_ID'] if patient_ids is not None else None,
                     title=title,
                     color_discrete_sequence=px.colors.qualitative.Set2)
    
    fig.update_layout(width=800, height=600)
    
    return fig


def plot_dendrogram(linkage_matrix, figsize=(14, 6), title='Hierarchical Clustering Dendrogram',
                   truncate_mode='level', p=5):
    """
    Plot dendrogram for hierarchical clustering.
    
    Parameters:
    -----------
    linkage_matrix : np.ndarray
        Linkage matrix from hierarchical clustering
    figsize : tuple
        Figure size
    title : str
        Plot title
    truncate_mode : str
        Truncation mode
    p : int
        Truncation parameter
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    dendrogram(linkage_matrix, 
              truncate_mode=truncate_mode, 
              p=p, 
              ax=ax,
              color_threshold=0.7*max(linkage_matrix[:,2]))
    
    ax.set_title(title)
    ax.set_xlabel('Samples')
    ax.set_ylabel('Distance')
    plt.tight_layout()
    
    return fig, ax


def plot_silhouette_analysis(X, labels, figsize=(10, 6)):
    """
    Plot silhouette analysis.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    labels : np.ndarray
        Cluster labels
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    from sklearn.metrics import silhouette_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)
    
    y_lower = 10
    n_clusters = len(np.unique(labels[labels >= 0]))
    
    for i in range(n_clusters):
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 
                        0, ith_cluster_silhouette_values, alpha=0.7)
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color='red', linestyle='--', 
              label=f'Average: {silhouette_avg:.3f}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster')
    ax.set_title('Silhouette Analysis')
    ax.legend()
    plt.tight_layout()
    
    return fig, ax


def plot_elbow_curve(k_range, scores, metric_name='Inertia', figsize=(8, 5)):
    """
    Plot elbow curve for determining optimal k.
    
    Parameters:
    -----------
    k_range : list or range
        Range of k values
    scores : list
        Corresponding scores
    metric_name : str
        Metric name
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(k_range, scores, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel(metric_name)
    ax.set_title(f'Elbow Method - {metric_name}')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig, ax


def plot_cluster_comparison(comparison_df, metrics=['silhouette', 'davies_bouldin'], 
                           figsize=(12, 5)):
    """
    Plot comparison of clustering methods.
    
    Parameters:
    -----------
    comparison_df : pd.DataFrame
        Comparison dataframe from compare_clustering_methods
    metrics : list
        Metrics to plot
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    
    if len(metrics) == 1:
        axes = [axes]
    
    for ax, metric in zip(axes, metrics):
        if metric in comparison_df.columns:
            bars = ax.bar(comparison_df['Method'], comparison_df[metric])
            
            # Highlight best
            if metric == 'davies_bouldin':
                best_idx = comparison_df[metric].idxmin()
            else:
                best_idx = comparison_df[metric].idxmax()
            bars[best_idx].set_color('green')
            
            ax.set_title(metric.replace('_', ' ').title())
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig, axes


def plot_cluster_profiles(profiles_df, figsize=(12, 6)):
    """
    Plot cluster profiles as heatmap.
    
    Parameters:
    -----------
    profiles_df : pd.DataFrame
        Cluster profiles with features as columns
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(profiles_df, annot=True, fmt='.2f', cmap='RdBu_r', 
               center=0, ax=ax, cbar_kws={'label': 'Value'})
    ax.set_title('Cluster Profiles')
    ax.set_xlabel('Features')
    ax.set_ylabel('Cluster')
    plt.tight_layout()
    
    return fig, ax


def plot_radar_chart(profiles_df, figsize=(8, 8)):
    """
    Create radar chart for cluster profiles.
    
    Parameters:
    -----------
    profiles_df : pd.DataFrame
        Normalized cluster profiles
    figsize : tuple
        Figure size
        
    Returns:
    --------
    fig, ax : matplotlib figure and axes
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize if not already
    scaler = MinMaxScaler()
    profiles_norm = pd.DataFrame(
        scaler.fit_transform(profiles_df),
        index=profiles_df.index,
        columns=profiles_df.columns
    )
    
    categories = list(profiles_norm.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(range(len(profiles_norm)))
    
    for idx, (cluster_id, row) in enumerate(profiles_norm.iterrows()):
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=f'Cluster {cluster_id}', color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title('Cluster Profiles (Normalized)')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.tight_layout()
    
    return fig, ax
