"""
Cluster validation metrics and utilities.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score,
    confusion_matrix
)


def compute_internal_metrics(X, labels):
    """
    Compute internal validation metrics.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    labels : np.ndarray
        Cluster labels
        
    Returns:
    --------
    metrics : dict
        Dictionary of metric scores
    """
    # Filter out noise points (-1 labels) if present
    valid_mask = labels >= 0
    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    
    if len(np.unique(labels_valid)) < 2:
        return {
            'silhouette': np.nan,
            'davies_bouldin': np.nan,
            'calinski_harabasz': np.nan
        }
    
    metrics = {
        'silhouette': silhouette_score(X_valid, labels_valid),
        'davies_bouldin': davies_bouldin_score(X_valid, labels_valid),
        'calinski_harabasz': calinski_harabasz_score(X_valid, labels_valid)
    }
    
    return metrics


def compute_external_metrics(y_true, y_pred):
    """
    Compute external validation metrics.
    
    Parameters:
    -----------
    y_true : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted cluster labels
        
    Returns:
    --------
    metrics : dict
        Dictionary of metric scores
    """
    metrics = {
        'ari': adjusted_rand_score(y_true, y_pred),
        'nmi': normalized_mutual_info_score(y_true, y_pred)
    }
    
    return metrics


def compute_silhouette_per_cluster(X, labels):
    """
    Compute silhouette score per cluster.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    labels : np.ndarray
        Cluster labels
        
    Returns:
    --------
    cluster_silhouettes : dict
        Silhouette score per cluster
    """
    sample_silhouettes = silhouette_samples(X, labels)
    
    cluster_silhouettes = {}
    for cluster_id in np.unique(labels):
        if cluster_id >= 0:  # Ignore noise (-1)
            cluster_mask = labels == cluster_id
            cluster_silhouettes[cluster_id] = sample_silhouettes[cluster_mask].mean()
    
    return cluster_silhouettes


def create_confusion_matrix(y_true, y_pred, label_names_true=None, label_names_pred=None):
    """
    Create confusion matrix with labels.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    label_names_true : list, optional
        Names for true labels
    label_names_pred : list, optional
        Names for predicted labels
        
    Returns:
    --------
    cm_df : pd.DataFrame
        Confusion matrix as DataFrame
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if label_names_true is None:
        label_names_true = [f'True_{i}' for i in range(cm.shape[0])]
    if label_names_pred is None:
        label_names_pred = [f'Pred_{i}' for i in range(cm.shape[1])]
    
    cm_df = pd.DataFrame(cm, index=label_names_true, columns=label_names_pred)
    
    return cm_df


def compare_clustering_methods(X, y_true=None, methods_labels=None):
    """
    Compare multiple clustering methods.
    
    Parameters:
    -----------
    X : np.ndarray
        Data matrix
    y_true : np.ndarray, optional
        True labels for external validation
    methods_labels : dict
        Dictionary {method_name: labels}
        
    Returns:
    --------
    comparison_df : pd.DataFrame
        Comparison table
    """
    if methods_labels is None:
        raise ValueError("methods_labels dictionary is required")
    
    results = []
    
    for method_name, labels in methods_labels.items():
        row = {'Method': method_name}
        
        # Internal metrics
        internal = compute_internal_metrics(X, labels)
        row.update(internal)
        
        # External metrics if ground truth available
        if y_true is not None:
            external = compute_external_metrics(y_true, labels)
            row.update(external)
        
        # Cluster distribution
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        row['n_clusters'] = len(unique)
        row['noise_points'] = np.sum(labels == -1)
        
        results.append(row)
    
    comparison_df = pd.DataFrame(results)
    
    return comparison_df


def profile_clusters(df, labels, feature_cols):
    """
    Create cluster profiles with mean values.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with features
    labels : np.ndarray
        Cluster labels
    feature_cols : list
        List of feature column names
        
    Returns:
    --------
    profiles : pd.DataFrame
        Cluster profiles
    """
    df_temp = df[feature_cols].copy()
    df_temp['Cluster'] = labels
    
    profiles = df_temp.groupby('Cluster')[feature_cols].mean()
    
    return profiles
