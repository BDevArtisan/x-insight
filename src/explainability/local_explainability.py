"""
Local explainability for individual patient cluster assignments.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import euclidean

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from lime import lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False


def explain_patient_shap(X, labels, patient_idx, feature_names=None, random_state=42):
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP is required for this function. Install with: pip install shap")
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    tree.fit(X_arr, labels)
    
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(X_arr)
    
    patient_cluster = labels[patient_idx]
    
    if isinstance(shap_values, list):
        patient_shap = shap_values[patient_cluster][patient_idx]
    else:
        patient_shap = shap_values[patient_idx]
    
    explanation = {
        'patient_idx': patient_idx,
        'cluster': patient_cluster,
        'feature_names': feature_names,
        'shap_values': patient_shap,
        'feature_values': X_arr[patient_idx],
        'base_value': explainer.expected_value if not isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value[patient_cluster]
    }
    
    return explanation


def explain_patient_lime(X, labels, patient_idx, feature_names=None, random_state=42):
    if not LIME_AVAILABLE:
        raise ImportError("LIME is required for this function. Install with: pip install lime")
    
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    tree.fit(X_arr, labels)
    
    explainer = lime_tabular.LimeTabularExplainer(
        X_arr,
        feature_names=feature_names,
        class_names=[f"Cluster_{i}" for i in np.unique(labels)],
        mode='classification',
        random_state=random_state
    )
    
    exp = explainer.explain_instance(
        X_arr[patient_idx],
        tree.predict_proba,
        num_features=len(feature_names)
    )
    
    patient_cluster = labels[patient_idx]
    lime_values = dict(exp.as_list())
    
    feature_contributions = []
    for fname in feature_names:
        contrib = 0
        for key, val in lime_values.items():
            if fname in key:
                contrib = val
                break
        feature_contributions.append(contrib)
    
    explanation = {
        'patient_idx': patient_idx,
        'cluster': patient_cluster,
        'feature_names': feature_names,
        'lime_values': np.array(feature_contributions),
        'feature_values': X_arr[patient_idx],
        'probability': tree.predict_proba(X_arr[patient_idx:patient_idx+1])[0]
    }
    
    return explanation


def explain_distance_to_centroid(X, labels, patient_idx, clustering_model, feature_names=None):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]
    
    patient_cluster = labels[patient_idx]
    patient_features = X_arr[patient_idx]
    
    if hasattr(clustering_model, 'get_cluster_centers'):
        centroid = clustering_model.get_cluster_centers()[patient_cluster]
    elif hasattr(clustering_model, 'get_medoids'):
        centroid = clustering_model.get_medoids()[patient_cluster]
    elif hasattr(clustering_model, 'get_means'):
        centroid = clustering_model.get_means()[patient_cluster]
    else:
        cluster_points = X_arr[labels == patient_cluster]
        centroid = np.mean(cluster_points, axis=0)
    
    feature_distances = np.abs(patient_features - centroid)
    
    total_distance = euclidean(patient_features, centroid)
    
    feature_contributions = feature_distances / (feature_distances.sum() + 1e-10)
    
    sorted_indices = np.argsort(feature_contributions)[::-1]
    
    explanation = {
        'patient_idx': patient_idx,
        'cluster': patient_cluster,
        'feature_names': feature_names,
        'feature_distances': feature_distances,
        'feature_contributions': feature_contributions,
        'total_distance': total_distance,
        'patient_values': patient_features,
        'centroid_values': centroid,
        'sorted_indices': sorted_indices
    }
    
    return explanation


def explain_probabilistic_membership(X, labels, patient_idx, clustering_model, feature_names=None):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f"Feature_{i}" for i in range(X_arr.shape[1])]
    
    patient_cluster = labels[patient_idx]
    patient_features = X_arr[patient_idx:patient_idx+1]
    
    if hasattr(clustering_model, 'predict_proba'):
        probabilities = clustering_model.predict_proba(patient_features)[0]
    elif hasattr(clustering_model, 'get_probabilities'):
        all_probs = clustering_model.get_probabilities()
        probabilities = all_probs[patient_idx]
    else:
        n_clusters = len(np.unique(labels))
        probabilities = np.zeros(n_clusters)
        probabilities[patient_cluster] = 1.0
    
    if hasattr(clustering_model, 'get_cluster_centers'):
        centers = clustering_model.get_cluster_centers()
    elif hasattr(clustering_model, 'get_medoids'):
        centers = clustering_model.get_medoids()
    elif hasattr(clustering_model, 'get_means'):
        centers = clustering_model.get_means()
    else:
        centers = []
        for c in np.unique(labels):
            cluster_points = X_arr[labels == c]
            centers.append(np.mean(cluster_points, axis=0))
        centers = np.array(centers)
    
    distances_to_centers = []
    for center in centers:
        dist = euclidean(patient_features[0], center)
        distances_to_centers.append(dist)
    
    explanation = {
        'patient_idx': patient_idx,
        'cluster': patient_cluster,
        'feature_names': feature_names,
        'probabilities': probabilities,
        'distances_to_centers': np.array(distances_to_centers),
        'patient_values': patient_features[0],
        'cluster_centers': centers,
        'membership_certainty': probabilities[patient_cluster]
    }
    
    return explanation


def explain_patient(X, labels, patient_idx, clustering_model, 
                   methods=['shap', 'lime', 'distance', 'probabilistic'],
                   feature_names=None, random_state=42):
    explanations = {}
    
    if 'shap' in methods:
        try:
            explanations['shap'] = explain_patient_shap(X, labels, patient_idx, feature_names, random_state)
        except Exception as e:
            explanations['shap'] = {'error': str(e)}
    
    if 'lime' in methods:
        try:
            explanations['lime'] = explain_patient_lime(X, labels, patient_idx, feature_names, random_state)
        except Exception as e:
            explanations['lime'] = {'error': str(e)}
    
    if 'distance' in methods:
        try:
            explanations['distance'] = explain_distance_to_centroid(X, labels, patient_idx, clustering_model, feature_names)
        except Exception as e:
            explanations['distance'] = {'error': str(e)}
    
    if 'probabilistic' in methods:
        try:
            explanations['probabilistic'] = explain_probabilistic_membership(X, labels, patient_idx, clustering_model, feature_names)
        except Exception as e:
            explanations['probabilistic'] = {'error': str(e)}
    
    return explanations


def plot_shap_explanation(explanation, top_n=10):
    if 'error' in explanation:
        print(f"Error: {explanation['error']}")
        return None
    
    feature_names = explanation['feature_names']
    shap_values = np.asarray(explanation['shap_values']).flatten()
    feature_values = explanation['feature_values']
    
    n_features = min(top_n, len(shap_values), len(feature_names))
    sorted_idx = np.argsort(np.abs(shap_values))[::-1][:n_features]
    sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
    n_features = len(sorted_idx)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap_vals = shap_values[sorted_idx]
    colors = ['red' if v < 0 else 'green' for v in shap_vals]
    
    ax.barh(range(n_features), shap_vals, color=colors)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f"{feature_names[i]} = {feature_values[i]:.2f}" for i in sorted_idx])
    ax.set_xlabel('SHAP Value (Impact on Cluster Assignment)')
    ax.set_title(f"Patient {explanation['patient_idx']} - Cluster {explanation['cluster']}")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_lime_explanation(explanation, top_n=10):
    if 'error' in explanation:
        print(f"Error: {explanation['error']}")
        return None
    
    feature_names = explanation['feature_names']
    lime_values = np.asarray(explanation['lime_values']).flatten()
    feature_values = explanation['feature_values']
    
    n_features = min(top_n, len(lime_values), len(feature_names))
    sorted_idx = np.argsort(np.abs(lime_values))[::-1][:n_features]
    sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
    n_features = len(sorted_idx)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lime_vals = lime_values[sorted_idx]
    colors = ['red' if v < 0 else 'green' for v in lime_vals]
    
    ax.barh(range(n_features), lime_vals, color=colors)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f"{feature_names[i]} = {feature_values[i]:.2f}" for i in sorted_idx])
    ax.set_xlabel('LIME Value (Impact on Cluster Assignment)')
    ax.set_title(f"Patient {explanation['patient_idx']} - Cluster {explanation['cluster']}")
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_distance_explanation(explanation, top_n=10):
    if 'error' in explanation:
        print(f"Error: {explanation['error']}")
        return None
    
    feature_names = explanation['feature_names']
    feature_contributions = explanation['feature_contributions']
    patient_values = explanation['patient_values']
    centroid_values = explanation['centroid_values']
    
    n_features = min(top_n, len(feature_contributions), len(feature_names))
    sorted_idx = explanation['sorted_indices'][:n_features]
    sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
    n_features = len(sorted_idx)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_features)
    width = 0.35
    
    ax.bar(x - width/2, patient_values[sorted_idx], width, label='Patient Value', alpha=0.8)
    ax.bar(x + width/2, centroid_values[sorted_idx], width, label='Cluster Centroid', alpha=0.8)
    
    ax.set_xlabel('Features')
    ax.set_ylabel('Value')
    ax.set_title(f"Patient {explanation['patient_idx']} - Distance to Cluster {explanation['cluster']}")
    ax.set_xticks(x)
    ax.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_probabilistic_explanation(explanation):
    if 'error' in explanation:
        print(f"Error: {explanation['error']}")
        return None
    
    probabilities = explanation['probabilities']
    patient_cluster = explanation['cluster']
    
    fig = go.Figure(data=[
        go.Bar(
            x=[f"Cluster {i}" for i in range(len(probabilities))],
            y=probabilities,
            marker_color=['green' if i == patient_cluster else 'lightblue' for i in range(len(probabilities))]
        )
    ])
    
    fig.update_layout(
        title=f"Patient {explanation['patient_idx']} - Cluster Membership Probabilities",
        xaxis_title="Cluster",
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1]),
        showlegend=False
    )
    
    return fig


def plot_patient_explanation_summary(explanations, top_n=10):
    fig = plt.figure(figsize=(16, 10))
    
    plot_idx = 1
    n_plots = sum([1 for k, v in explanations.items() if 'error' not in v])
    
    if n_plots == 0:
        print("No valid explanations to plot")
        return None
    
    rows = (n_plots + 1) // 2
    cols = min(2, n_plots)
    
    if 'shap' in explanations and 'error' not in explanations['shap']:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        exp = explanations['shap']
        feature_names = exp['feature_names']
        shap_values = np.asarray(exp['shap_values']).flatten()
        feature_values = exp['feature_values']
        
        n_features = min(top_n, len(shap_values), len(feature_names))
        sorted_idx = np.argsort(np.abs(shap_values))[::-1][:n_features]
        sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
        n_features = len(sorted_idx)
        
        shap_vals = shap_values[sorted_idx]
        colors = ['red' if v < 0 else 'green' for v in shap_vals]
        
        ax.barh(range(n_features), shap_vals, color=colors)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f"{feature_names[i]}" for i in sorted_idx], fontsize=8)
        ax.set_xlabel('SHAP Value')
        ax.set_title('SHAP Explanation')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
    
    if 'lime' in explanations and 'error' not in explanations['lime']:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        exp = explanations['lime']
        feature_names = exp['feature_names']
        lime_values = np.asarray(exp['lime_values']).flatten()
        
        n_features = min(top_n, len(lime_values), len(feature_names))
        sorted_idx = np.argsort(np.abs(lime_values))[::-1][:n_features]
        sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
        n_features = len(sorted_idx)
        
        lime_vals = lime_values[sorted_idx]
        colors = ['red' if v < 0 else 'green' for v in lime_vals]
        
        ax.barh(range(n_features), lime_vals, color=colors)
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f"{feature_names[i]}" for i in sorted_idx], fontsize=8)
        ax.set_xlabel('LIME Value')
        ax.set_title('LIME Explanation')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
    
    if 'distance' in explanations and 'error' not in explanations['distance']:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        exp = explanations['distance']
        feature_names = exp['feature_names']
        feature_contributions = exp['feature_contributions']
        
        n_features = min(top_n, len(feature_contributions), len(feature_names))
        sorted_idx = exp['sorted_indices'][:n_features]
        sorted_idx = sorted_idx[sorted_idx < len(feature_names)]
        n_features = len(sorted_idx)
        
        ax.barh(range(n_features), feature_contributions[sorted_idx], color='steelblue')
        ax.set_yticks(range(n_features))
        ax.set_yticklabels([f"{feature_names[i]}" for i in sorted_idx], fontsize=8)
        ax.set_xlabel('Contribution to Distance')
        ax.set_title('Distance to Centroid')
        ax.invert_yaxis()
    
    if 'probabilistic' in explanations and 'error' not in explanations['probabilistic']:
        ax = plt.subplot(rows, cols, plot_idx)
        plot_idx += 1
        
        exp = explanations['probabilistic']
        probabilities = exp['probabilities']
        patient_cluster = exp['cluster']
        
        colors = ['green' if i == patient_cluster else 'lightblue' for i in range(len(probabilities))]
        ax.bar([f"C{i}" for i in range(len(probabilities))], probabilities, color=colors)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Probability')
        ax.set_title('Cluster Membership Probabilities')
        ax.set_ylim([0, 1])
    
    plt.tight_layout()
    return fig
