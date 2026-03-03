"""
Counterfactual explanations for cluster assignments.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import plotly.graph_objects as go


class ClinicalConstraints:
    
    def __init__(self, feature_names=None):
        self.feature_names = feature_names or []
        self.constraints = {}
        self.immutable_features = set()
        self.categorical_features = {}
        
    def set_range(self, feature, min_val, max_val):
        if feature in self.feature_names or isinstance(feature, int):
            idx = feature if isinstance(feature, int) else self.feature_names.index(feature)
            self.constraints[idx] = {'min': min_val, 'max': max_val}
    
    def set_immutable(self, feature):
        if feature in self.feature_names or isinstance(feature, int):
            idx = feature if isinstance(feature, int) else self.feature_names.index(feature)
            self.immutable_features.add(idx)
    
    def set_categorical(self, feature, allowed_values):
        if feature in self.feature_names or isinstance(feature, int):
            idx = feature if isinstance(feature, int) else self.feature_names.index(feature)
            self.categorical_features[idx] = allowed_values
    
    def apply_constraints(self, x):
        x_constrained = x.copy()
        
        for idx, bounds in self.constraints.items():
            if idx < len(x_constrained):
                x_constrained[idx] = np.clip(x_constrained[idx], bounds['min'], bounds['max'])
        
        for idx in self.categorical_features:
            if idx < len(x_constrained):
                allowed = self.categorical_features[idx]
                closest = min(allowed, key=lambda v: abs(v - x_constrained[idx]))
                x_constrained[idx] = closest
        
        return x_constrained
    
    def check_validity(self, x):
        for idx, bounds in self.constraints.items():
            if idx < len(x):
                if x[idx] < bounds['min'] or x[idx] > bounds['max']:
                    return False
        
        for idx in self.categorical_features:
            if idx < len(x):
                if x[idx] not in self.categorical_features[idx]:
                    return False
        
        return True


def generate_counterfactual_optimization(
    X, labels, patient_idx, clustering_model,
    target_cluster=None, constraints=None,
    max_iterations=1000, lambda_sparse=0.1, lambda_distance=1.0
):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    x_original = X_arr[patient_idx].copy()
    current_cluster = labels[patient_idx]
    
    if target_cluster is None:
        unique_clusters = np.unique(labels)
        possible_targets = [c for c in unique_clusters if c != current_cluster]
        if not possible_targets:
            return None
        target_cluster = possible_targets[0]
    
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
    
    target_center = centers[target_cluster]
    
    def objective(x):
        dist_to_target = euclidean(x, target_center)
        
        sparsity = np.sum(np.abs(x - x_original) > 1e-6)
        
        distance_from_original = euclidean(x, x_original)
        
        return lambda_distance * dist_to_target + lambda_sparse * sparsity + 0.5 * distance_from_original
    
    def cluster_constraint(x):
        if hasattr(clustering_model, 'predict'):
            pred_cluster = clustering_model.predict(x.reshape(1, -1))[0]
        else:
            distances = [euclidean(x, center) for center in centers]
            pred_cluster = np.argmin(distances)
        
        return 1.0 if pred_cluster == target_cluster else -1.0
    
    bounds = [(None, None)] * len(x_original)
    
    if constraints:
        for idx, constraint in constraints.constraints.items():
            if idx < len(bounds):
                bounds[idx] = (constraint['min'], constraint['max'])
        
        for idx in constraints.immutable_features:
            if idx < len(bounds):
                bounds[idx] = (x_original[idx], x_original[idx])
    
    result = minimize(
        objective,
        x_original,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iterations}
    )
    
    x_counterfactual = result.x
    
    if constraints:
        x_counterfactual = constraints.apply_constraints(x_counterfactual)
    
    if hasattr(clustering_model, 'predict'):
        predicted_cluster = clustering_model.predict(x_counterfactual.reshape(1, -1))[0]
    else:
        distances = [euclidean(x_counterfactual, center) for center in centers]
        predicted_cluster = np.argmin(distances)
    
    if predicted_cluster != target_cluster:
        target_points = X_arr[labels == target_cluster]
        if len(target_points) > 0:
            distances_to_target = [euclidean(x_original, tp) for tp in target_points]
            nearest_target_idx = np.argmin(distances_to_target)
            nearest_target = target_points[nearest_target_idx]
            
            for alpha in [0.6, 0.7, 0.8, 0.9]:
                x_candidate = x_original + alpha * (nearest_target - x_original)
                
                if constraints:
                    x_candidate = constraints.apply_constraints(x_candidate)
                
                if hasattr(clustering_model, 'predict'):
                    pred = clustering_model.predict(x_candidate.reshape(1, -1))[0]
                else:
                    dists = [euclidean(x_candidate, center) for center in centers]
                    pred = np.argmin(dists)
                
                if pred == target_cluster:
                    x_counterfactual = x_candidate
                    predicted_cluster = pred
                    break
    
    changes = x_counterfactual - x_original
    feature_changes = np.where(np.abs(changes) > 1e-6)[0]
    
    return {
        'original': x_original,
        'counterfactual': x_counterfactual,
        'changes': changes,
        'feature_changes': feature_changes,
        'original_cluster': current_cluster,
        'target_cluster': target_cluster,
        'predicted_cluster': predicted_cluster,
        'success': predicted_cluster == target_cluster,
        'distance': euclidean(x_original, x_counterfactual),
        'n_changes': len(feature_changes)
    }


def generate_diverse_counterfactuals(
    X, labels, patient_idx, clustering_model,
    target_cluster=None, constraints=None,
    n_counterfactuals=3, diversity_weight=0.5
):
    counterfactuals = []
    
    for i in range(n_counterfactuals):
        lambda_sparse = 0.1 + i * 0.05
        lambda_distance = 1.0 - i * 0.1
        
        cf = generate_counterfactual_optimization(
            X, labels, patient_idx, clustering_model,
            target_cluster=target_cluster,
            constraints=constraints,
            lambda_sparse=lambda_sparse,
            lambda_distance=lambda_distance
        )
        
        if cf and cf['success']:
            is_diverse = True
            for existing_cf in counterfactuals:
                similarity = 1 - euclidean(cf['counterfactual'], existing_cf['counterfactual']) / (
                    euclidean(cf['counterfactual'], cf['original']) + 1e-10
                )
                if similarity > (1 - diversity_weight):
                    is_diverse = False
                    break
            
            if is_diverse:
                counterfactuals.append(cf)
    
    return counterfactuals


def generate_counterfactual_dice_style(
    X, labels, patient_idx, clustering_model,
    target_cluster=None, constraints=None,
    n_counterfactuals=3, proximity_weight=0.5, diversity_weight=0.5
):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    x_original = X_arr[patient_idx].copy()
    current_cluster = labels[patient_idx]
    
    if target_cluster is None:
        unique_clusters = np.unique(labels)
        possible_targets = [c for c in unique_clusters if c != current_cluster]
        if not possible_targets:
            return []
        target_cluster = possible_targets[0]
    
    target_points = X_arr[labels == target_cluster]
    
    if len(target_points) == 0:
        return []
    
    distances = [euclidean(x_original, tp) for tp in target_points]
    sorted_indices = np.argsort(distances)
    
    candidate_indices = sorted_indices[:min(20, len(sorted_indices))]
    
    counterfactuals = []
    
    for idx in candidate_indices[:n_counterfactuals * 3]:
        candidate = target_points[idx]
        
        direction = candidate - x_original
        
        for alpha in [0.5, 0.7, 0.9]:
            x_cf = x_original + alpha * direction
            
            if constraints:
                x_cf = constraints.apply_constraints(x_cf)
            
            if hasattr(clustering_model, 'predict'):
                pred_cluster = clustering_model.predict(x_cf.reshape(1, -1))[0]
            else:
                if hasattr(clustering_model, 'get_cluster_centers'):
                    centers = clustering_model.get_cluster_centers()
                elif hasattr(clustering_model, 'get_medoids'):
                    centers = clustering_model.get_medoids()
                elif hasattr(clustering_model, 'get_means'):
                    centers = clustering_model.get_means()
                else:
                    centers = []
                    for c in np.unique(labels):
                        cluster_pts = X_arr[labels == c]
                        centers.append(np.mean(cluster_pts, axis=0))
                    centers = np.array(centers)
                
                distances_to_centers = [euclidean(x_cf, center) for center in centers]
                pred_cluster = np.argmin(distances_to_centers)
            
            if pred_cluster == target_cluster:
                changes = x_cf - x_original
                feature_changes = np.where(np.abs(changes) > 1e-6)[0]
                
                cf_dict = {
                    'original': x_original,
                    'counterfactual': x_cf,
                    'changes': changes,
                    'feature_changes': feature_changes,
                    'original_cluster': current_cluster,
                    'target_cluster': target_cluster,
                    'predicted_cluster': pred_cluster,
                    'success': True,
                    'distance': euclidean(x_original, x_cf),
                    'n_changes': len(feature_changes)
                }
                
                is_diverse = True
                if diversity_weight > 0:
                    for existing_cf in counterfactuals:
                        similarity = 1 - euclidean(x_cf, existing_cf['counterfactual']) / (
                            euclidean(x_cf, x_original) + 1e-10
                        )
                        if similarity > (1 - diversity_weight):
                            is_diverse = False
                            break
                
                if is_diverse:
                    counterfactuals.append(cf_dict)
                
                if len(counterfactuals) >= n_counterfactuals:
                    break
        
        if len(counterfactuals) >= n_counterfactuals:
            break
    
    return counterfactuals


def format_counterfactual_explanation(counterfactual, feature_names=None):
    if not counterfactual or not counterfactual['success']:
        return "No valid counterfactual found"
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(counterfactual['original']))]
    
    changes = counterfactual['changes']
    feature_changes = counterfactual['feature_changes']
    
    explanation_parts = []
    
    for idx in feature_changes:
        feature_name = feature_names[idx]
        original_value = counterfactual['original'][idx]
        new_value = counterfactual['counterfactual'][idx]
        change = changes[idx]
        
        if change > 0:
            explanation_parts.append(f"Increase {feature_name} by {abs(change):.2f} (from {original_value:.2f} to {new_value:.2f})")
        else:
            explanation_parts.append(f"Decrease {feature_name} by {abs(change):.2f} (from {original_value:.2f} to {new_value:.2f})")
    
    changes_text = " and ".join(explanation_parts)
    
    explanation = f"{changes_text} → move from Cluster {counterfactual['original_cluster']} to Cluster {counterfactual['target_cluster']}"
    
    return explanation


def plot_counterfactual_comparison(counterfactual, feature_names=None, top_n=10):
    if not counterfactual or not counterfactual['success']:
        print("No valid counterfactual to plot")
        return None
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(counterfactual['original']))]
    
    changes = counterfactual['changes']
    abs_changes = np.abs(changes)
    sorted_idx = np.argsort(abs_changes)[::-1][:top_n]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(sorted_idx))
    width = 0.35
    
    original_vals = counterfactual['original'][sorted_idx]
    cf_vals = counterfactual['counterfactual'][sorted_idx]
    
    ax1.bar(x - width/2, original_vals, width, label='Original', alpha=0.8)
    ax1.bar(x + width/2, cf_vals, width, label='Counterfactual', alpha=0.8)
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Value')
    ax1.set_title('Original vs Counterfactual Values')
    ax1.set_xticks(x)
    ax1.set_xticklabels([feature_names[i] for i in sorted_idx], rotation=45, ha='right')
    ax1.legend()
    
    change_vals = changes[sorted_idx]
    colors = ['green' if c > 0 else 'red' for c in change_vals]
    
    ax2.barh(range(len(sorted_idx)), change_vals, color=colors)
    ax2.set_yticks(range(len(sorted_idx)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax2.set_xlabel('Change in Value')
    ax2.set_title('Feature Changes Required')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    return fig


def plot_diverse_counterfactuals(counterfactuals, feature_names=None, top_n=5):
    if not counterfactuals:
        print("No counterfactuals to plot")
        return None
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(counterfactuals[0]['original']))]
    
    n_cf = len(counterfactuals)
    
    all_changes = []
    for cf in counterfactuals:
        all_changes.append(np.abs(cf['changes']))
    all_changes = np.array(all_changes)
    
    max_changes_per_feature = np.max(all_changes, axis=0)
    top_features = np.argsort(max_changes_per_feature)[::-1][:top_n]
    
    fig = go.Figure()
    
    for i, cf in enumerate(counterfactuals):
        changes = cf['changes'][top_features]
        
        fig.add_trace(go.Bar(
            name=f'CF {i+1} (dist={cf["distance"]:.2f})',
            x=[feature_names[j] for j in top_features],
            y=changes,
            text=[f'{c:+.2f}' for c in changes],
            textposition='auto',
        ))
    
    fig.update_layout(
        title='Diverse Counterfactual Explanations',
        xaxis_title='Features',
        yaxis_title='Change in Value',
        barmode='group',
        height=500
    )
    
    return fig


def evaluate_counterfactual_quality(counterfactual, constraints=None):
    if not counterfactual:
        return {'valid': False}
    
    metrics = {
        'valid': counterfactual['success'],
        'proximity': counterfactual['distance'],
        'sparsity': counterfactual['n_changes'],
        'sparsity_ratio': counterfactual['n_changes'] / len(counterfactual['original'])
    }
    
    if constraints:
        metrics['constraints_satisfied'] = constraints.check_validity(counterfactual['counterfactual'])
    else:
        metrics['constraints_satisfied'] = True
    
    return metrics
