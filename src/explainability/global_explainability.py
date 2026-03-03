import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score
import shap


def train_surrogate_tree(X, labels, feature_names=None, max_depth=5, random_state=42):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X_arr, labels)

    # Fidelity: how well the surrogate reproduces the clustering labels
    fidelity = accuracy_score(labels, tree.predict(X_arr))

    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_arr.shape[1])]

    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': tree.feature_importances_
    }).sort_values('importance', ascending=False)

    return tree, importances, fidelity


def compute_permutation_importance(X, labels, feature_names=None, n_repeats=10, random_state=42):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    tree.fit(X_arr, labels)
    
    perm_importance = permutation_importance(
        tree, X_arr, labels,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_arr.shape[1])]
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=False)
    
    return importances


def compute_shap_importance(X, labels, feature_names=None, random_state=42):
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    tree = DecisionTreeClassifier(max_depth=5, random_state=random_state)
    tree.fit(X_arr, labels)
    
    explainer = shap.TreeExplainer(tree)
    shap_values = explainer.shap_values(X_arr)
    
    if feature_names is None:
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
        else:
            feature_names = [f'Feature_{i}' for i in range(X_arr.shape[1])]
    
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    mean_abs_shap = np.asarray(mean_abs_shap).flatten()
    
    if len(mean_abs_shap) != len(feature_names):
        mean_abs_shap = mean_abs_shap[:len(feature_names)]
    
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap.tolist()
    }).sort_values('importance', ascending=False)
    
    return importances, shap_values, explainer


def compute_feature_importance(X, labels, feature_names=None, method='all', **kwargs):
    results = {}
    
    if method in ['surrogate', 'all']:
        _, results['surrogate'], results['surrogate_fidelity'] = train_surrogate_tree(X, labels, feature_names, **kwargs)
    
    if method in ['permutation', 'all']:
        results['permutation'] = compute_permutation_importance(X, labels, feature_names, **kwargs)
    
    if method in ['shap', 'all']:
        results['shap'], _, _ = compute_shap_importance(X, labels, feature_names, **kwargs)
    
    return results


def compute_cluster_profiles(df, labels, feature_cols):
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    profiles = df_with_labels.groupby('cluster')[feature_cols].agg(['mean', 'median', 'std'])
    
    return profiles


def compute_contrastive_differences(df, labels, feature_cols):
    df_with_labels = df.copy()
    df_with_labels['cluster'] = labels
    
    cluster_means = df_with_labels.groupby('cluster')[feature_cols].mean()
    global_mean = df_with_labels[feature_cols].mean()
    
    differences = cluster_means - global_mean
    
    std_per_cluster = df_with_labels.groupby('cluster')[feature_cols].std()
    effect_sizes = differences / df_with_labels[feature_cols].std()
    
    return differences, effect_sizes


def plot_feature_importance(importance_dict, top_n=10, figsize=(12, 6)):
    filtered_dict = {k: v for k, v in importance_dict.items() if isinstance(v, pd.DataFrame)}
    n_methods = len(filtered_dict)
    
    if n_methods == 1:
        method_name, importances = list(filtered_dict.items())[0]
        top_features = importances.head(top_n)
        
        colors = px.colors.sequential.Viridis_r[:len(top_features)]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=top_features['importance'].values,
            y=top_features['feature'].values,
            orientation='h',
            marker=dict(color=colors, line=dict(color='white', width=1)),
            text=top_features['importance'].round(3).values,
            textposition='auto',
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f'{method_name.capitalize()} Feature Importance', font=dict(size=18, color='#2c3e50')),
            xaxis_title='Importance Score',
            yaxis_title='',
            yaxis=dict(autorange='reversed'),
            template='plotly_white',
            height=max(400, 40 * top_n),
            margin=dict(l=150, r=50, t=80, b=50),
            font=dict(size=12)
        )
        
        return fig
    else:
        fig = make_subplots(
            rows=1, cols=n_methods,
            subplot_titles=[f'{name.capitalize()}' for name in filtered_dict.keys()],
            horizontal_spacing=0.12
        )
        
        colors_palette = [px.colors.sequential.Viridis_r, px.colors.sequential.Plasma_r, px.colors.sequential.Turbo_r]
        
        for idx, (method_name, importances) in enumerate(filtered_dict.items()):
            top_features = importances.head(top_n)
            colors = colors_palette[idx % len(colors_palette)][:len(top_features)]
            
            fig.add_trace(
                go.Bar(
                    x=top_features['importance'].values,
                    y=top_features['feature'].values,
                    orientation='h',
                    marker=dict(color=colors, line=dict(color='white', width=1)),
                    text=top_features['importance'].round(3).values,
                    textposition='auto',
                    hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=idx+1
            )
            
            fig.update_yaxes(autorange='reversed', row=1, col=idx+1)
        
        fig.update_layout(
            title=dict(text='Feature Importance Comparison', font=dict(size=20, color='#2c3e50')),
            template='plotly_white',
            height=max(500, 40 * top_n),
            margin=dict(l=150, r=50, t=100, b=50),
            font=dict(size=11)
        )
        
        fig.update_xaxes(title_text='Importance', row=1, col=2)
        
        return fig


def plot_cluster_comparison_bars(profiles, feature_cols=None, figsize=(14, 6)):
    if feature_cols is None:
        feature_cols = profiles.columns.get_level_values(0).unique().tolist()
    
    means = profiles.xs('mean', level=1, axis=1)[feature_cols]
    
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    
    for idx, (cluster, row) in enumerate(means.iterrows()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Bar(
            name=f'Cluster {cluster}',
            x=feature_cols,
            y=row.values,
            marker=dict(color=color, line=dict(color='white', width=1.5)),
            text=row.round(2).values,
            textposition='outside',
            hovertemplate='<b>Cluster %{fullData.name}</b><br>Feature: %{x}<br>Mean: %{y:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(text='Feature Comparison Across Clusters', font=dict(size=20, color='#2c3e50')),
        xaxis_title='Features',
        yaxis_title='Mean Value (Standardized)',
        barmode='group',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            font=dict(size=12)
        ),
        margin=dict(l=80, r=50, t=100, b=120),
        xaxis=dict(tickangle=-45),
        font=dict(size=12)
    )
    
    return fig


def plot_contrastive_heatmap(differences, figsize=(10, 6), cmap='RdBu_r'):
    max_abs = np.abs(differences.values).max()
    
    annotations = []
    for i, cluster in enumerate(differences.index):
        for j, feature in enumerate(differences.columns):
            value = differences.loc[cluster, feature]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f'{value:.2f}',
                    showarrow=False,
                    font=dict(
                        color='white' if abs(value) > max_abs * 0.5 else 'black',
                        size=11,
                        family='Arial'
                    )
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=differences.values,
        x=differences.columns.tolist(),
        y=[f'Cluster {c}' for c in differences.index],
        colorscale='RdBu_r',
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        colorbar=dict(
            title=dict(text='Deviation<br>from Mean', side='right'),
            tickmode='linear',
            tick0=-max_abs,
            dtick=max_abs/2,
            thickness=20,
            len=0.7
        ),
        hovertemplate='<b>%{y}</b><br>Feature: %{x}<br>Deviation: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Contrastive Analysis: Cluster Deviations from Global Mean',
            font=dict(size=18, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Features',
        yaxis_title='',
        template='plotly_white',
        height=max(400, 80 * len(differences)),
        margin=dict(l=100, r=120, t=80, b=120),
        xaxis=dict(tickangle=-45, side='bottom'),
        font=dict(size=12),
        annotations=annotations
    )
    
    return fig


def extract_decision_rules(tree, feature_names, class_names=None):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != -2 else "undefined"
        for i in tree_.feature
    ]
    
    if class_names is None:
        class_names = [f'Cluster {i}' for i in range(tree.n_classes_)]
    
    rules = []
    
    def recurse(node, depth, conditions):
        indent = "  " * depth
        if tree_.feature[node] != -2:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            
            left_conditions = conditions + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], depth + 1, left_conditions)
            
            right_conditions = conditions + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], depth + 1, right_conditions)
        else:
            samples = tree_.n_node_samples[node]
            class_idx = np.argmax(tree_.value[node])
            
            rule = {
                'conditions': ' AND '.join(conditions),
                'predicted_class': class_names[class_idx],
                'samples': samples,
                'depth': depth
            }
            rules.append(rule)
    
    recurse(0, 0, [])
    
    return pd.DataFrame(rules).sort_values('samples', ascending=False)


def plot_cluster_radar(profiles, feature_cols=None, normalize=True):
    """
    Create interactive radar chart comparing cluster profiles.
    """
    if feature_cols is None:
        feature_cols = profiles.columns.get_level_values(0).unique().tolist()
    
    means = profiles.xs('mean', level=1, axis=1)[feature_cols]
    
    if normalize:
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        means_norm = pd.DataFrame(
            scaler.fit_transform(means.T).T,
            index=means.index,
            columns=means.columns
        )
    else:
        means_norm = means
    
    colors = px.colors.qualitative.Set2
    
    fig = go.Figure()
    
    for idx, (cluster, row) in enumerate(means_norm.iterrows()):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Scatterpolar(
            r=row.values.tolist() + [row.values[0]],
            theta=feature_cols + [feature_cols[0]],
            fill='toself',
            name=f'Cluster {cluster}',
            line=dict(color=color, width=2),
            fillcolor=color,
            opacity=0.3,
            hovertemplate='<b>Cluster %{fullData.name}</b><br>Feature: %{theta}<br>Value: %{r:.3f}<extra></extra>'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1] if normalize else [means_norm.values.min(), means_norm.values.max()],
                showticklabels=True,
                ticks='outside',
                gridcolor='lightgray'
            ),
            angularaxis=dict(
                gridcolor='lightgray'
            )
        ),
        showlegend=True,
        title=dict(
            text='Cluster Profile Comparison (Normalized)' if normalize else 'Cluster Profile Comparison',
            font=dict(size=18, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        ),
        height=600,
        template='plotly_white',
        font=dict(size=12)
    )
    
    return fig


def plot_shap_summary(shap_values, X, feature_names, max_display=10):
    """
    Create interactive SHAP summary plot with directional values.
    Shows both positive (increase) and negative (decrease) impacts.
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else X
    
    # Handle multi-class SHAP values
    if isinstance(shap_values, list):
        # Average across classes
        shap_matrix = np.mean([sv for sv in shap_values], axis=0)
    else:
        shap_matrix = shap_values
    
    # Select top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:max_display]
    top_features = [feature_names[i] for i in top_indices]
    
    # Create figure with subplots for each feature
    fig = make_subplots(
        rows=len(top_features), cols=1,
        subplot_titles=top_features,
        vertical_spacing=0.02,
        shared_xaxes=True
    )
    
    for idx, (feat_idx, feat_name) in enumerate(zip(top_indices, top_features), 1):
        shap_vals = shap_matrix[:, feat_idx]
        feature_vals = X_arr[:, feat_idx]
        
        # Normalize feature values for color mapping
        feat_min, feat_max = feature_vals.min(), feature_vals.max()
        if feat_max - feat_min > 0:
            feat_norm = (feature_vals - feat_min) / (feat_max - feat_min)
        else:
            feat_norm = np.zeros_like(feature_vals)
        
        # Add scatter plot
        fig.add_trace(
            go.Scatter(
                x=shap_vals,
                y=np.random.randn(len(shap_vals)) * 0.2,  # Add jitter
                mode='markers',
                marker=dict(
                    size=6,
                    color=feat_norm,
                    colorscale='RdBu_r',
                    showscale=(idx == 1),
                    colorbar=dict(
                        title=dict(text='Feature<br>Value', side='right'),
                        thickness=15,
                        len=0.3,
                        y=0.85
                    ) if idx == 1 else None,
                    line=dict(width=0.5, color='white'),
                    opacity=0.6
                ),
                text=[f'{feat_name}<br>SHAP: {sv:.3f}<br>Value: {fv:.2f}' 
                      for sv, fv in zip(shap_vals, feature_vals)],
                hovertemplate='%{text}<extra></extra>',
                showlegend=False
            ),
            row=idx, col=1
        )
        
        # Update y-axis for this subplot
        fig.update_yaxes(visible=False, row=idx, col=1)
    
    # Update layout
    fig.update_xaxes(
        title_text='SHAP Value (impact on model output)',
        row=len(top_features), col=1,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor='gray'
    )
    
    fig.update_layout(
        title=dict(
            text='SHAP Summary - Feature Impact Distribution',
            font=dict(size=18, color='#2c3e50'),
            x=0.5,
            xanchor='center'
        ),
        template='plotly_white',
        height=max(500, 80 * len(top_features)),
        margin=dict(l=150, r=100, t=80, b=80),
        font=dict(size=11)
    )
    
    return fig


def plot_feature_importance_comparison(importance_dict, top_n=10):
    """
    Create side-by-side comparison of all importance methods.
    """
    # Filter out scalar metadata entries (e.g. surrogate_fidelity float)
    importance_dict = {k: v for k, v in importance_dict.items() if hasattr(v, 'columns')}

    all_features = set()
    for imp_df in importance_dict.values():
        all_features.update(imp_df.head(top_n)['feature'].tolist())
    
    comparison_data = []
    for method_name, imp_df in importance_dict.items():
        imp_dict = dict(zip(imp_df['feature'], imp_df['importance']))
        for feature in all_features:
            comparison_data.append({
                'Method': method_name.capitalize(),
                'Feature': feature,
                'Importance': imp_dict.get(feature, 0)
            })
    
    comp_df = pd.DataFrame(comparison_data)
    
    pivot_df = comp_df.pivot(index='Feature', columns='Method', values='Importance').fillna(0)
    pivot_df['Mean'] = pivot_df.mean(axis=1)
    pivot_df = pivot_df.sort_values('Mean', ascending=False).head(top_n)
    pivot_df = pivot_df.drop('Mean', axis=1)
    
    colors = px.colors.qualitative.Plotly
    
    fig = go.Figure()
    
    for idx, method in enumerate(pivot_df.columns):
        color = colors[idx % len(colors)]
        fig.add_trace(go.Bar(
            name=method,
            x=pivot_df.index,
            y=pivot_df[method].values,
            marker=dict(color=color, line=dict(color='white', width=1)),
            text=pivot_df[method].round(3).values,
            textposition='outside',
            hovertemplate=f'<b>{method}</b><br>Feature: %{{x}}<br>Importance: %{{y:.4f}}<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text='Feature Importance - Method Comparison',
            font=dict(size=20, color='#2c3e50')
        ),
        xaxis_title='Features',
        yaxis_title='Importance Score',
        barmode='group',
        template='plotly_white',
        height=500,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=80, r=50, t=100, b=120),
        xaxis=dict(tickangle=-45),
        font=dict(size=12)
    )
    
    return fig
