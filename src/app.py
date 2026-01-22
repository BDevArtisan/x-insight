"""
X-Insight: Explainable AI Platform for Unsupervised Learning
Streamlit Application for SSc Clinical Data Clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data.preprocessing import SScPreprocessor
from clustering import (
    KMeansClustering, HierarchicalClustering, KMedoidsClustering,
    GMMClustering, SpectralClusteringWrapper,
    compute_internal_metrics, compare_clustering_methods, profile_clusters
)
from visualization import (
    DimensionalityReducer,
    plot_clusters_2d, plot_clusters_interactive,
    plot_dendrogram, plot_silhouette_analysis,
    plot_elbow_curve, plot_cluster_comparison,
    plot_cluster_profiles, plot_radar_chart
)
from explainability import (
    compute_feature_importance,
    train_surrogate_tree,
    compute_cluster_profiles,
    compute_contrastive_differences,
    plot_feature_importance,
    plot_cluster_comparison_bars,
    plot_contrastive_heatmap,
    plot_cluster_radar,
    plot_shap_summary,
    plot_feature_importance_comparison,
    extract_decision_rules
)

# Page configuration
st.set_page_config(
    page_title="X-Insight: SSc Clustering",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 1rem;
}
.sub-header {
    font-size: 1.2rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔬 X-Insight</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explainable AI for Systemic Sclerosis Patient Clustering</p>', 
           unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload SSc Data (CSV)", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Loaded {len(df)} patients")
    
    # Store in session state
    if 'df' not in st.session_state:
        st.session_state.df = df
else:
    # Load default data
    data_path = Path(__file__).parent.parent / 'data' / 'ssc_synthetic_data.csv'
    if data_path.exists():
        df = pd.read_csv(data_path)
        st.session_state.df = df
        st.sidebar.info(f"Using default data: {len(df)} patients")
    else:
        st.warning("Please upload a CSV file to begin")
        st.stop()

df = st.session_state.df

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data Overview", "🔧 Preprocessing", "🎯 Clustering", "📈 Visualization", "🔍 Explainability"])

# TAB 1: Data Overview
with tab1:
    st.header("Data Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 2)  # Excluding Patient_ID and True_Phenotype
    with col3:
        missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
        st.metric("Missing Data", f"{missing_pct:.1f}%")
    
    st.subheader("Data Sample")
    st.dataframe(df.head(10))
    
    st.subheader("Data Statistics")
    st.dataframe(df.describe())
    
    # Missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame({
        'Feature': missing.index,
        'Missing': missing.values,
        'Percentage': (missing.values / len(df) * 100).round(2)
    })
    missing_df = missing_df[missing_df['Missing'] > 0]
    if not missing_df.empty:
        st.dataframe(missing_df)
    else:
        st.success("No missing values")

# TAB 2: Preprocessing
with tab2:
    st.header("Data Preprocessing")
    
    # Feature selection
    st.subheader("Feature Selection")
    default_features = ['Age', 'Gender', 'mRSS', 'FVC_predicted', 'DLCO_predicted', 
                       'ANA_titer', 'Anti_Scl_70', 'Anti_Centromere', 'CRP', 'ESR', 
                       'Raynauds', 'Digital_Ulcers']
    available_features = [col for col in default_features if col in df.columns]
    
    selected_features = st.multiselect(
        "Select features for clustering",
        available_features,
        default=available_features
    )
    
    if st.button("Preprocess Data", type="primary"):
        with st.spinner("Preprocessing..."):
            # Preprocess
            preprocessor = SScPreprocessor()
            X_scaled, df_imputed = preprocessor.fit_transform(df, selected_features)
            
            # Store in session state
            st.session_state.X_scaled = X_scaled
            st.session_state.df_imputed = df_imputed
            st.session_state.preprocessor = preprocessor
            st.session_state.selected_features = selected_features
            
            st.success("✅ Preprocessing complete!")
            
            # Show results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Before Preprocessing")
                st.dataframe(df[selected_features].head())
            
            with col2:
                st.subheader("After Preprocessing (Imputed)")
                st.dataframe(df_imputed.head())

# TAB 3: Clustering
with tab3:
    st.header("Clustering Analysis")
    
    if 'X_scaled' not in st.session_state:
        st.warning("⚠️ Please preprocess data first (Tab 2)")
        st.stop()
    
    X_scaled = st.session_state.X_scaled
    df_imputed = st.session_state.df_imputed
    
    # Clustering method selection
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Method Selection")
        method = st.selectbox(
            "Clustering Method",
            ["K-Means", "Hierarchical", "K-Medoids", "GMM", "Spectral"]
        )
        
        n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    
    with col2:
        st.subheader("Method Description")
        descriptions = {
            "K-Means": "Partitions data into k clusters by minimizing within-cluster variance. Fast and simple.",
            "Hierarchical": "Builds a tree of clusters. Useful for understanding data hierarchy.",
            "K-Medoids": "Similar to K-Means but uses actual data points as cluster centers.",
            "GMM": "Probabilistic model assuming Gaussian distributions. Provides soft clustering.",
            "Spectral": "Uses graph theory for clustering. Good for non-convex clusters."
        }
        st.info(descriptions[method])
    
    if st.button("Run Clustering", type="primary"):
        with st.spinner(f"Running {method}..."):
            # Run clustering
            if method == "K-Means":
                model = KMeansClustering(n_clusters=n_clusters)
            elif method == "Hierarchical":
                model = HierarchicalClustering(n_clusters=n_clusters)
            elif method == "K-Medoids":
                model = KMedoidsClustering(n_clusters=n_clusters)
            elif method == "GMM":
                model = GMMClustering(n_components=n_clusters)
            elif method == "Spectral":
                model = SpectralClusteringWrapper(n_clusters=n_clusters)
            
            labels = model.fit_predict(X_scaled)
            
            # Store in session state
            st.session_state.labels = labels
            st.session_state.model = model
            st.session_state.method = method
            
            st.success(f"✅ {method} clustering complete!")
            
            # Compute metrics
            metrics = compute_internal_metrics(X_scaled, labels)
            
            # Display metrics
            st.subheader("Clustering Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
                st.caption("Mesure la cohésion et la séparation des clusters. Intervalle: [-1, 1]. Plus élevé est mieux (proche de 1).")
            with col2:
                st.metric("Davies-Bouldin Index", f"{metrics['davies_bouldin']:.3f}")
                st.caption("Mesure la séparation entre clusters. Intervalle: [0, ∞]. Plus faible est mieux (proche de 0).")
            with col3:
                st.metric("Calinski-Harabasz", f"{metrics['calinski_harabasz']:.1f}")
                st.caption("Ratio variance inter-cluster / intra-cluster. Intervalle: [0, ∞]. Plus élevé est mieux.")
            
            # Cluster distribution
            st.subheader("Cluster Distribution")
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            cluster_dist = pd.DataFrame({
                'Cluster': unique,
                'Count': counts,
                'Percentage': (counts / len(labels) * 100).round(1)
            })
            st.dataframe(cluster_dist)
            
            # Cluster profiles
            st.subheader("Cluster Profiles")
            profiles = profile_clusters(df_imputed, labels, st.session_state.selected_features)
            st.dataframe(profiles.round(2))

# TAB 4: Visualization
with tab4:
    st.header("Cluster Visualization")
    
    if 'labels' not in st.session_state:
        st.warning("⚠️ Please run clustering first (Tab 3)")
        st.stop()
    
    X_scaled = st.session_state.X_scaled
    labels = st.session_state.labels
    
    # Dimensionality reduction
    st.subheader("Dimensionality Reduction")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        reduction_method = st.selectbox(
            "Reduction Method",
            ["PCA", "t-SNE", "UMAP"]
        )
    
    if st.button("Generate Visualization", type="primary"):
        with st.spinner(f"Applying {reduction_method}..."):
            # Apply reduction
            method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}
            reducer = DimensionalityReducer(method=method_map[reduction_method], n_components=2)
            X_reduced = reducer.fit_transform(X_scaled)
            
            st.session_state.X_reduced = X_reduced
            st.session_state.reduction_method = reduction_method
            
            # Plot
            st.subheader(f"{reduction_method} Projection")
            
            # Interactive plot
            fig = plot_clusters_interactive(
                X_reduced, labels,
                patient_ids=df['Patient_ID'].values if 'Patient_ID' in df.columns else None,
                method_name=reduction_method,
                title=f'{st.session_state.method} Clustering - {reduction_method} Projection'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Explained variance for PCA
            if reduction_method == "PCA":
                variance = reducer.get_explained_variance()
                st.info(f"Explained Variance: PC1={variance[0]:.1%}, PC2={variance[1]:.1%}, Total={variance.sum():.1%}")
    
    # Additional plots
    if 'X_reduced' in st.session_state:
        st.subheader("Additional Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Silhouette analysis
            st.write("**Silhouette Analysis**")
            fig_sil, _ = plot_silhouette_analysis(X_scaled, labels)
            st.pyplot(fig_sil)
        
        with col2:
            # Cluster profiles radar
            if len(st.session_state.selected_features) <= 8:
                st.write("**Cluster Profiles (Radar)**")
                profiles = profile_clusters(df_imputed, labels, st.session_state.selected_features)
                fig_radar, _ = plot_radar_chart(profiles)
                st.pyplot(fig_radar)

# TAB 5: Explainability
with tab5:
    st.header("Global Explainability")
    
    if 'labels' not in st.session_state or 'X_scaled' not in st.session_state:
        st.warning("Please run clustering first (Tab 3)")
        st.stop()
    
    X_scaled = st.session_state.X_scaled
    labels = st.session_state.labels
    df_imputed = st.session_state.df_imputed
    feature_names = st.session_state.selected_features
    
    st.subheader("Feature Importance")
    
    importance_method = st.selectbox(
        "Importance Method",
        ["All Methods", "Surrogate Tree", "Permutation", "SHAP"]
    )
    
    if st.button("Compute Feature Importance", type="primary"):
        with st.spinner("Computing feature importance..."):
            method_map = {
                "All Methods": "all",
                "Surrogate Tree": "surrogate",
                "Permutation": "permutation",
                "SHAP": "shap"
            }
            
            importances = compute_feature_importance(
                X_scaled, labels,
                feature_names=feature_names,
                method=method_map[importance_method]
            )
            
            st.session_state.importances = importances
            
            st.success("Feature importance computed")
            
            # Show individual method plots if single method selected
            if importance_method != "All Methods":
                fig = plot_feature_importance(importances, top_n=10)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Show comparison plot for all methods
                fig_comp = plot_feature_importance_comparison(importances, top_n=10)
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.subheader("Individual Method Details")
                cols = st.columns(len(importances))
                for idx, (method_name, imp_df) in enumerate(importances.items()):
                    with cols[idx]:
                        st.write(f"**{method_name.capitalize()}**")
                        fig = plot_feature_importance({method_name: imp_df}, top_n=8)
                        st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Importance Scores")
            for method_name, imp_df in importances.items():
                with st.expander(f"{method_name.capitalize()} Importance"):
                    st.dataframe(imp_df)
    
    st.subheader("Cluster Profiling")
    
    if st.button("Generate Cluster Profiles", type="primary"):
        with st.spinner("Computing cluster profiles..."):
            profiles = compute_cluster_profiles(df_imputed, labels, feature_names)
            differences, effect_sizes = compute_contrastive_differences(df_imputed, labels, feature_names)
            
            st.session_state.profiles = profiles
            st.session_state.differences = differences
            st.session_state.effect_sizes = effect_sizes
            
            st.success("Cluster profiles computed")
            
            # Radar chart
            st.subheader("Cluster Profile Overview")
            fig_radar = plot_cluster_radar(profiles, feature_names, normalize=True)
            st.plotly_chart(fig_radar, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Mean Values per Cluster**")
                st.dataframe(profiles.xs('mean', level=1, axis=1).round(2))
            
            with col2:
                st.write("**Standard Deviation per Cluster**")
                st.dataframe(profiles.xs('std', level=1, axis=1).round(2))
            
            st.subheader("Feature Comparison Across Clusters")
            fig_comp = plot_cluster_comparison_bars(profiles, feature_names)
            st.plotly_chart(fig_comp, use_container_width=True)
            
            st.subheader("Contrastive Analysis")
            st.write("Differences from global mean")
            
            fig_heatmap = plot_contrastive_heatmap(differences)
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            st.write("**Effect Sizes**")
            st.dataframe(effect_sizes.round(2))
    
    st.subheader("Decision Rules")
    
    if st.button("Extract Decision Rules", type="primary"):
        with st.spinner("Training surrogate tree and extracting rules..."):
            tree, _ = train_surrogate_tree(X_scaled, labels, feature_names, max_depth=4)
            
            class_names = [f'Cluster {i}' for i in range(len(np.unique(labels)))]
            rules_df = extract_decision_rules(tree, feature_names, class_names)
            
            st.success("Decision rules extracted")
            
            st.write(f"**Top {min(10, len(rules_df))} Decision Rules**")
            st.dataframe(rules_df.head(10))
            
            st.subheader("Decision Tree Visualization")
            fig, ax = plt.subplots(figsize=(20, 10))
            from sklearn.tree import plot_tree
            plot_tree(tree, feature_names=feature_names, class_names=class_names,
                     filled=True, rounded=True, ax=ax, fontsize=10)
            st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>X-Insight - Explainable AI Platform for Systemic Sclerosis Analysis</p>
    <p>Developed for unsupervised learning with transparency and clinical interpretability</p>
</div>
""", unsafe_allow_html=True)
