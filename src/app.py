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
from ydata_profiling import ProfileReport

# Add src to path
src_path = Path(__file__).parent
sys.path.insert(0, str(src_path))

from data.preprocessing import BasePreprocessor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from clustering import (
    KMeansClustering,
    HierarchicalClustering,
    KMedoidsClustering,
    GMMClustering,
    SpectralClusteringWrapper,
    DECClustering,
    IDECClustering,
    DCNClustering,
    HDBSCANClustering,
    GMMAutoClustering,
    compute_internal_metrics,
    compare_clustering_methods,
    profile_clusters,
    determine_optimal_k,
)
from visualization import (
    DimensionalityReducer,
    plot_clusters_2d,
    plot_clusters_interactive,
    plot_dendrogram,
    plot_silhouette_analysis,
    plot_elbow_curve,
    plot_optimal_k_analysis,
    plot_cluster_comparison,
    plot_cluster_profiles,
    plot_radar_chart,
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
    extract_decision_rules,
    explain_patient,
    plot_shap_explanation,
    plot_lime_explanation,
    plot_distance_explanation,
    plot_probabilistic_explanation,
    plot_patient_explanation_summary,
)
from counterfactuals import (
    ClinicalConstraints,
    generate_counterfactual_optimization,
    generate_counterfactual_dice_style,
    format_counterfactual_explanation,
    plot_counterfactual_comparison,
    plot_diverse_counterfactuals,
)
from llm import interpret_visualization
from llm.prompts import (
    VISUALIZATION_PROMPT,
    GLOBAL_EXPLAINABILITY_PROMPT,
    LOCAL_EXPLAINABILITY_PROMPT,
    COUNTERFACTUAL_PROMPT,
)


def extract_figure(plot_result):
    """Extract matplotlib/plotly figure from plot result (handles tuples)"""
    if isinstance(plot_result, tuple):
        return plot_result[0]
    return plot_result


# Page configuration
st.set_page_config(
    page_title="X-Insight: SSc Clustering",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# Title
st.markdown('<h1 class="main-header">🔬 X-Insight</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">Explainable AI for Systemic Sclerosis Patient Clustering</p>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Configuration")

st.sidebar.subheader("LLM Settings")
api_key = st.sidebar.text_input(
    "Google AI API Key", type="password", help="Enter your Google AI Studio API key"
)

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload SSc Data",
    type=["xlsx", "xls", "csv"],
    help="Upload your file with patients data (.xlsx, .xls or .csv)",
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.sidebar.success(
            f"Uploaded file: {len(df)} patients, {len(df.columns)} columns"
        )
    except Exception as e:
        st.sidebar.error(f"Download error: {e}")
        st.stop()
else:
    if "df" not in st.session_state:
        st.info("Please upload a file in the sidebar to begin the analysis")
        st.stop()

df = st.session_state.df

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [
        "📊 Data Overview",
        "🔧 Preprocessing",
        "🎯 Clustering",
        "📈 Visualization",
        "🔍 Global Explainability",
        "👤 Local Explainability",
        "🔄 Counterfactuals",
    ]
)

# TAB 1: Data Overview
with tab1:
    st.header("Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric(
            "Features", len(df.columns) - 2
        )  # Excluding Patient_ID and True_Phenotype
    with col3:
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
        st.metric("Missing Data", f"{missing_pct:.1f}%")

    st.subheader("Data Sample")
    st.dataframe(df.head(10))

    st.subheader("Data Statistics")
    st.dataframe(df.describe())

    # Missing values
    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing_df = pd.DataFrame(
        {
            "Feature": missing.index,
            "Missing": missing.values,
            "Percentage": (missing.values / len(df) * 100).round(2),
        }
    )
    missing_df = missing_df[missing_df["Missing"] > 0]
    if not missing_df.empty:
        st.dataframe(missing_df)
    else:
        st.success("No missing values")

    st.subheader("Data Profiling Report")
    if st.button("Generate profiling report", type="primary"):
        with st.spinner("Generating profiling report..."):
            profile = ProfileReport(
                df, minimal=False, explorative=True, html={"minify_html": False}
            )

            html_report = profile.to_html()

            st.download_button(
                label="Download profiling report",
                data=html_report,
                file_name="profiling_report.html",
                mime="text/html",
            )

# TAB 2: Preprocessing
with tab2:
    st.header("Data Preprocessing")

    use_preprocessing = st.checkbox("Enable preprocessing", value=True)

    if use_preprocessing:
        st.subheader("K Selection for KNN Imputation")

        if st.button("Find Optimal K", type="secondary"):
            with st.spinner("Testing different K values..."):
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                numeric_data = df[numeric_cols].copy()

                if numeric_data.isnull().sum().sum() == 0:
                    st.warning(
                        "No missing values found. K optimization requires missing data."
                    )
                else:
                    k_range = range(2, 16)
                    mse_scores = []

                    for k in k_range:
                        imputer = KNNImputer(n_neighbors=k)

                        mask = ~numeric_data.isnull()
                        available_data = numeric_data[mask]

                        if len(available_data) > 100:
                            sample_indices = np.random.choice(
                                len(available_data), 100, replace=False
                            )
                        else:
                            sample_indices = range(len(available_data))

                        errors = []
                        for idx in sample_indices:
                            row_idx, col_idx = np.unravel_index(
                                np.where(mask.values.flatten())[0][idx], mask.shape
                            )

                            test_data = numeric_data.copy()
                            true_value = test_data.iloc[row_idx, col_idx]
                            test_data.iloc[row_idx, col_idx] = np.nan

                            try:
                                imputed = imputer.fit_transform(test_data)
                                predicted_value = imputed[row_idx, col_idx]
                                errors.append((true_value - predicted_value) ** 2)
                            except:
                                continue

                        if errors:
                            mse_scores.append(np.mean(errors))
                        else:
                            mse_scores.append(np.nan)

                    valid_scores = [
                        (k, score)
                        for k, score in zip(k_range, mse_scores)
                        if not np.isnan(score)
                    ]

                    if valid_scores:
                        optimal_k = min(valid_scores, key=lambda x: x[1])[0]

                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.plot(
                            k_range,
                            mse_scores,
                            marker="o",
                            color="steelblue",
                            linewidth=2,
                        )
                        ax.axvline(
                            optimal_k,
                            color="red",
                            linestyle="--",
                            linewidth=2,
                            label=f"Optimal K={optimal_k}",
                        )
                        ax.scatter(
                            [optimal_k],
                            [mse_scores[optimal_k - 2]],
                            color="red",
                            s=100,
                            zorder=5,
                        )
                        ax.set_xlabel("Number of Neighbors (K)", fontsize=12)
                        ax.set_ylabel("Mean Squared Error", fontsize=12)
                        ax.set_title(
                            "KNN Imputation - K Selection via Cross-Validation",
                            fontsize=14,
                        )
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        st.pyplot(fig)

                        st.success(f"Suggested optimal K: {optimal_k}")
                        st.info(
                            f"MSE at K={optimal_k}: {mse_scores[optimal_k - 2]:.4f}"
                        )
                    else:
                        st.error("Could not compute optimal K. Check your data.")

        st.subheader("Preprocessing Options")

        col1, col2 = st.columns(2)

        with col1:
            remove_duplicates = st.checkbox("Remove duplicates", value=True)
            n_neighbors = st.slider("KNN imputation — number of neighbors", 2, 15, 5)

        with col2:
            outlier_method = st.selectbox(
                "Outlier detection method", ["None", "Z-Score", "IQR"], index=0
            )
            if outlier_method != "None":
                outlier_threshold = st.slider(
                    "Outlier threshold",
                    1.5,
                    5.0,
                    3.0,
                    0.5,
                    help="Z-Score: typically 3.0 | IQR: typically 1.5",
                )
                remove_outlier_patients = st.checkbox(
                    "Remove outlier patients",
                    value=False,
                    help="Remove patients with outlier values from the analysis",
                )
            else:
                outlier_threshold = 3.0
                remove_outlier_patients = False

    if st.button("Process Data", type="primary"):
        with st.spinner("Processing..."):
            if use_preprocessing:
                outlier_method_map = {"None": None, "Z-Score": "zscore", "IQR": "iqr"}

                preprocessor = BasePreprocessor(
                    n_neighbors=n_neighbors,
                    remove_duplicates=remove_duplicates,
                    outlier_method=outlier_method_map[outlier_method],
                    outlier_threshold=outlier_threshold,
                )

                X_scaled, df_processed, feature_names = preprocessor.fit_transform(
                    df, remove_outlier_rows=remove_outlier_patients
                )

                st.session_state.X_scaled = X_scaled
                st.session_state.df_imputed = df_processed
                st.session_state.preprocessor = preprocessor
                st.session_state.selected_features = feature_names
                st.session_state.df_raw = df

                st.success(
                    f"Preprocessing done — {X_scaled.shape[0]} patients, {X_scaled.shape[1]} features"
                )

                if preprocessor.duplicates_removed > 0:
                    st.info(f"Duplicates removed: {preprocessor.duplicates_removed}")

                if len(preprocessor.outlier_indices) > 0:
                    n_outlier_patients = len(preprocessor.outlier_indices)
                    total_outlier_values = sum(preprocessor.outliers_detected.values())

                    if remove_outlier_patients:
                        st.info(
                            f"Outliers removed: {n_outlier_patients} patient(s) with {total_outlier_values} outlier value(s)"
                        )
                    else:
                        st.warning(
                            f"Outliers detected: {n_outlier_patients} patient(s) with {total_outlier_values} outlier value(s)"
                        )

                    with st.expander("Outlier details"):
                        col_a, col_b = st.columns(2)

                        with col_a:
                            st.write("**Outliers by feature:**")
                            outlier_df = pd.DataFrame(
                                [
                                    {"Feature": k, "Outliers": v}
                                    for k, v in preprocessor.outliers_detected.items()
                                    if v > 0
                                ]
                            )
                            st.dataframe(outlier_df, use_container_width=True)

                        with col_b:
                            st.write("**Patient IDs with outliers:**")
                            patient_id_col = (
                                "Patient_ID" if "Patient_ID" in df.columns else None
                            )

                            if patient_id_col:
                                outlier_patient_ids = df.iloc[
                                    preprocessor.outlier_indices
                                ][patient_id_col].values
                                outlier_list_df = pd.DataFrame(
                                    {
                                        "Patient_ID": outlier_patient_ids,
                                        "Index": preprocessor.outlier_indices,
                                    }
                                )
                                st.dataframe(outlier_list_df, use_container_width=True)
                            else:
                                outlier_list_df = pd.DataFrame(
                                    {"Index": preprocessor.outlier_indices}
                                )
                                st.dataframe(outlier_list_df, use_container_width=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Raw data (first 5 rows)")
                    st.dataframe(df.head())
                with col2:
                    st.subheader("After preprocessing (first 5 rows)")
                    st.dataframe(df_processed.head())

                st.subheader("Column classification")
                col_info = []
                for c in preprocessor.binary_cols:
                    col_info.append(
                        {
                            "Column": c,
                            "Type": "binary",
                            "Encoding": "LabelEncoder (0/1)",
                            "Scaling": "none",
                        }
                    )
                for c in preprocessor.ohe_cols:
                    col_info.append(
                        {
                            "Column": c,
                            "Type": "nominal",
                            "Encoding": "OneHotEncoder (drop first)",
                            "Scaling": "none",
                        }
                    )
                for c in preprocessor.continuous_cols:
                    scaler_type = type(
                        preprocessor.scalers.get(c, StandardScaler())
                    ).__name__
                    col_info.append(
                        {
                            "Column": c,
                            "Type": "continuous",
                            "Encoding": "none",
                            "Scaling": scaler_type,
                        }
                    )
                st.dataframe(pd.DataFrame(col_info), use_container_width=True)
            else:
                numeric_df = df.select_dtypes(include=[np.number])
                numeric_df = numeric_df.fillna(0)

                st.session_state.X_scaled = numeric_df.values
                st.session_state.df_imputed = numeric_df
                st.session_state.preprocessor = None
                st.session_state.selected_features = list(numeric_df.columns)

                st.success(
                    f"Data ready — {numeric_df.shape[0]} patients, {numeric_df.shape[1]} features"
                )
                st.dataframe(numeric_df.head())

# TAB 3: Clustering
with tab3:
    st.header("Clustering Analysis")

    if "X_scaled" not in st.session_state:
        st.warning("Please process data first (Tab 2)")
    else:
        X_scaled = st.session_state.X_scaled
        df_imputed = st.session_state.df_imputed

        # Methods that need the user to pick k
        ELBOW_METHODS = ["K-Means", "Hierarchical", "K-Medoids"]  # elbow curve + slider
        MANUAL_K_METHODS = ELBOW_METHODS + [
            "GMM",
            "Spectral",
            "DEC",
            "IDEC",
            "DCN",
        ]  # slider only
        AUTO_K_METHODS = ["HDBSCAN", "GMM (Auto-BIC)"]  # no slider
        COMPARE_MODE = "🔬 Compare All Methods"

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Method Selection")
            method = st.selectbox(
                "Clustering Method", [COMPARE_MODE] + MANUAL_K_METHODS + AUTO_K_METHODS
            )

            if method == COMPARE_MODE:
                st.info("Compare all clustering algorithms with the same k value.")
                n_clusters = st.slider(
                    "Number of Clusters (k) for comparison", 2, 10, 3
                )
                hdbscan_min_size = 5  # Default for HDBSCAN

                # Add Analyze Optimal K button for comparison mode
                if st.button("Analyze Optimal K (K-Means)"):
                    with st.spinner("Computing optimal k analysis..."):
                        k_range = range(2, 11)
                        results = determine_optimal_k(
                            X_scaled, k_range=k_range, method="kmeans", metric="multi"
                        )

                        # Display individual recommendations
                        st.subheader("📊 Optimal K Recommendations by Metric")

                        fig, recommendations = plot_optimal_k_analysis(results)
                        st.pyplot(fig)

                        # Display recommendations table
                        col_a, col_b, col_c, col_d = st.columns(4)
                        with col_a:
                            st.metric(
                                "🔵 Silhouette",
                                f"k = {recommendations['silhouette']}",
                                help="Measure clusters separation (↑ higher is better)",
                            )
                        with col_b:
                            st.metric(
                                "🟣 Davies-Bouldin",
                                f"k = {recommendations['davies_bouldin']}",
                                help="Intra/inter clusters dispersion ratio (↓ lower is better)",
                            )
                        with col_c:
                            st.metric(
                                "🟠 Calinski-Harabasz",
                                f"k = {recommendations['calinski_harabasz']}",
                                help="Inter/intra clusters variance ratio (↑ higher is better)",
                            )
                        with col_d:
                            st.metric(
                                "🟢 Combined Score",
                                f"k = {recommendations['combined']}",
                                help="Weighted score combining Silhouette, Davies-Bouldin and Calinski",
                            )

                        # Voting summary
                        votes = list(recommendations.values())
                        most_common = max(set(votes), key=votes.count)
                        vote_count = votes.count(most_common)

                        if vote_count >= 3:
                            st.success(
                                f"✅ Strong consensus: k = {most_common} recommended by your {vote_count}/4 metrics"
                            )
                        elif vote_count == 2:
                            st.info(
                                f"⚖️ Unclear consensus: k = {most_common} suggested by 2 metrics. You can consider: {set(votes) - {most_common}}"
                            )
                        else:
                            st.warning(
                                f"⚠️ No consensus: You can use your expertise or the weighted-score (k = {recommendations['combined']})"
                            )

                        # Quality assessment
                        quality = results.get("quality", "unknown")
                        quality_emojis = {
                            "excellent": "🌟",
                            "good": "✅",
                            "moderate": "⚠️",
                            "weak": "❌",
                        }
                        quality_msgs = {
                            "excellent": "Excellent cluster structure (Silhouette ≥ 0.7)",
                            "good": "Good cluster structure (Silhouette ≥ 0.5)",
                            "moderate": "Moderate cluster structure (Silhouette ≥ 0.25)",
                            "weak": "Weak cluster structure (Silhouette < 0.25)",
                        }
                        if quality in quality_emojis:
                            st.info(
                                f"{quality_emojis[quality]} {quality_msgs[quality]}"
                            )
            elif method in MANUAL_K_METHODS:
                if method in ELBOW_METHODS:
                    st.info(
                        "This method requires choosing the number of clusters. Use the analysis below as a guide."
                    )
                    if st.button("Analyze Optimal K"):
                        with st.spinner("Computing optimal k analysis..."):
                            method_map = {
                                "K-Means": "kmeans",
                                "Hierarchical": "hierarchical",
                                "K-Medoids": "kmedoids",
                            }
                            k_range = range(2, 11)
                            results = determine_optimal_k(
                                X_scaled,
                                k_range=k_range,
                                method=method_map[method],
                                metric="multi",
                            )

                            # Display individual recommendations
                            st.subheader("📊 Optimal K Recommendations by Metric")

                            fig, recommendations = plot_optimal_k_analysis(results)
                            st.pyplot(fig)

                            # Display recommendations table
                            col_a, col_b, col_c, col_d = st.columns(4)
                            with col_a:
                                st.metric(
                                    "🔵 Silhouette",
                                    f"k = {recommendations['silhouette']}",
                                    help="Measure clusters separation (↑ higher is better)",
                                )
                            with col_b:
                                st.metric(
                                    "🟣 Davies-Bouldin",
                                    f"k = {recommendations['davies_bouldin']}",
                                    help="Intra/inter clusters dispersion ratio (↓ lower is better)",
                                )
                            with col_c:
                                st.metric(
                                    "🟠 Calinski-Harabasz",
                                    f"k = {recommendations['calinski_harabasz']}",
                                    help="Inter/intra clusters variance ratio (↑ higher is better)",
                                )
                            with col_d:
                                st.metric(
                                    "🟢 Combined Score",
                                    f"k = {recommendations['combined']}",
                                    help="Weighted score combining Silhouette, Davies-Bouldin and Calinski",
                                )

                            # Voting summary
                            votes = list(recommendations.values())
                            most_common = max(set(votes), key=votes.count)
                            vote_count = votes.count(most_common)

                            if vote_count >= 3:
                                st.success(
                                    f"✅ Strong consensus: k = {most_common} recommended by your {vote_count}/4 metrics"
                                )
                            elif vote_count == 2:
                                st.info(
                                    f"⚖️ Unclear consensus: k = {most_common} suggested by 2 metrics. You can consider: {set(votes) - {most_common}}"
                                )
                            else:
                                st.warning(
                                    f"⚠️ No consensus: You can use your expertise or the weighted-score (k = {recommendations['combined']})"
                                )

                            # Quality assessment
                            quality = results.get("quality", "unknown")
                            quality_emojis = {
                                "excellent": "🌟",
                                "good": "✅",
                                "moderate": "⚠️",
                                "weak": "❌",
                            }
                            quality_msgs = {
                                "excellent": "Excellent cluster structure (Silhouette ≥ 0.7)",
                                "good": "Good cluster structure (Silhouette ≥ 0.5)",
                                "moderate": "Moderate cluster structure (Silhouette ≥ 0.25)",
                                "weak": "Weak cluster structure (Silhouette < 0.25)",
                            }
                            if quality in quality_emojis:
                                st.info(
                                    f"{quality_emojis[quality]} {quality_msgs[quality]}"
                                )
                else:
                    st.info(
                        "This method requires specifying the number of clusters (k)."
                    )
                n_clusters = st.slider("Number of Clusters (k)", 2, 10, 3)
                hdbscan_min_size = None  # not used
            else:
                # Auto-k methods
                n_clusters = None
                if method == "HDBSCAN":
                    st.info(
                        "HDBSCAN automatically finds the number of clusters based on data density. "
                        "Set the minimum number of patients required to form a cluster."
                    )
                    hdbscan_min_size = st.slider(
                        "Minimum Cluster Size (patients)", 2, 20, 5
                    )
                else:  # GMM Auto-BIC
                    st.info(
                        "GMM with BIC automatically tests k = 2 to 10 and selects the number of "
                        "clusters that best fits the data statistically."
                    )
                    hdbscan_min_size = None

        with col2:
            if method == COMPARE_MODE:
                st.subheader("Comparison Mode")
                st.info(
                    "🔬 **Comparison Mode** will execute all clustering algorithms and display a comparative table with metrics.\n\n"
                    "**Algorithms included:**\n"
                    "- K-Means, Hierarchical, K-Medoids\n"
                    "- GMM, Spectral\n"
                    "- DEC, IDEC, DCN (Deep Learning)\n"
                    "- HDBSCAN (auto-k), GMM Auto-BIC (auto-k)\n\n"
                    "⏱️ This may take a few moments, especially for deep learning methods."
                )
            else:
                descriptions = {
                    "K-Means": "Partitions data into k clusters by minimizing within-cluster variance. Fast and simple. Use the elbow curve to choose k.",
                    "Hierarchical": "Builds a tree of clusters (dendrogram). Useful for understanding data hierarchy. Use the elbow curve to choose k.",
                    "K-Medoids": "Like K-Means but uses actual data points as cluster centers — more robust to outliers. Use the elbow curve to choose k.",
                    "GMM": "Probabilistic model that fits k Gaussian distributions to the data. Provides soft (probabilistic) cluster membership. k must be specified.",
                    "Spectral": "Graph-based method that clusters based on data connectivity. Effective for non-convex cluster shapes. k must be specified.",
                    "DEC": "Deep Embedded Clustering: trains an autoencoder to learn a compact representation, then clusters in latent space. k sets the number of latent clusters.",
                    "IDEC": "Improved DEC: adds a reconstruction loss to better preserve local data structure during clustering. k sets the number of latent clusters.",
                    "DCN": "Deep Clustering Network: jointly optimizes reconstruction and clustering objectives with structural constraints. k sets the number of latent clusters.",
                    "HDBSCAN": "🤖 AUTO-K — Density-based clustering that automatically finds the number of clusters. Robust to noise and outlier patients. No k needed.",
                    "GMM (Auto-BIC)": "🤖 AUTO-K — Gaussian Mixture Model that automatically selects the optimal number of clusters by minimizing the Bayesian Information Criterion (BIC). No k needed.",
                }
                st.subheader("Method Description")
                st.info(descriptions[method])

        button_label = "Run Comparison" if method == COMPARE_MODE else "Run Clustering"
        if st.button(button_label, type="primary"):
            if method == COMPARE_MODE:
                # Comparison mode - run all methods
                with st.spinner("Running all clustering algorithms..."):
                    methods_labels = {}
                    methods_models = {}

                    # Traditional methods with k
                    progress_bar = st.progress(0)
                    total_methods = 10
                    current = 0

                    st.write("Running K-Means...")
                    model = KMeansClustering(n_clusters=n_clusters)
                    methods_labels["K-Means"] = model.fit_predict(X_scaled)
                    methods_models["K-Means"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running Hierarchical...")
                    model = HierarchicalClustering(n_clusters=n_clusters)
                    methods_labels["Hierarchical"] = model.fit_predict(X_scaled)
                    methods_models["Hierarchical"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running K-Medoids...")
                    model = KMedoidsClustering(n_clusters=n_clusters)
                    methods_labels["K-Medoids"] = model.fit_predict(X_scaled)
                    methods_models["K-Medoids"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running GMM...")
                    model = GMMClustering(n_components=n_clusters)
                    methods_labels["GMM"] = model.fit_predict(X_scaled)
                    methods_models["GMM"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running Spectral...")
                    model = SpectralClusteringWrapper(n_clusters=n_clusters)
                    methods_labels["Spectral"] = model.fit_predict(X_scaled)
                    methods_models["Spectral"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running DEC (Deep Learning)...")
                    try:
                        model = DECClustering(
                            n_clusters=n_clusters,
                            pretrain_epochs=10,
                            clustering_epochs=20,
                        )
                        methods_labels["DEC"] = model.fit_predict(X_scaled)
                        methods_models["DEC"] = model
                    except Exception as e:
                        st.warning(f"DEC failed: {e}")
                        methods_labels["DEC"] = np.zeros(len(X_scaled))
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running IDEC (Deep Learning)...")
                    try:
                        model = IDECClustering(
                            n_clusters=n_clusters,
                            pretrain_epochs=10,
                            clustering_epochs=20,
                        )
                        methods_labels["IDEC"] = model.fit_predict(X_scaled)
                        methods_models["IDEC"] = model
                    except Exception as e:
                        st.warning(f"IDEC failed: {e}")
                        methods_labels["IDEC"] = np.zeros(len(X_scaled))
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running DCN (Deep Learning)...")
                    try:
                        model = DCNClustering(
                            n_clusters=n_clusters,
                            pretrain_epochs=10,
                            clustering_epochs=20,
                        )
                        methods_labels["DCN"] = model.fit_predict(X_scaled)
                        methods_models["DCN"] = model
                    except Exception as e:
                        st.warning(f"DCN failed: {e}")
                        methods_labels["DCN"] = np.zeros(len(X_scaled))
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running HDBSCAN (Auto-k)...")
                    model = HDBSCANClustering(min_cluster_size=hdbscan_min_size)
                    methods_labels["HDBSCAN"] = model.fit_predict(X_scaled)
                    methods_models["HDBSCAN"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    st.write("Running GMM Auto-BIC (Auto-k)...")
                    model = GMMAutoClustering(max_k=10)
                    methods_labels["GMM Auto-BIC"] = model.fit_predict(X_scaled)
                    methods_models["GMM Auto-BIC"] = model
                    current += 1
                    progress_bar.progress(current / total_methods)

                    progress_bar.progress(1.0)
                    st.success("✅ All algorithms completed!")

                    # Generate comparison table
                    comparison_df = compare_clustering_methods(
                        X_scaled, methods_labels=methods_labels
                    )

                    # Sort by silhouette score (descending)
                    comparison_sorted = comparison_df.sort_values(
                        "silhouette", ascending=False
                    )

                    # Select best method based on Silhouette score for subsequent tabs
                    best_method_name = comparison_sorted.iloc[0]["Method"]
                    best_labels = methods_labels[best_method_name]
                    best_model = methods_models[best_method_name]

                    # Store best method in session state for other tabs
                    st.session_state.labels = best_labels
                    st.session_state.model = best_model
                    st.session_state.method = (
                        f"{best_method_name} (Best from Comparison)"
                    )

                    # Store comparison results in session state
                    st.session_state.comparison_df = comparison_df
                    st.session_state.methods_labels = methods_labels
                    st.session_state.methods_models = methods_models

                    # Display results
                    st.subheader("📊 Comparative Analysis")

                    # Highlight best method for each metric
                    def highlight_best(s):
                        if s.name == "silhouette":
                            return [
                                "background-color: lightgreen" if v == s.max() else ""
                                for v in s
                            ]
                        elif s.name == "davies_bouldin":
                            return [
                                "background-color: lightgreen" if v == s.min() else ""
                                for v in s
                            ]
                        elif s.name == "calinski_harabasz":
                            return [
                                "background-color: lightgreen" if v == s.max() else ""
                                for v in s
                            ]
                        else:
                            return ["" for _ in s]

                    # Format and display
                    styled_df = comparison_sorted.style.format(
                        {
                            "silhouette": "{:.3f}",
                            "davies_bouldin": "{:.3f}",
                            "calinski_harabasz": "{:.1f}",
                        }
                    ).apply(
                        highlight_best,
                        subset=["silhouette", "davies_bouldin", "calinski_harabasz"],
                    )

                    st.dataframe(styled_df, use_container_width=True)

                    # Recommendations
                    best_silhouette = comparison_sorted.iloc[0]["Method"]
                    best_db = comparison_df.loc[
                        comparison_df["davies_bouldin"].idxmin(), "Method"
                    ]
                    best_ch = comparison_df.loc[
                        comparison_df["calinski_harabasz"].idxmax(), "Method"
                    ]

                    st.subheader("🏆 Recommendations")
                    col_rec1, col_rec2, col_rec3 = st.columns(3)
                    with col_rec1:
                        st.metric(
                            "🥇 Best Silhouette",
                            best_silhouette,
                            f"{comparison_df[comparison_df['Method'] == best_silhouette]['silhouette'].values[0]:.3f}",
                        )
                    with col_rec2:
                        st.metric(
                            "🥇 Best Davies-Bouldin",
                            best_db,
                            f"{comparison_df[comparison_df['Method'] == best_db]['davies_bouldin'].values[0]:.3f}",
                        )
                    with col_rec3:
                        st.metric(
                            "🥇 Best Calinski-Harabasz",
                            best_ch,
                            f"{comparison_df[comparison_df['Method'] == best_ch]['calinski_harabasz'].values[0]:.1f}",
                        )

                    # Visual comparison (always shown)
                    st.subheader("📈 Visual Comparison")
                    fig_comp, _ = plot_cluster_comparison(
                        comparison_sorted,
                        metrics=["silhouette", "davies_bouldin", "calinski_harabasz"],
                    )
                    st.pyplot(fig_comp)

                    st.success(
                        f"✅ **{best_method_name}** selected for subsequent analysis tabs (best Silhouette score)"
                    )

                    # Display metrics for the selected best method
                    st.subheader(f"📊 {best_method_name} - Detailed Metrics")
                    valid_mask = best_labels >= 0
                    if valid_mask.sum() > 0 and len(set(best_labels[valid_mask])) > 1:
                        metrics = compute_internal_metrics(
                            X_scaled[valid_mask], best_labels[valid_mask]
                        )

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Silhouette Score", f"{metrics['silhouette']:.3f}"
                            )
                            st.caption("[-1, 1] — higher is better")
                        with col2:
                            st.metric(
                                "Davies-Bouldin Index",
                                f"{metrics['davies_bouldin']:.3f}",
                            )
                            st.caption("[0, inf] — lower is better")
                        with col3:
                            st.metric(
                                "Calinski-Harabasz",
                                f"{metrics['calinski_harabasz']:.1f}",
                            )
                            st.caption("[0, inf] — higher is better")

                    # Display cluster distribution
                    st.subheader("Cluster Distribution")
                    n_found = len(set(best_labels) - {-1})
                    noise_count = int(np.sum(best_labels == -1))
                    unique, counts = np.unique(
                        best_labels[best_labels >= 0], return_counts=True
                    )
                    cluster_dist = pd.DataFrame(
                        {
                            "Cluster": unique,
                            "Count": counts,
                            "Percentage": (counts / len(best_labels) * 100).round(1),
                        }
                    )
                    if noise_count:
                        noise_row = pd.DataFrame(
                            [
                                {
                                    "Cluster": -1,
                                    "Count": noise_count,
                                    "Percentage": round(
                                        noise_count / len(best_labels) * 100, 1
                                    ),
                                }
                            ]
                        )
                        cluster_dist = pd.concat(
                            [cluster_dist, noise_row], ignore_index=True
                        )
                    st.dataframe(cluster_dist)

                    st.info(
                        "💡 **Tip:** This method will be used in the Visualization, Explainability, and Counterfactual tabs. "
                        "You can view individual cluster details for this method in the following tabs."
                    )

            else:
                # Single method mode (original behavior)
                with st.spinner(f"Running {method}..."):
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
                    elif method == "DEC":
                        model = DECClustering(
                            n_clusters=n_clusters,
                            pretrain_epochs=10,
                            clustering_epochs=20,
                        )
                    elif method == "IDEC":
                        model = IDECClustering(
                            n_clusters=n_clusters,
                            pretrain_epochs=10,
                            clustering_epochs=20,
                        )
                    elif method == "DCN":
                        model = DCNClustering(
                            n_clusters=n_clusters,
                            pretrain_epochs=10,
                            clustering_epochs=20,
                        )
                    elif method == "HDBSCAN":
                        model = HDBSCANClustering(min_cluster_size=hdbscan_min_size)
                    elif method == "GMM (Auto-BIC)":
                        model = GMMAutoClustering(max_k=10)

                    labels = model.fit_predict(X_scaled)

                    st.session_state.labels = labels
                    st.session_state.model = model
                    st.session_state.method = method

                    n_found = len(set(labels) - {-1})
                    noise_count = int(np.sum(labels == -1))

                    if method == "HDBSCAN":
                        st.success(
                            f"HDBSCAN found **{n_found} clusters** automatically."
                            + (
                                f" {noise_count} patient(s) marked as noise (outliers)."
                                if noise_count
                                else ""
                            )
                        )
                    elif method == "GMM (Auto-BIC)":
                        st.success(
                            f"GMM (Auto-BIC) selected **k = {model.best_k_}** clusters automatically."
                        )
                    else:
                        st.success(
                            f"{method} clustering complete — {len(np.unique(labels))} clusters found"
                        )

                    # Only compute metrics on non-noise points
                    valid_mask = labels >= 0
                    if valid_mask.sum() > 0 and len(set(labels[valid_mask])) > 1:
                        metrics = compute_internal_metrics(
                            X_scaled[valid_mask], labels[valid_mask]
                        )

                        st.subheader("Clustering Metrics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Silhouette Score", f"{metrics['silhouette']:.3f}"
                            )
                            st.caption("[-1, 1] — higher is better")
                        with col2:
                            st.metric(
                                "Davies-Bouldin Index",
                                f"{metrics['davies_bouldin']:.3f}",
                            )
                            st.caption("[0, inf] — lower is better")
                        with col3:
                            st.metric(
                                "Calinski-Harabasz",
                                f"{metrics['calinski_harabasz']:.1f}",
                            )
                            st.caption("[0, inf] — higher is better")

                    st.subheader("Cluster Distribution")
                    unique, counts = np.unique(labels[labels >= 0], return_counts=True)
                    cluster_dist = pd.DataFrame(
                        {
                            "Cluster": unique,
                            "Count": counts,
                            "Percentage": (counts / len(labels) * 100).round(1),
                        }
                    )
                    if noise_count:
                        noise_row = pd.DataFrame(
                            [
                                {
                                    "Cluster": -1,
                                    "Count": noise_count,
                                    "Percentage": round(
                                        noise_count / len(labels) * 100, 1
                                    ),
                                }
                            ]
                        )
                        cluster_dist = pd.concat(
                            [cluster_dist, noise_row], ignore_index=True
                        )
                    st.dataframe(cluster_dist)

                    st.subheader("Cluster Profiles")
                    profiles = profile_clusters(
                        df_imputed[valid_mask],
                        labels[valid_mask],
                        st.session_state.selected_features,
                    )
                    st.dataframe(profiles.round(2))

# TAB 4: Visualization
with tab4:
    st.header("Cluster Visualization")

    if "labels" not in st.session_state:
        st.warning("Please run clustering first (Tab 3)")
    else:
        X_scaled = st.session_state.X_scaled
        labels = st.session_state.labels
        df_imputed = st.session_state.df_imputed

        st.subheader("Dimensionality Reduction")

        col1, col2 = st.columns([1, 2])

        with col1:
            reduction_method = st.selectbox(
                "Reduction Method", ["PCA", "t-SNE", "UMAP"]
            )

        if st.button("Generate Visualization", type="primary"):
            with st.spinner(f"Applying {reduction_method}..."):
                method_map = {"PCA": "pca", "t-SNE": "tsne", "UMAP": "umap"}
                reducer = DimensionalityReducer(
                    method=method_map[reduction_method], n_components=2
                )
                X_reduced = reducer.fit_transform(X_scaled)

                st.session_state.X_reduced = X_reduced
                st.session_state.reduction_method = reduction_method

                st.subheader(f"{reduction_method} Projection")

                fig = plot_clusters_interactive(
                    X_reduced,
                    labels,
                    patient_ids=df["Patient_ID"].values
                    if "Patient_ID" in df.columns
                    else None,
                    method_name=reduction_method,
                    title=f"{st.session_state.method} Clustering - {reduction_method} Projection",
                )
                st.plotly_chart(fig, use_container_width=True)

                if reduction_method == "PCA":
                    variance = reducer.get_explained_variance()
                    st.info(
                        f"Explained Variance: PC1={variance[0]:.1%}, PC2={variance[1]:.1%}, Total={variance.sum():.1%}"
                    )

        if "X_reduced" in st.session_state:
            st.subheader("Additional Visualizations")

            col1, col2 = st.columns(2)

            with col1:
                st.write("**Silhouette Analysis**")
                fig_sil, _ = plot_silhouette_analysis(X_scaled, labels)
                st.pyplot(fig_sil)

            with col2:
                if len(st.session_state.selected_features) <= 8:
                    st.write("**Cluster Profiles (Radar)**")
                    profiles = profile_clusters(
                        df_imputed, labels, st.session_state.selected_features
                    )
                    fig_radar, _ = plot_radar_chart(profiles)
                    st.pyplot(fig_radar)

        if "X_reduced" in st.session_state and api_key:
            st.markdown("---")
            if st.button("🤖 Interpret with LLM", key="llm_viz"):
                with st.spinner("Running analysis..."):
                    try:
                        fig_viz = extract_figure(
                            plot_silhouette_analysis(X_scaled, labels)
                        )
                        interpretation = interpret_visualization(
                            api_key, VISUALIZATION_PROMPT, fig_viz
                        )
                        st.subheader("💡 LLM interpretation")
                        st.write(interpretation)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# TAB 5: Global Explainability
with tab5:
    st.header("Global Explainability")

    if "labels" not in st.session_state or "X_scaled" not in st.session_state:
        st.warning("Please run clustering first (Tab 3)")
    else:
        X_scaled = st.session_state.X_scaled
        labels = st.session_state.labels
        df_imputed = st.session_state.df_imputed
        feature_names = st.session_state.selected_features

        st.subheader("Feature Importance")

        importance_method = st.selectbox(
            "Importance Method",
            ["All Methods", "Surrogate Tree", "Permutation", "SHAP"],
        )

        # --- Surrogate Tree concept explanation (always visible) ---
        with st.expander("ℹ️ What is a Surrogate Tree?", expanded=False):
            st.markdown("""
**Surrogate Tree concept**

Clustering is an **unsupervised** algorithm: it does not have an explicit decision boundary that can be 
directly explained with SHAP or LIME. To work around this, a **supervised decision tree** is trained
to reproduce the cluster labels - this is the *surrogate* model.

Feature importances, SHAP values, and decision rules are then extracted **from this proxy tree**, not from the clustering itself.

> ⚠️ **Implications**: If the proxy tree does not perfectly imitate the clustering,
> the explanations may deviate from the actual clustering behavior. This is why **fidelity** is measured and reported.
> A score ≥ 80 % is considered acceptable in the literature.
            """)

        if st.button("Compute Feature Importance", type="primary"):
            with st.spinner("Computing feature importance..."):
                method_map = {
                    "All Methods": "all",
                    "Surrogate Tree": "surrogate",
                    "Permutation": "permutation",
                    "SHAP": "shap",
                }

                importances = compute_feature_importance(
                    X_scaled,
                    labels,
                    feature_names=feature_names,
                    method=method_map[importance_method],
                )

                st.session_state.importances = importances

                # --- Surrogate fidelity badge ---
                surrogate_fidelity = importances.get("surrogate_fidelity", None)
                if surrogate_fidelity is not None:
                    fidelity_pct = surrogate_fidelity * 100
                    if surrogate_fidelity >= 0.80:
                        st.success(
                            f"✅ Surrogate fildeity: **{fidelity_pct:.1f}%** — Reliable explanation."
                        )
                    elif surrogate_fidelity >= 0.65:
                        st.warning(
                            f"⚠️ Surrogate fidelity: **{fidelity_pct:.1f}%** — Interpret results with caution."
                        )
                    else:
                        st.error(
                            f"❌ Surrogate fidelity: **{fidelity_pct:.1f}%** — Low fidelity = unreliable explanation."
                        )

                st.success("Feature importance computed")

                if importance_method != "All Methods":
                    fig = plot_feature_importance(importances, top_n=10)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig_comp = plot_feature_importance_comparison(importances, top_n=10)
                    st.plotly_chart(fig_comp, use_container_width=True)

                    st.subheader("Individual Method Details")
                    imp_methods = {
                        k: v for k, v in importances.items() if hasattr(v, "columns")
                    }
                    cols = st.columns(len(imp_methods))
                    for idx, (method_name, imp_df) in enumerate(imp_methods.items()):
                        with cols[idx]:
                            st.write(f"**{method_name.capitalize()}**")
                            fig = plot_feature_importance(
                                {method_name: imp_df}, top_n=8
                            )
                            st.plotly_chart(fig, use_container_width=True)

                st.subheader("Importance Scores")
                imp_methods_all = {
                    k: v for k, v in importances.items() if hasattr(v, "columns")
                }
                for method_name, imp_df in imp_methods_all.items():
                    with st.expander(f"{method_name.capitalize()} Importance"):
                        st.dataframe(imp_df)

        st.subheader("Cluster Profiling")

        if st.button("Generate Cluster Profiles", type="primary"):
            with st.spinner("Computing cluster profiles..."):
                profiles = compute_cluster_profiles(df_imputed, labels, feature_names)
                differences, effect_sizes = compute_contrastive_differences(
                    df_imputed, labels, feature_names
                )

                st.session_state.profiles = profiles
                st.session_state.differences = differences
                st.session_state.effect_sizes = effect_sizes

                st.success("Cluster profiles computed")

                st.subheader("Cluster Profile Overview")
                fig_radar = plot_cluster_radar(profiles, feature_names, normalize=True)
                st.plotly_chart(fig_radar, use_container_width=True)

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Mean Values per Cluster**")
                    st.dataframe(profiles.xs("mean", level=1, axis=1).round(2))

                with col2:
                    st.write("**Standard Deviation per Cluster**")
                    st.dataframe(profiles.xs("std", level=1, axis=1).round(2))

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
                tree, _, fidelity_rules = train_surrogate_tree(
                    X_scaled, labels, feature_names, max_depth=4
                )

                class_names = [f"Cluster {i}" for i in range(len(np.unique(labels)))]
                rules_df = extract_decision_rules(tree, feature_names, class_names)

                st.success("Decision rules extracted")

                # Fidelity badge for decision rules
                fidelity_pct = fidelity_rules * 100
                if fidelity_rules >= 0.80:
                    st.success(f"✅ Surrogate fidelity: **{fidelity_pct:.1f}%**")
                elif fidelity_rules >= 0.65:
                    st.warning(
                        f"⚠️ Surrogate fidelity: **{fidelity_pct:.1f}%** — Interpret with caution."
                    )
                else:
                    st.error(
                        f"❌ Surrogate fidelity: **{fidelity_pct:.1f}%** — Rules may not capture the actual clustering."
                    )

                st.write(f"**Top {min(10, len(rules_df))} Decision Rules**")
                st.dataframe(rules_df.head(10))

                st.subheader("Decision Tree Visualization")
                fig, ax = plt.subplots(figsize=(20, 10))
                from sklearn.tree import plot_tree

                plot_tree(
                    tree,
                    feature_names=feature_names,
                    class_names=class_names,
                    filled=True,
                    rounded=True,
                    ax=ax,
                    fontsize=10,
                )
                st.pyplot(fig)

        if api_key and (
            "profiles" in st.session_state or "importances" in st.session_state
        ):
            st.markdown("---")
            if st.button("🤖 Interpret with LLM", key="llm_global"):
                with st.spinner("Running analysis..."):
                    try:
                        fig_to_send = None
                        if "profiles" in st.session_state:
                            fig_to_send = extract_figure(
                                plot_contrastive_heatmap(st.session_state.differences)
                            )
                        elif "importances" in st.session_state:
                            fig_to_send = extract_figure(
                                plot_feature_importance(
                                    st.session_state.importances, top_n=10
                                )
                            )
                        interpretation = interpret_visualization(
                            api_key, GLOBAL_EXPLAINABILITY_PROMPT, fig_to_send
                        )
                        st.subheader("💡 LLM interpretation")
                        st.write(interpretation)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# TAB 6: Local Explainability
with tab6:
    st.header("Local Explainability - Patient-Level Explanations")

    if "labels" not in st.session_state or "X_scaled" not in st.session_state:
        st.warning("Please run clustering first (Tab 3)")
    else:
        X_scaled = st.session_state.X_scaled
        labels = st.session_state.labels
        df_imputed = st.session_state.df_imputed
        feature_names = st.session_state.selected_features
        model = st.session_state.model

        st.subheader("Patient Selection")

        col1, col2 = st.columns([2, 1])

        with col1:
            patient_id_col = "Patient_ID" if "Patient_ID" in df.columns else None
            if patient_id_col:
                patient_ids = df[patient_id_col].values
                selected_patient_id = st.selectbox(
                    "Select Patient",
                    patient_ids,
                    format_func=lambda x: (
                        f"Patient {x} - Cluster {labels[np.where(patient_ids == x)[0][0]]}"
                    ),
                )
                patient_idx = np.where(patient_ids == selected_patient_id)[0][0]
            else:
                patient_idx = st.number_input("Patient Index", 0, len(labels) - 1, 0)

        with col2:
            st.metric("Patient Cluster", labels[patient_idx])
            st.metric("Total Features", len(feature_names))

        explanation_methods = st.multiselect(
            "Explanation Methods",
            ["SHAP", "LIME", "Distance to Centroid", "Probabilistic Membership"],
            default=["SHAP", "Distance to Centroid"],
        )

        if st.button("Explain Patient Assignment", type="primary"):
            with st.spinner("Generating patient explanations..."):
                method_map = {
                    "SHAP": "shap",
                    "LIME": "lime",
                    "Distance to Centroid": "distance",
                    "Probabilistic Membership": "probabilistic",
                }

                selected_methods = [method_map[m] for m in explanation_methods]

                explanations = explain_patient(
                    X_scaled,
                    labels,
                    patient_idx,
                    model,
                    methods=selected_methods,
                    feature_names=feature_names,
                )

                st.session_state.patient_explanations = explanations
                st.session_state.explained_patient_idx = patient_idx

                st.success(f"Explanations generated for Patient {patient_idx}")

                # --- Surrogate Tree concept explanation (local level) ---
                with st.expander(
                    "ℹ️ Method: Surrogate Tree + SHAP/LIME", expanded=False
                ):
                    st.markdown("""
**How does local explanation work?**

For SHAP and LIME, a **decision tree** is trained to mimic the clustering
(excluding the patient being explained, to avoid data leakage). SHAP/LIME are then
applied to this proxy tree.

The **fidelity** score measures how well this tree reproduces the clustering labels.
The recommended threshold is **≥ 80%** for the explanations to be considered reliable.

> **Distance to Centroid** is the only method applied directly to the clustering, without a proxy.
                    """)

                # --- Fidelity badges ---
                if "shap" in explanations and "error" not in explanations["shap"]:
                    fid = explanations["shap"].get("fidelity")
                    if fid is not None:
                        fid_pct = fid * 100
                        if fid >= 0.80:
                            st.success(
                                f"✅ Surrogate fidelity (SHAP): **{fid_pct:.1f}%** — Reliable explanation."
                            )
                        elif fid >= 0.65:
                            st.warning(
                                f"⚠️ Surrogate fidelity (SHAP): **{fid_pct:.1f}%** — Interpret with caution."
                            )
                        else:
                            st.error(
                                f"❌ Surrogate fidelity (SHAP): **{fid_pct:.1f}%** —Low fidelity proxy = unreliable explanation."
                            )

                if "lime" in explanations and "error" not in explanations["lime"]:
                    fid = explanations["lime"].get("fidelity")
                    if fid is not None:
                        fid_pct = fid * 100
                        if fid >= 0.80:
                            st.success(
                                f"✅ Surrogate fidelity (LIME): **{fid_pct:.1f}%** — Reliable explanation."
                            )
                        elif fid >= 0.65:
                            st.warning(
                                f"⚠️ Surrogate fidelity (LIME): **{fid_pct:.1f}%** — Interpret with caution."
                            )
                        else:
                            st.error(
                                f"❌ Surrogate fidelity (LIME): **{fid_pct:.1f}%** — Low fidelity proxy = unreliable explanation."
                            )

                st.subheader("Explanation Summary")
                fig_summary = plot_patient_explanation_summary(explanations, top_n=8)
                st.pyplot(fig_summary)

                st.subheader("Detailed Explanations")

                if "shap" in explanations and "error" not in explanations["shap"]:
                    with st.expander("SHAP Explanation", expanded=True):
                        st.write(
                            "SHAP values show how each feature contributes to the cluster assignment"
                        )

                        fig_shap = plot_shap_explanation(explanations["shap"], top_n=10)
                        st.pyplot(fig_shap)

                        import plotly.graph_objects as go

                        shap_vals = np.asarray(
                            explanations["shap"]["shap_values"]
                        ).flatten()
                        feature_vals = np.asarray(
                            explanations["shap"]["feature_values"]
                        ).flatten()
                        n_vals = min(
                            len(shap_vals), len(feature_vals), len(feature_names)
                        )

                        sorted_idx_full = np.argsort(np.abs(shap_vals))[::-1]
                        sorted_idx = sorted_idx_full[
                            sorted_idx_full < len(feature_names)
                        ][:10]
                        colors = [
                            "#ff4d4d" if v < 0 else "#4dff4d"
                            for v in shap_vals[sorted_idx]
                        ]

                        fig_interactive = go.Figure()
                        fig_interactive.add_trace(
                            go.Bar(
                                y=[feature_names[i] for i in sorted_idx],
                                x=shap_vals[sorted_idx],
                                orientation="h",
                                marker=dict(color=colors),
                                text=[f"{v:.3f}" for v in shap_vals[sorted_idx]],
                                textposition="auto",
                                hovertemplate="<b>%{y}</b><br>SHAP: %{x:.3f}<br>Value: %{customdata:.2f}<extra></extra>",
                                customdata=feature_vals[sorted_idx],
                            )
                        )
                        fig_interactive.update_layout(
                            title="Interactive SHAP Values",
                            xaxis_title="SHAP Impact",
                            yaxis_title="Features",
                            height=400,
                            showlegend=False,
                        )
                        st.plotly_chart(fig_interactive, use_container_width=True)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Strongest Positive",
                                feature_names[sorted_idx[0]]
                                if shap_vals[sorted_idx[0]] > 0
                                else "None",
                                f"+{shap_vals[sorted_idx[0]]:.3f}"
                                if shap_vals[sorted_idx[0]] > 0
                                else "",
                            )
                        with col2:
                            neg_vals = [v for v in shap_vals if v < 0]
                            if neg_vals:
                                # clamp min_idx to valid feature_names range
                                min_idx = int(
                                    np.argmin(shap_vals[: len(feature_names)])
                                )
                                st.metric(
                                    "Strongest Negative",
                                    feature_names[min_idx],
                                    f"{shap_vals[min_idx]:.3f}",
                                )
                        with col3:
                            st.metric(
                                "Total SHAP Impact", f"{np.sum(np.abs(shap_vals)):.3f}"
                            )

                        st.write("**Feature Values vs SHAP Impact**")
                        shap_data = pd.DataFrame(
                            {
                                "Feature": feature_names[:n_vals],
                                "Value": feature_vals[:n_vals],
                                "SHAP Impact": shap_vals[:n_vals],
                            }
                        )
                        shap_data["Abs Impact"] = np.abs(shap_data["SHAP Impact"])
                        shap_data["Direction"] = shap_data["SHAP Impact"].apply(
                            lambda x: "Positive" if x > 0 else "Negative"
                        )
                        shap_data = shap_data.sort_values("Abs Impact", ascending=False)
                        st.dataframe(
                            shap_data[
                                ["Feature", "Value", "SHAP Impact", "Direction"]
                            ].head(10),
                            use_container_width=True,
                        )

                if "lime" in explanations and "error" not in explanations["lime"]:
                    with st.expander("LIME Explanation"):
                        st.write("LIME explains the cluster assignment locally")
                        fig_lime = plot_lime_explanation(explanations["lime"], top_n=10)
                        st.pyplot(fig_lime)

                        import plotly.graph_objects as go

                        lime_vals = np.asarray(
                            explanations["lime"]["lime_values"]
                        ).flatten()
                        feature_vals = np.asarray(
                            explanations["lime"]["feature_values"]
                        ).flatten()

                        sorted_idx_lime = np.argsort(np.abs(lime_vals))[::-1]
                        # filter to valid feature_names indices only
                        sorted_idx = sorted_idx_lime[
                            sorted_idx_lime < len(feature_names)
                        ][:10]
                        colors = [
                            "#ff6b6b" if v < 0 else "#51cf66"
                            for v in lime_vals[sorted_idx]
                        ]

                        fig_lime_int = go.Figure()
                        fig_lime_int.add_trace(
                            go.Bar(
                                y=[feature_names[i] for i in sorted_idx],
                                x=lime_vals[sorted_idx],
                                orientation="h",
                                marker=dict(color=colors),
                                text=[f"{v:.3f}" for v in lime_vals[sorted_idx]],
                                textposition="auto",
                            )
                        )
                        fig_lime_int.update_layout(
                            title="Interactive LIME Explanation",
                            xaxis_title="LIME Weight",
                            yaxis_title="Features",
                            height=400,
                        )
                        st.plotly_chart(fig_lime_int, use_container_width=True)

                        if "probability" in explanations["lime"]:
                            st.write("**Cluster Probabilities**")
                            prob_fig = go.Figure(
                                data=[
                                    go.Bar(
                                        x=[
                                            f"Cluster {i}"
                                            for i in range(
                                                len(explanations["lime"]["probability"])
                                            )
                                        ],
                                        y=explanations["lime"]["probability"],
                                        marker_color=[
                                            "#22c55e"
                                            if i == explanations["lime"]["cluster"]
                                            else "#94a3b8"
                                            for i in range(
                                                len(explanations["lime"]["probability"])
                                            )
                                        ],
                                    )
                                ]
                            )
                            prob_fig.update_layout(
                                title="Prediction Confidence", height=300
                            )
                            st.plotly_chart(prob_fig, use_container_width=True)

                if (
                    "distance" in explanations
                    and "error" not in explanations["distance"]
                ):
                    with st.expander("Distance to Centroid Explanation", expanded=True):
                        st.write(
                            "Distance-based explanation shows which features differ most from cluster center"
                        )
                        fig_dist = plot_distance_explanation(
                            explanations["distance"], top_n=10
                        )
                        st.pyplot(fig_dist)

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(
                                "Total Distance",
                                f"{explanations['distance']['total_distance']:.3f}",
                            )
                        with col2:
                            # clamp top_feat to valid feature_names range
                            contribs = np.asarray(
                                explanations["distance"]["feature_contributions"]
                            )
                            top_feat = int(np.argmax(contribs[: len(feature_names)]))
                            st.metric("Top Contributor", feature_names[top_feat])
                        with col3:
                            top_contrib = (
                                np.max(
                                    explanations["distance"]["feature_contributions"]
                                )
                                * 100
                            )
                            st.metric("Max Contribution", f"{top_contrib:.1f}%")

                        import plotly.graph_objects as go

                        # filter sorted_indices to valid feature_names range
                        all_sorted = np.asarray(
                            explanations["distance"]["sorted_indices"]
                        )
                        top_n_idx = all_sorted[all_sorted < len(feature_names)][:8]

                        fig_radar = go.Figure()
                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=explanations["distance"]["patient_values"][top_n_idx],
                                theta=[feature_names[i] for i in top_n_idx],
                                fill="toself",
                                name="Patient",
                                line_color="#3b82f6",
                            )
                        )
                        fig_radar.add_trace(
                            go.Scatterpolar(
                                r=explanations["distance"]["centroid_values"][
                                    top_n_idx
                                ],
                                theta=[feature_names[i] for i in top_n_idx],
                                fill="toself",
                                name="Cluster Center",
                                line_color="#ef4444",
                            )
                        )
                        fig_radar.update_layout(
                            polar=dict(radialaxis=dict(visible=True)),
                            title="Patient vs Cluster Center (Top 8 Features)",
                            height=450,
                        )
                        st.plotly_chart(fig_radar, use_container_width=True)

                        st.write("**Top Contributing Features**")
                        dist_data = pd.DataFrame(
                            {
                                "Feature": feature_names,
                                "Patient Value": explanations["distance"][
                                    "patient_values"
                                ],
                                "Centroid Value": explanations["distance"][
                                    "centroid_values"
                                ],
                                "Distance": explanations["distance"][
                                    "feature_distances"
                                ],
                                "Contribution %": explanations["distance"][
                                    "feature_contributions"
                                ]
                                * 100,
                            }
                        )
                        dist_data["Difference"] = (
                            dist_data["Patient Value"] - dist_data["Centroid Value"]
                        )
                        dist_data = dist_data.sort_values(
                            "Contribution %", ascending=False
                        )
                        st.dataframe(
                            dist_data[
                                [
                                    "Feature",
                                    "Patient Value",
                                    "Centroid Value",
                                    "Difference",
                                    "Contribution %",
                                ]
                            ].head(10),
                            use_container_width=True,
                        )

                if (
                    "probabilistic" in explanations
                    and "error" not in explanations["probabilistic"]
                ):
                    with st.expander("Probabilistic Membership Explanation"):
                        st.write("Cluster membership probabilities")
                        fig_prob = plot_probabilistic_explanation(
                            explanations["probabilistic"]
                        )
                        st.plotly_chart(fig_prob, use_container_width=True)

                        st.metric(
                            "Membership Certainty",
                            f"{explanations['probabilistic']['membership_certainty']:.2%}",
                        )

                        st.write("**Distances to All Cluster Centers**")
                        prob_data = pd.DataFrame(
                            {
                                "Cluster": range(
                                    len(explanations["probabilistic"]["probabilities"])
                                ),
                                "Probability": explanations["probabilistic"][
                                    "probabilities"
                                ],
                                "Distance": explanations["probabilistic"][
                                    "distances_to_centers"
                                ],
                            }
                        )
                        st.dataframe(prob_data)

        if api_key and "patient_explanations" in st.session_state:
            st.markdown("---")
            if st.button("🤖 Interpret with LLM", key="llm_local"):
                with st.spinner("Running analysis..."):
                    try:
                        fig_to_send = extract_figure(
                            plot_patient_explanation_summary(
                                st.session_state.patient_explanations, top_n=8
                            )
                        )
                        interpretation = interpret_visualization(
                            api_key, LOCAL_EXPLAINABILITY_PROMPT, fig_to_send
                        )
                        st.subheader("💡 LLM interpretation")
                        st.write(interpretation)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# TAB 7: Counterfactual Explanations
with tab7:
    st.header("Counterfactual Explanations - What-If Scenarios")

    if "labels" not in st.session_state or "X_scaled" not in st.session_state:
        st.warning("Please run clustering first (Tab 3)")
    else:
        X_scaled = st.session_state.X_scaled
        labels = st.session_state.labels
        df_imputed = st.session_state.df_imputed
        feature_names = st.session_state.selected_features
        model = st.session_state.model
        preprocessor = st.session_state.get("preprocessor", None)

        st.subheader("Patient and Target Selection")

        col1, col2, col3 = st.columns(3)

        with col1:
            patient_id_col = "Patient_ID" if "Patient_ID" in df.columns else None
            if patient_id_col:
                patient_ids = df[patient_id_col].values
                selected_patient_id = st.selectbox(
                    "Select Patient",
                    patient_ids,
                    format_func=lambda x: (
                        f"Patient {x} - Cluster {labels[np.where(patient_ids == x)[0][0]]}"
                    ),
                    key="cf_patient",
                )
                patient_idx = np.where(patient_ids == selected_patient_id)[0][0]
            else:
                patient_idx = st.number_input(
                    "Patient Index", 0, len(labels) - 1, 0, key="cf_patient_idx"
                )

        with col2:
            current_cluster = labels[patient_idx]
            st.metric("Current Cluster", current_cluster)

            available_clusters = [c for c in np.unique(labels) if c != current_cluster]
            target_cluster = st.selectbox("Target Cluster", available_clusters)

        with col3:
            cf_method = st.selectbox(
                "Generation Method", ["DiCE-Style (Fast)", "Optimization-Based"]
            )

            n_counterfactuals = st.slider("Number of Counterfactuals", 1, 5, 3)

        st.subheader("Clinical Constraints (Optional)")

        with st.expander("Configure Constraints"):
            st.write("Set constraints to ensure clinically plausible counterfactuals")

            constraints = ClinicalConstraints(feature_names)

            st.write("**Feature Ranges**")
            constraint_features = st.multiselect(
                "Features with constraints", feature_names, default=[]
            )

            for feat in constraint_features:
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.number_input(
                        f"{feat} - Min", value=0.0, key=f"min_{feat}"
                    )
                with col2:
                    max_val = st.number_input(
                        f"{feat} - Max", value=100.0, key=f"max_{feat}"
                    )
                constraints.set_range(feat, min_val, max_val)

            st.write("**Immutable Features**")
            immutable_features = st.multiselect(
                "Features that cannot change (e.g., Age, Gender)",
                feature_names,
                default=[],
            )

            for feat in immutable_features:
                constraints.set_immutable(feat)

        use_constraints = len(constraint_features) > 0 or len(immutable_features) > 0

        if st.button("Generate Counterfactuals", type="primary"):
            with st.spinner(f"Generating {n_counterfactuals} counterfactual(s)..."):
                try:
                    if cf_method == "DiCE-Style (Fast)":
                        counterfactuals = generate_counterfactual_dice_style(
                            X_scaled,
                            labels,
                            patient_idx,
                            model,
                            target_cluster=target_cluster,
                            constraints=constraints if use_constraints else None,
                            n_counterfactuals=n_counterfactuals,
                        )
                    else:
                        from counterfactuals import generate_diverse_counterfactuals

                        counterfactuals = generate_diverse_counterfactuals(
                            X_scaled,
                            labels,
                            patient_idx,
                            model,
                            target_cluster=target_cluster,
                            constraints=constraints if use_constraints else None,
                            n_counterfactuals=n_counterfactuals,
                        )
                except Exception as e:
                    st.error(f"Error generating counterfactuals: {str(e)}")
                    counterfactuals = []

                st.session_state.counterfactuals = counterfactuals

                if counterfactuals and len(counterfactuals) > 0:
                    st.success(
                        f"✅ Generated {len(counterfactuals)} valid counterfactual(s)"
                    )

                    st.subheader("Quality Metrics")
                    from counterfactuals import evaluate_counterfactual_quality

                    quality_data = []
                    for idx, cf in enumerate(counterfactuals):
                        quality = evaluate_counterfactual_quality(
                            cf, constraints if use_constraints else None
                        )
                        quality_data.append(
                            {
                                "CF #": idx + 1,
                                "Valid": "Yes" if quality["valid"] else "No",
                                "Distance": quality["proximity"],
                                "Changes": quality["sparsity"],
                                "Sparsity %": f"{quality['sparsity_ratio'] * 100:.1f}%",
                            }
                        )

                    quality_df = pd.DataFrame(quality_data)
                    st.dataframe(quality_df, use_container_width=True)

                    st.subheader("Counterfactual Scenarios")

                    # C2: Apply inverse_transform to show values in original clinical scale
                    def to_original_scale(scaled_values):
                        if preprocessor is not None:
                            try:
                                arr = np.array(scaled_values).reshape(1, -1)
                                return preprocessor.inverse_transform(arr).flatten()
                            except Exception:
                                pass  # Fallback to scaled values if inverse_transform fails or preprocessor is None
                        return np.array(scaled_values)

                    for idx, cf in enumerate(counterfactuals):
                        with st.expander(
                            f"🔄 Counterfactual #{idx + 1} - Distance: {cf['distance']:.3f} | Changes: {cf['n_changes']}",
                            expanded=(idx == 0),
                        ):
                            explanation_text = format_counterfactual_explanation(
                                cf, feature_names
                            )
                            st.info(explanation_text)

                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Changes", cf["n_changes"])
                            with col2:
                                st.metric("Distance", f"{cf['distance']:.3f}")
                            with col3:
                                st.metric("From Cluster", cf["original_cluster"])
                            with col4:
                                st.metric(
                                    "To Cluster",
                                    cf["predicted_cluster"],
                                    delta="Success" if cf["success"] else "Failed",
                                    delta_color="normal"
                                    if cf["success"]
                                    else "inverse",
                                )

                            # Interactive comparison
                            import plotly.graph_objects as go

                            # filter feature_changes to valid feature_names indices
                            all_changes = [
                                i
                                for i in cf["feature_changes"]
                                if i < len(feature_names)
                            ]
                            feat_idx = all_changes[:10]  # Top 10 changes

                            orig_display = to_original_scale(cf["original"])
                            cf_display = to_original_scale(cf["counterfactual"])

                            fig_comp = go.Figure()
                            fig_comp.add_trace(
                                go.Bar(
                                    name="Original",
                                    x=[feature_names[i] for i in feat_idx],
                                    y=[orig_display[i] for i in feat_idx],
                                    marker_color="#64748b",
                                )
                            )
                            fig_comp.add_trace(
                                go.Bar(
                                    name="Counterfactual",
                                    x=[feature_names[i] for i in feat_idx],
                                    y=[cf_display[i] for i in feat_idx],
                                    marker_color="#22c55e",
                                )
                            )
                            fig_comp.update_layout(
                                title="Feature Values Comparison (Original Scale)",
                                barmode="group",
                                height=400,
                                xaxis_tickangle=-45,
                            )
                            st.plotly_chart(fig_comp, use_container_width=True)

                            # Changes magnitude
                            changes_display = cf_display - orig_display
                            colors_changes = [
                                "#ef4444" if c < 0 else "#22c55e"
                                for c in [changes_display[i] for i in feat_idx]
                            ]
                            fig_changes = go.Figure()
                            fig_changes.add_trace(
                                go.Bar(
                                    x=[feature_names[i] for i in feat_idx],
                                    y=[changes_display[i] for i in feat_idx],
                                    marker_color=colors_changes,
                                    text=[
                                        f"{changes_display[i]:+.2f}" for i in feat_idx
                                    ],
                                    textposition="auto",
                                )
                            )
                            fig_changes.update_layout(
                                title="Change Magnitude (Original Scale)",
                                height=350,
                                xaxis_tickangle=-45,
                                yaxis_title="Change Value",
                            )
                            st.plotly_chart(fig_changes, use_container_width=True)

                            st.write("**Detailed Changes (Original Clinical Scale)**")
                            changes_data = []
                            for feat_idx_val in cf["feature_changes"]:
                                if feat_idx_val >= len(feature_names):
                                    continue
                                orig_val = orig_display[feat_idx_val]
                                cf_val = cf_display[feat_idx_val]
                                delta = cf_val - orig_val
                                change_pct = delta / (abs(orig_val) + 1e-10) * 100
                                changes_data.append(
                                    {
                                        "Feature": feature_names[feat_idx_val],
                                        "Original": f"{orig_val:.3f}",
                                        "Counterfactual": f"{cf_val:.3f}",
                                        "Δ": f"{delta:+.3f}",
                                        "Δ %": f"{change_pct:+.1f}%",
                                        "Direction": "Increase"
                                        if delta > 0
                                        else "Decrease",
                                    }
                                )
                            changes_df = pd.DataFrame(changes_data)
                            st.dataframe(changes_df, use_container_width=True)

                    if len(counterfactuals) > 1:
                        st.subheader("Comparison of Counterfactuals")
                        fig_diverse = plot_diverse_counterfactuals(
                            counterfactuals, feature_names, top_n=8
                        )
                        st.plotly_chart(fig_diverse, use_container_width=True)
                else:
                    st.warning("⚠️ Could not generate valid counterfactuals")
                    st.info("""
**Possible reasons:**
- Target cluster may be too different from current cluster
- Constraints may be too restrictive
- Not enough data points in target cluster

**Suggestions:**
- Try a different target cluster
- Relax or remove some constraints
- Use DiCE-Style method if using Optimization-Based (or vice versa)
                    """)

        if api_key and "counterfactuals" in st.session_state:
            st.markdown("---")
            if st.button("🤖 Interpret with LLM", key="llm_cf"):
                with st.spinner("Running analysis..."):
                    try:
                        cf_list = st.session_state.counterfactuals
                        if cf_list and len(cf_list) > 0:
                            fig_to_send = extract_figure(
                                plot_counterfactual_comparison(
                                    cf_list[0], feature_names
                                )
                            )
                            interpretation = interpret_visualization(
                                api_key, COUNTERFACTUAL_PROMPT, fig_to_send
                            )
                            st.subheader("💡 LLM interpretation")
                            st.write(interpretation)
                        else:
                            st.warning("No contrefactual to analyze")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown(
    """
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>X-Insight - Explainable AI Platform for Systemic Sclerosis Analysis</p>
    <p>Developed for unsupervised learning with transparency and clinical interpretability</p>
</div>
""",
    unsafe_allow_html=True,
)
