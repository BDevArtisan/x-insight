# X-Insight

Explainable AI Platform for Unsupervised Learning on Clinical Data

## Overview

X-Insight is a research platform designed to provide transparent and interpretable clustering analysis for Systemic Sclerosis (SSc) clinical data. The platform combines traditional and advanced clustering methods with explainability techniques to support clinical decision-making.

## Project Structure

```
x-insight/
├── data/                           # Datasets
│   ├── ssc_synthetic_data.csv      # Synthetic SSc patient data (300 samples)
│   └── synt_data.py                # Data generation script
├── notebooks/                      # Exploratory analysis
│   ├── 01_EDA_SSc_Data.ipynb       # Exploratory data analysis
│   └── 02_Clustering_Traditional.ipynb  # Clustering experiments
├── src/                            # Source code
│   ├── app.py                      # Streamlit web interface (7 tabs)
│   ├── data/
│   │   └── preprocessing.py        # Data preprocessing pipeline
│   ├── clustering/
│   │   ├── traditional.py          # K-Means, Hierarchical, K-Medoids
│   │   ├── advanced.py             # GMM, Spectral, DEC, IDEC, DCN, HDBSCAN, GMM-Auto
│   │   └── validation.py           # Clustering metrics and evaluation
│   ├── explainability/
│   │   ├── global_explainability.py  # Feature importance, cluster profiling
│   │   └── local_explainability.py   # Patient-level explanations (SHAP, LIME)
│   ├── counterfactuals/
│   │   └── counterfactual_explanations.py  # What-if scenarios
│   └── visualization/
│       ├── dimensionality_reduction.py  # PCA, t-SNE, UMAP
│       └── plots.py                # Visualization functions
└── tests/                          # Unit tests
    ├── test_preprocessing.py
    ├── test_clustering.py
    ├── test_explainability.py
    ├── test_local_explainability.py
    └── test_counterfactual.py
```

## Features

### Data Processing
- KNN imputation for missing values
- Categorical variable encoding
- Feature standardization
- Preprocessing pipeline persistence

### Clustering Algorithms

The platform separates clustering methods into two categories:

#### Manual-k Methods *(user selects number of clusters)*
These methods require specifying `k`. An **elbow curve** is provided as guidance for K-Means, Hierarchical, and K-Medoids.

| Method | Description |
|---|---|
| **K-Means** | Partitions data into k clusters by minimizing within-cluster variance |
| **Hierarchical** | Builds a dendrogram tree; useful for understanding data hierarchy |
| **K-Medoids** | Like K-Means but uses real data points as centers — robust to outliers |
| **GMM** | Probabilistic model fitting k Gaussian distributions (soft membership) |
| **Spectral** | Graph-based method effective for non-convex cluster shapes |
| **DEC** | Deep Embedded Clustering using autoencoders (ClustPy/PyTorch) |
| **IDEC** | Improved DEC with better local structure preservation |
| **DCN** | Deep Clustering Network with joint reconstruction and clustering |

#### Auto-k Methods *(number of clusters determined automatically)*
These methods require **no k selection** — the algorithm finds the structure on its own.

| Method | Description |
|---|---|
| **HDBSCAN** | Density-based; finds clusters of varying density, marks noise patients as outliers |
| **GMM (Auto-BIC)** | Tests k = 2..10 and selects the optimal k by minimizing the Bayesian Information Criterion |

### Validation Metrics
- Internal metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
- External metrics: Adjusted Rand Index, Normalized Mutual Information
- Per-cluster silhouette analysis
- Cluster profiling and comparison
- Noise/outlier patient detection (HDBSCAN)

### Visualization
- Dimensionality reduction: PCA, t-SNE, UMAP
- Interactive cluster visualizations (Plotly)
- Dendrograms for hierarchical clustering
- Elbow curves and silhouette plots
- Radar charts for cluster profiles

### Explainability

#### Global Explainability
- Global feature importance (Surrogate Trees, Permutation, SHAP)
- Cluster profiling with statistical summaries
- Contrastive analysis (cluster vs global mean)
- Decision rule extraction from surrogate trees
- Interactive visualizations for interpretability

#### Local Explainability
- Patient-level explanations with SHAP values
- LIME explanations for individual cluster assignments
- Distance-to-centroid analysis (feature contributions)
- Probabilistic membership explanations (soft assignments)
- Combined multi-method explanations per patient

#### Counterfactual Explanations
- DiCE-style diverse counterfactual generation
- Optimization-based minimal feature changes
- Clinical constraint management (ranges, immutability)
- What-if scenario analysis for cluster transitions
- Multiple diverse alternatives for actionable insights

### Web Interface (7 Tabs)

| Tab | Description |
|---|---|
| **1 — Data** | Upload and explore the dataset |
| **2 — Preprocessing** | Configure and apply the preprocessing pipeline |
| **3 — Clustering** | Select algorithm, configure parameters, run clustering |
| **4 — Visualization** | PCA/t-SNE/UMAP cluster plots, radar charts, silhouette analysis |
| **5 — Global Explainability** | Feature importance, cluster profiles, contrastive heatmaps |
| **6 — Local Explainability** | Patient-level SHAP, LIME, distance, and probabilistic explanations |
| **7 — Counterfactuals** | What-if scenarios and cluster transition recommendations |

## Installation

### Prerequisites
- Python 3.10 or higher
- Virtual environment recommended

### Setup

1. Clone the repository and navigate to the project directory:
```bash
cd x-insight
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

> **Windows users:** ClustPy (required for DEC/IDEC/DCN) requires Microsoft Visual C++ 14.0 or greater.
> Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select "Desktop development with C++" before running `pip install`.

## Usage

### Running the Application

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

### Running Tests

```bash
pytest tests/ -v
```

## Technical Stack

| Layer | Technologies |
|---|---|
| **Core** | Python 3.10+, NumPy, Pandas |
| **Machine Learning** | scikit-learn, scipy |
| **Deep Clustering** | ClustPy (DEC, IDEC, DCN) with PyTorch backend |
| **Density Clustering** | hdbscan |
| **Explainability** | SHAP, LIME, DiCE, scikit-learn (surrogate models) |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Interface** | Streamlit |
| **Testing** | pytest |

## Current Status

### Implemented
- ✅ Complete data preprocessing pipeline (KNN imputation, encoding, scaling)
- ✅ **11 clustering algorithms** — 9 manual-k + 2 true auto-k (HDBSCAN, GMM-Auto)
- ✅ Elbow curve guidance for manual-k methods
- ✅ Deep clustering methods via ClustPy (DEC, IDEC, DCN)
- ✅ Comprehensive validation metrics with noise-aware evaluation
- ✅ Multiple dimensionality reduction methods (PCA, t-SNE, UMAP)
- ✅ Global explainability (feature importance, cluster profiling, contrastive analysis)
- ✅ Local explainability (SHAP, LIME, distance-based, probabilistic)
- ✅ Counterfactual explanations with clinical constraints
- ✅ Interactive web interface with 7 tabs
- ✅ Robust index-safe visualization (no IndexError on any cluster count)

### In Development
- 🔄 LLM integration for natural language explanations

## Contact

Project developed as part of a Master's thesis (PFE) on Explainable AI for clinical data analysis.
