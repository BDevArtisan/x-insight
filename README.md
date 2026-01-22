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
│   ├── app.py                      # Streamlit web interface
│   ├── data/
│   │   └── preprocessing.py        # Data preprocessing pipeline
│   ├── clustering/
│   │   ├── traditional.py          # K-Means, Hierarchical, K-Medoids
│   │   ├── advanced.py             # HDBSCAN, GMM, Spectral
│   │   └── validation.py           # Clustering metrics and evaluation
│   ├── explainability/
│   │   └── global_explainability.py  # Feature importance, cluster profiling
│   └── visualization/
│       ├── dimensionality_reduction.py  # PCA, t-SNE, UMAP
│       └── plots.py                # Visualization functions
└── tests/                          # Unit tests
    ├── test_preprocessing.py
    ├── test_clustering.py
    └── test_explainability.py
```

## Features

### Data Processing
- KNN imputation for missing values
- Categorical variable encoding
- Feature standardization
- Preprocessing pipeline persistence

### Clustering Algorithms
- Traditional methods: K-Means, Hierarchical (Ward/Complete/Average), K-Medoids
- Advanced methods: Gaussian Mixture Models, Spectral Clustering, HDBSCAN (optional)
- Automatic optimal cluster number determination

### Validation Metrics
- Internal metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
- External metrics: Adjusted Rand Index, Normalized Mutual Information
- Per-cluster silhouette analysis
- Cluster profiling and comparison

### Visualization
- Dimensionality reduction: PCA, t-SNE, UMAP
- Interactive cluster visualizations (Plotly)
- Dendrograms for hierarchical clustering
- Elbow curves and silhouette plots
- Radar charts for cluster profiles

### Explainability
- Global feature importance (Surrogate Trees, Permutation, SHAP)
- Cluster profiling with statistical summaries
- Contrastive analysis (cluster vs global mean)
- Decision rule extraction from surrogate trees
- Interactive visualizations for interpretability

### Web Interface
- Streamlit-based interactive application
- Five main sections: Data, Preprocessing, Clustering, Visualization, Explainability
- Data upload and exploration
- Real-time clustering and visualization
- Comprehensive explainability analysis
- Downloadable results

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

Note: Some packages (HDBSCAN, UMAP) may require Microsoft C++ Build Tools on Windows. These are optional and the platform provides graceful fallbacks.

## Usage

### Running the Streamlit Application

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`


### Running Tests

```bash
pytest tests/ -v
```

All 43 tests currently pass (11 preprocessing, 22 clustering, 10 explainability).

### Implemented
- Complete data preprocessing pipeline
- Six clustering algorithms (K-Means, Hierarchical, K-Medoids, GMM, Spectral, HDBSCAN)
- Comprehensive validation metrics
- Multiple dimensionality reduction methods (PCA, t-SNE, UMAP)
- Global explainability module with feature importance and decision rules
- Interactive web interface with 5 tabs
- Full test coverage (43 unit tests)

### In Development
- Local explainability (LIME, instance-level explanations)
- Counterfactual generation
- LLM integration for natural language explanations
- Deep clustering methods
- REST API

## Technical Stack

- Core: Python 3.12, NumPy, Pandas
- Machine Learning: scikit-learn, scipy
- Explainability: SHAP, scikit-learn (surrogate models, permutation importance)
- Visualization: Matplotlib, Seaborn, Plotly
- Web Interface: Streamlit
- Testing: pytest

## Contact

Project developed as part of a Master's thesis (PFE) on Explainable AI for clinical data analysis.
