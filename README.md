# X-Insight

**Explainable AI Platform for Unsupervised Learning on Clinical Data**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-FF4B4B.svg)](https://streamlit.io)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Project Structure](#project-structure)
- [Features](#features)
  - [Data Processing](#data-processing)
  - [Clustering Algorithms](#clustering-algorithms)
  - [Validation Metrics](#validation-metrics)
  - [Visualization](#visualization)
  - [Explainability](#explainability)
  - [LLM-Powered Interpretations](#llm-powered-interpretations)
  - [Web Interface](#web-interface-7-tabs)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Technical Stack](#technical-stack)
- [Current Status](#current-status)
- [Research Context](#research-context)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

X-Insight is a research platform designed to provide transparent and interpretable clustering analysis for Systemic Sclerosis (SSc) clinical data. The platform combines traditional and advanced clustering methods with explainability techniques to support clinical decision-making.

## Key Features

🔬 **10 Clustering Algorithms** — From traditional (K-Means, Hierarchical) to deep learning (DEC, IDEC, DCN), including auto-k methods (HDBSCAN, GMM-Auto)

🔍 **Multi-Level Explainability** — Global feature importance, local patient-level explanations (SHAP, LIME), and counterfactual scenarios

🤖 **AI-Powered Interpretations** — Integrated Google Gemini 2.5 Flash for natural language explanations of all analyses

📊 **Interactive Visualizations** — PCA, t-SNE, UMAP projections with interactive Plotly charts

🧪 **Clinical Focus** — Constraint-aware counterfactuals and clinical data preprocessing pipeline

✅ **Production Ready** — Comprehensive test suite, modular architecture, and easy-to-use Streamlit interface

## Project Structure

```
x-insight/
├── src/                            # Source code
│   ├── app.py                      # Streamlit web interface (7 tabs)
│   ├── data/
│   │   └── preprocessing.py        # Data preprocessing pipeline
│   ├── clustering/
│   │   ├── __init__.py             # Module exports
│   │   ├── traditional.py          # K-Means, Hierarchical, K-Medoids
│   │   ├── advanced.py             # GMM, Spectral, DEC, IDEC, DCN, HDBSCAN, GMM-Auto
│   │   └── validation.py           # Clustering metrics and evaluation
│   ├── explainability/
│   │   ├── __init__.py             # Module exports
│   │   ├── global_explainability.py  # Feature importance, cluster profiling
│   │   └── local_explainability.py   # Patient-level explanations (SHAP, LIME)
│   ├── counterfactuals/
│   │   ├── __init__.py             # Module exports
│   │   └── counterfactual_explanations.py  # What-if scenarios
│   ├── visualization/
│   │   ├── __init__.py             # Module exports
│   │   ├── dimensionality_reduction.py  # PCA, t-SNE, UMAP
│   │   └── plots.py                # Visualization functions
│   └── llm/
│       ├── __init__.py             # Module exports
│       ├── gemini_client.py        # Google Gemini API client
│       └── prompts.py              # LLM prompts for interpretations
├── tests/                          # Unit tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_clustering.py
│   ├── test_explainability.py
│   ├── test_local_explainability.py
│   └── test_counterfactual.py
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Features

### Data Processing
- Support for multiple file formats (CSV, Excel .xlsx/.xls)
- KNN imputation for missing values
- Categorical variable encoding (binary and one-hot)
- Feature standardization and normalization
- Preprocessing pipeline persistence
- Comprehensive data profiling with ydata-profiling

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

### LLM-Powered Interpretations
- **Natural language explanations** for all visualizations and analyses
- **Context-aware prompts** tailored to each analysis type:
  - Clustering visualization analysis (separation quality, patterns)
  - Global explainability interpretation (feature importance, cluster profiles)
  - Local explainability explanation (patient-specific assignments)
  - Counterfactual scenario evaluation (feasibility, clinical implications)
- **Gemini 2.5 Flash** integration for fast, accurate interpretations
- **French language** outputs optimized for clinical context
- **Image-based analysis** using vision capabilities for chart interpretation

### Web Interface (7 Tabs)

| Tab | Description |
|---|---|
| **1 — Data** | Upload and explore the dataset (CSV, Excel), data profiling |
| **2 — Preprocessing** | Configure and apply the preprocessing pipeline |
| **3 — Clustering** | Select algorithm, configure parameters, run clustering |
| **4 — Visualization** | PCA/t-SNE/UMAP cluster plots, radar charts, silhouette analysis + **LLM interpretation** |
| **5 — Global Explainability** | Feature importance, cluster profiles, contrastive heatmaps + **LLM interpretation** |
| **6 — Local Explainability** | Patient-level SHAP, LIME, distance, and probabilistic explanations + **LLM interpretation** |
| **7 — Counterfactuals** | What-if scenarios and cluster transition recommendations + **LLM interpretation** |

**LLM Integration:** All visualization and explainability tabs include AI-powered interpretation buttons that use **Google Gemini 2.5 Flash** to provide natural language explanations of the results in French.

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

## Quick Start

1. **Launch the application:**
   ```bash
   streamlit run src/app.py
   ```

2. **Upload your data:**
   - Navigate to the **Data** tab
   - Upload a CSV or Excel file with patient clinical data
   - Review the automatic data profiling report

3. **Preprocess your data:**
   - Go to the **Preprocessing** tab
   - Configure imputation, encoding, and scaling options
   - Apply preprocessing and save the pipeline

4. **Run clustering:**
   - Select the **Clustering** tab
   - Choose an algorithm (try K-Means with elbow curve first)
   - View validation metrics

5. **Explore results:**
   - **Visualization** tab: See clusters in 2D/3D space
   - **Global Explainability** tab: Understand which features define clusters
   - **Local Explainability** tab: Explain individual patient assignments
   - **Counterfactuals** tab: Generate what-if scenarios

6. **Get AI insights (optional):**
   - Add your Google API key in the sidebar
   - Click any "🤖 Interpréter avec LLM" button for natural language explanations

## Usage

### Running the Application

```bash
streamlit run src/app.py
```

The application will be available at `http://localhost:8501`

### LLM Integration Setup

To use the AI-powered natural language interpretations:

1. Obtain a Google API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. In the Streamlit sidebar, enter your API key in the "LLM Settings" section
3. Click the "🤖 Interpréter avec LLM" button on any visualization or analysis to get AI-powered explanations

**Note:** The LLM feature is optional. All core functionality works without an API key.

### Running Tests

```bash
pytest tests/ -v
```

### Troubleshooting

**ClustPy installation fails on Windows:**
- Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
- Select "Desktop development with C++" workload
- Restart your terminal and retry `pip install clustpy`

**CUDA/PyTorch issues for deep clustering:**
- The platform uses CPU by default for deep clustering algorithms
- For GPU acceleration, install PyTorch with CUDA support: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

**SHAP warnings:**
- SHAP may show warnings with newer scikit-learn versions; these are usually harmless
- Explanations will still work correctly

**Memory issues with large datasets:**
- Deep clustering (DEC/IDEC/DCN) requires significant RAM for large datasets (>10,000 samples)
- Consider using traditional methods first or reducing batch size in the code

## Technical Stack

| Layer | Technologies |
|---|---|
| **Core** | Python 3.10+, NumPy, Pandas |
| **Machine Learning** | scikit-learn, scipy |
| **Deep Clustering** | ClustPy (DEC, IDEC, DCN) with PyTorch backend |
| **Density Clustering** | hdbscan |
| **Explainability** | SHAP, LIME, DiCE, scikit-learn (surrogate models) |
| **LLM Integration** | Google Generative AI (Gemini 2.5 Flash) |
| **Visualization** | Matplotlib, Seaborn, Plotly, Kaleido |
| **Data Profiling** | ydata-profiling |
| **Web Interface** | Streamlit |
| **Testing** | pytest |

## Current Status

### Implemented
- ✅ Complete data preprocessing pipeline (KNN imputation, encoding, scaling)
- ✅ Multi-format data import (CSV, Excel .xlsx/.xls)
- ✅ **10 clustering algorithms** — 8 manual-k + 2 true auto-k (HDBSCAN, GMM-Auto)
- ✅ Elbow curve guidance for manual-k methods
- ✅ Deep clustering methods via ClustPy (DEC, IDEC, DCN)
- ✅ Comprehensive validation metrics with noise-aware evaluation
- ✅ Multiple dimensionality reduction methods (PCA, t-SNE, UMAP)
- ✅ Global explainability (feature importance, cluster profiling, contrastive analysis)
- ✅ Local explainability (SHAP, LIME, distance-based, probabilistic)
- ✅ Counterfactual explanations with clinical constraints
- ✅ Interactive web interface with 7 tabs
- ✅ Robust index-safe visualization (no IndexError on any cluster count)
- ✅ **LLM integration with Google Gemini 2.5 Flash** for natural language explanations across all analysis tabs
- ✅ Comprehensive test suite with pytest


## Research Context

This platform was developed for the analysis of Systemic Sclerosis (SSc) clinical data, focusing on patient stratification and phenotype discovery. The combination of multiple clustering approaches with comprehensive explainability methods enables researchers and clinicians to:

- **Discover patient subgroups** with distinct clinical characteristics
- **Validate findings** across multiple algorithmic approaches
- **Understand cluster assignments** through interpretable explanations
- **Explore interventions** via counterfactual scenarios
- **Communicate results** with AI-generated natural language summaries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/ -v --cov=src

# Format code
black src/ tests/
```
## Supervision
This work was supervised by Prof Zaineb GARCIA, Univ. Lille, CNRS, Inria, Centrale Lille, UMR  9189 CRIStAL, F-59000 Lille, France


## Contact

For questions, issues, or collaborations, please open an issue on GitHub.

**Project Information:** Developed as part of a Master's thesis (PFE) on Explainable AI for clinical data analysis at FHU-PRECISE, focusing on Systemic Sclerosis patient stratification.

---

**Built with ❤️ for advancing explainable AI in healthcare**
