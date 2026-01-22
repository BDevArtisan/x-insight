"""
Dimensionality reduction techniques for visualization.
Includes PCA, t-SNE, and UMAP.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class DimensionalityReducer:
    """Base class for dimensionality reduction."""
    
    def __init__(self, method='pca', n_components=2, **kwargs):
        """
        Initialize dimensionality reducer.
        
        Parameters:
        -----------
        method : str
            Reduction method (pca, tsne, umap)
        n_components : int
            Number of components/dimensions
        **kwargs : dict
            Additional method-specific parameters
        """
        self.method = method.lower()
        self.n_components = n_components
        self.kwargs = kwargs
        self.model = None
        self.X_reduced = None
        
        self._init_model()
    
    def _init_model(self):
        """Initialize the appropriate model."""
        if self.method == 'pca':
            self.model = PCA(n_components=self.n_components, **self.kwargs)
        elif self.method == 'tsne':
            default_params = {'random_state': 42, 'perplexity': 30}
            default_params.update(self.kwargs)
            self.model = TSNE(n_components=self.n_components, **default_params)
        elif self.method == 'umap':
            if not UMAP_AVAILABLE:
                raise ImportError("UMAP not available. Install with: pip install umap-learn")
            default_params = {'random_state': 42, 'n_neighbors': 15, 'min_dist': 0.1}
            default_params.update(self.kwargs)
            self.model = umap.UMAP(n_components=self.n_components, **default_params)
        else:
            raise ValueError(f"Unknown method: {self.method}. Use 'pca', 'tsne', or 'umap'")
    
    def fit_transform(self, X):
        """
        Fit and transform data.
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        X_reduced : np.ndarray
            Reduced data
        """
        self.X_reduced = self.model.fit_transform(X)
        return self.X_reduced
    
    def transform(self, X):
        """
        Transform new data (only for PCA and UMAP).
        
        Parameters:
        -----------
        X : np.ndarray
            Input data
            
        Returns:
        --------
        X_reduced : np.ndarray
            Reduced data
        """
        if self.method == 'tsne':
            raise ValueError("t-SNE does not support transform on new data. Use fit_transform.")
        return self.model.transform(X)
    
    def get_explained_variance(self):
        """Get explained variance ratio (PCA only)."""
        if self.method != 'pca':
            raise ValueError("Explained variance only available for PCA")
        return self.model.explained_variance_ratio_
    
    def get_loadings(self):
        """Get principal component loadings (PCA only)."""
        if self.method != 'pca':
            raise ValueError("Loadings only available for PCA")
        return self.model.components_


def apply_pca(X, n_components=2):
    """
    Apply PCA reduction.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    n_components : int
        Number of components
        
    Returns:
    --------
    X_pca : np.ndarray
        PCA-reduced data
    pca_model : PCA
        Fitted PCA model
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca


def apply_tsne(X, n_components=2, perplexity=30, random_state=42):
    """
    Apply t-SNE reduction.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    n_components : int
        Number of components
    perplexity : float
        t-SNE perplexity parameter
    random_state : int
        Random seed
        
    Returns:
    --------
    X_tsne : np.ndarray
        t-SNE-reduced data
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    X_tsne = tsne.fit_transform(X)
    return X_tsne


def apply_umap(X, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Apply UMAP reduction.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    n_components : int
        Number of components
    n_neighbors : int
        Number of neighbors
    min_dist : float
        Minimum distance
    random_state : int
        Random seed
        
    Returns:
    --------
    X_umap : np.ndarray
        UMAP-reduced data
    umap_model : UMAP
        Fitted UMAP model
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP not available. Install with: pip install umap-learn")
    
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state
    )
    X_umap = reducer.fit_transform(X)
    return X_umap, reducer


def compare_reduction_methods(X, labels=None):
    """
    Apply all reduction methods and return results.
    
    Parameters:
    -----------
    X : np.ndarray
        Input data
    labels : np.ndarray, optional
        Cluster labels for visualization
        
    Returns:
    --------
    reductions : dict
        Dictionary with reduction results
    """
    reductions = {}
    
    # PCA
    X_pca, pca_model = apply_pca(X, n_components=2)
    reductions['pca'] = {
        'data': X_pca,
        'model': pca_model,
        'explained_variance': pca_model.explained_variance_ratio_
    }
    
    # t-SNE
    X_tsne = apply_tsne(X, n_components=2)
    reductions['tsne'] = {'data': X_tsne}
    
    # UMAP (if available)
    if UMAP_AVAILABLE:
        try:
            X_umap, umap_model = apply_umap(X, n_components=2)
            reductions['umap'] = {'data': X_umap, 'model': umap_model}
        except Exception as e:
            print(f"UMAP failed: {e}")
    
    return reductions
