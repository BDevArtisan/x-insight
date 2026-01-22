"""
Preprocessing module for SSc clinical data.
Handles encoding, imputation, and scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import joblib


class SScPreprocessor:
    """Preprocessor for Systemic Sclerosis clinical data."""
    
    def __init__(self, n_neighbors=5):
        """
        Initialize preprocessor.
        
        Parameters:
        -----------
        n_neighbors : int
            Number of neighbors for KNN imputation
        """
        self.n_neighbors = n_neighbors
        self.scaler = StandardScaler()
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.fitted = False
        
    def fit_transform(self, df, feature_cols=None):
        """
        Fit and transform data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        feature_cols : list, optional
            List of feature columns. If None, uses all except Patient_ID and True_Phenotype
            
        Returns:
        --------
        X_scaled : np.ndarray
            Scaled features
        df_processed : pd.DataFrame
            Processed dataframe
        """
        if feature_cols is None:
            # Default feature columns
            feature_cols = ['Age', 'Gender', 'mRSS', 'FVC_predicted', 'DLCO_predicted', 
                           'ANA_titer', 'Anti_Scl_70', 'Anti_Centromere', 'CRP', 'ESR', 
                           'Raynauds', 'Digital_Ulcers']
        
        self.feature_cols = feature_cols
        df_features = df[feature_cols].copy()
        
        # Encode Gender if present
        if 'Gender' in df_features.columns:
            df_features['Gender'] = self.label_encoder.fit_transform(df_features['Gender'])
        
        # Impute missing values
        df_imputed = pd.DataFrame(
            self.imputer.fit_transform(df_features),
            columns=df_features.columns,
            index=df_features.index
        )
        
        # Scale features
        X_scaled = self.scaler.fit_transform(df_imputed)
        
        self.fitted = True
        
        return X_scaled, df_imputed
    
    def transform(self, df):
        """
        Transform new data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        X_scaled : np.ndarray
            Scaled features
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        df_features = df[self.feature_cols].copy()
        
        # Encode Gender if present
        if 'Gender' in df_features.columns:
            df_features['Gender'] = self.label_encoder.transform(df_features['Gender'])
        
        # Impute and scale
        df_imputed = pd.DataFrame(
            self.imputer.transform(df_features),
            columns=df_features.columns,
            index=df_features.index
        )
        
        X_scaled = self.scaler.transform(df_imputed)
        
        return X_scaled
    
    def inverse_transform(self, X_scaled):
        """
        Inverse transform scaled data back to original scale.
        
        Parameters:
        -----------
        X_scaled : np.ndarray
            Scaled features
            
        Returns:
        --------
        df_original : pd.DataFrame
            Data in original scale
        """
        X_original = self.scaler.inverse_transform(X_scaled)
        df_original = pd.DataFrame(X_original, columns=self.feature_cols)
        
        # Decode Gender if present
        if 'Gender' in df_original.columns:
            df_original['Gender'] = self.label_encoder.inverse_transform(
                df_original['Gender'].astype(int)
            )
        
        return df_original
    
    def save(self, filepath):
        """Save preprocessor to file."""
        joblib.dump({
            'scaler': self.scaler,
            'imputer': self.imputer,
            'label_encoder': self.label_encoder,
            'feature_cols': self.feature_cols,
            'fitted': self.fitted,
            'n_neighbors': self.n_neighbors
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load preprocessor from file."""
        data = joblib.load(filepath)
        preprocessor = cls(n_neighbors=data['n_neighbors'])
        preprocessor.scaler = data['scaler']
        preprocessor.imputer = data['imputer']
        preprocessor.label_encoder = data['label_encoder']
        preprocessor.feature_cols = data['feature_cols']
        preprocessor.fitted = data['fitted']
        return preprocessor


def preprocess_data(df, feature_cols=None, return_preprocessor=False):
    """
    Convenience function for preprocessing.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : list, optional
        Feature columns
    return_preprocessor : bool
        Whether to return the preprocessor object
        
    Returns:
    --------
    X_scaled : np.ndarray
        Scaled features
    df_imputed : pd.DataFrame
        Imputed dataframe (optional)
    preprocessor : SScPreprocessor
        Fitted preprocessor (optional)
    """
    preprocessor = SScPreprocessor()
    X_scaled, df_imputed = preprocessor.fit_transform(df, feature_cols)
    
    if return_preprocessor:
        return X_scaled, df_imputed, preprocessor
    return X_scaled, df_imputed
