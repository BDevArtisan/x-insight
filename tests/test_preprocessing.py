"""
Unit tests for SScPreprocessor module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data.preprocessing import SScPreprocessor


@pytest.fixture
def sample_data():
    """Create sample SSc data for testing"""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Patient_ID': [f'P{i:03d}' for i in range(n_samples)],
        'Age': np.random.randint(20, 80, n_samples),
        'Gender': np.random.choice(['Female', 'Male'], n_samples),
        'mRSS': np.random.randint(0, 40, n_samples),
        'FVC_predicted': np.random.uniform(50, 100, n_samples),
        'DLCO_predicted': np.random.uniform(40, 90, n_samples),
        'ANA_titer': np.random.choice([80, 160, 320, 640, 1280], n_samples),
        'Anti_Scl_70': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'Anti_Centromere': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'CRP': np.random.uniform(0, 50, n_samples),
        'ESR': np.random.uniform(5, 80, n_samples),
        'True_Phenotype': np.random.choice(['Limited', 'Diffuse'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add some missing values
    missing_cols = ['CRP', 'ESR', 'DLCO_predicted']
    for col in missing_cols:
        missing_idx = np.random.choice(df.index, size=5, replace=False)
        df.loc[missing_idx, col] = np.nan
    
    return df


@pytest.fixture
def feature_columns():
    """Feature columns for testing"""
    return ['Age', 'Gender', 'mRSS', 'FVC_predicted', 'DLCO_predicted', 
            'ANA_titer', 'Anti_Scl_70', 'Anti_Centromere', 'CRP', 'ESR']


class TestSScPreprocessor:
    """Test suite for SScPreprocessor"""
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = SScPreprocessor()
        
        assert preprocessor.imputer is not None
        assert preprocessor.scaler is not None
        assert preprocessor.feature_cols is None
        assert preprocessor.fitted is False
    
    def test_fit_transform_basic(self, sample_data, feature_columns):
        """Test basic fit_transform operation"""
        preprocessor = SScPreprocessor()
        X_scaled, df_imputed = preprocessor.fit_transform(sample_data, feature_columns)
        
        # Check output shapes
        assert X_scaled.shape[0] == len(sample_data)
        assert X_scaled.shape[1] == len(feature_columns)
        assert len(df_imputed) == len(sample_data)
        
        # Check no missing values after imputation
        assert df_imputed[feature_columns].isnull().sum().sum() == 0
        
        # Check scaling (mean ~ 0, std ~ 1)
        assert np.abs(X_scaled.mean()) < 0.1
        assert np.abs(X_scaled.std() - 1.0) < 0.1
    
    def test_fit_transform_gender_encoding(self, sample_data, feature_columns):
        """Test Gender encoding"""
        preprocessor = SScPreprocessor()
        _, df_imputed = preprocessor.fit_transform(sample_data, feature_columns)
        
        # Check Gender is encoded as 0/1 (may be float after imputation)
        assert set(df_imputed['Gender'].unique()).issubset({0.0, 1.0})
        assert df_imputed['Gender'].dtype in [int, np.int64, float, np.float64]
    
    def test_fit_transform_stores_attributes(self, sample_data, feature_columns):
        """Test that fit_transform stores necessary attributes"""
        preprocessor = SScPreprocessor()
        preprocessor.fit_transform(sample_data, feature_columns)
        
        assert preprocessor.imputer is not None
        assert preprocessor.scaler is not None
        assert preprocessor.feature_cols == feature_columns
        assert preprocessor.fitted is True
    
    def test_transform_consistency(self, sample_data, feature_columns):
        """Test that transform produces consistent results"""
        preprocessor = SScPreprocessor()
        X_scaled1, _ = preprocessor.fit_transform(sample_data, feature_columns)
        X_scaled2 = preprocessor.transform(sample_data)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(X_scaled1, X_scaled2, decimal=5)
    
    def test_transform_new_data(self, sample_data, feature_columns):
        """Test transform on new data"""
        # Split data
        train_data = sample_data.iloc[:80]
        test_data = sample_data.iloc[80:]
        
        preprocessor = SScPreprocessor()
        preprocessor.fit_transform(train_data, feature_columns)
        
        # Transform test data
        X_test = preprocessor.transform(test_data)
        
        assert X_test.shape[0] == len(test_data)
        assert X_test.shape[1] == len(feature_columns)
    
    def test_inverse_transform(self, sample_data, feature_columns):
        """Test inverse transformation"""
        preprocessor = SScPreprocessor()
        X_scaled, _ = preprocessor.fit_transform(sample_data, feature_columns)
        
        # Inverse transform
        df_original = preprocessor.inverse_transform(X_scaled)
        
        # Check shape
        assert len(df_original) == len(sample_data)
        assert len(df_original.columns) == len(feature_columns)
        
        # Check column names
        assert list(df_original.columns) == feature_columns
    
    def test_save_load(self, sample_data, feature_columns, tmp_path):
        """Test saving and loading preprocessor"""
        # Fit preprocessor
        preprocessor1 = SScPreprocessor()
        X_scaled1, _ = preprocessor1.fit_transform(sample_data, feature_columns)
        
        # Save
        save_path = tmp_path / "preprocessor.pkl"
        preprocessor1.save(str(save_path))
        
        # Load
        preprocessor2 = SScPreprocessor.load(str(save_path))
        
        # Transform with loaded preprocessor
        X_scaled2 = preprocessor2.transform(sample_data)
        
        # Should produce same results
        np.testing.assert_array_almost_equal(X_scaled1, X_scaled2, decimal=5)
        
        # Check attributes
        assert preprocessor2.feature_cols == feature_columns
    
    def test_missing_values_handling(self):
        """Test handling of data with different missing patterns"""
        # Create data with various missing patterns
        data = pd.DataFrame({
            'Patient_ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            'Age': [50, 60, np.nan, 55, 45],
            'Gender': ['Female', 'Male', 'Female', np.nan, 'Male'],
            'mRSS': [15, np.nan, 10, 20, np.nan],
            'CRP': [5, 10, np.nan, np.nan, 8]
        })
        
        feature_cols = ['Age', 'Gender', 'mRSS', 'CRP']
        
        preprocessor = SScPreprocessor()
        X_scaled, df_imputed = preprocessor.fit_transform(data, feature_cols)
        
        # Check no missing values
        assert df_imputed[feature_cols].isnull().sum().sum() == 0
        
        # Check output shape
        assert X_scaled.shape == (5, 4)
    
    def test_edge_case_single_sample(self, feature_columns):
        """Test with single sample"""
        data = pd.DataFrame({
            'Patient_ID': ['P001'],
            'Age': [50],
            'Gender': ['Female'],
            'mRSS': [15],
            'FVC_predicted': [80],
            'DLCO_predicted': [70],
            'ANA_titer': [320],
            'Anti_Scl_70': [1],
            'Anti_Centromere': [0],
            'CRP': [10],
            'ESR': [20]
        })
        
        preprocessor = SScPreprocessor(n_neighbors=1)
        X_scaled, df_imputed = preprocessor.fit_transform(data, feature_columns)
        
        assert X_scaled.shape == (1, len(feature_columns))
    
    def test_edge_case_all_same_values(self):
        """Test with features having all same values"""
        data = pd.DataFrame({
            'Patient_ID': ['P001', 'P002', 'P003'],
            'Age': [50, 50, 50],
            'Gender': ['Female', 'Female', 'Female'],
            'mRSS': [15, 15, 15]
        })
        
        feature_cols = ['Age', 'Gender', 'mRSS']
        
        preprocessor = SScPreprocessor()
        X_scaled, _ = preprocessor.fit_transform(data, feature_cols)
        
        # Constant features should be scaled to 0
        assert X_scaled.shape == (3, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
