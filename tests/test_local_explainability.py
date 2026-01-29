"""
Unit tests for local explainability module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from explainability.local_explainability import (
    explain_patient_shap,
    explain_patient_lime,
    explain_distance_to_centroid,
    explain_probabilistic_membership,
    explain_patient
)
from clustering.traditional import KMeansClustering
from clustering.advanced import GMMClustering


@pytest.fixture
def sample_clustered_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    cluster_0 = np.random.randn(60, n_features) + [0, 0, 0, 0, 0]
    cluster_1 = np.random.randn(70, n_features) + [3, 3, 3, 3, 3]
    cluster_2 = np.random.randn(70, n_features) + [6, 0, 6, 0, 6]
    
    X = np.vstack([cluster_0, cluster_1, cluster_2])
    labels = np.array([0]*60 + [1]*70 + [2]*70)
    
    feature_names = [f'Feature_{i}' for i in range(n_features)]
    
    return X, labels, feature_names


@pytest.fixture
def kmeans_model(sample_clustered_data):
    X, labels, _ = sample_clustered_data
    model = KMeansClustering(n_clusters=3, random_state=42)
    model.fit(X)
    return model


@pytest.fixture
def gmm_model(sample_clustered_data):
    X, labels, _ = sample_clustered_data
    model = GMMClustering(n_components=3, random_state=42)
    model.fit(X)
    return model


class TestShapExplanation:
    
    def test_explain_patient_shap(self, sample_clustered_data):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 0
        
        explanation = explain_patient_shap(X, labels, patient_idx, feature_names)
        
        assert 'patient_idx' in explanation
        assert 'cluster' in explanation
        assert 'shap_values' in explanation
        assert 'feature_names' in explanation
        assert len(explanation['shap_values']) == len(feature_names)
        assert explanation['patient_idx'] == patient_idx
    
    def test_shap_with_dataframe(self, sample_clustered_data):
        X, labels, feature_names = sample_clustered_data
        df = pd.DataFrame(X, columns=feature_names)
        patient_idx = 5
        
        explanation = explain_patient_shap(df, labels, patient_idx)
        
        assert explanation['feature_names'] == feature_names
        assert len(explanation['shap_values']) == len(feature_names)


class TestLimeExplanation:
    
    def test_explain_patient_lime(self, sample_clustered_data):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 10
        
        explanation = explain_patient_lime(X, labels, patient_idx, feature_names)
        
        assert 'patient_idx' in explanation
        assert 'cluster' in explanation
        assert 'lime_values' in explanation
        assert 'probability' in explanation
        assert len(explanation['lime_values']) == len(feature_names)
    
    def test_lime_with_dataframe(self, sample_clustered_data):
        X, labels, feature_names = sample_clustered_data
        df = pd.DataFrame(X, columns=feature_names)
        patient_idx = 15
        
        explanation = explain_patient_lime(df, labels, patient_idx)
        
        assert explanation['feature_names'] == feature_names
        assert len(explanation['probability']) == 3


class TestDistanceExplanation:
    
    def test_distance_to_centroid_kmeans(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 20
        
        explanation = explain_distance_to_centroid(X, labels, patient_idx, kmeans_model, feature_names)
        
        assert 'patient_idx' in explanation
        assert 'cluster' in explanation
        assert 'feature_distances' in explanation
        assert 'feature_contributions' in explanation
        assert 'total_distance' in explanation
        assert len(explanation['feature_distances']) == len(feature_names)
        assert np.isclose(explanation['feature_contributions'].sum(), 1.0, atol=1e-5)
    
    def test_distance_with_dataframe(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        df = pd.DataFrame(X, columns=feature_names)
        patient_idx = 25
        
        explanation = explain_distance_to_centroid(df, labels, patient_idx, kmeans_model)
        
        assert explanation['feature_names'] == feature_names
        assert 'centroid_values' in explanation


class TestProbabilisticExplanation:
    
    def test_probabilistic_membership_gmm(self, sample_clustered_data, gmm_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 30
        
        explanation = explain_probabilistic_membership(X, labels, patient_idx, gmm_model, feature_names)
        
        assert 'patient_idx' in explanation
        assert 'cluster' in explanation
        assert 'probabilities' in explanation
        assert 'distances_to_centers' in explanation
        assert 'membership_certainty' in explanation
        assert len(explanation['probabilities']) == 3
        assert np.isclose(explanation['probabilities'].sum(), 1.0, atol=1e-5)
    
    def test_probabilistic_with_kmeans(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 35
        
        explanation = explain_probabilistic_membership(X, labels, patient_idx, kmeans_model, feature_names)
        
        assert 'probabilities' in explanation
        assert explanation['membership_certainty'] == 1.0


class TestCombinedExplanation:
    
    def test_explain_patient_all_methods(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 40
        
        explanations = explain_patient(
            X, labels, patient_idx, kmeans_model,
            methods=['shap', 'lime', 'distance', 'probabilistic'],
            feature_names=feature_names
        )
        
        assert 'shap' in explanations
        assert 'lime' in explanations
        assert 'distance' in explanations
        assert 'probabilistic' in explanations
    
    def test_explain_patient_subset_methods(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 45
        
        explanations = explain_patient(
            X, labels, patient_idx, kmeans_model,
            methods=['distance', 'probabilistic'],
            feature_names=feature_names
        )
        
        assert 'shap' not in explanations
        assert 'lime' not in explanations
        assert 'distance' in explanations
        assert 'probabilistic' in explanations


class TestEdgeCases:
    
    def test_patient_at_boundary(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 59
        
        explanation = explain_distance_to_centroid(X, labels, patient_idx, kmeans_model, feature_names)
        
        assert explanation['total_distance'] >= 0
    
    def test_different_patient_indices(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        
        for patient_idx in [0, 50, 100, 150, 199]:
            explanation = explain_distance_to_centroid(X, labels, patient_idx, kmeans_model, feature_names)
            assert explanation['patient_idx'] == patient_idx
            assert explanation['cluster'] in [0, 1, 2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
