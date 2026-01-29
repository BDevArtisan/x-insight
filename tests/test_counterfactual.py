"""
Unit tests for counterfactual explanations module
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from counterfactuals import (
    ClinicalConstraints,
    generate_counterfactual_optimization,
    generate_diverse_counterfactuals,
    generate_counterfactual_dice_style,
    format_counterfactual_explanation,
    evaluate_counterfactual_quality
)
from clustering.traditional import KMeansClustering
from clustering.advanced import GMMClustering


@pytest.fixture
def sample_clustered_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    cluster_0 = np.random.randn(60, n_features) * 0.5 + [0, 0, 0, 0, 0]
    cluster_1 = np.random.randn(70, n_features) * 0.5 + [3, 3, 3, 3, 3]
    cluster_2 = np.random.randn(70, n_features) * 0.5 + [6, 0, 6, 0, 6]
    
    X = np.vstack([cluster_0, cluster_1, cluster_2])
    labels = np.array([0]*60 + [1]*70 + [2]*70)
    
    feature_names = ['Age', 'ESR', 'ANA', 'CRP', 'Hemoglobin']
    
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


class TestClinicalConstraints:
    
    def test_initialization(self):
        constraints = ClinicalConstraints(['Age', 'ESR'])
        
        assert len(constraints.feature_names) == 2
        assert len(constraints.constraints) == 0
        assert len(constraints.immutable_features) == 0
    
    def test_set_range(self):
        constraints = ClinicalConstraints(['Age', 'ESR', 'ANA'])
        constraints.set_range('Age', 18, 90)
        constraints.set_range(1, 0, 100)
        
        assert 0 in constraints.constraints
        assert 1 in constraints.constraints
        assert constraints.constraints[0]['min'] == 18
        assert constraints.constraints[0]['max'] == 90
    
    def test_set_immutable(self):
        constraints = ClinicalConstraints(['Age', 'Gender'])
        constraints.set_immutable('Gender')
        
        assert 1 in constraints.immutable_features
    
    def test_apply_constraints(self):
        constraints = ClinicalConstraints(['Age', 'ESR'])
        constraints.set_range('Age', 18, 90)
        constraints.set_range('ESR', 0, 100)
        
        x = np.array([100, -10])
        x_constrained = constraints.apply_constraints(x)
        
        assert x_constrained[0] == 90
        assert x_constrained[1] == 0
    
    def test_check_validity(self):
        constraints = ClinicalConstraints(['Age', 'ESR'])
        constraints.set_range('Age', 18, 90)
        
        assert constraints.check_validity(np.array([50, 20]))
        assert not constraints.check_validity(np.array([100, 20]))


class TestCounterfactualOptimization:
    
    def test_generate_counterfactual_basic(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 0
        
        counterfactual = generate_counterfactual_optimization(
            X, labels, patient_idx, kmeans_model,
            target_cluster=1
        )
        
        assert counterfactual is not None
        assert 'original' in counterfactual
        assert 'counterfactual' in counterfactual
        assert 'changes' in counterfactual
        assert 'original_cluster' in counterfactual
        assert 'target_cluster' in counterfactual
        assert counterfactual['original_cluster'] == 0
        assert counterfactual['target_cluster'] == 1
    
    def test_counterfactual_with_constraints(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 5
        
        constraints = ClinicalConstraints(feature_names)
        constraints.set_range('Age', 0, 100)
        constraints.set_range('ESR', 0, 150)
        
        counterfactual = generate_counterfactual_optimization(
            X, labels, patient_idx, kmeans_model,
            target_cluster=2,
            constraints=constraints
        )
        
        assert counterfactual is not None
        assert constraints.check_validity(counterfactual['counterfactual'])
    
    def test_counterfactual_with_immutable(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 10
        
        constraints = ClinicalConstraints(feature_names)
        constraints.set_immutable('Age')
        
        counterfactual = generate_counterfactual_optimization(
            X, labels, patient_idx, kmeans_model,
            target_cluster=1,
            constraints=constraints
        )
        
        if counterfactual:
            assert np.isclose(counterfactual['original'][0], counterfactual['counterfactual'][0], atol=1e-6)


class TestDiverseCounterfactuals:
    
    def test_generate_diverse_counterfactuals(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 15
        
        counterfactuals = generate_diverse_counterfactuals(
            X, labels, patient_idx, kmeans_model,
            target_cluster=1,
            n_counterfactuals=3
        )
        
        assert isinstance(counterfactuals, list)
        assert len(counterfactuals) <= 3
        
        for cf in counterfactuals:
            assert 'original' in cf
            assert 'counterfactual' in cf
            assert cf['target_cluster'] == 1


class TestDiceStyleCounterfactuals:
    
    def test_generate_dice_style(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 20
        
        counterfactuals = generate_counterfactual_dice_style(
            X, labels, patient_idx, kmeans_model,
            target_cluster=2,
            n_counterfactuals=3
        )
        
        assert isinstance(counterfactuals, list)
        
        for cf in counterfactuals:
            assert cf['success']
            assert cf['target_cluster'] == 2
            assert cf['predicted_cluster'] == 2
    
    def test_dice_with_constraints(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 25
        
        constraints = ClinicalConstraints(feature_names)
        constraints.set_range('Age', 0, 100)
        constraints.set_range('ESR', 0, 150)
        
        counterfactuals = generate_counterfactual_dice_style(
            X, labels, patient_idx, kmeans_model,
            target_cluster=1,
            constraints=constraints,
            n_counterfactuals=2
        )
        
        for cf in counterfactuals:
            assert constraints.check_validity(cf['counterfactual'])


class TestCounterfactualFormatting:
    
    def test_format_explanation(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 30
        
        counterfactual = generate_counterfactual_optimization(
            X, labels, patient_idx, kmeans_model,
            target_cluster=1
        )
        
        if counterfactual and counterfactual['success']:
            explanation = format_counterfactual_explanation(counterfactual, feature_names)
            
            assert isinstance(explanation, str)
            assert 'Cluster' in explanation
            assert '->' in explanation or '→' in explanation
    
    def test_format_unsuccessful_counterfactual(self):
        cf = {'success': False}
        explanation = format_counterfactual_explanation(cf)
        
        assert 'No valid counterfactual' in explanation


class TestCounterfactualQuality:
    
    def test_evaluate_quality(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 35
        
        counterfactual = generate_counterfactual_optimization(
            X, labels, patient_idx, kmeans_model,
            target_cluster=2
        )
        
        if counterfactual:
            metrics = evaluate_counterfactual_quality(counterfactual)
            
            assert 'valid' in metrics
            assert 'proximity' in metrics
            assert 'sparsity' in metrics
            assert 'sparsity_ratio' in metrics
            assert metrics['proximity'] >= 0
            assert metrics['sparsity'] >= 0
    
    def test_evaluate_with_constraints(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        patient_idx = 40
        
        constraints = ClinicalConstraints(feature_names)
        constraints.set_range('Age', 0, 100)
        
        counterfactual = generate_counterfactual_optimization(
            X, labels, patient_idx, kmeans_model,
            target_cluster=1,
            constraints=constraints
        )
        
        if counterfactual:
            metrics = evaluate_counterfactual_quality(counterfactual, constraints)
            
            assert 'constraints_satisfied' in metrics
            assert isinstance(metrics['constraints_satisfied'], bool)


class TestEdgeCases:
    
    def test_no_valid_target_cluster(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        
        single_cluster_labels = np.zeros(len(labels), dtype=int)
        patient_idx = 0
        
        counterfactual = generate_counterfactual_optimization(
            X, single_cluster_labels, patient_idx, kmeans_model
        )
        
        assert counterfactual is None
    
    def test_empty_target_cluster(self, sample_clustered_data, kmeans_model):
        X, labels, feature_names = sample_clustered_data
        
        labels_modified = labels.copy()
        labels_modified[labels == 2] = 1
        
        patient_idx = 0
        
        counterfactuals = generate_counterfactual_dice_style(
            X, labels_modified, patient_idx, kmeans_model,
            target_cluster=2,
            n_counterfactuals=2
        )
        
        assert isinstance(counterfactuals, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
