import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from explainability import (
    train_surrogate_tree,
    compute_permutation_importance,
    compute_shap_importance,
    compute_feature_importance,
    compute_cluster_profiles,
    compute_contrastive_differences,
    extract_decision_rules
)


@pytest.fixture
def sample_clustered_data():
    np.random.seed(42)
    n_samples = 200
    n_features = 5
    
    cluster1 = np.random.randn(70, n_features) + np.array([0, 0, 0, 0, 0])
    cluster2 = np.random.randn(70, n_features) + np.array([3, 3, 3, 3, 3])
    cluster3 = np.random.randn(60, n_features) + np.array([-3, -3, -3, -3, -3])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    labels = np.array([0]*70 + [1]*70 + [2]*60)
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    
    return X, labels, df, feature_names


class TestSurrogateTree:
    
    def test_train_surrogate_tree(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        tree, importances = train_surrogate_tree(X, labels, feature_names)
        
        assert tree is not None
        assert len(importances) == len(feature_names)
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
        assert importances['importance'].sum() > 0
    
    def test_surrogate_tree_with_dataframe(self, sample_clustered_data):
        _, labels, df, _ = sample_clustered_data
        
        tree, importances = train_surrogate_tree(df, labels)
        
        assert len(importances) == len(df.columns)
        assert set(importances['feature']) == set(df.columns)


class TestPermutationImportance:
    
    def test_compute_permutation_importance(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        importances = compute_permutation_importance(X, labels, feature_names, n_repeats=5)
        
        assert len(importances) == len(feature_names)
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
        assert 'std' in importances.columns
    
    def test_permutation_with_dataframe(self, sample_clustered_data):
        _, labels, df, _ = sample_clustered_data
        
        importances = compute_permutation_importance(df, labels, n_repeats=5)
        
        assert set(importances['feature']) == set(df.columns)


class TestShapImportance:
    
    def test_compute_shap_importance(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        importances, shap_values, explainer = compute_shap_importance(X, labels, feature_names)
        
        assert len(importances) == len(feature_names)
        assert 'feature' in importances.columns
        assert 'importance' in importances.columns
        assert shap_values is not None
        assert explainer is not None


class TestFeatureImportance:
    
    def test_all_methods(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        results = compute_feature_importance(X, labels, feature_names, method='all')
        
        assert 'surrogate' in results
        assert 'permutation' in results
        assert 'shap' in results
    
    def test_single_method(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        results = compute_feature_importance(X, labels, feature_names, method='surrogate')
        
        assert 'surrogate' in results
        assert len(results) == 1


class TestClusterProfiling:
    
    def test_compute_cluster_profiles(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        profiles = compute_cluster_profiles(df, labels, feature_names)
        
        assert len(profiles) == 3
        assert 'mean' in profiles.columns.get_level_values(1)
        assert 'median' in profiles.columns.get_level_values(1)
        assert 'std' in profiles.columns.get_level_values(1)
    
    def test_contrastive_differences(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        differences, effect_sizes = compute_contrastive_differences(df, labels, feature_names)
        
        assert differences.shape[0] == 3
        assert differences.shape[1] == len(feature_names)
        assert effect_sizes.shape == differences.shape


class TestDecisionRules:
    
    def test_extract_decision_rules(self, sample_clustered_data):
        X, labels, df, feature_names = sample_clustered_data
        
        tree, _ = train_surrogate_tree(X, labels, feature_names, max_depth=3)
        rules_df = extract_decision_rules(tree, feature_names)
        
        assert len(rules_df) > 0
        assert 'conditions' in rules_df.columns
        assert 'predicted_class' in rules_df.columns
        assert 'samples' in rules_df.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
