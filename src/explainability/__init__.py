from .global_explainability import (
    compute_feature_importance,
    train_surrogate_tree,
    compute_permutation_importance,
    compute_shap_importance,
    compute_cluster_profiles,
    compute_contrastive_differences,
    plot_feature_importance,
    plot_cluster_comparison_bars,
    plot_contrastive_heatmap,
    plot_cluster_radar,
    plot_shap_summary,
    plot_feature_importance_comparison,
    extract_decision_rules
)

from .local_explainability import (
    explain_patient,
    explain_patient_shap,
    explain_patient_lime,
    explain_distance_to_centroid,
    explain_probabilistic_membership,
    plot_shap_explanation,
    plot_lime_explanation,
    plot_distance_explanation,
    plot_probabilistic_explanation,
    plot_patient_explanation_summary
)

__all__ = [
    'compute_feature_importance',
    'train_surrogate_tree',
    'compute_permutation_importance',
    'compute_shap_importance',
    'compute_cluster_profiles',
    'compute_contrastive_differences',
    'plot_feature_importance',
    'plot_cluster_comparison_bars',
    'plot_contrastive_heatmap',
    'plot_cluster_radar',
    'plot_shap_summary',
    'plot_feature_importance_comparison',
    'extract_decision_rules',
    'explain_patient',
    'explain_patient_shap',
    'explain_patient_lime',
    'explain_distance_to_centroid',
    'explain_probabilistic_membership',
    'plot_shap_explanation',
    'plot_lime_explanation',
    'plot_distance_explanation',
    'plot_probabilistic_explanation',
    'plot_patient_explanation_summary'
]
