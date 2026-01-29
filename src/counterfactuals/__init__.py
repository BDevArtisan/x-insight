"""
Counterfactual explanations module for cluster assignments.
"""

from .counterfactual_explanations import (
    ClinicalConstraints,
    generate_counterfactual_optimization,
    generate_diverse_counterfactuals,
    generate_counterfactual_dice_style,
    format_counterfactual_explanation,
    plot_counterfactual_comparison,
    plot_diverse_counterfactuals,
    evaluate_counterfactual_quality
)

__all__ = [
    'ClinicalConstraints',
    'generate_counterfactual_optimization',
    'generate_diverse_counterfactuals',
    'generate_counterfactual_dice_style',
    'format_counterfactual_explanation',
    'plot_counterfactual_comparison',
    'plot_diverse_counterfactuals',
    'evaluate_counterfactual_quality'
]
