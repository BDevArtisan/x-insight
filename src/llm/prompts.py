VISUALIZATION_PROMPT = """Analyze this clustering visualization of patients with systemic sclerosis.

Context:
The patients were grouped using unsupervised clustering based on their clinical characteristics.
The visualization is intended for exploratory clinical interpretation, not diagnostic classification.

Instructions:
1. Describe the observed cluster separation using only what is visible in the figure.
2. Comment on clustering quality: overlap, density, compactness, outliers, and ambiguous regions.
3. Identify any visible patterns or structures that may be clinically relevant.
4. Suggest cautious clinical interpretations, clearly distinguishing observation from speculation.
5. Mention key limitations of the visualization, especially if clusters overlap or separation is weak.

Output format:
- Cluster separation:
- Clustering quality:
- Notable patterns:
- Possible clinical interpretation:
- Limitations / caution:

Be concise, precise, and avoid causal claims unless they are directly supported by the visualization."""

GLOBAL_EXPLAINABILITY_PROMPT = """Analyze this global explainability visualization of the clustering.

Context:
This visualization shows feature importance, cluster profiles, or contrastive differences between clusters.
The clustering is unsupervised, so the results should be interpreted as exploratory associations rather than validated clinical phenotypes.

Instructions:
1. Identify the most important features shown in the visualization.
2. Explain how these features distinguish the clusters, referring to direction and relative magnitude when visible.
3. Comment on the consistency of the results across features or clusters.
4. Propose a cautious clinical interpretation of the observed differences.
5. State any limitations, including possible redundancy between features, scaling effects, or weak cluster separation.

Output format:
- Most important features:
- How they distinguish clusters:
- Consistency of the pattern:
- Possible clinical interpretation:
- Limitations / caution:

Be concise, clinically oriented, and avoid overinterpreting feature importance as causality."""

LOCAL_EXPLAINABILITY_PROMPT = """Analyze this local explanation for a specific patient.

Context:
This visualization explains why a specific patient was assigned to their cluster.
For SHAP and LIME, the explanation may come from a surrogate decision tree trained to approximate the clustering.
Therefore, interpretation depends on surrogate fidelity and should be treated cautiously if fidelity is low.

Instructions:
1. Identify the features that contributed most to this patient's cluster assignment.
2. Explain whether each major feature supports or opposes the assigned cluster, if visible.
3. Comment on the coherence of the explanation: do the main features form a clinically plausible pattern?
4. Suggest cautious clinical implications for this patient.
5. Mention limitations, especially if the explanation is based on a surrogate model or if feature effects are small.

Output format:
- Main contributing features:
- Direction of contribution:
- Coherence of the explanation:
- Possible clinical implications:
- Limitations / caution:

Be concise and precise. Do not present the explanation as a diagnosis or causal mechanism."""

COUNTERFACTUAL_PROMPT = """Analyze these counterfactual scenarios for a patient.

Context:
These visualizations show which feature changes would move the patient from their current cluster to another cluster.
Counterfactuals in this setting are exploratory and should not be interpreted as causal or directly actionable treatment recommendations.

Instructions:
1. Identify the features proposed for modification and the magnitude/direction of the changes.
2. Assess whether the proposed changes appear clinically plausible.
3. Comment on whether the counterfactual scenario is sparse and realistic, or whether it requires many large changes.
4. Suggest possible therapeutic or clinical implications only if they are plausible and clearly framed as hypotheses.
5. Mention limitations, especially that changing a feature in the model does not imply that changing it clinically would move the patient between disease states.

Output format:
- Proposed changes:
- Clinical feasibility:
- Plausibility of the scenario:
- Possible clinical implications:
- Limitations / caution:

Be concise, pragmatic, and avoid causal or prescriptive medical claims."""
