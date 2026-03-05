VISUALIZATION_PROMPT = """Analyse cette visualisation de clustering de patients atteints de sclérose systémique.

Contexte: Les patients ont été regroupés par clustering automatique basé sur leurs caractéristiques cliniques.

Instructions:
1. Décris la séparation des clusters observée
2. Commente la qualité du clustering (chevauchement, densité)
3. Identifie les patterns ou structures intéressantes
4. Suggère des interprétations cliniques possibles

Sois concis et précis dans ton analyse."""

GLOBAL_EXPLAINABILITY_PROMPT = """Analyse cette visualisation d'explicabilité globale du clustering.

Contexte: Cette visualisation montre l'importance des features ou les profils des clusters.

Instructions:
1. Identifie les features les plus importantes
2. Explique comment ces features distinguent les clusters
3. Commente la cohérence des résultats
4. Propose une interprétation clinique des différences observées

Sois concis et orienté vers l'interprétation clinique."""

LOCAL_EXPLAINABILITY_PROMPT = """Analyse cette explication locale pour un patient spécifique.

Contexte: Cette visualisation explique pourquoi ce patient a été assigné à son cluster.

Instructions:
1. Identifie les features qui ont le plus contribué à l'assignation
2. Explique l'impact de chaque feature principale
3. Commente la cohérence de l'explication
4. Suggère des implications cliniques pour ce patient

Sois concis et précis."""

COUNTERFACTUAL_PROMPT = """Analyse ces scénarios contrefactuels pour un patient.

Contexte: Ces visualisations montrent quels changements de features feraient passer le patient à un autre cluster.

Instructions:
1. Identifie les features à modifier et leur amplitude
2. Évalue la faisabilité clinique des changements proposés
3. Commente la plausibilité des scénarios
4. Suggère des implications thérapeutiques potentielles

Sois concis et pragmatique."""
