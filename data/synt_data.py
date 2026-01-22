import pandas as pd
import numpy as np

def generate_ssc_synthetic_data(n_samples=300):
    # Pour la reproductibilité
    np.random.seed(42)
    
    # Répartition équilibrée pour commencer (ajustable)
    n_per_cluster = n_samples // 3

    # --- CLUSTER 1: Limited SSc (Forme limitée) ---
    # Profil: Anti-Centromere +, mRSS bas, Poumons sains
    c1 = pd.DataFrame({
        'Patient_ID': [f'P{i:03d}' for i in range(n_per_cluster)],
        'Age': np.random.normal(55, 10, n_per_cluster).astype(int),
        'Gender': np.random.choice(['Female', 'Male'], n_per_cluster, p=[0.8, 0.2]),
        'mRSS': np.random.normal(8, 4, n_per_cluster).clip(0, 51).round(1), # Score cutané bas
        'FVC_predicted': np.random.normal(95, 10, n_per_cluster).clip(50, 120).round(1), # FVC normal
        'DLCO_predicted': np.random.normal(90, 10, n_per_cluster).clip(40, 120).round(1), # DLCO normal
        'ANA_titer': np.random.choice([160, 320, 640], n_per_cluster),
        'Anti_Scl_70': np.random.choice([0, 1], n_per_cluster, p=[0.9, 0.1]), # Rarement positif
        'Anti_Centromere': np.random.choice([0, 1], n_per_cluster, p=[0.1, 0.9]), # Souvent positif
        'CRP': np.random.normal(3, 2, n_per_cluster).clip(0, 20).round(1), # Inflammation faible
        'ESR': np.random.normal(15, 10, n_per_cluster).clip(0, 50).round(1),
        'Raynauds': 1, # Quasi constant
        'Digital_Ulcers': np.random.choice([0, 1], n_per_cluster, p=[0.8, 0.2]), # Moins fréquent
        'True_Phenotype': 'Limited_SSc' # Pour validation (à retirer avant clustering non-supervisé)
    })

    # --- CLUSTER 2: Diffuse SSc - Modéré ---
    # Profil: Anti-Scl-70 +, mRSS modéré, atteinte pulmonaire débutante
    c2 = pd.DataFrame({
        'Patient_ID': [f'P{i:03d}' for i in range(n_per_cluster, 2*n_per_cluster)],
        'Age': np.random.normal(48, 12, n_per_cluster).astype(int),
        'Gender': np.random.choice(['Female', 'Male'], n_per_cluster, p=[0.7, 0.3]),
        'mRSS': np.random.normal(20, 6, n_per_cluster).clip(0, 51).round(1),
        'FVC_predicted': np.random.normal(75, 12, n_per_cluster).clip(40, 110).round(1),
        'DLCO_predicted': np.random.normal(70, 12, n_per_cluster).clip(30, 110).round(1),
        'ANA_titer': np.random.choice([320, 640, 1280], n_per_cluster),
        'Anti_Scl_70': np.random.choice([0, 1], n_per_cluster, p=[0.2, 0.8]), # Souvent positif
        'Anti_Centromere': np.random.choice([0, 1], n_per_cluster, p=[0.9, 0.1]),
        'CRP': np.random.normal(8, 5, n_per_cluster).clip(0, 50).round(1),
        'ESR': np.random.normal(30, 15, n_per_cluster).clip(0, 100).round(1),
        'Raynauds': 1,
        'Digital_Ulcers': np.random.choice([0, 1], n_per_cluster, p=[0.6, 0.4]),
        'True_Phenotype': 'Diffuse_Moderate'
    })

    # --- CLUSTER 3: Diffuse SSc - Sévère/Fibrotique ---
    # Profil: mRSS élevé, ILD (FVC/DLCO bas), Inflammation
    c3 = pd.DataFrame({
        'Patient_ID': [f'P{i:03d}' for i in range(2*n_per_cluster, 3*n_per_cluster)],
        'Age': np.random.normal(52, 11, n_per_cluster).astype(int),
        'Gender': np.random.choice(['Female', 'Male'], n_per_cluster, p=[0.6, 0.4]),
        'mRSS': np.random.normal(35, 8, n_per_cluster).clip(0, 51).round(1),
        'FVC_predicted': np.random.normal(55, 10, n_per_cluster).clip(25, 100).round(1), # Atteinte sévère
        'DLCO_predicted': np.random.normal(45, 10, n_per_cluster).clip(20, 90).round(1),
        'ANA_titer': np.random.choice([640, 1280, 2560], n_per_cluster),
        'Anti_Scl_70': np.random.choice([0, 1], n_per_cluster, p=[0.1, 0.9]), # Très positif
        'Anti_Centromere': np.random.choice([0, 1], n_per_cluster, p=[0.95, 0.05]),
        'CRP': np.random.normal(20, 10, n_per_cluster).clip(0, 100).round(1),
        'ESR': np.random.normal(50, 20, n_per_cluster).clip(0, 140).round(1),
        'Raynauds': 1,
        'Digital_Ulcers': np.random.choice([0, 1], n_per_cluster, p=[0.4, 0.6]),
        'True_Phenotype': 'Diffuse_Severe'
    })

    # Fusion des clusters
    df = pd.concat([c1, c2, c3], ignore_index=True)
    
    # Mélange aléatoire des lignes
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Simulation de données manquantes (NaN) dans 5% des cas pour CRP, ESR et DLCO
    # Cela vous oblige à gérer l'imputation dans votre preprocessing.py
    for col in ['CRP', 'ESR', 'DLCO_predicted']:
        df.loc[df.sample(frac=0.05).index, col] = np.nan

    return df

# Génération et sauvegarde
df_synthetic = generate_ssc_synthetic_data()
df_synthetic.to_csv("ssc_synthetic_data.csv", index=False)

print("Dataset généré : 'ssc_synthetic_data.csv'")
print(df_synthetic.head())