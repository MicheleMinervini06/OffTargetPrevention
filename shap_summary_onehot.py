"""
Script per generare un SHAP summary plot usando il modello best_xgb_regressor_change_OneHot.joblib
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import joblib

from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.utilities import load_order_sg_rnas, order_sg_rnas, build_sequence_features, \
    create_nucleotides_to_position_mapping


def create_feature_names_onehot(include_distance=True):
    """
    Crea i nomi delle feature per l'encoding OneHot.
    Per ogni posizione (1-23) ci sono 4 feature (A, C, G, T) dopo l'OR tra target e off-target.
    """
    nucleotides = ['A', 'C', 'G', 'T']
    feature_names = []
    
    for pos in range(1, 24):  # Posizioni 1-23
        for nuc in nucleotides:
            feature_names.append(f"Pos{pos}_{nuc}")
    
    if include_distance:
        feature_names.append("Distance")
    
    return feature_names


def run_shap_summary():
    """
    Carica il modello best_xgb_regressor_change_OneHot.joblib ed esegue uno SHAP summary plot.
    """
    # Carica il modello
    model_path = "best_xgb_regressor_change_OneHot.joblib"
    print(f"Caricamento modello da: {model_path}")
    model = joblib.load(model_path)
    print(f"Modello caricato: {type(model)}")
    
    # Carica i dataset
    data_type = "CHANGEseq"
    datasets_dir_path = general_utilities.DATASETS_PATH + 'include_on_targets/'
    
    print(f"Caricamento dataset da: {datasets_dir_path}")
    positive_df = pd.read_csv(datasets_dir_path + f'{data_type}_positive.csv', index_col=0)
    negative_df = pd.read_csv(datasets_dir_path + f'{data_type}_negative.csv', index_col=0)
    
    # Rimuovi sequenze con 'N'
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find('N') == -1]
    
    print(f"Positive samples: {len(positive_df)}")
    print(f"Negative samples: {len(negative_df)}")
    
    # Combina i dataset
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    
    # Campiona se troppo grande (per velocizzare SHAP)
    sample_size = min(5000, len(combined_df))
    if len(combined_df) > sample_size:
        print(f"Campionamento di {sample_size} samples per SHAP...")
        combined_df = combined_df.sample(n=sample_size, random_state=42)
    
    # Crea le feature con encoding OneHot
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    
    print("Costruzione features con encoding OneHot...")
    features = build_sequence_features(
        combined_df, 
        nucleotides_to_position_mapping,
        include_distance_feature=True,
        include_sequence_features=True,
        encoding='OneHot'
    )
    
    print(f"Shape delle features: {features.shape}")
    
    # Crea i nomi delle feature
    feature_names = create_feature_names_onehot(include_distance=True)
    print(f"Numero di feature names: {len(feature_names)}")
    
    # Crea l'explainer SHAP
    print("Creazione SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Calcola i valori SHAP
    print("Calcolo dei valori SHAP (potrebbe richiedere qualche minuto)...")
    shap_values = explainer.shap_values(features)
    
    print(f"Shape dei SHAP values: {shap_values.shape}")
    
    # Plot summary
    print("Generazione SHAP Summary Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, features, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary_onehot.png", dpi=150, bbox_inches='tight')
    print("Plot salvato come 'shap_summary_onehot.png'")
    plt.show()
    
    # Bar plot delle feature importance
    print("Generazione SHAP Bar Plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, features, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("shap_bar_onehot.png", dpi=150, bbox_inches='tight')
    print("Plot salvato come 'shap_bar_onehot.png'")
    plt.show()
    
    return shap_values, features, feature_names


if __name__ == '__main__':
    shap_values, features, feature_names = run_shap_summary()
