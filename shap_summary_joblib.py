"""
Script per generare SHAP summary plot dal modello best_xgb_regressor_change_OneHot.joblib
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from tqdm import tqdm
import xgboost as xgb
import os

from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.utilities import (
    build_sequence_features,
    create_nucleotides_to_position_mapping
)
from collections import defaultdict

FIGURES_PATH = general_utilities.FIGURES_PATH


def plot_shap_summary(shap_values, X_sample=None, feature_names=None, output_path=None, 
                      max_display=20, title=None, plot_type="dot", encoding=None, 
                      include_distance=None, task_type="Regression"):
    """
    Generate a SHAP summary plot consistent with the paper style.
    """
    print("🔍 Generating SHAP summary plot...")

    # Set white background style without grid
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.grid'] = False
    
    # Create the summary plot with appropriate size
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.grid(False)
    
    # Create the SHAP summary plot
    shap.summary_plot(shap_values, 
                     features=X_sample,
                     feature_names=feature_names, 
                     plot_type=plot_type,
                     max_display=max_display,
                     show=False)

    # Auto-generate title if not provided but encoding is available
    if title is None and encoding is not None:
        distance_text = "seq-dist" if include_distance else "seq"
        title_line1 = f"{task_type}-{distance_text}"
        title_line2 = f"{encoding} encoding"
        title = f"{title_line1}\n{title_line2}"
    
    if title:
        plt.title(title, fontsize=12, pad=10, loc='left', fontweight='normal')
    
    plt.xlabel("SHAP value (impact on model output)", fontsize=12)
    plt.ylabel("", fontsize=12)
    
    plt.tight_layout()

    # Save or display the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor='white', edgecolor='none')
        print(f"✅ Plot saved to {output_path}")
    else:
        plt.show()

    print("✅ SHAP summary plot generated successfully!")


def aggregate_shap_by_kmer(shap_values, X, combined_df, k=3, include_distance=True):
    """
    Aggrega i valori SHAP per k-mer specifico invece che per posizione.
    
    Per ogni k-mer trovato nelle sequenze, somma i valori SHAP di tutte le posizioni
    dove quel k-mer appare (confronto tra target e offtarget).
    
    IMPORTANTE: Preserva la feature Distance se presente (ultima colonna)
    
    Returns:
        aggregated_shap: array di SHAP values aggregati per k-mer (+ Distance se presente)
        kmer_names: lista dei nomi delle feature (k-mer + Distance se presente)
        kmer_features: matrice delle feature aggregate per k-mer (+ Distance se presente)
    """
    print("\n🔧 Aggregating SHAP values by k-mer...")
    
    # Calcola il numero di feature k-mer (escludendo distance se presente)
    n_kmer_features = (23 - k + 1) * k  # Per k=3: 21*3 = 63
    
    # Separa le feature k-mer dalla distance
    if include_distance and X.shape[1] > n_kmer_features:
        print(f"   📏 Distance feature detected at position {n_kmer_features}")
        shap_values_kmers = shap_values[:, :n_kmer_features]
        shap_values_distance = shap_values[:, n_kmer_features:]
        X_kmers = X[:, :n_kmer_features]
        X_distance = X[:, n_kmer_features:]
    else:
        shap_values_kmers = shap_values
        shap_values_distance = None
        X_kmers = X
        X_distance = None
    
    # Dizionario per raccogliere i valori SHAP per ogni k-mer per ogni campione
    # Format: {sample_idx: {kmer: shap_value}}
    sample_kmer_shap = defaultdict(lambda: defaultdict(float))
    sample_kmer_features = defaultdict(lambda: defaultdict(float))
    all_kmers = set()
    
    def get_kmers(sequence, k):
        """Estrae tutti i k-mer da una sequenza"""
        return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
    
    # Per ogni campione
    for sample_idx in tqdm(range(len(combined_df)), desc="Processing k-mers"):
        target_seq = combined_df.iloc[sample_idx]['target']
        offtarget_seq = combined_df.iloc[sample_idx]['offtarget_sequence']
        
        # Estrai k-mer dalla sequenza target
        target_kmers = get_kmers(target_seq, k)
        
        # Per ogni posizione di k-mer
        for kmer_idx, t_kmer in enumerate(target_kmers):
            # Usa il k-mer del target come chiave
            kmer_key = t_kmer
            all_kmers.add(kmer_key)
            
            # Gli indici delle feature per questo k-mer sono [kmer_idx * k : (kmer_idx + 1) * k]
            start_idx = kmer_idx * k
            end_idx = start_idx + k
            
            # Somma i valori SHAP per tutte le posizioni di questo k-mer
            kmer_shap_contribution = shap_values_kmers[sample_idx, start_idx:end_idx].sum()
            sample_kmer_shap[sample_idx][kmer_key] += kmer_shap_contribution
            
            # Anche le feature (per colorazione)
            kmer_feature_value = X_kmers[sample_idx, start_idx:end_idx].sum()
            sample_kmer_features[sample_idx][kmer_key] += kmer_feature_value
    
    # Crea array dalle strutture dati
    kmer_names = sorted(all_kmers)  # Ordina alfabeticamente
    n_samples = len(combined_df)
    n_kmers = len(kmer_names)
    
    aggregated_shap = np.zeros((n_samples, n_kmers))
    aggregated_features = np.zeros((n_samples, n_kmers))
    
    # Riempi le matrici
    for sample_idx in range(n_samples):
        for kmer_idx, kmer in enumerate(kmer_names):
            aggregated_shap[sample_idx, kmer_idx] = sample_kmer_shap[sample_idx].get(kmer, 0)
            aggregated_features[sample_idx, kmer_idx] = sample_kmer_features[sample_idx].get(kmer, 0)
    
    # Aggiungi prefisso "kmer_" ai nomi
    kmer_names = [f"kmer_{kmer}" for kmer in kmer_names]
    
    # Aggiungi Distance se presente
    if shap_values_distance is not None:
        aggregated_shap = np.hstack([aggregated_shap, shap_values_distance])
        aggregated_features = np.hstack([aggregated_features, X_distance])
        kmer_names.append("Distance")
        print(f"   📏 Distance feature preserved")
    
    print(f"   Found {len(kmer_names) - (1 if shap_values_distance is not None else 0)} unique k-mers")
    print(f"   Aggregated SHAP shape: {aggregated_shap.shape}")
    
    return aggregated_shap, aggregated_features, kmer_names


def generate_feature_names(encoding='OneHot', include_distance=True, n_features=None):
    """
    Genera i nomi delle feature basandosi sull'encoding definito in utilities.py.
    
    IMPORTANTE: Deve essere coerente con build_sequence_features() in utilities.py
    
    OneHot: 23 posizioni x 4 nucleotidi = 92 feature + distance (se include_distance=True)
    kmer: 21 k-mers x 3 posizioni = 63 feature + distance (se include_distance=True)
    LabelEncodingPairwise: 23 posizioni = 23 feature + distance
    OneHot5Channel: 23 posizioni x 5 canali = 115 feature + distance
    """
    feature_names = []
    nucleotides = ['A', 'C', 'G', 'T']
    
    if encoding == 'OneHot':
        # OneHot encoding: logical OR tra target e offtarget per ogni posizione e nucleotide
        # Result shape: (n_samples, 23, 4) -> flattened to (n_samples, 92)
        for pos in range(1, 24):
            for nuc in nucleotides:
                feature_names.append(f"Pos{pos}_{nuc}")
                
    elif encoding == 'kmer':
        # k-mer encoding con k=3
        # Genera (23-3+1) = 21 k-mers, ognuno confrontato position-wise = 21*3 = 63 feature
        # Nomi indicano la posizione nella sequenza (es. pos1-3 significa nucleotidi 1,2,3)
        k = 3
        n_kmers = 23 - k + 1  # 21 k-mers
        for kmer_idx in range(n_kmers):
            start_pos = kmer_idx + 1
            end_pos = kmer_idx + k
            for pos_in_kmer in range(k):
                abs_pos = start_pos + pos_in_kmer
                feature_names.append(f"kmer_pos{start_pos}-{end_pos}_nt{abs_pos}")
                
    elif encoding == 'LabelEncodingPairwise':
        # Label encoding: ogni posizione è mappata a un intero (0-15) che rappresenta la coppia di nucleotidi
        # 23 posizioni -> 23 feature
        for pos in range(1, 24):
            feature_names.append(f"Pos{pos}_pair")
            
    elif encoding == 'OneHot5Channel':
        # OneHot con canale extra per la direzione
        # 23 posizioni x (4 nucleotidi + 1 direzione) = 115 feature
        for pos in range(1, 24):
            for nuc in nucleotides:
                feature_names.append(f"Pos{pos}_{nuc}")
            feature_names.append(f"Pos{pos}_dir")
            
    elif encoding == 'OneHotVstack':
        # OneHot con concatenazione verticale (invece di OR)
        # 23 posizioni x 4 nucleotidi x 2 sequenze = 184 feature
        for pos in range(1, 24):
            for nuc in nucleotides:
                feature_names.append(f"Pos{pos}_target_{nuc}")
        for pos in range(1, 24):
            for nuc in nucleotides:
                feature_names.append(f"Pos{pos}_offtarget_{nuc}")
                
    elif encoding == 'NPM':
        # Nucleotide Position Mapping: 23 posizioni x 4x4 matrix = 368 feature
        for pos in range(1, 24):
            for nuc1 in nucleotides:
                for nuc2 in nucleotides:
                    feature_names.append(f"Pos{pos}_{nuc1}{nuc2}")
                    
    elif encoding == 'bulges':
        # Bulges encoding: NPM with bulge support using 4x5 matrix
        # 23 positions x 4x5 = 460 features
        nucleotides_with_bulge = ['A', 'C', 'G', 'T', '-']
        for pos in range(1, 24):
            for nuc1 in nucleotides:
                for nuc2 in nucleotides_with_bulge:
                    feature_names.append(f"Pos{pos}_{nuc1}{nuc2}")
                    
    else:
        # Fallback: genera nomi generici basandosi sul numero di feature effettive
        if n_features:
            n = n_features - 1 if include_distance else n_features
            feature_names = [f"Feature_{i}" for i in range(n)]
    
    # Aggiungi distance come ultima feature se richiesto
    if include_distance:
        feature_names.append("Distance")
    
    return feature_names


def run_shap_analysis(model_path, encoding='OneHot', include_distance=True, 
                      data_type='CHANGEseq', sample_size=1000, output_path=None,
                      aggregate_kmers=True, task_type="Regression"):
    """
    Esegue l'analisi SHAP sul modello specificato.
    
    Args:
        aggregate_kmers: Se True e encoding='kmer', aggrega i valori SHAP per k-mer specifico
        task_type: "Classification" o "Regression" per il titolo del plot
    """
    print("=" * 60)
    print("🚀 SHAP Analysis for XGBoost Model")
    print("=" * 60)
    
    # 1. Carica il modello (supporta sia .joblib che .json)
    print(f"\n📂 Loading model from: {model_path}")
    
    if model_path.endswith('.json'):
        # Carica modello XGBoost da JSON
        if 'classifier' in model_path.lower():
            model = xgb.XGBClassifier()
        else:
            model = xgb.XGBRegressor()
        model.load_model(model_path)
        print("✅ Model loaded from JSON successfully!")
    else:
        # Carica modello da joblib
        model = joblib.load(model_path)
        print("✅ Model loaded from joblib successfully!")
    
    # 2. Carica i dataset
    print(f"\n📊 Loading {data_type} dataset...")
    datasets_dir_path = general_utilities.DATASETS_PATH + 'include_on_targets/'
    
    positive_df = pd.read_csv(datasets_dir_path + f'{data_type}_positive.csv', index_col=0)
    negative_df = pd.read_csv(datasets_dir_path + f'{data_type}_negative.csv', index_col=0)
    
    # Rimuovi sequenze con 'N'
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find('N') == -1]
    
    print(f"   Positive samples: {len(positive_df)}")
    print(f"   Negative samples: {len(negative_df)}")
    
    # 3. Combina i dataset
    print("\n🔄 Combining datasets...")
    combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
    print(f"   Total samples: {len(combined_df)}")
    
    # 4. Campiona se necessario
    if sample_size and len(combined_df) > sample_size:
        print(f"\n🎲 Sampling {sample_size} samples for SHAP analysis...")
        combined_df = combined_df.sample(n=sample_size, random_state=42)
    
    # 5. Costruisci le feature
    print(f"\n🔧 Building features with {encoding} encoding...")
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    
    X = build_sequence_features(
        combined_df, 
        nucleotides_to_position_mapping,
        include_distance_feature=include_distance,
        include_sequence_features=True,
        encoding=encoding
    )
    print(f"   Feature matrix shape: {X.shape}")
    
    # 6. Calcola i valori SHAP con barra di progresso
    print("\n⏳ Computing SHAP values (this may take a while)...")
    
    # Crea l'explainer
    explainer = shap.TreeExplainer(model)
    
    # Calcola SHAP values con progress bar
    batch_size = 100
    n_samples = X.shape[0]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    shap_values_list = []
    
    for i in tqdm(range(n_batches), desc="Computing SHAP values", unit="batch"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n_samples)
        batch = X[start_idx:end_idx]
        batch_shap = explainer.shap_values(batch)
        shap_values_list.append(batch_shap)
    
    # Combina tutti i batch
    shap_values = np.vstack(shap_values_list)
    print(f"✅ SHAP values computed! Shape: {shap_values.shape}")
    
    # 7. Se encoding è kmer e aggregate_kmers=True, aggrega per k-mer specifico
    if encoding == 'kmer' and aggregate_kmers:
        shap_values, X, feature_names = aggregate_shap_by_kmer(
            shap_values, X, combined_df, k=3, include_distance=include_distance
        )
    else:
        # 7. Genera i nomi delle feature normalmente
        feature_names = generate_feature_names(encoding=encoding, include_distance=include_distance)
        print(f"\n📋 Number of features: {len(feature_names)}")
        
        # Verifica che il numero di feature corrisponda
        if len(feature_names) != X.shape[1]:
            print(f"⚠️ Warning: Feature names ({len(feature_names)}) don't match data ({X.shape[1]})")
            # Genera nomi generici
            feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    # 8. Genera il plot
    if output_path is None:
        output_path = f"shap_summary_{encoding}_{data_type}.png"
    
    plot_shap_summary(
        shap_values=shap_values,
        X_sample=X,
        feature_names=feature_names,
        output_path=output_path,
        max_display=20,
        encoding=encoding,
        include_distance=include_distance,
        task_type=task_type
    )
    
    return shap_values, X, feature_names


def run_shap_analysis_multi_fold(model_path_pattern, n_folds=10, encoding='OneHot', 
                                  include_distance=True, data_type='CHANGEseq', 
                                  sample_size=1000, output_path=None,
                                  aggregate_kmers=True, task_type="Regression"):
    """
    Esegue l'analisi SHAP su modelli multi-fold e aggrega i risultati.
    
    Args:
        model_path_pattern: Pattern del path con {fold} come placeholder (es. "path/model_fold_{fold}.json")
        n_folds: Numero di fold da analizzare
        Altri parametri: come run_shap_analysis
    """
    print("=" * 70)
    print(f"🚀 SHAP Analysis for Multi-Fold XGBoost Model ({n_folds} folds)")
    print("=" * 70)
    
    all_shap_values = []
    all_X = []
    feature_names = None
    
    # Processa ogni fold
    for fold_idx in range(n_folds):
        model_path = model_path_pattern.format(fold=fold_idx)
        
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found: {model_path} - Skipping fold {fold_idx}...")
            continue
            
        print(f"\n{'='*60}")
        print(f"📊 Processing Fold {fold_idx}/{n_folds-1}")
        print(f"{'='*60}")
        
        try:
            # Esegui analisi SHAP per questo fold (senza salvare il plot)
            shap_values, X, feat_names = run_shap_analysis(
                model_path=model_path,
                encoding=encoding,
                include_distance=include_distance,
                data_type=data_type,
                sample_size=sample_size,
                output_path=None,  # Non salvare plot individuali
                aggregate_kmers=aggregate_kmers,
                task_type=task_type
            )
            
            all_shap_values.append(shap_values)
            all_X.append(X)
            
            if feature_names is None:
                feature_names = feat_names
                
            print(f"✅ Fold {fold_idx} completed!")
            
        except Exception as e:
            print(f"❌ Error processing fold {fold_idx}: {e}")
            continue
    
    if len(all_shap_values) == 0:
        raise ValueError("No folds were successfully processed!")
    
    # Aggrega i risultati di tutti i fold
    print(f"\n{'='*60}")
    print(f"📊 Aggregating results from {len(all_shap_values)} folds...")
    print(f"{'='*60}")
    
    aggregated_shap_values = np.vstack(all_shap_values)
    aggregated_X = np.vstack(all_X)
    
    print(f"   Aggregated SHAP values shape: {aggregated_shap_values.shape}")
    print(f"   Aggregated features shape: {aggregated_X.shape}")
    
    # Genera il plot aggregato
    if output_path is None:
        output_path = f"shap_summary_{task_type.lower()}_{encoding}_{data_type}_multifold.png"
    
    plot_shap_summary(
        shap_values=aggregated_shap_values,
        X_sample=aggregated_X,
        feature_names=feature_names,
        output_path=output_path,
        max_display=20,
        encoding=encoding,
        include_distance=include_distance,
        task_type=task_type
    )
    
    return aggregated_shap_values, aggregated_X, feature_names


if __name__ == '__main__':
    
    # Configurazione per modelli multi-fold
    multifold_models_config = [
        # {
        #     "model_path_pattern": "files/models_10fold/CHANGEseq/include_on_targets/classifier/classifier_xgb_model_fold_{fold}_with_distance_imbalanced_with_OneHotEncoding.json",
        #     "encoding": "OneHot",
        #     "output_path": FIGURES_PATH + "shap_summary_classifier_OneHot_CHANGEseq_10fold.png",
        #     "task_type": "Classification",
        #     "aggregate_kmers": False,
        #     "n_folds": 10
        # },
        #  {
        #     "model_path_pattern":"files\models_10fold\CHANGEseq\include_on_targets\classifier\classifier_catboost_model_fold_{fold}_with_distance_imbalanced_with_kmerEncoding_tuned_early_stopping.json",
        #     "encoding": "kmer",
        #     "output_path": FIGURES_PATH + "shap_summary_classifier_kmer_CHANGEseq_10fold.png",
        #     "task_type": "Classification",
        #     "aggregate_kmers": True,
        #     "n_folds": 10
        # },
        # {
        #     "model_path_pattern": "files/models_10fold/CHANGEseq/include_on_targets/classifier/classifier_catboost_model_fold_{fold}_with_distance_imbalanced_with_bulgesEncoding_tuned_early_stopping.json",
        #     "encoding": "bulges",
        #     "output_path": FIGURES_PATH + "shap_summary_classifier_bulges_CHANGEseq_10fold.png",
        #     "task_type": "Classification",
        #     "aggregate_kmers": False,
        #     "n_folds": 10
        # },
        # {
        #     "model_path_pattern": "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives/regression_with_negatives_xgb_model_fold_{fold}_with_distance_imbalanced_with_OneHotEncoding.json",
        #     "encoding": "OneHot",
        #     "output_path": FIGURES_PATH + "shap_summary_regression_OneHot_CHANGEseq_10fold.png",
        #     "task_type": "Regression",
        #     "aggregate_kmers": False,
        #     "n_folds": 10
        # },
         {
            "model_path_pattern": "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives/regression_with_negatives_catboost_model_fold_{fold}_with_distance_imbalanced_with_kmerEncoding_tuned_early_stopping.json",
            "encoding": "kmer",
            "output_path": FIGURES_PATH + "shap_summary_regression_kmer_CHANGEseq_10fold.png",
            "task_type": "Regression",
            "aggregate_kmers": True,
            "n_folds": 10
        },
        # {
        #     "model_path_pattern": "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives/regression_with_negatives_catboost_model_fold_{fold}_with_distance_imbalanced_with_bulgesEncoding_tuned_early_stopping.json",
        #     "encoding": "bulges",
        #     "output_path": FIGURES_PATH + "shap_summary_regression_bulges_CHANGEseq_10fold.png",
        #     "task_type": "Regression",
        #     "aggregate_kmers": False,
        #     "n_folds": 10
        # }
    ]
    
    # Esegui analisi SHAP multi-fold
    print("\n" + "=" * 70)
    print("📊 MULTI-FOLD SHAP ANALYSIS")
    print("=" * 70)
    
    for config in multifold_models_config:
        print("\n" + "=" * 70)
        print(f"📊 Analyzing: {config['model_path_pattern']}")
        print("=" * 70)
        
        try:
            shap_values, X, feature_names = run_shap_analysis_multi_fold(
                model_path_pattern=config["model_path_pattern"],
                n_folds=config.get("n_folds", 10),
                encoding=config["encoding"],
                include_distance=True,
                data_type='CHANGEseq',
                sample_size=2000,
                output_path=config["output_path"],
                aggregate_kmers=config.get("aggregate_kmers", False),
                task_type=config.get("task_type", "Regression")
            )
            print(f"✅ Successfully generated: {config['output_path']}")
        except FileNotFoundError as e:
            print(f"⚠️ Models not found - Skipping...")
        except Exception as e:
            print(f"❌ Error analyzing models: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🎉 Analysis completed!")
    print("=" * 60)
