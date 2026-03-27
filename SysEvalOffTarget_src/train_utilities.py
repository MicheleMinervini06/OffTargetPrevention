"""
    This module contains the function for training all the xgboost model variants
"""

import random
import time
from collections import defaultdict
from xml.parsers.expat import model
from idlelib.iomenu import encoding

from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve
import catboost as cb

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.utils import shuffle
import joblib

from SysEvalOffTarget_src.utilities import create_fold_sets, extract_model_path, \
    build_sequence_features, build_sampleweight, transformer_generator, transform
from SysEvalOffTarget_src import general_utilities
from plot_data import plot_execution_times, plot_boxplot_with_mean
from catboost import CatBoostClassifier, CatBoostRegressor
import hashlib
import pickle
import os

random.seed(general_utilities.SEED)


# Tuned Decision Tree baseline params obtained from random search on CHANGEseq regression_with_negatives (OneHot).
DT_TUNED_COMMON_PARAMS = {
    'splitter': 'random',
    'min_samples_split': 55,
    'min_samples_leaf': 17,
    'max_features': 'log2',
    'max_depth': 20,
    'ccp_alpha': 0.011,
    'random_state': 42,
}

DT_TUNED_REGRESSOR_PARAMS = {
    **DT_TUNED_COMMON_PARAMS,
    'criterion': 'absolute_error',
}

DT_TUNED_CLASSIFIER_PARAMS = {
    **DT_TUNED_COMMON_PARAMS,
    # Classifier cannot use "absolute_error"; keep the same structure and use a valid impurity criterion.
    'criterion': 'log_loss',
}


def data_preprocessing(positive_df, negative_df, trans_type, data_type, trans_all_fold, trans_only_positive):
    """
    data_preprocessing
    """
    data_type = "" if data_type is None else data_type + "_"
    reads_col = "{}reads".format(data_type)
    # it might include, but just confirm:
    positive_df.loc[:, "label"] = 1
    negative_df.loc[:, "label"] = 0
    negative_df.loc[:, reads_col] = 0

    positive_labels_df = positive_df[["target", "offtarget_sequence", "label", reads_col]]
    if trans_only_positive:
        labels_df = positive_labels_df
    else:
        negative_labels_df = negative_df[["target", "offtarget_sequence", "label", reads_col]]
        labels_df = pd.concat([positive_labels_df, negative_labels_df])

    if trans_all_fold:
        labels = labels_df[reads_col].values
        transformer = transformer_generator(labels, trans_type)
        labels_df[reads_col] = transform(labels, transformer)
    else:
        # preform the preprocessing on each sgRNA data individually
        for target in labels_df["target"].unique():
            target_df = labels_df[labels_df["target"] == target]
            target_labels = target_df[reads_col].values
            transformer = transformer_generator(target_labels, trans_type)
            # Change the DataFrame column type to float if necessary
            labels_df[reads_col] = labels_df[reads_col].astype(float)
            labels_df.loc[labels_df["target"] == target, reads_col] = transform(target_labels, transformer)

    if trans_only_positive:
        positive_df.loc[:, reads_col] = labels_df[reads_col].astype(float)
    else:
        positive_labels_df = labels_df[labels_df["label"] == 1]
        negative_labels_df = labels_df[labels_df["label"] == 0]
        positive_df[reads_col] = positive_labels_df[reads_col]
        negative_df[reads_col] = negative_labels_df[reads_col]

    return positive_df, negative_df

def plot_learning_curve(max_depth_range, train_sizes, train_errors, validation_errors):
    # Plot learning curves
    plt.figure(figsize=(12, 8))
    for i, max_depth in enumerate(max_depth_range):
        plt.plot(train_sizes, train_errors[i], label=f'Train Error (max_depth={max_depth})')
        plt.plot(train_sizes, validation_errors[i], label=f'Validation Error (max_depth={max_depth})', linestyle='--')

    plt.title("Learning Curves per diversi valori di max_depth")
    plt.xlabel("Dimensione del training set")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid()
    plt.show()


def generate_cache_key(positive_df, negative_df, nucleotides_to_position_mapping, 
                      include_distance_feature, include_sequence_features, encoding,
                      model_type, k_fold_number, targets, balanced, exclude_targets_without_positives):
    """
    Genera una chiave univoca per il caching basata sui parametri di input
    """
    # Crea un hash degli elementi che influenzano le feature
    key_elements = [
        str(sorted(targets)),
        str(include_distance_feature),
        str(include_sequence_features), 
        str(encoding),
        str(model_type),
        str(k_fold_number),
        str(balanced),
        str(exclude_targets_without_positives),
        str(len(positive_df)),
        str(len(negative_df))
    ]
    
    key_string = "_".join(key_elements)
    return hashlib.md5(key_string.encode()).hexdigest()


def get_cache_path(cache_key, encoding):
    """
    Restituisce il percorso del file cache
    """
    cache_dir = "cache/features"
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"features_{encoding}_{cache_key}.pkl")


def save_features_to_cache(features_data, cache_path):
    """
    Salva le feature pre-calcolate nel cache
    """
    print(f"💾 Salvando feature nel cache: {cache_path}")
    with open(cache_path, 'wb') as f:
        pickle.dump(features_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Cache salvato con successo!")


def load_features_from_cache(cache_path):
    """
    Carica le feature pre-calcolate dal cache
    """
    print(f"⚡ Caricando feature dal cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        features_data = pickle.load(f)
    print(f"✅ Cache caricato con successo!")
    return features_data


def precompute_all_features(positive_df, negative_df, targets, nucleotides_to_position_mapping,
                           include_distance_feature, include_sequence_features, encoding,
                           model_type, k_fold_number, balanced, exclude_targets_without_positives):
    """
    Pre-calcola tutte le feature per tutti i fold e le salva nel cache
    """
    print("🚀 Avvio pre-calcolo feature per tutti i fold...")
    start_time = time.time()
    
    target_folds_list = np.array_split(targets, k_fold_number) if k_fold_number > 1 else [[]]
    all_features = {}
    
    for i, target_fold in enumerate(target_folds_list):
        print(f"📊 Pre-calcolando feature per fold {i}/{len(target_folds_list)-1}")
        
        # Crea i set per questo fold
        negative_df_train, positive_df_train, _, _ = create_fold_sets(
            target_fold, targets, positive_df, negative_df, balanced,
            exclude_targets_without_positives)
        
        # Calcola feature positive
        positive_features = build_sequence_features(
            positive_df_train, nucleotides_to_position_mapping,
            include_distance_feature=include_distance_feature,
            include_sequence_features=include_sequence_features, 
            encoding=encoding)
        
        # Calcola feature negative se necessario
        negative_features = None
        if model_type in ("classifier", "regression_with_negatives"):
            negative_features = build_sequence_features(
                negative_df_train, nucleotides_to_position_mapping,
                include_distance_feature=include_distance_feature,
                include_sequence_features=include_sequence_features, 
                encoding=encoding)
        
        # Salva nel dizionario
        all_features[i] = {
            'positive_features': positive_features,
            'negative_features': negative_features,
            'positive_df': positive_df_train,
            'negative_df': negative_df_train
        }
    
    end_time = time.time()
    print(f"✅ Pre-calcolo completato in {end_time - start_time:.2f} secondi")
    
    return all_features


def clear_feature_cache(cache_dir="cache/features"):
    """
    Pulisce la cache delle feature (utile quando cambi i dati)
    """
    if os.path.exists(cache_dir):
        import shutil
        shutil.rmtree(cache_dir)
        print("🧹 Cache delle feature pulita!")
    else:
        print("📁 Nessuna cache da pulire.")


def train(positive_df, negative_df, targets, nucleotides_to_position_mapping,
          data_type='CHANGEseq', model_type="classifier", k_fold_number=10,
          include_distance_feature=False, include_sequence_features=True,
          balanced=False, trans_type="ln_x_plus_one_trans", trans_all_fold=False,
          trans_only_positive=False, exclude_targets_without_positives=False, skip_num_folds=0,
          path_prefix="", xgb_model=None, transfer_learning_type="add", save_model=False, n_trees=1000,
          encoding="NPM", use_xgboost=True, model_backend=None):
    """
    The train function
    """
    # To avoid a situation in which they are not defined (not really needed)
    global total_time
    total_time = 0
    models = []
    negative_sequence_features_train, sequence_labels_train = None, None\

    # Dictionary to store timing information for ea
    timing_info = defaultdict(float)
    fold_times = defaultdict(list)

    # Backward-compatible backend selection:
    # if model_backend is not provided, preserve old behavior driven by use_xgboost.
    if model_backend is None:
        model_backend = "xgboost" if use_xgboost else "decision_tree"
    if not isinstance(model_backend, str):
        raise ValueError("model_backend must be a string")
    model_backend = model_backend.lower()
    if model_backend == "xgb":
        model_backend = "xgboost"
    if model_backend not in ("xgboost", "catboost", "decision_tree"):
        raise ValueError("model_backend must be one of: 'xgboost', 'catboost', 'decision_tree'")
    if model_backend != "xgboost" and xgb_model is not None:
        raise ValueError("xgb_model transfer learning is currently supported only with model_backend='xgboost'")

    # set transfer_learning setting if needed
    # 'tree_method': 'gpu_hist' is deprecated, changed in 'device': 'cuda'
    if xgb_model is not None:
        # update the trees or train additional trees
        transfer_learning_args = {'process_type': 'update', 'updater': 'refresh'} \
            if transfer_learning_type == 'update' \
            else {'device': 'cuda'}
    else:
        transfer_learning_args = {'device': 'cuda'}

    # Assicurati che encoding sia sempre una lista
    if isinstance(encoding, str):
        encoding = [encoding]
    
    for enc in encoding:
        # 🚀 FEATURE CACHING: disabilitato temporaneamente
        # Per riabilitare: impostare USE_FEATURE_CACHE = True
        USE_FEATURE_CACHE = False

        if USE_FEATURE_CACHE:
            cache_key = generate_cache_key(
                positive_df, negative_df, nucleotides_to_position_mapping,
                include_distance_feature, include_sequence_features, enc,
                model_type, k_fold_number, targets, balanced, exclude_targets_without_positives
            )
            cache_path = get_cache_path(cache_key, enc)
            if os.path.exists(cache_path):
                print(f"⚡ CACHE HIT! Caricando feature pre-calcolate per encoding {enc}")
                cached_features = load_features_from_cache(cache_path)
            else:
                print(f"💾 CACHE MISS! Pre-calcolando feature per encoding {enc}")
                cached_features = precompute_all_features(
                    positive_df, negative_df, targets, nucleotides_to_position_mapping,
                    include_distance_feature, include_sequence_features, enc,
                    model_type, k_fold_number, balanced, exclude_targets_without_positives
                )
                save_features_to_cache(cached_features, cache_path)
        else:
            print(f"🔄 Cache disabilitato. Pre-calcolando feature per encoding {enc}")
            cached_features = precompute_all_features(
                positive_df, negative_df, targets, nucleotides_to_position_mapping,
                include_distance_feature, include_sequence_features, enc,
                model_type, k_fold_number, balanced, exclude_targets_without_positives
            )
        # model_type can get: 'classifier, regression_with_negatives, regression_without_negatives
        # in case we don't have k_fold, we train all the dataset with test set.
        target_folds_list = np.array_split(
            targets, k_fold_number) if k_fold_number > 1 else [[]]

        # Start global timing
        start_time = time.time()

        cat_feature_indices = []
        if enc == "CatBoost":
            cat_feature_indices = [i for i in range(0,23)]

        for i, target_fold in enumerate(target_folds_list[skip_num_folds:]):
            print(f"🏋️ Training fold {i + skip_num_folds} with encoding: {enc}")

            # Avvia il timer per il fold corrente e l'encoding specificato
            encoding_start_time = time.time()

            # 🚀 USA FEATURE CACHED invece di ricalcolarle
            fold_data = cached_features[i + skip_num_folds]
            positive_df_train = fold_data['positive_df']
            negative_df_train = fold_data['negative_df']
            positive_sequence_features_train = fold_data['positive_features']
            negative_sequence_features_train = fold_data['negative_features']
            
            # Costruisci le feature finali
            if model_type in ("classifier", "regression_with_negatives"):
                sequence_features_train = np.concatenate(
                    (negative_sequence_features_train, positive_sequence_features_train))
            elif model_type == 'regression_without_negatives':
                sequence_features_train = positive_sequence_features_train
            else:
                raise ValueError('model_type is invalid.')
            
            print(f"⚡ Feature caricate dal cache in {time.time() - encoding_start_time:.3f}s")

            # obtain classes
            negative_class_train = negative_df_train["label"].values
            positive_class_train = positive_df_train["label"].values
            sequence_class_train = \
                np.concatenate((negative_class_train, positive_class_train)) if \
                    model_type != "regression_without_negatives" else positive_class_train

            # obtain regression labels
            if model_type == "regression_with_negatives":
                positive_df_train, negative_df_train = \
                    data_preprocessing(positive_df_train, negative_df_train, trans_type=trans_type, data_type=data_type,
                                       trans_all_fold=trans_all_fold, trans_only_positive=trans_only_positive)
                negative_labels_train = negative_df_train[data_type +
                                                          "_reads"].values
                positive_labels_train = positive_df_train[data_type +
                                                          "_reads"].values
                sequence_labels_train = np.concatenate(
                    (negative_labels_train, positive_labels_train))
            elif model_type == "regression_without_negatives":
                positive_df_train, negative_df_train = \
                    data_preprocessing(positive_df_train, negative_df_train,
                                       trans_type=trans_type, data_type=data_type,
                                       trans_all_fold=trans_all_fold,
                                       trans_only_positive=True)
                sequence_labels_train = positive_df_train[data_type +
                                                          "_reads"].values

            if model_type == "classifier":
                sequence_class_train, sequence_features_train = shuffle(
                    sequence_class_train, sequence_features_train,
                    random_state=general_utilities.SEED)
            else:
                sequence_class_train, sequence_features_train, sequence_labels_train = shuffle(
                    sequence_class_train, sequence_features_train, sequence_labels_train,
                    random_state=general_utilities.SEED)

            negative_num = 0 if model_type == "regression_without_negatives" else len(
                negative_sequence_features_train)
            print("train fold ", i + skip_num_folds, " positive:",
                  len(positive_sequence_features_train), ", negative:", negative_num)

            if model_backend == "xgboost":
                if model_type == "classifier":
                    # model = xgb.XGBClassifier(max_depth=10,
                    #                           learning_rate=0.1,
                    #                           n_estimators=n_trees,
                    #                           nthread=55,
                    #                           **transfer_learning_args)

                    if enc == "OneHot":
                        # Hyperparameters for fine-tuned model using One Hot encoding with XGBoost
                        model = xgb.XGBClassifier(max_depth=12,
                                                 learning_rate=0.2,
                                                 n_estimators=1000,
                                                 nthread=55,
                                                 colsample_bytree=0.8,
                                                 subsample=1.0,
                                                 **transfer_learning_args)

                    if enc == "OneHot5Channel":
                        # Hyperparameters for fine-tuned model using One Hot 5 Channel encoding
                        model = xgb.XGBClassifier(max_depth=12,
                                                learning_rate=0.1,
                                                n_estimators=1000,
                                                nthread=55,
                                                colsample_bytree=1.0,
                                                subsample=1.0,
                                                **transfer_learning_args)

                    if enc == "kmer":
                        # Hyperparameters for fine-tuned model using Kmer encoding
                        model = xgb.XGBClassifier(max_depth=12,
                                                 learning_rate=0.2,
                                                 n_estimators=1000,
                                                 nthread=55,
                                                 colsample_bytree=0.8,
                                                 subsample=1.0,
                                                 **transfer_learning_args)

                    if enc == "LabelEncodingPairwise":
                        # Hyperparameters for fine-tuned model using Label Encoding Pairwise
                        model = xgb.XGBClassifier(max_depth=12,
                                                  learning_rate=0.2,
                                                  n_estimators=1000,
                                                  nthread=55,
                                                  colsample_bytree=0.8,
                                                  subsample=1.0,
                                                  **transfer_learning_args)    

                    else:
                        #Hyperparameters for fine-tuned model using NPM encoding
                        model = xgb.XGBClassifier(max_depth=12,
                                                learning_rate=0.1,
                                                n_estimators=500,
                                                nthread=55,
                                                colsample_bytree=1.0,
                                                subsample=1.0,
                                                **transfer_learning_args)


                    # # Hyperparameters for fine-tuned model using One Higher Depth (Same parameters for One Hot Encoding, One Hot 5 Channel Encoding, Label Encoding Pairwise)
                    # model = xgb.XGBClassifier(max_depth=17,
                    #                           learning_rate=0.2,
                    #                           n_estimators=2000,
                    #                           nthread=55,
                    #                           colsample_bytree=1.0,
                    #                           subsample=0.8,
                    #                           **transfer_learning_args)

                    start = time.time()
                    
                    # Crea validation set per early stopping (80% train, 20% val)
                    from sklearn.model_selection import train_test_split
                    X_train, X_val, y_train, y_val = train_test_split(
                        sequence_features_train, sequence_class_train, 
                        test_size=0.2, random_state=42, stratify=sequence_class_train
                    )
                    
                    # Crea sample weights per train e validation
                    train_weights = build_sampleweight(y_train)
                    
                    model.fit(X_train, y_train,
                              sample_weight=train_weights,
                              eval_set=[(X_val, y_val)],
                              verbose=100)
                    
                    end = time.time()
                    print("************** training time:", end - start, "**************")
                else:
                    # model = xgb.XGBRegressor(max_depth=10,
                    #                          learning_rate=0.1,
                    #                          n_estimators=n_trees,
                    #                          nthread=55,
                    #                          **transfer_learning_args)
        
                    if enc == "OneHot":
                        # Hyperparameters for fine-tuned model using One Hot encoding with XGBoost
                        model = xgb.XGBRegressor(max_depth=12,
                                                learning_rate=0.2,
                                                n_estimators=1000,
                                                nthread=55,
                                                colsample_bytree=0.8,
                                                subsample=1.0,
                                                **transfer_learning_args)
                        
                    if enc == "OneHot5Channel":
                        # Hyperparameters for fine-tuned model using One Hot 5 Channel encoding
                        model = xgb.XGBRegressor(max_depth=12,
                                                    learning_rate=0.1,
                                                    n_estimators=1000,
                                                    nthread=55,
                                                    colsample_bytree=1.0,
                                                    subsample=1.0,
                                                    **transfer_learning_args)
                    if enc == "kmer":
                        # Hyperparameters for fine-tuned model using Kmer encoding
                        model = xgb.XGBRegressor(max_depth=12,
                                                 learning_rate=0.2,
                                                 n_estimators=1000,
                                                 nthread=55,
                                                 colsample_bytree=0.8,
                                                 subsample=1.0,
                                                 **transfer_learning_args)

                    if enc == "LabelEncodingPairwise":
                        # Hyperparameters for fine-tuned model using Label Encoding Pairwise
                        model = xgb.XGBRegressor(max_depth=12,
                                                 learning_rate=0.2,
                                                 n_estimators=1000,
                                                 nthread=55,
                                                 colsample_bytree=0.8,
                                                 subsample=1.0,
                                                 **transfer_learning_args)

                    else:    
                        # Hyperparameters for fine tuned model using NPM encoding
                        model = xgb.XGBRegressor(max_depth=12,
                                                learning_rate=0.1,
                                                n_estimators=500,
                                                nthread=55,
                                                colsample_bytree=1.0,
                                                subsample=1.0,
                                                **transfer_learning_args)
                        
                    # # Hyperparameters for fine-tuned model using One Higher Depth (Same parameters for One Hot Encoding, One Hot 5 Channel Encoding, Label Encoding Pairwise)
                    # model = xgb.XGBRegressor(max_depth=17,
                    #                          learning_rate=0.2,
                    #                          n_estimators=2000,
                    #                          nthread=55,
                    #                          colsample_bytree=1.0,
                    #                          subsample=0.8,
                    #                          **transfer_learning_args)


                    start = time.time()

                    # # Imposta il range di max_depth
                    # max_depth_range = [3, 5, 7, 10, 12, 15, 20]
                    # train_sizes = np.linspace(0.1, 1.0, 5)  # Diverse dimensioni del training set
                    #
                    # # Inizializza una lista per salvare gli errori
                    # train_errors, validation_errors = [], []
                    #
                    # for max_depth in max_depth_range:
                    #     model = xgb.XGBRegressor(nthread=55,
                    #                             max_depth=max_depth,
                    #                             learning_rate=0.2,
                    #                             n_estimators=1000,
                    #                             colsample_bytree=0.8,
                    #                             subsample=1.0,
                    #                             **transfer_learning_args)
                    #
                    #     # Calcola le learning curves
                    #     train_sizes, train_scores, validation_scores = learning_curve(
                    #         model,
                    #         sequence_features_train,
                    #         sequence_labels_train,
                    #         train_sizes=train_sizes,
                    #         cv=10,
                    #         scoring='neg_mean_absolute_error',
                    #         n_jobs=-1  # Usa tutti i thread disponibili
                    #     )
                    #
                    # # Calcola la media e la deviazione standard dei punteggi
                    # train_errors.append(-train_scores.mean(axis=1))  # Trasformiamo in positivo
                    # validation_errors.append(-validation_scores.mean(axis=1))
                    #
                    # # Plot learning curves
                    # plot_learning_curve(max_depth_range, train_sizes, train_errors, validation_errors)


                    # # Transfer data to GPU using Cupy
                    # def to_gpu_array(x):
                    #     return cp.asarray(x)
                    #
                    # # Use a specific GPU device
                    # gpu_id = 0
                    # cp.cuda.Device(gpu_id).use()
                    #
                    # # Convert training data to GPU arrays
                    # sequence_features_train_gpu = to_gpu_array(sequence_features_train)
                    # sequence_labels_train_gpu = to_gpu_array(sequence_labels_train)

                    # Crea validation set per early stopping (80% train, 20% val)
                    from sklearn.model_selection import train_test_split
                    if model_type == "regression_with_negatives":
                        X_train, X_val, y_train, y_val, class_train, class_val = train_test_split(
                            sequence_features_train, sequence_labels_train, sequence_class_train,
                            test_size=0.2, random_state=42
                        )
                        train_weights = build_sampleweight(class_train)
                        model.fit(X_train, y_train,
                                  sample_weight=train_weights,
                                  eval_set=[(X_val, y_val)],
                                  verbose=100)  # Mostra solo ogni 100 iterazioni
                    else:
                        X_train, X_val, y_train, y_val = train_test_split(
                            sequence_features_train, sequence_labels_train,
                            test_size=0.2, random_state=42
                        )
                        model.fit(X_train, y_train,
                                  eval_set=[(X_val, y_val)],
                                  verbose=100)  # Mostra solo ogni 100 iterazioni
                    end = time.time()


                    # # Ensure you have CUDA enabled XGBoost and Cupy installed
                    
                    # # Define the parameter grid or distribution
                    # param_dist = {
                    #     #'max_depth': [5, 7, 10, 12],
                    #     'max_depth': [10, 12, 15, 17, 20],
                    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #     'n_estimators': [100, 500, 1000, 2000],
                    #     'subsample': [0.6, 0.8, 1.0],
                    #     'colsample_bytree': [0.6, 0.8, 1.0]
                    # }
                    
                    # # Transfer data to GPU using Cupy
                    # def to_gpu_array(x):
                    #     return cp.asarray(x)
                    
                    # # Use a specific GPU device
                    # gpu_id = 0
                    # cp.cuda.Device(gpu_id).use()
                    
                    # # # Convert training data to GPU arrays
                    # # sequence_features_train_gpu = to_gpu_array(sequence_features_train)
                    # # sequence_labels_train_gpu = to_gpu_array(sequence_labels_train)
                    
                    # # Initialize the regressor with GPU settings
                    # model = xgb.XGBRegressor(nthread=55, **transfer_learning_args)
                    
                    # # Setup randomized search with MAE as the scoring metric
                    # random_search = RandomizedSearchCV(model,param_distributions=param_dist, n_iter=20, scoring='neg_mean_absolute_error', cv=10, verbose=2,random_state=42)
                    
                    # # Fit the randomized search model
                    # start = time.time()
                    # if model_type == "regression_with_negatives":
                    #     random_search.fit(sequence_features_train, sequence_labels_train,
                    #                       sample_weight=build_sampleweight(sequence_class_train))
                    # else:
                    #     random_search.fit(sequence_features_train, sequence_labels_train)
                    # end = time.time()
                    
                    # print("************** training time:", end - start, "**************")
                    
                    # # Save the best model
                    # joblib.dump(random_search.best_estimator_, f"best_xgb_regressor_change_{enc}.joblib")
                    
                    # # Save the best hyperparameters
                    # best_params = random_search.best_params_
                    # joblib.dump(best_params, f"best_hyperparameters_regressor_change_{enc}.joblib")
                    
                    # print("Best parameters found: ", best_params)
                    # print("Best model saved to disk.")

                    # param_dist = {
                    #     'depth': [6, 8, 10, 12],
                    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #     'iterations': [100, 500, 1000, 2000],
                    #     'l2_leaf_reg': [1, 3, 5, 7, 9],
                        
                    #     # Includi bootstrap_type e subsample nella ricerca
                    #     'bootstrap_type': ['Bernoulli', 'MVS', 'Poisson'], # Tipi compatibili con subsample
                    #     'subsample': [0.6, 0.8, 1.0]
                    # }

                    # model = cb.CatBoostRegressor(
                    #     task_type="GPU",
                    #     devices='0',
                    #     verbose=0
                    # )

                    # # Imposta la ricerca randomizzata con MAE come metrica di valutazione
                    # random_search = RandomizedSearchCV(
                    #     model,
                    #     param_distributions=param_dist,
                    #     n_iter=40,
                    #     scoring='neg_mean_absolute_error',
                    #     cv=5,
                    #     verbose=2,
                    #     random_state=42
                    # )

                    # # Avvia l'addestramento del modello di ricerca randomizzata
                    # start = time.time()
                    # if model_type == "regression_with_negatives":
                    #     random_search.fit(sequence_features_train, sequence_labels_train,
                    #                     sample_weight=build_sampleweight(sequence_class_train))
                    # else:
                    #     random_search.fit(sequence_features_train, sequence_labels_train)
                    # end = time.time()

                    # print("************** tempo di addestramento:", end - start, "**************")

                    # # Salva il miglior modello
                    # # CatBoost ha i suoi metodi nativi per salvare i modelli, che sono spesso preferibili.
                    # # Usare joblib potrebbe non essere sempre compatibile tra versioni.
                    # # Metodo nativo di CatBoost:
                    # random_search.best_estimator_.save_model(f"best_catboost_regressor_change_{enc}.cbm")

                    # # Salva i migliori iperparametri
                    # best_params = random_search.best_params_
                    # joblib.dump(best_params, f"best_hyperparameters_regressor_change_{enc}.joblib")

                    # print("Migliori parametri trovati: ", best_params)
                    # print("Miglior modello salvato su disco.")

            elif model_backend == "catboost":
                start = time.time()
                from sklearn.model_selection import train_test_split

                use_gpu = transfer_learning_args.get('device') == 'cuda'
                catboost_common_args = {
                    'depth': 12,
                    'learning_rate': 0.05,
                    'iterations': 3000,
                    'l2_leaf_reg': 9,
                    'subsample': 0.8,
                    'bootstrap_type': 'Bernoulli',
                    'thread_count': -1,
                    'verbose': 100,
                    'random_seed': 42,
                    'task_type': 'GPU' if use_gpu else 'CPU'
                }

                # Only categorical for the dedicated CatBoost encoding; other encodings are numeric.
                cat_feature_indices = [i for i in range(0, 23)] if enc == "CatBoost" and include_sequence_features else []

                if model_type == "classifier":
                    model = CatBoostClassifier(cat_features=cat_feature_indices, **catboost_common_args)
                    X_train, X_val, y_train, y_val = train_test_split(
                        sequence_features_train, sequence_class_train,
                        test_size=0.2, random_state=42, stratify=sequence_class_train
                    )
                    train_weights = build_sampleweight(y_train)
                    model.fit(
                        X_train, y_train,
                        sample_weight=train_weights,
                        eval_set=(X_val, y_val),
                        use_best_model=True,
                        early_stopping_rounds=120,
                        verbose=100
                    )
                else:
                    model = CatBoostRegressor(cat_features=cat_feature_indices, **catboost_common_args)
                    if model_type == "regression_with_negatives":
                        X_train, X_val, y_train, y_val, class_train, class_val = train_test_split(
                            sequence_features_train, sequence_labels_train, sequence_class_train,
                            test_size=0.2, random_state=42
                        )
                        train_weights = build_sampleweight(class_train)
                        model.fit(
                            X_train, y_train,
                            sample_weight=train_weights,
                            eval_set=(X_val, y_val),
                            use_best_model=True,
                            early_stopping_rounds=120,
                            verbose=100
                        )
                    else:
                        X_train, X_val, y_train, y_val = train_test_split(
                            sequence_features_train, sequence_labels_train,
                            test_size=0.2, random_state=42
                        )
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_val, y_val),
                            use_best_model=True,
                            early_stopping_rounds=120,
                            verbose=100
                        )

                end = time.time()
                print("************** training time:", end - start, "**************")
            else:
                if model_type == "classifier":
                    model = DecisionTreeClassifier(**DT_TUNED_CLASSIFIER_PARAMS)
                else:
                    model = DecisionTreeRegressor(**DT_TUNED_REGRESSOR_PARAMS)

                # # Define the parameter grid for Decision Tree
                # param_dist = {
                #     'max_depth': [3, 5, 10, 20, 40, 50, None],  # Profondità dell'albero
                #     'min_samples_split': [2, 5, 10, 20, 30],  # Campioni minimi per split
                #     'min_samples_leaf': [1, 2, 5, 10, 20],  # Campioni minimi per foglia
                #     'max_features': ['sqrt', 'log2', None, 0.8, 0.9]  # Caratteristiche per split
                # }
                #
                # # Setup randomized search random_search = RandomizedSearchCV(model, param_distributions=param_dist,
                # n_iter=25, scoring='neg_mean_absolute_error' if model_type != "classifier" else 'accuracy', cv=3,
                # verbose=2, random_state=42)

                start = time.time()
                if model_type == "classifier":
                    train_weights = build_sampleweight(sequence_class_train)
                    model.fit(sequence_features_train, sequence_class_train,
                              sample_weight=train_weights)
                elif model_type == "regression_with_negatives":
                    train_weights = build_sampleweight(sequence_class_train)
                    model.fit(sequence_features_train, sequence_labels_train,
                              sample_weight=train_weights)
                else:
                    model.fit(sequence_features_train, sequence_labels_train)
                end = time.time()

                # model = random_search.best_estimator_
                #
                # # Save the best model
                # joblib.dump(random_search.best_estimator_, 'best_decision_tree_regressor_change.joblib')
                #
                # # Save the best hyperparameters
                # best_params = random_search.best_params_
                # joblib.dump(best_params, 'best_hyperparameters_decision_tree_regressor_change.joblib')
                #
                # print("Best parameters found: ", best_params)
                # print("Best model saved to disk.")

            # Registra il tempo totale per l'encoding corrente
            encoding_end_time = time.time()
            fold_time = encoding_end_time - encoding_start_time
            timing_info[enc] += fold_time
            fold_times[enc].append(fold_time)

            # Statistiche Early Stopping (solo per CatBoost)
            if isinstance(model, (CatBoostClassifier, CatBoostRegressor)):
                best_iter = model.get_best_iteration()
                best_score = model.get_best_score()
                
                print(f"🎯 Iterazioni usate: {best_iter if best_iter is not None else 'N/A'}")
                print(f"📊 Score migliore: {best_score}")
                
                # Calcola percentuale di riduzione iterazioni
                if best_iter is not None:
                    reduction = ((3000 - best_iter) / 3000) * 100
                    print(f"⚡ Riduzione iterazioni: {reduction:.1f}% ({3000 - best_iter} iterazioni risparmiate)")
            else:
                print("No early stopping")


            if save_model:
                dir_path = extract_model_path(model_type, k_fold_number, include_distance_feature,
                                              include_sequence_features, balanced, trans_type, trans_all_fold,
                                              trans_only_positive, exclude_targets_without_positives,
                                              i + skip_num_folds, path_prefix, enc,
                                              model_backend=model_backend)

                Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
                print(dir_path)

                if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
                    joblib.dump(model, dir_path)
                else:
                    model.save_model(dir_path)
            models.append(model)


        # End timing
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60
        print(f"Training completed in {elapsed_time:.2f} minutes.")
        print(f"Mean training time: {total_time / 10:.2f} seconds.")

    # Stampa i tempi per ciascun encoding
    print("\nExecution times per encoding:")
    for enc, time_taken in timing_info.items():
        print(f"Encoding {enc}: {time_taken:.2f} seconds")

    # # Genera i grafici alla fine dell'addestramento
    # plot_execution_times(timing_info)
    # plot_boxplot_with_mean(fold_times)

    if k_fold_number == 1:
        return models[0]
    else:
        return models