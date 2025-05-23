"""
    This module contains the function for training all the xgboost model variants
"""

import random
import time
from collections import defaultdict
from idlelib.iomenu import encoding

from pathlib import Path
import xgboost as xgb
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import learning_curve

import cupy as cp
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.utils import shuffle
import joblib

from SysEvalOffTarget_src.utilities import create_fold_sets, extract_model_path, \
    build_sequence_features, build_sampleweight, transformer_generator, transform
from SysEvalOffTarget_src import general_utilities
from plot_data import plot_execution_times, plot_boxplot_with_mean

random.seed(general_utilities.SEED)


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


def train(positive_df, negative_df, targets, nucleotides_to_position_mapping,
          data_type='CHANGEseq', model_type="classifier", k_fold_number=10,
          include_distance_feature=False, include_sequence_features=True,
          balanced=False, trans_type="ln_x_plus_one_trans", trans_all_fold=False,
          trans_only_positive=False, exclude_targets_without_positives=False, skip_num_folds=0,
          path_prefix="", xgb_model=None, transfer_learning_type="add", save_model=False, n_trees=1000,
          encoding="NPM", use_xgboost=True):
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

    # set transfer_learning setting if needed
    # 'tree_method': 'gpu_hist' is deprecated, changed in 'device': 'cuda'
    if xgb_model is not None:
        # update the trees or train additional trees
        transfer_learning_args = {'process_type': 'update', 'updater': 'refresh'} \
            if transfer_learning_type == 'update' \
            else {'device': 'cuda'}
    else:
        transfer_learning_args = {'device': 'cuda'}

    for encoding in encoding:
        # model_type can get: 'classifier, regression_with_negatives, regression_without_negatives
        # in case we don't have k_fold, we train all the dataset with test set.
        target_folds_list = np.array_split(
            targets, k_fold_number) if k_fold_number > 1 else [[]]

        # Start global timing
        start_time = time.time()

        # Start timing
        start_time = time.time()
        for i, target_fold in enumerate(target_folds_list[skip_num_folds:]):
            print(f"Training fold {i + skip_num_folds} with encoding: {encoding}")

            # Avvia il timer per il fold corrente e l'encoding specificato
            encoding_start_time = time.time()

            negative_df_train, positive_df_train, _, _ = create_fold_sets(
                target_fold, targets, positive_df, negative_df, balanced,
                exclude_targets_without_positives)
            # build features
            positive_sequence_features_train = build_sequence_features(
                positive_df_train, nucleotides_to_position_mapping,
                include_distance_feature=include_distance_feature,
                include_sequence_features=include_sequence_features, encoding=encoding)
            if model_type in ("classifier", "regression_with_negatives"):
                negative_sequence_features_train = build_sequence_features(
                    negative_df_train, nucleotides_to_position_mapping,
                    include_distance_feature=include_distance_feature,
                    include_sequence_features=include_sequence_features, encoding=encoding)
                sequence_features_train = np.concatenate(
                    (negative_sequence_features_train, positive_sequence_features_train))
            elif model_type == 'regression_without_negatives':
                sequence_features_train = positive_sequence_features_train
            else:
                raise ValueError('model_type is invalid.')

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

            if use_xgboost:
                if model_type == "classifier":
                    # model = xgb.XGBClassifier(max_depth=10,
                    #                           learning_rate=0.1,
                    #                           n_estimators=n_trees,
                    #                           nthread=55,
                    #                           **transfer_learning_args)

                    # Hyperparameters for fine-tuned model using NPM encoding
                    # model = xgb.XGBClassifier(max_depth=12,
                    #                           learning_rate=0.1,
                    #                           n_estimators=500,
                    #                           nthread=55,
                    #                           colsample_bytree=1.0,
                    #                           subsample=1.0,
                    #                           **transfer_learning_args)

                    # Hyperparameters for fine-tuned model using One Hot encoding
                    model = xgb.XGBClassifier(max_depth=12,
                                             learning_rate=0.2,
                                             n_estimators=1000,
                                             nthread=55,
                                             colsample_bytree=0.8,
                                             subsample=1.0,
                                             **transfer_learning_args)

                    # # Hyperparameters for fine-tuned model using One Hot 5 Channel encoding
                    # model = xgb.XGBClassifier(max_depth=12,
                    #                          learning_rate=0.1,
                    #                          n_estimators=1000,
                    #                          nthread=55,
                    #                          colsample_bytree=1.0,
                    #                          subsample=1.0,
                    #                          **transfer_learning_args)

                    # # Hyperparameters for fine-tuned model using Kmer encoding
                    # model = xgb.XGBClassifier(max_depth=12,
                    #                          learning_rate=0.2,
                    #                          n_estimators=1000,
                    #                          nthread=55,
                    #                          colsample_bytree=0.8,
                    #                          subsample=1.0,
                    #                          **transfer_learning_args)

                    # # Hyperparameters for fine-tuned model using Label Encoding Pairwise
                    # model = xgb.XGBClassifier(max_depth=12,
                    #                           learning_rate=0.2,
                    #                           n_estimators=1000,
                    #                           nthread=55,
                    #                           colsample_bytree=0.8,
                    #                           subsample=1.0,
                    #                           **transfer_learning_args)
                    #

                    # # Hyperparameters for fine-tuned model using One Higher Depth (Same parameters for One Hot Encoding, One Hot 5 Channel Encoding, Label Encoding Pairwise)
                    # model = xgb.XGBClassifier(max_depth=17,
                    #                           learning_rate=0.2,
                    #                           n_estimators=2000,
                    #                           nthread=55,
                    #                           colsample_bytree=1.0,
                    #                           subsample=0.8,
                    #                           **transfer_learning_args)

                    start = time.time()
                    model.fit(sequence_features_train, sequence_class_train,
                              sample_weight=build_sampleweight(sequence_class_train), xgb_model=xgb_model)
                    end = time.time()
                    print("************** training time:", end - start, "**************")
                else:
                    # model = xgb.XGBRegressor(max_depth=10,
                    #                          learning_rate=0.1,
                    #                          n_estimators=n_trees,
                    #                          nthread=55,
                    #                          **transfer_learning_args)

                    # # Hyperparameters for fine tuned model using NPM encoding
                    # model = xgb.XGBRegressor(max_depth=12,
                    #                          learning_rate=0.1,
                    #                          n_estimators=500,
                    #                          nthread=55,
                    #                          colsample_bytree=1.0,
                    #                          subsample=1.0,
                    #                          **transfer_learning_args)

                    # Hyperparameters for fine-tuned model using One Hot encoding
                    model = xgb.XGBRegressor(max_depth=12,
                                             learning_rate=0.2,
                                             n_estimators=1000,
                                             nthread=55,
                                             colsample_bytree=0.8,
                                             subsample=1.0,
                                             **transfer_learning_args)

                    # # Hyperparameters for fine-tuned model using One Hot 5 Channel encoding
                    # model = xgb.XGBRegressor(max_depth=12,
                    #                             learning_rate=0.1,
                    #                             n_estimators=1000,
                    #                             nthread=55,
                    #                             colsample_bytree=1.0,
                    #                             subsample=1.0,
                    #                             **transfer_learning_args)

                    # # Hyperparameters for fine-tuned model using Kmer encoding
                    # model = xgb.XGBRegressor(max_depth=12,
                    #                          learning_rate=0.2,
                    #                          n_estimators=1000,
                    #                          nthread=55,
                    #                          colsample_bytree=0.8,
                    #                          subsample=1.0,
                    #                          **transfer_learning_args)

                    # # Hyperparameters for fine-tuned model using Label Encoding Pairwise
                    # model = xgb.XGBRegressor(max_depth=12,
                    #                          learning_rate=0.2,
                    #                          n_estimators=1000,
                    #                          nthread=55,
                    #                          colsample_bytree=0.8,
                    #                          subsample=1.0,
                    #                          **transfer_learning_args)

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

                    if model_type == "regression_with_negatives":
                        model.fit(sequence_features_train, sequence_labels_train,
                                  sample_weight=build_sampleweight(sequence_class_train),
                                  xgb_model=xgb_model)
                    else:
                        model.fit(sequence_features_train,
                                  sequence_labels_train, xgb_model=xgb_model)
                    end = time.time()


                    # # Ensure you have CUDA enabled XGBoost and Cupy installed
                    #
                    # # Define the parameter grid or distribution
                    # param_dist = {
                    #     #'max_depth': [5, 7, 10, 12],
                    #     'max_depth': [10, 12, 15, 17, 20],
                    #     'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    #     'n_estimators': [100, 500, 1000, 2000],
                    #     'subsample': [0.6, 0.8, 1.0],
                    #     'colsample_bytree': [0.6, 0.8, 1.0]
                    # }
                    #
                    # # Transfer data to GPU using Cupy
                    # def to_gpu_array(x):
                    #     return cp.asarray(x)
                    #
                    # # Use a specific GPU device
                    # gpu_id = 0
                    # cp.cuda.Device(gpu_id).use()
                    #
                    # # # Convert training data to GPU arrays
                    # # sequence_features_train_gpu = to_gpu_array(sequence_features_train)
                    # # sequence_labels_train_gpu = to_gpu_array(sequence_labels_train)
                    #
                    # # Initialize the regressor with GPU settings
                    # model = xgb.XGBRegressor(nthread=55, **transfer_learning_args)
                    #
                    # # Setup randomized search with MAE as the scoring metric
                    # random_search = RandomizedSearchCV(model,param_distributions=param_dist, n_iter=20, scoring='neg_mean_absolute_error', cv=10, verbose=2,random_state=42)
                    #
                    # # Fit the randomized search model
                    # start = time.time()
                    # if model_type == "regression_with_negatives":
                    #     random_search.fit(sequence_features_train, sequence_labels_train,
                    #                       sample_weight=build_sampleweight(sequence_class_train))
                    # else:
                    #     random_search.fit(sequence_features_train, sequence_labels_train)
                    # end = time.time()
                    #
                    # print("************** training time:", end - start, "**************")
                    #
                    # # Save the best model
                    # joblib.dump(random_search.best_estimator_, f"best_xgb_regressor_change_{encoding}.joblib")
                    #
                    # # Save the best hyperparameters
                    # best_params = random_search.best_params_
                    # joblib.dump(best_params, f"best_hyperparameters_regressor_change_{encoding}.joblib")
                    #
                    # print("Best parameters found: ", best_params)
                    # print("Best model saved to disk.")
            else:
                if model_type == "classifier":
                    model = DecisionTreeClassifier(max_depth=None, min_samples_split=5, min_samples_leaf=2,
                                                   max_features=None, random_state=42)
                else:
                    model = DecisionTreeRegressor(max_depth=None, min_samples_split=5, min_samples_leaf=2,
                                                  max_features=None, random_state=42)

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
                    # random_search.fit(sequence_features_train, sequence_class_train)
                    model.fit(sequence_features_train, sequence_class_train)
                else:
                    # random_search.fit(sequence_features_train, sequence_labels_train)
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
            timing_info[encoding] += fold_time
            fold_times[encoding].append(fold_time)

            if save_model:
                if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
                    print("The model is a Decision Tree.")
                    # Save the model
                    if model_type == "classifier":
                        joblib.dump(model, f'decision_tree_{i}_classifier.pkl')
                    elif model_type == "regression_with_negatives":
                        joblib.dump(model, f'decision_tree_{i}_regression_with_negatives.pkl')
                    else:
                        joblib.dump(model, f'decision_tree_{i}_regression_without_negatives.pkl')
                else:
                    dir_path = extract_model_path(model_type, k_fold_number, include_distance_feature,
                                                  include_sequence_features, balanced, trans_type, trans_all_fold,
                                                  trans_only_positive, exclude_targets_without_positives,
                                                  i + skip_num_folds, path_prefix, encoding)

                    #dir_path = dir_path.replace(".json", "_tunedHigherDepth.json")

                    Path(dir_path).parent.mkdir(parents=True, exist_ok=True)
                    # Changed format to avoid warning
                    print(dir_path)
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

    # Genera i grafici alla fine dell'addestramento
    plot_execution_times(timing_info)
    plot_boxplot_with_mean(fold_times)

    if k_fold_number == 1:
        return models[0]
    else:
        return models