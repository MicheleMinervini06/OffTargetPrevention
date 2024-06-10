import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from PyALE import ale

from xgboost import XGBRegressor, XGBClassifier

from SysEvalOffTarget_src import utilities, general_utilities
from SysEvalOffTarget_src.test_utilities import load_model, load_fold_dataset
from SysEvalOffTarget_src.utilities import load_order_sg_rnas, order_sg_rnas, build_sequence_features, \
    create_nucleotides_to_position_mapping


def PDP(model_type="classifier", include_distance_feature=True, include_sequence_features=True, balanced=False,
        trans_type="ln_x_plus_one_trans",
        trans_all_fold=False, trans_only_positive=False, exclude_targets_without_positives=False, k_fold_number=10,
        gpu=True, encoding='NPM', data_type="CHANGEseq"):
    # Feature index for the PDP
    feature_index = 368
    all_pdp_results = []

    # Defining prefix to load the model
    path_prefix = 'CHANGEseq/include_on_targets/' + model_type + "/"

    # Loading the targets (i.e. sgRNAs)
    try:
        targets = load_order_sg_rnas()
    except FileNotFoundError:
        targets = order_sg_rnas()

    # Loading the dataset
    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    target_folds_list = np.array_split(targets, k_fold_number)

    for i, target_fold in enumerate(target_folds_list):
        # we don't exclude the targets without positives from the prediction stage.
        # if required, it is done in the evaluation stage
        negative_df_test, positive_df_test = load_fold_dataset(data_type, target_fold, targets, positive_df,
                                                               negative_df, balanced=False,
                                                               evaluate_only_distance=None,
                                                               exclude_targets_without_positives=False)
        model = load_model(model_type, k_fold_number, i, gpu, trans_type, balanced,
                           include_distance_feature, include_sequence_features, path_prefix,
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives, encoding=encoding)
        # predict and insert the predictions into the predictions dfs

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            sequence_features_test = build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features,
                                                             encoding=encoding)

            pdp_result = partial_dependence(model, sequence_features_test, [feature_index], grid_resolution=50)
            all_pdp_results.append(pdp_result['average'][0])

    # Ensure the loop was executed
    if all_pdp_results:

        # Determine the maximum size
        max_size = max(arr.shape[0] for arr in all_pdp_results)

        # Pad arrays with zeros to match the maximum size
        padded_arrays = []
        for arr in all_pdp_results:
            if arr.shape[0] < max_size:
                padding = (0, max_size - arr.shape[0])
                padded_arrays.append(np.pad(arr, padding, mode='constant'))
            else:
                padded_arrays.append(arr)

        # Stack the arrays into a 2D array
        stacked_arrays = np.vstack(padded_arrays)

        # Calculate the mean for each position, ignoring zeros
        def mean_ignore_zeros(arr):
            valid_elements = arr[arr != 0]
            if len(valid_elements) == 0:
                return 0
            return np.mean(valid_elements)

        # Compute the mean for each element position across all arrays
        average_pdp = np.apply_along_axis(mean_ignore_zeros, axis=0, arr=stacked_arrays)

        for i in range(len(average_pdp)):
            average_pdp[i] = np.expm1(average_pdp[i])

        grid = np.array(range(7))

        # Plot the averaged PDP
        plt.figure(figsize=(10, 6))
        plt.plot(grid, average_pdp, marker='o')
        plt.xlabel('Distance')
        plt.ylabel('Partial dependence')
        plt.title('Average Partial Dependence Plot for distance')
        plt.show()
    else:
        print("No models found. Please ensure the 'models' list is not empty.")


def ALE(model_type="classifier", include_distance_feature=True, include_sequence_features=True, balanced=False,
        trans_type="ln_x_plus_one_trans",
        trans_all_fold=False, trans_only_positive=False, exclude_targets_without_positives=False, k_fold_number=10,
        gpu=True, encoding='NPM', data_type="CHANGEseq"):
    # Feature index for the PDP
    feature_index = 368
    all_ale_results = []

    # Defining prefix to load the model
    path_prefix = 'CHANGEseq/include_on_targets/' + model_type + "/"

    # Loading the targets (i.e. sgRNAs)
    try:
        targets = load_order_sg_rnas()
    except FileNotFoundError:
        targets = order_sg_rnas()

    # Loading the dataset
    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    target_folds_list = np.array_split(targets, k_fold_number)

    for i, target_fold in enumerate(target_folds_list):
        # we don't exclude the targets without positives from the prediction stage.
        # if required, it is done in the evaluation stage
        negative_df_test, positive_df_test = load_fold_dataset(data_type, target_fold, targets, positive_df,
                                                               negative_df, balanced=False,
                                                               evaluate_only_distance=None,
                                                               exclude_targets_without_positives=False)
        model = load_model(model_type, k_fold_number, i, gpu, trans_type, balanced,
                           include_distance_feature, include_sequence_features, path_prefix,
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives, encoding=encoding)
        # predict and insert the predictions into the predictions dfs

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            sequence_features_test = build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features,
                                                             encoding=encoding)

            # 1D - continuous - no CI
            pdp_result = ale(X=pd.DataFrame(sequence_features_test), model=model, feature=[feature_index])
            all_ale_results.append(pdp_result['eff'])

    # Ensure the loop was executed
    if all_ale_results:

        # Determine the maximum size
        max_size = max(arr.shape[0] for arr in all_ale_results)

        # Pad arrays with zeros to match the maximum size
        padded_arrays = []
        for arr in all_ale_results:
            if arr.shape[0] < max_size:
                padding = (0, max_size - arr.shape[0])
                padded_arrays.append(np.pad(arr, padding, mode='constant'))
            else:
                padded_arrays.append(arr)

        # Stack the arrays into a 2D array
        stacked_arrays = np.vstack(padded_arrays)

        # Calculate the mean for each position, ignoring zeros
        def mean_ignore_zeros(arr):
            valid_elements = arr[arr != 0]
            if len(valid_elements) == 0:
                return 0
            return np.mean(valid_elements)

        # Compute the mean for each element position across all arrays
        average_pdp = np.apply_along_axis(mean_ignore_zeros, axis=0, arr=stacked_arrays)

        for i in range(len(average_pdp)):
            average_pdp[i] = np.expm1(average_pdp[i])

        grid = np.array(range(7))

        # Plot the averaged PDP
        plt.figure(figsize=(10, 6))
        plt.plot(grid, average_pdp, marker='o')
        plt.xlabel('Distance')
        plt.ylabel('Effect on prediction')
        plt.title('Accumulated Local Effects plot for distance')
        plt.show()
    else:
        print("No models found. Please ensure the 'models' list is not empty.")


if __name__ == '__main__':
    ALE(model_type="regression_with_negatives")
