import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from PyALE import ale
from sklearn.inspection import partial_dependence

from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.test_utilities import load_model, load_fold_dataset
from SysEvalOffTarget_src.utilities import load_order_sg_rnas, order_sg_rnas, build_sequence_features, \
    create_nucleotides_to_position_mapping

from deeplift.visualization import viz_sequence


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


def SHAP_explain(model_type="classifier", include_distance_feature=True, include_sequence_features=True, balanced=False,
                 trans_type="ln_x_plus_one_trans", trans_all_fold=False, trans_only_positive=False,
                 exclude_targets_without_positives=False, k_fold_number=10, gpu=True, encoding='NPM',
                 data_type="GUIDEseq", single_grna=None):
    # Feature index for the SHAP explanation
    all_shap_values = []
    all_data_samples = []

    # Feature names as the nucleotide position
    feature = [int(i / 23) for i in range(368)]
    # mapping
    feature_names = [str(i + 1) for i in feature]

    # # Definisco i nucleotidi possibili
    # nucleotides = ['A', 'T', 'C', 'G']
    #
    # # Creo tutte le combinazioni di dinucleotidi
    # nucleotide_combinations = [f'{n1}->{n2}' for n1 in nucleotides for n2 in nucleotides]
    #
    # # Ogni feature rappresenta una coppia di nucleotidi in una posizione del vettore
    # # Ad esempio: '1_A->A', '1_A->T', ..., '1_T->G', ..., '16_A->A', ecc.
    # feature_names = []
    # num_positions = 23
    # for i in range(num_positions):
    #     for nucleotide_pair in nucleotide_combinations:
    #         feature_names.append(f'{i + 1}_{nucleotide_pair}')
    feature_names.append("Distance")

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
    positive_df = pd.read_csv(datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
    negative_df = pd.read_csv(datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find('N') == -1]  # Remove invalid sequences

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    target_folds_list = np.array_split(targets, k_fold_number)

    for i, target_fold in enumerate(target_folds_list):
        if single_grna is None:
            negative_df_test, positive_df_test = load_fold_dataset(data_type, target_fold, targets, positive_df,
                                                                   negative_df, balanced=False,
                                                                   evaluate_only_distance=None,
                                                                   exclude_targets_without_positives=False)
        elif single_grna in target_fold:
            positive_df_test = positive_df[positive_df["target"] == single_grna]
            negative_df_test = negative_df[negative_df["target"] == single_grna]
        else:
            continue  # Skip if the target is not in the current fold

        model = load_model(model_type, k_fold_number, i, gpu, trans_type, balanced,
                           include_distance_feature, include_sequence_features, path_prefix,
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives, encoding=encoding)

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            # Build sequence features for the positive test dataset
            sequence_features_test = build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features,
                                                             encoding=encoding)

            shap_values, data_sample = compute_shap_values(model, sequence_features_test, 1000)
            all_shap_values.append(shap_values)
            all_data_samples.append(data_sample)

            dataset = "positive" if j == 0 else "negative"
            print(f"Computed SHAP values for {dataset} subset {i + 1}")
    # Combine the SHAP values and data samples
    combined_shap_values = np.vstack(all_shap_values)
    combined_data_samples = np.vstack(all_data_samples)

    # Plot summary plot for combined SHAP values
    shap.summary_plot(combined_shap_values, combined_data_samples, feature_names=feature_names)

    return combined_shap_values


def compute_shap_values(model, data, sample_size=100):
    """
    Compute SHAP values for a subset of the data.

    Parameters:
    - model: Trained XGBoost model.
    - data: numpy.ndarray containing the genomic data with 368 features.
    - sample_size: Number of samples to use for SHAP computation.

    Returns:
    - shap_values: SHAP values computed for the input data.
    - data_sample: The subset of the data used for SHAP computation.
    """
    # Use a subset of the data if it's too large
    # data_sample = sample_numpy_array(data, sample_size)

    # Create a SHAP explainer for the XGBoost model
    explainer = shap.TreeExplainer(model)

    # Compute SHAP values
    shap_values = explainer.shap_values(data)

    return shap_values, data


def sample_numpy_array(data, sample_size):
    """
    Sample a subset from a numpy.ndarray.

    Parameters:
    - data: numpy.ndarray, the original array to sample from.
    - sample_size: int, the number of samples to draw.

    Returns:
    - sampled_data: numpy.ndarray, the sampled subset of the original array.
    """
    # Ensure the sample size is not greater than the number of rows in the array
    if sample_size > data.shape[0]:
        raise ValueError("Sample size cannot be greater than the number of rows in the array.")

    # Generate random indices
    indices = np.random.choice(data.shape[0], size=sample_size, replace=False)

    # Select the subset using the indices
    sampled_data = data[indices, :]

    return sampled_data


def SHAP_explain_single_prediction(model_type="classifier", include_distance_feature=True,
                                   include_sequence_features=True, balanced=False,
                                   trans_type="ln_x_plus_one_trans", trans_all_fold=False, trans_only_positive=False,
                                   exclude_targets_without_positives=False, k_fold_number=10, gpu=True, encoding='NPM',
                                   data_type="GUIDEseq", single_grna=None):
    # Feature index for the SHAP explanation
    all_shap_values = []
    all_data_samples = []

    # # Feature names as the nucleotide position
    # feature = [int(i / 23) for i in range(368)]
    # # mapping
    # feature_names = [str(i + 1) for i in feature]
    # feature_names.append("Distance")

    # Definisco i nucleotidi possibili
    nucleotides = ['A', 'T', 'C', 'G']

    # Creo tutte le combinazioni di dinucleotidi
    nucleotide_combinations = [f'{n1}->{n2}' for n1 in nucleotides for n2 in nucleotides]

    # Ogni feature rappresenta una coppia di nucleotidi in una posizione del vettore
    # Ad esempio: '1_A->A', '1_A->T', ..., '1_T->G', ..., '16_A->A', ecc.
    feature_names = []
    num_positions = 23
    for i in range(num_positions):
        for nucleotide_pair in nucleotide_combinations:
            feature_names.append(f'{i + 1}_{nucleotide_pair}')
    feature_names.append("Distance")

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
    positive_df = pd.read_csv(datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
    negative_df = pd.read_csv(datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find('N') == -1]  # Remove invalid sequences

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()

    target_folds_list = np.array_split(targets, k_fold_number)

    for i, target_fold in enumerate(target_folds_list):
        if single_grna is None:
            negative_df_test, positive_df_test = load_fold_dataset(data_type, target_fold, targets, positive_df,
                                                                   negative_df, balanced=False,
                                                                   evaluate_only_distance=None,
                                                                   exclude_targets_without_positives=False)
        elif single_grna in target_fold:
            positive_df_test = positive_df[positive_df["target"] == single_grna]
            negative_df_test = negative_df[negative_df["target"] == single_grna]
        else:
            continue  # Skip if the target is not in the current fold

        model = load_model(model_type, k_fold_number, i, gpu, trans_type, balanced,
                           include_distance_feature, include_sequence_features, path_prefix,
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives, encoding=encoding)

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            # Build sequence features for the positive test dataset
            sequence_features_test = build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features,
                                                             encoding=encoding)

            # Convert sequence_features_test to a numpy array if it's a list
            if isinstance(sequence_features_test, list):
                sequence_features_test = np.array(sequence_features_test)

            # Instantiate the DeepExplainer
            explainer = shap.DeepExplainer(model, sequence_features_test)

            # Compute SHAP values using DeepExplainer
            shap_values = explainer.shap_values(sequence_features_test)
            data_sample = sequence_features_test  # Assuming sequence_features_test is the input sample

            all_shap_values.append(shap_values)
            all_data_samples.append(data_sample)

            dataset = "positive" if j == 0 else "negative"
            print(f"Computed SHAP values for {dataset} subset {i + 1}")

    # Combine the SHAP values and data samples
    combined_shap_values = np.vstack(all_shap_values)
    combined_data_samples = np.vstack(all_data_samples)

    # Plot summary plot for combined SHAP values
    shap.summary_plot(combined_shap_values, combined_data_samples, feature_names=feature_names)

    return combined_shap_values


if __name__ == '__main__':
    #SHAP_explain_single_prediction(model_type="regression_with_negatives")
    SHAP_explain(model_type="regression_with_negatives", single_grna="GGTGAGGGAGGAGAGATGCCNGG")
