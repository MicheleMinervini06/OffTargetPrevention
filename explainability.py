import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import time
#from PyALE import ale
from sklearn.inspection import partial_dependence

from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.test_utilities import load_model, load_fold_dataset
from SysEvalOffTarget_src.utilities import load_order_sg_rnas, order_sg_rnas, build_sequence_features, \
    create_nucleotides_to_position_mapping

#from deeplift.visualization import viz_sequence


def PDP(model_type="classifier", include_distance_feature=True, include_sequence_features=True, balanced=False,
        trans_type="ln_x_plus_one_trans",
        trans_all_fold=False, trans_only_positive=False, exclude_targets_without_positives=False, k_fold_number=10,
    gpu=True, encoding='NPM', data_type="CHANGEseq", model_backend="xgboost"):
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

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping(encoding=encoding)

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
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives,
                           encoding=encoding, model_backend=model_backend)
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
    gpu=True, encoding='NPM', data_type="CHANGEseq", model_backend="xgboost"):
    try:
        from PyALE import ale as pyale
    except ImportError as exc:
        raise ImportError("ALE requires PyALE. Install it with: pip install PyALE") from exc

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

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping(encoding=encoding)

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
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives,
                           encoding=encoding, model_backend=model_backend)
        # predict and insert the predictions into the predictions dfs

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            sequence_features_test = build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features,
                                                             encoding=encoding)

            # 1D - continuous - no CI
            pdp_result = pyale(X=pd.DataFrame(sequence_features_test), model=model, feature=[feature_index])
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
                 data_type="GUIDEseq", single_grna=None, model_backend="xgboost", max_samples=None):
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

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping(encoding=encoding)

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
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives,
                           encoding=encoding, model_backend=model_backend)

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            # Sample rows before feature construction to avoid encoding entire large subsets.
            sampled_df = dataset_df
            if max_samples is not None and len(dataset_df) > max_samples:
                sampled_df = dataset_df.sample(n=max_samples, random_state=42)
                print(f"  Sampled {max_samples} from {len(dataset_df)} samples")

            # Build sequence features for the positive test dataset
            sequence_features_test = build_sequence_features(sampled_df, nucleotides_to_position_mapping,
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


def create_feature_names_for_encoding(encoding, include_distance=True):
    """
    Create appropriate feature names based on encoding type.
    """
    feature_names = []
    
    if encoding == 'OneHot5Channel':
        # OneHot5Channel: 5 channels (A, C, G, T, match/mismatch) × 23 positions = 115 features
        channels = ['A', 'C', 'G', 'T', 'Match']
        for pos in range(23):
            for channel in channels:
                feature_names.append(f'Pos{pos+1}_{channel}')
    
    elif encoding == 'bulges':
        # Bulges encoding: 5x5 matrix (ACGT + bulge '-') for each of 23 positions
        # Each position has a 5x5 matrix representing guide_nuc × target_nuc alignment
        nucleotides_with_bulge = ['A', 'C', 'G', 'T', '-']
        
        for pos in range(23):
            for guide_nuc in nucleotides_with_bulge:
                for target_nuc in nucleotides_with_bulge:
                    if guide_nuc == '-':
                        label = f'Pos{pos+1}_DNA_bulge_{target_nuc}'
                    elif target_nuc == '-':
                        label = f'Pos{pos+1}_RNA_bulge_{guide_nuc}'
                    elif guide_nuc == target_nuc:
                        label = f'Pos{pos+1}_match_{guide_nuc}'
                    else:
                        label = f'Pos{pos+1}_mismatch_{guide_nuc}→{target_nuc}'
                    feature_names.append(label)
        
        # Note: actual implementation may have fewer features due to sparsity
        # If actual features < generated names, will be truncated automatically
    
    elif encoding == 'OneHot':
        # Standard OneHot: 4 nucleotides × 23 positions = 92 features
        nucleotides = ['A', 'C', 'G', 'T']
        for pos in range(23):
            for nuc in nucleotides:
                feature_names.append(f'Pos{pos+1}_{nuc}')
    
    elif encoding == 'NPM':
        # NPM: similar to OneHot but with position matrix
        # Definisco i nucleotidi possibili
        nucleotides = ['A', 'T', 'C', 'G']
        # Creo tutte le combinazioni di dinucleotidi
        nucleotide_combinations = [f'{n1}→{n2}' for n1 in nucleotides for n2 in nucleotides]
        # Ogni feature rappresenta una coppia di nucleotidi in una posizione
        num_positions = 23
        for i in range(num_positions):
            for nucleotide_pair in nucleotide_combinations:
                feature_names.append(f'Pos{i + 1}_{nucleotide_pair}')
    
    elif encoding == 'OneHotVstack':
        # OneHotVstack: guide + target stacked = 184 features
        nucleotides = ['A', 'C', 'G', 'T']
        for seq_type in ['Guide', 'Target']:
            for pos in range(23):
                for nuc in nucleotides:
                    feature_names.append(f'{seq_type}_Pos{pos+1}_{nuc}')
    
    elif encoding == 'kmer':
        # k-mer implementation uses k=3 and compares aligned 3-mers over 23nt sequences.
        # This yields (23 - 3 + 1) * 3 = 63 binary features.
        feature_names = [f'kmer_bit_{i+1}' for i in range(63)]
    
    elif encoding == 'LabelEncodingPairwise':
        # LabelEncodingPairwise: ~138 features
        # Positions × alignment states
        for pos in range(23):
            feature_names.extend([
                f'Pos{pos+1}_match',
                f'Pos{pos+1}_mismatch',
                f'Pos{pos+1}_transition',  # A↔G or C↔T
                f'Pos{pos+1}_transversion',  # Other mismatches
                f'Pos{pos+1}_DNA_bulge',
                f'Pos{pos+1}_RNA_bulge'
            ])
    
    elif encoding == 'MM':
        # MM encoding: mismatch features
        for pos in range(23):
            feature_names.append(f'Pos{pos+1}_mismatch_type')
        # Add aggregate features
        feature_names.extend([
            'total_mismatches',
            'PAM_proximal_mismatches',
            'PAM_distal_mismatches',
            'seed_region_mismatches',
            'total_bulges',
            'DNA_bulges',
            'RNA_bulges',
            'GC_content_guide',
            'GC_content_target',
            'GC_content_diff',
            'distance_to_PAM'
        ])
    
    else:
        # Generic fallback - will be adjusted based on actual data
        feature_names = None
    
    if include_distance and feature_names is not None:
        feature_names.append('Distance')
    
    return feature_names


def SHAP_explain_single_prediction(model_type="classifier", include_distance_feature=True,
                                   include_sequence_features=True, balanced=False,
                                   trans_type="ln_x_plus_one_trans", trans_all_fold=False, trans_only_positive=False,
                                   exclude_targets_without_positives=False, k_fold_number=10, gpu=True, encoding='NPM',
                                   data_type="GUIDEseq", single_grna=None, model_backend="xgb", max_samples=None):
    # Feature index for the SHAP explanation
    all_shap_values = []
    all_data_samples = []

    # # Feature names as the nucleotide position
    # feature = [int(i / 23) for i in range(368)]
    # # mapping
    # feature_names = [str(i + 1) for i in feature]
    # feature_names.append("Distance")

    # Generate feature names dynamically based on encoding
    feature_names = create_feature_names_for_encoding(encoding, include_distance_feature)

    # Defining prefix to load the model
    # IMPORTANT: All models are trained on CHANGEseq, so path_prefix is always CHANGEseq
    # But we can evaluate them on different data_type (CHANGEseq or GUIDEseq)
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

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping(encoding=encoding)

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
                           trans_all_fold, trans_only_positive, exclude_targets_without_positives,
                           encoding=encoding, model_backend=model_backend)

        # Build the SHAP explainer once per fold/model to avoid repeated initialization overhead.
        if model_backend in ("catboost", "decision_tree", "xgboost", "xgb"):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = None

        for j, dataset_df in enumerate((positive_df_test, negative_df_test)):
            # Sample rows before feature construction to avoid encoding entire large subsets.
            sampled_df = dataset_df
            if max_samples is not None and len(dataset_df) > max_samples:
                sampled_df = dataset_df.sample(n=max_samples, random_state=42)
                print(f"  Sampled {max_samples} from {len(dataset_df)} samples")

            # Build sequence features for the positive test dataset
            sequence_features_test = build_sequence_features(sampled_df, nucleotides_to_position_mapping,
                                                             include_distance_feature=include_distance_feature,
                                                             include_sequence_features=include_sequence_features,
                                                             encoding=encoding)

            # Convert sequence_features_test to a numpy array if it's a list
            if isinstance(sequence_features_test, list):
                sequence_features_test = np.array(sequence_features_test)

            # Choose appropriate explainer based on model backend
            if explainer is None:
                # Fallback for non-tree models.
                explainer = shap.DeepExplainer(model, sequence_features_test)

            # Compute SHAP values
            dataset = "positive" if j == 0 else "negative"
            print(f"Computing SHAP values for {dataset} subset {i + 1} ({sequence_features_test.shape[0]} samples)...")
            start_time = time.perf_counter()
            try:
                shap_values = explainer.shap_values(sequence_features_test, check_additivity=False)
            except TypeError:
                # Compatibility fallback for SHAP versions without this argument.
                shap_values = explainer.shap_values(sequence_features_test)
            elapsed = time.perf_counter() - start_time
            
            # Handle multi-output models (classification may return list)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification
             
            data_sample = sequence_features_test

            all_shap_values.append(shap_values)
            all_data_samples.append(data_sample)

            print(f"Computed SHAP values for {dataset} subset {i + 1} in {elapsed:.1f}s")

    # Combine the SHAP values and data samples
    combined_shap_values = np.vstack(all_shap_values)
    combined_data_samples = np.vstack(all_data_samples)

    print(f"\nCombined SHAP values shape: {combined_shap_values.shape}")
    print(f"Combined data samples shape: {combined_data_samples.shape}")

    # Generate feature names dynamically based on actual data shape if needed
    if feature_names is None:
        print(f"Generating generic feature names for {combined_shap_values.shape[1]} features")
        feature_names = [f'Feature_{i+1}' for i in range(combined_shap_values.shape[1])]
    elif len(feature_names) != combined_shap_values.shape[1]:
        print(f"Adjusting feature names: generated {len(feature_names)}, actual {combined_shap_values.shape[1]}")
        if len(feature_names) > combined_shap_values.shape[1]:
            # Truncate if we generated too many
            feature_names = feature_names[:combined_shap_values.shape[1]]
        else:
            # Pad with generic names if we generated too few
            for i in range(len(feature_names), combined_shap_values.shape[1]):
                feature_names.append(f'Feature_{i+1}')
    
    print(f"Using {len(feature_names)} feature names for plotting")

    # Save SHAP summary plot with descriptive filename
    output_filename = f'shap_summary_{encoding}_{data_type}_{model_backend}_{model_type}.png'
    plt.figure(figsize=(12, 8))
    shap.summary_plot(combined_shap_values, combined_data_samples, 
                     feature_names=feature_names, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved SHAP summary plot to: {output_filename}")
    plt.close()
    
    # Also create a bar plot showing absolute feature importance
    output_filename_bar = f'shap_bar_{encoding}_{data_type}_{model_backend}_{model_type}.png'
    plt.figure(figsize=(10, 8))
    shap.summary_plot(combined_shap_values, combined_data_samples, 
                     feature_names=feature_names, plot_type="bar", 
                     show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_filename_bar, dpi=300, bbox_inches='tight')
    print(f"Saved SHAP bar plot to: {output_filename_bar}")
    plt.close()

    return combined_shap_values


if __name__ == '__main__':
    # GENERATE SHAP PLOTS FOR CATBOOST MODELS
    # Note: All models are trained on CHANGEseq, but we can evaluate them on GUIDEseq
    
    # Example 1: OneHot5Channel encoding on GUIDEseq
    print("="*80)
    print("GENERATING SHAP PLOTS FOR OneHot5Channel on GUIDEseq with CatBoost")
    print("="*80)
    SHAP_explain_single_prediction(
        model_type="classifier",
        encoding='OneHot5Channel',
        data_type='GUIDEseq',  # Using GUIDEseq data with CHANGEseq-trained model
        model_backend='catboost',
        single_grna=None,  # Analyze all sgRNAs
        max_samples=1000   # Limit samples for speed
    )
    
    # # Example 2: bulges encoding on GUIDEseq
    # print("\n" + "="*80)
    # print("GENERATING SHAP PLOTS FOR bulges on GUIDEseq with CatBoost")
    # print("="*80)
    # SHAP_explain_single_prediction(
    #     model_type="classifier",
    #     encoding='bulges',
    #     data_type='GUIDEseq',
    #     model_backend='catboost',
    #     single_grna=None,
    #     max_samples=1000
    # )
    
    # Example 3: kmer encoding on GUIDEseq
    # TEMPORARILY DISABLED: kmer has dimension mismatch between CHANGEseq and GUIDEseq
    # This is because kmer encoding may generate different features based on the dataset
    # Run debug_kmer_dimensions.py to check the exact dimensions
    # 
    # Option 1: Use kmer on CHANGEseq instead:
    print("\n" + "="*80)
    print("GENERATING SHAP PLOTS FOR kmer on GUIDEseq with CatBoost")
    print("="*80)
    SHAP_explain_single_prediction(
        model_type="classifier",
        encoding='kmer',
        data_type='GUIDEseq', 
        model_backend='catboost',
        single_grna=None,
        max_samples=1000
    )
    #
    # Option 2: If you need GUIDEseq analysis, first fix the kmer encoding
    # to use a fixed vocabulary across both datasets
    
    # Uncomment for regression task:
    # print("\n" + "="*80)
    # print("GENERATING SHAP PLOTS FOR OneHot5Channel (REGRESSION)")
    # print("="*80)
    # SHAP_explain_single_prediction(
    #     model_type="regression_with_negatives",
    #     encoding='OneHot5Channel',
    #     data_type='GUIDEseq',
    #     model_backend='catboost',
    #     single_grna=None,
    #     max_samples=1000
    # )
