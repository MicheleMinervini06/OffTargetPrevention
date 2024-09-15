"""
This module contains the utilizes functions for training and training all the xgboost model variants
"""
import random
import itertools
import pandas as pd
import numpy as np

from sklearn.preprocessing import PowerTransformer, FunctionTransformer, MaxAbsScaler, StandardScaler

from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def load_order_sg_rnas(data_type='CHANGE'):
    """
    load and return the sgRNAs in certain order for the k-fold training
    """
    data_type = 'CHANGE' if data_type.lower() in ('changeseq', 'change-seq', 'change_seq') else data_type
    data_type = 'GUIDE' if data_type.lower() in ('guideseq', 'guide-seq', 'guide_seq') else data_type
    sg_rnas_s = pd.read_csv(general_utilities.DATASETS_PATH + data_type + '-seq_sgRNAs_ordering.csv', header=None)
    # Modificata la conversione del dataframe in lista
    return sg_rnas_s.iloc[:, 0].tolist()


def order_sg_rnas(data_type='CHANGE'):
    """
    Create and return the sgRNAs in certain order for the k-fold training
    """
    data_type = 'CHANGE' if data_type.lower() in ('changeseq', 'change-seq', 'change_seq') else data_type
    data_type = 'GUIDE' if data_type.lower() in ('guideseq', 'guide-seq', 'guide_seq') else data_type
    dataset_df = pd.read_excel(
        general_utilities.DATASETS_PATH + data_type + '-seq.xlsx', index_col=0)
    sg_rnas = list(dataset_df["target"].unique())
    print("There are", len(sg_rnas), "unique sgRNAs in the", data_type, "dataset")

    # sort the sgRNAs and shuffle them
    sg_rnas.sort()
    random.shuffle(sg_rnas)

    # save the sgRNAs order into csv file
    sg_rnas_s = pd.Series(sg_rnas)
    # to csv - you can read this to Series using -
    # pd.read_csv("file_name.csv", header=None, squeeze=True)
    sg_rnas_s.to_csv(general_utilities.DATASETS_PATH + data_type + '-seq_sgRNAs_ordering.csv',
                     header=False, index=False)

    return sg_rnas


def create_nucleotides_to_position_mapping():
    """
    Return the nucleotides to position mapping
    """
    # matrix positions for ('A','A'), ('A','C'),...
    # tuples of ('A','A'), ('A','C'),...
    nucleotides_product = list(itertools.product(*(['ACGT'] * 2)))
    # tuples of (0,0), (0,1), ...
    position_product = [(int(x[0]), int(x[1]))
                        for x in list(itertools.product(*(['0123'] * 2)))]
    nucleotides_to_position_mapping = dict(
        zip(nucleotides_product, position_product))

    # tuples of ('N','A'), ('N','C'),...
    n_mapping_nucleotides_list = [('N', char) for char in ['A', 'C', 'G', 'T']]
    # list of tuples positions corresponding to ('A','A'), ('C','C'), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ['A', 'C', 'G', 'T']]

    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    # tuples of ('A','N'), ('C','N'),...
    n_mapping_nucleotides_list = [(char, 'N') for char in ['A', 'C', 'G', 'T']]
    # list of tuples positions corresponding to ('A','A'), ('C','C'), ...
    n_mapping_position_list = [nucleotides_to_position_mapping[(char, char)]
                               for char in ['A', 'C', 'G', 'T']]
    nucleotides_to_position_mapping.update(
        dict(zip(n_mapping_nucleotides_list, n_mapping_position_list)))

    return nucleotides_to_position_mapping


def build_sequence_features(dataset_df, nucleotides_to_position_mapping,
                            include_distance_feature=False,
                            include_sequence_features=True,
                            encoding="NPM"):
    if encoding == "NPM":
        """
        Build sequence features using the nucleotides to position mapping
        """
        if (not include_distance_feature) and (not include_sequence_features):
            raise ValueError(
                'include_distance_feature and include_sequence_features can not be both False')

        # convert dataset_df["target"] -3 position to 'N'
        print("Converting the [-3] positions in each sgRNA sequence to 'N'")
        dataset_df.loc[:, 'target'] = dataset_df['target'].apply(lambda s: s[:-3] + 'N' + s[-2:])

        if include_sequence_features:
            final_result = np.zeros((len(dataset_df), (23 * 16) + 1),
                                    dtype=np.int8) if include_distance_feature else \
                np.zeros((len(dataset_df), 23 * 16), dtype=np.int8)
        else:
            final_result = np.zeros((len(dataset_df), 1), dtype=np.int8)
        for i, (seq1, seq2) in enumerate(zip(dataset_df["target"], dataset_df["offtarget_sequence"])):
            if include_sequence_features:
                intersection_matrices = np.zeros((23, 4, 4), dtype=np.int8)
                for j in range(23):
                    matrix_positions = nucleotides_to_position_mapping[(
                        seq1[j], seq2[j])]
                    intersection_matrices[j, matrix_positions[0],
                    matrix_positions[1]] = 1
            else:
                intersection_matrices = None

            if include_distance_feature:
                if include_sequence_features:
                    final_result[i, :-1] = intersection_matrices.flatten()
                final_result[i, -1] = dataset_df["distance"].iloc[i]
            else:
                final_result[i, :] = intersection_matrices.flatten()

        return final_result

    """Including the k-mer encoding"""
    if encoding == "kmer":
        def get_kmers(sequence, k):
            """Convert a sequence into a list of overlapping k-mers"""
            return [sequence[i:i + k] for i in range(len(sequence) - k + 1)]

        def compare_kmers(kmers1, kmers2):
            """Compare two k-mer lists and return a binary vector representing match/mismatch"""
            binary_encoding = []
            for kmer1, kmer2 in zip(kmers1, kmers2):
                binary_kmer = ''.join(['1' if nuc1 == nuc2 else '0' for nuc1, nuc2 in zip(kmer1, kmer2)])
                binary_encoding.append(binary_kmer)
            return ''.join(binary_encoding)

        k = 3  # k-mer length
        if include_sequence_features:
            final_result = np.zeros((len(dataset_df), (23 - k + 1) * k), dtype=np.int8)
        else:
            final_result = np.zeros((len(dataset_df), 1), dtype=np.int8)

        for i, (seq1, seq2) in enumerate(zip(dataset_df["target"], dataset_df["offtarget_sequence"])):
            kmers_seq1 = get_kmers(seq1, k)
            kmers_seq2 = get_kmers(seq2, k)
            binary_encoding = compare_kmers(kmers_seq1, kmers_seq2)

            # Convert the binary encoding string to a list of integers
            encoded_array = np.array([int(b) for b in binary_encoding], dtype=np.int8)

            if include_sequence_features:
                final_result[i, :len(encoded_array)] = encoded_array
            else:
                final_result[i, :] = encoded_array

            if include_distance_feature:
                final_result[i, -1] = dataset_df["distance"].iloc[i]

        return final_result

    """INcluding the Label Encoding Pairwise"""
    if encoding == 'LabelEncodingPairwise':
        # Definisci la mappatura delle coppie di nucleotidi
        nucleotide_pairs_mapping = {
            ('A', 'A'): 0, ('A', 'C'): 1, ('A', 'G'): 2, ('A', 'T'): 3,
            ('C', 'A'): 4, ('C', 'C'): 5, ('C', 'G'): 6, ('C', 'T'): 7,
            ('G', 'A'): 8, ('G', 'C'): 9, ('G', 'G'): 10, ('G', 'T'): 11,
            ('T', 'A'): 12, ('T', 'C'): 13, ('T', 'G'): 14, ('T', 'T'): 15
        }

        # Funzione per mappare le coppie di nucleotidi con il trattamento di 'N'
        def pairwise_sequence_encoding(seq1, seq2):
            encoded_seq = []
            for n1, n2 in zip(seq1, seq2):
                if n1 == 'N':
                    n1 = n2  # Se n1 è 'N', trattalo come n2
                elif n2 == 'N':
                    n2 = n1  # Se n2 è 'N', trattalo come n1
                encoded_seq.append(nucleotide_pairs_mapping[(n1, n2)])
            return encoded_seq

        # Applica il pairwise encoding alle sequenze target e off-target
        pairwise_encoded = np.array([pairwise_sequence_encoding(seq1, seq2)
                                     for seq1, seq2 in zip(dataset_df['target'], dataset_df['offtarget_sequence'])])

        # Aggiungi opzionalmente la feature di distanza
        if include_distance_feature:
            flattened_results = np.hstack([pairwise_encoded, dataset_df['distance'].values[:, np.newaxis]])
        else:
            flattened_results = pairwise_encoded

        return flattened_results

    if encoding == 'OneHot' or encoding == 'OneHot5Channel' or encoding == 'OneHotVstack':
        # Define the mapping from nucleotides to one-hot encoding using a numpy array for direct indexing
        nucleotide_mapping = np.array([[1, 0, 0, 0],  # A 0
                                       [0, 1, 0, 0],  # C 1
                                       [0, 0, 1, 0],  # G 2
                                       [0, 0, 0, 1],  # T 3
                                       [0, 0, 0, 0]])  # N 4

        # Create a mapper from nucleotide characters to indices [A, C, G, T, N]
        char_to_index = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

        # Function to convert sequence string to index array
        def sequence_to_index_array(sequence):
            if sequence is None:
                return np.array([], dtype=int)
            return np.array([char_to_index.get(nuc, 4) for nuc in sequence], dtype=int)

        # Convert sequences to indices
        target_indices = dataset_df['target'].apply(sequence_to_index_array)
        offtarget_indices = dataset_df['offtarget_sequence'].apply(sequence_to_index_array)

        # Convert indices to one-hot encoded form
        target_encoded = np.array([nucleotide_mapping[indices] for indices in target_indices])
        offtarget_encoded = np.array([nucleotide_mapping[indices] for indices in offtarget_indices])

        if encoding == 'OneHotVstack':
            # Stack the target and offtarget sequences vertically
            result = np.concatenate((target_encoded, offtarget_encoded), axis=-1)
        else:
            # Perform a logical OR operation
            result = np.logical_or(target_encoded, offtarget_encoded).astype(int)

        if encoding == 'OneHot5Channel':
            # Determine the direction for each position
            direction_indicators = []
            for t_idx, o_idx in zip(target_indices, offtarget_indices):
                direction_indicator = np.zeros((len(t_idx), 1), dtype=int)  # 5th position for the direction
                for i, (t, o) in enumerate(zip(t_idx, o_idx)):
                    if t != 4 and o != 4:  # Ignore 'N' characters
                        if t <= o:  # If they match
                            # Determine the order of the nucleotides
                            direction_indicator[i, 0] = 0
                        else:
                            direction_indicator[i, 0] = 1
                direction_indicators.append(direction_indicator)

            direction_indicators = np.array(direction_indicators)

            # Append the direction indicators to the OR result
            result = np.concatenate((result, direction_indicators), axis=-1)

        # Flatten the results and optionally include the distance feature
        flattened_result = result.reshape(result.shape[0], -1)  # Vettorizzazione

        if include_distance_feature:
            flattened_results = np.hstack([flattened_result, dataset_df['distance'].values[:, np.newaxis]])
        else:
            flattened_results = flattened_result

        # Convert the results into a NumPy array with dtype set for memory efficiency
        final_result = np.array(flattened_results, dtype=np.int8)

        return final_result


##########################################################################


def create_fold_sets(target_fold, targets, positive_df, negative_df,
                     balanced, exclude_targets_without_positives):
    """
    Create fold sets for train/test
    remove_targets_without_positives: only from the train test.
        It doesn't matter in the test set, as we can't evaluate the performance when evaluating per sgRNA.
        Moreover, we can always remove them in the evaluation stage.
    """
    test_targets = target_fold
    train_targets = [target for target in targets if target not in target_fold]
    if exclude_targets_without_positives:
        for target in train_targets.copy():
            if len(positive_df[positive_df["target"] == target]) == 0:
                print("removing target:", target, "from training set, since it has no positives")
                train_targets.remove(target)

    positive_df_test = positive_df[positive_df['target'].isin(test_targets)]
    positive_df_train = positive_df[positive_df['target'].isin(train_targets)]

    if balanced:
        # obtain the negative samples for train
        # (for each target the positive and negative samples numbers is equal)
        negative_indices_train = []
        for target in targets:
            if target in test_targets:
                continue
            negative_indices_train = negative_indices_train + \
                                     list(negative_df[(negative_df['target'] == target)].sample(
                                         n=len(positive_df_train[(positive_df_train['target'] == target)])).index)
        negative_df_train = negative_df.loc[negative_indices_train]

        # obtain the negative samples for test (for test take all negatives not in the trains set)
        negative_df_test = negative_df[negative_df['target'].isin(
            test_targets)]

        # negative_indices_test = []
        # for target in target_fold:
        #     negative_indices_test = negative_indices_test + \
        #         list(negative_df[negative_df['target']==target].sample(
        #             n=len(positive_df_test[positive_df_test['target']==target])).index)
        # negative_df_test = negative_df.loc[negative_indices_test]
    else:
        negative_df_test = negative_df[negative_df['target'].isin(
            test_targets)]
        negative_df_train = negative_df[negative_df['target'].isin(
            train_targets)]

    return negative_df_train, positive_df_train, negative_df_test, positive_df_test


##########################################################################
def build_sampleweight(y_values):
    """
    Sample weight according to class
    """
    vec = np.zeros((len(y_values)))
    for values_class in np.unique(y_values):
        vec[y_values == values_class] = np.sum(
            y_values != values_class) / len(y_values)
    return vec


##########################################################################


def extract_model_name(model_type, include_distance_feature, include_sequence_features, balanced, trans_type,
                       trans_all_fold, trans_only_positive, exclude_targets_without_positives):
    """
    extract model name
    """
    model_name = "Classification" if model_type == "classifier" else "Regression"
    model_name += "-no-negatives" if model_type == "regression_without_negatives" else ""
    model_name += "-seq" if include_sequence_features else ""
    model_name += "-dist" if include_distance_feature else ""
    model_name += "-positiveSgRNAs" if exclude_targets_without_positives else ""
    if model_type != "classifier":
        model_name += "-noTrans" if trans_type == "no_trans" else ""
        model_name += "-log1pMaxTrans" if trans_type == "ln_x_plus_one_and_max_trans" else ""
        model_name += "-maxTrans" if trans_type == "max_trans" else ""
        model_name += "-standardTrans" if trans_type == "standard_trans" else ""
        model_name += "-boxTrans" if trans_type == "box_cox_trans" else ""
        model_name += "-yeoTrans" if trans_type == "yeo_johnson_trans" else ""
        model_name += "-balanced" if balanced else ""
        model_name += "-foldTrans" if trans_all_fold else ""
        model_name += "-positiveTrans" if trans_only_positive else ""

    return model_name


##########################################################################


def prefix_and_suffix_path(model_type, k_fold_number, include_distance_feature, include_sequence_features, balanced,
                           trans_type, trans_all_fold, trans_only_positive, exclude_targets_without_positives,
                           path_prefix, encoding="NPM"):
    suffix = "_with_distance" if include_distance_feature else ""
    suffix += "" if include_sequence_features else "_without_sequence_features"
    suffix += ("_without_Kfold" if k_fold_number == 1 else "")
    suffix += ("" if balanced == 1 else "_imbalanced")
    if encoding == "OneHot":
        suffix += "_with_OneHotEncoding"
    elif encoding == "OneHot5Channel":
        suffix += "_with_OneHotEncoding5Channel"
    elif encoding == "kmer":
        suffix += "_with_kmerEncoding"
    elif encoding == "OneHotVstack":
        suffix += "_with_OneHotEncodingVstack"
    elif encoding == "LabelEncodingPairwise":
        suffix += "_with_LabelEncodingPairwise"
    if trans_type != "ln_x_plus_one_trans" and model_type != "classifier":
        suffix += "_" + trans_type
    path_prefix = "trans_only_positive/" + path_prefix if trans_only_positive else path_prefix
    path_prefix = "trans_on_entire_train_or_test_fold/" + path_prefix if trans_all_fold else path_prefix
    path_prefix = "drop_sg_rna_with_non_positives/" + path_prefix if exclude_targets_without_positives else path_prefix

    return path_prefix, suffix


def extract_model_path(model_type, k_fold_number, include_distance_feature, include_sequence_features,
                       balanced, trans_type, trans_all_fold, trans_only_positive, exclude_targets_without_positives,
                       fold_index, path_prefix, encoding="NPM"):
    """
    extract model path
    """
    path_prefix, suffix = prefix_and_suffix_path(model_type, k_fold_number, include_distance_feature,
                                                 include_sequence_features, balanced, trans_type, trans_all_fold,
                                                 trans_only_positive, exclude_targets_without_positives, path_prefix,
                                                 encoding)
    dir_path = general_utilities.FILES_DIR + "models_" + \
               str(k_fold_number) + "fold/" + path_prefix + model_type + \
               "_xgb_model_fold_" + str(fold_index) + suffix + ".json"

    return dir_path


def extract_model_results_path(model_type, data_type, k_fold_number, include_distance_feature,
                               include_sequence_features, balanced, trans_type, trans_all_fold, trans_only_positive,
                               exclude_targets_without_positives, evaluate_only_distance, suffix_add, path_prefix,
                               encoding):
    """
    extract model results path
    """
    path_prefix, suffix = prefix_and_suffix_path(model_type, k_fold_number, include_distance_feature,
                                                 include_sequence_features, balanced, trans_type, trans_all_fold,
                                                 trans_only_positive, exclude_targets_without_positives, path_prefix,
                                                 encoding)
    suffix = suffix + ("" if evaluate_only_distance is None else "_distance_" + str(evaluate_only_distance))
    suffix = suffix + suffix_add
    dir_path = general_utilities.FILES_DIR + "models_" + str(k_fold_number) + \
               "fold/" + path_prefix + data_type + "_" + model_type + \
               "_results_xgb_model_all_" + str(k_fold_number) + "_folds" + suffix + ".csv"

    return dir_path


##########################################################################


def transformer_generator(data, trans_type):
    """
    Create create data transformer
    """
    data = data.reshape(-1, 1)
    if trans_type == "no_trans":
        # identity transformer
        transformer = FunctionTransformer(validate=False)
    elif trans_type == "ln_x_plus_one_trans":
        transformer = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
    elif trans_type == "ln_x_plus_one_and_max_trans":
        transformer_1 = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        data = transformer_1.transform(data)
        transformer_2 = MaxAbsScaler()
        transformer_2.fit(data)
        transformer = (transformer_1, transformer_2)
    elif trans_type == "standard_trans":
        transformer = StandardScaler()
        transformer.fit(data)
    elif trans_type == "max_trans":
        transformer = MaxAbsScaler()
        transformer.fit(data)
    elif trans_type == "box_cox_trans":
        if np.all(data == data[0]):
            # if the input data is constant, the return identity transformer
            print("identity transformer (instead of box-cox) was returned since the input data is constant")
            transformer = FunctionTransformer()
        else:
            # we balance the negatives and positives and then fit the transformation.
            data = data[data > 0]
            data = data.reshape(-1, 1)
            data = np.concatenate([data, np.zeros(data.shape)])
            # we perform box-cox on data+1
            transformer_1 = FunctionTransformer(func=lambda x: x + 1, inverse_func=lambda x: x - 1)
            data = transformer_1.transform(data)
            transformer_2 = PowerTransformer(method='box-cox')
            transformer_2.fit(data)
            transformer = (transformer_1, transformer_2)
    elif trans_type == "yeo_johnson_trans":
        if np.all(data == data[0]):
            # if the input data is constant, the return identity transformer
            print("identity transformer (instead of yeo-johnson) was returned since the input data is constant")
            transformer = FunctionTransformer()
        else:
            # we balance the negatives and positives and then fit the transformation.
            data = data[data > 0]
            data = data.reshape(-1, 1)
            data = np.concatenate([data, np.zeros(data.shape)])
            transformer = PowerTransformer(method='yeo-johnson')
            transformer.fit(data)
    else:
        raise ValueError("Invalid trans_type")

    return transformer


def transform(data, transformer, inverse=False):
    """
    transform function
    """
    data = data.reshape(-1, 1)
    if not isinstance(transformer, (list, tuple)):
        transformer = [transformer]
    if not inverse:
        for transformer_i in transformer:
            data = transformer_i.transform(data)
    else:
        for transformer_i in transformer[::-1]:
            data = transformer_i.inverse_transform(data)

    return np.squeeze(data)
