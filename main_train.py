"""
Contains function for training the models. Note this code show partial examples of the test options.
 You can see the options in the function's documentation.
"""

import random
import numpy as np
import pandas as pd
from SysEvalOffTarget_src.train_utilities import train

from SysEvalOffTarget_src.utilities import create_nucleotides_to_position_mapping
from SysEvalOffTarget_src.utilities import order_sg_rnas, load_order_sg_rnas
from SysEvalOffTarget_src import general_utilities

random.seed(general_utilities.SEED)


def load_train_datasets(union_model, data_type, exclude_on_targets):
    """
    Load datasets for the regular training.
    :param union_model: bool. train the CS-GS-union model if True. Default: False
    :param data_type: str. The data type on which the models are trained. "CHANGEseq" or "GUIDEseq".
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True.
    :return: (targets, positive_df, negative_df)
        targets is a list of the sgRNAs which we will train on.
        positive_df, negative_df are Pandas dataframes used for training the models.
    """
    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    if not union_model:
        # Train CHANGE-seq/GUIDE-seq model
        try:
            targets = load_order_sg_rnas(data_type)
        except FileNotFoundError:
            targets = order_sg_rnas(data_type)

        positive_df = pd.read_csv(
            datasets_dir_path + '{}_positive.csv'.format(data_type), index_col=0)
        negative_df = pd.read_csv(
            datasets_dir_path + '{}_negative.csv'.format(data_type), index_col=0)
        negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
            'N') == -1]  # some off_targets contains N's. we drop them
    else:
        try:
            targets_change_seq = load_order_sg_rnas("CHANGEseq")
            targets_guide_seq = load_order_sg_rnas("GUIDEseq")
        except FileNotFoundError:
            targets_change_seq = order_sg_rnas("CHANGEseq")
            targets_guide_seq = order_sg_rnas("GUIDEseq")

        targets_guide_seq_train = targets_guide_seq
        targets_change_seq_train = [target for target in targets_change_seq if target not in targets_guide_seq]
        targets_train = targets_guide_seq_train + targets_change_seq_train
        print("sgRNAs in train:", len(targets_train))

        # load CHANGE-seq dataset
        positive_change_seq_df = pd.read_csv(
            datasets_dir_path + '{}_positive.csv'.format("CHANGEseq"), index_col=0)
        negative_change_seq_df = pd.read_csv(
            datasets_dir_path + '{}_negative.csv'.format("CHANGEseq"), index_col=0)
        negative_change_seq_df = negative_change_seq_df[negative_change_seq_df["offtarget_sequence"].str.find(
            'N') == -1]  # some off_targets contains N's. we drop them
        positive_change_seq_df = positive_change_seq_df[positive_change_seq_df["target"].isin(targets_change_seq_train)]
        negative_change_seq_df = negative_change_seq_df[negative_change_seq_df["target"].isin(targets_change_seq_train)]
        # load GUIDE-seq dataset
        positive_guide_seq_df = pd.read_csv(
            datasets_dir_path + '{}_positive.csv'.format("GUIDEseq"), index_col=0)
        negative_guide_seq_df = pd.read_csv(
            datasets_dir_path + '{}_negative.csv'.format("GUIDEseq"), index_col=0)
        negative_guide_seq_df = negative_guide_seq_df[negative_guide_seq_df["offtarget_sequence"].str.find(
            'N') == -1]  # some off_targets contains N's. we drop them
        positive_guide_seq_df = positive_guide_seq_df[positive_guide_seq_df["target"].isin(targets_guide_seq_train)]
        negative_guide_seq_df = negative_guide_seq_df[negative_guide_seq_df["target"].isin(targets_guide_seq_train)]

        # concat CHANGE-seq and GUIDE-seq
        positive_guide_seq_df = positive_guide_seq_df[
            ["chrom", "chromStart", "GUIDEseq_reads", "target", "offtarget_sequence", "distance", "label"]]
        # just for simplicity, assume GUIDEseq_reads are CHANGEseq_reads
        positive_guide_seq_df = positive_guide_seq_df.rename({"GUIDEseq_reads": "CHANGEseq_reads"}, axis="columns")
        positive_change_seq_df = positive_change_seq_df[
            ["chrom", "chromStart", "CHANGEseq_reads", "target", "offtarget_sequence", "distance", "label"]]
        positive_df = pd.concat([positive_change_seq_df, positive_guide_seq_df])

        negative_guide_seq_df = negative_guide_seq_df[
            ["chrom", "chromStart", "target", "offtarget_sequence", "distance", "label"]]
        negative_change_seq_df = negative_change_seq_df[
            ["chrom", "chromStart", "target", "offtarget_sequence", "distance", "label"]]
        negative_df = pd.concat([negative_change_seq_df, negative_guide_seq_df])

        targets = targets_change_seq

    return targets, positive_df, negative_df


def regular_train_models(
        models_options=("regression_with_negatives", "classifier", "regression_without_negatives"), union_model=False,
        include_distance_feature_options=(True, False), include_sequence_features_options=(True, False),
        n_trees=1000, trans_type="ln_x_plus_one_trans", trans_all_fold=False, trans_only_positive=False,
        exclude_targets_without_positives=False, exclude_on_targets=False, k_fold_number=10, data_type="CHANGEseq",
        xgb_model=None, transfer_learning_type="add", exclude_targets_with_rhampseq_exp=False, save_model=True,
        encoding="NPM"):
    """
    Function for training the models. This performs k-fold training.
    :param models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier", "regression_without_negatives")
    :param union_model: bool. train the CS-GS-union model if True. Default: False
    :param include_distance_feature_options:  tuple. A tuple that determinate whether to add the distance feature.
        The tuple can contain both True and False. Default: (True, False)
    :param include_sequence_features_options:  tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. In case False is included in the tuple, False cannot be
        included in include_distance_feature_options. Default: (True, False)
    :param n_trees: int. number XGBoost trees estimators. Default: 1000
    :param trans_type: str. define which transformation is applied on the read counts.
        Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param trans_all_fold: bool. apply the data transformation on each sgRNA dataset if False,
        else apply on the entire train fold. Default: False
    :param trans_only_positive: bool. apply the data transformation on only on active sites if True. Default: False
    :param exclude_targets_without_positives: bool. exclude sgRNAs data without positives if True. Default: False
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param k_fold_number: int. number of k-folds. Default: 10
    :param data_type: str. The data type on which the models are trained. "CHANGEseq" or "GUIDEseq".
        Default: "CHANGEseq"
    :param xgb_model: str or None. Path of the pretrained model (used for transfer learning.
        Assuming that (models_options + include_distance_feature_options + include_sequence_features_options)
        has one option that corresponds to the pre-trained model.
        Default: None
    :param transfer_learning_type: str. Transfer learning type. can be "add" or "update".
        Relevant if xgb_model is not None. Default: "add"
    :param exclude_targets_with_rhampseq_exp: bool. exclude the targets that appear in the rhAmpSeq experiment if True.
        Default: False
    :param save_model: bool. Save the models if True
    :param encoding: str. The encoding type. Options : "NPM" or "OneHot" Default: "NPM"
    :return: None
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    targets, positive_df, negative_df = load_train_datasets(union_model, data_type, exclude_on_targets)
    data_type = "CHANGEseq" if union_model else data_type

    if exclude_targets_with_rhampseq_exp:
        targets_list = ["GTCAGGGTTCTGGATATCTGNGG",  # TRAC_site_1
                        "GCTGGTACACGGCAGGGTCANGG",  # TRAC_site_2
                        "GAGAATCAAAATCGGTGAATNGG"  # TRAC_site_3,
                        "GAAGGCTGAGATCCTGGAGGNGG",  # LAG3_site_9
                        "GGACTGAGGGCCATGGACACNGG"  # CTLA4_site_9
                        "GTCCCTAGTGGCCCCACTGTNGG"  # AAVS1_site_2
                        ]
        positive_df = positive_df[~positive_df["target"].isin(targets_list)]
        negative_df = negative_df[~negative_df["target"].isin(targets_list)]

    save_model_dir_path_prefix = 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    save_model_dir_path_prefix = "trained_without_rhampseq_exp_targets/" + save_model_dir_path_prefix \
        if exclude_targets_with_rhampseq_exp else save_model_dir_path_prefix
    save_model_dir_path_prefix = data_type + '/' + save_model_dir_path_prefix if not union_model else \
        "GUIDE_and_CHANGE_seq_{}_trees".format(str(n_trees)) + '/' + save_model_dir_path_prefix

    save_model_dir_path_prefix += "TL_" + transfer_learning_type + '/' if xgb_model is not None else ""
    for model_type in models_options:
        path_prefix = save_model_dir_path_prefix + model_type + "/"
        for include_distance_feature in include_distance_feature_options:
            for include_sequence_features in include_sequence_features_options:
                if (not include_distance_feature) and (not include_sequence_features):
                    continue

                train(positive_df, negative_df, targets, nucleotides_to_position_mapping, data_type=data_type,
                      model_type=model_type, k_fold_number=k_fold_number,
                      include_distance_feature=include_distance_feature,
                      include_sequence_features=include_sequence_features, balanced=False, trans_type=trans_type,
                      trans_all_fold=trans_all_fold, trans_only_positive=trans_only_positive,
                      exclude_targets_without_positives=exclude_targets_without_positives, path_prefix=path_prefix,
                      xgb_model=xgb_model, transfer_learning_type=transfer_learning_type, save_model=save_model,
                      n_trees=n_trees, encoding=encoding)


def incremental_pretrain_base_models(models_options=("regression_with_negatives", "classifier"),
                                     trans_type="ln_x_plus_one_trans", exclude_on_targets=False, **kwargs):
    """
    Pre-train the base models for the TL incremental training. save the model in files.
    :param models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier")
    :param trans_type: str. define which transformation is applied on the read counts.
        Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param kwargs: Additional train settings. See SysEvalOffTarget_src.train_utilities.train for additional arguments.
    :return: None
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    try:
        targets_change_seq = load_order_sg_rnas("CHANGE")
    except FileNotFoundError:
        targets_change_seq = order_sg_rnas("CHANGE")

    try:
        targets_guide_seq = load_order_sg_rnas("GUIDE")
    except FileNotFoundError:
        targets_guide_seq = order_sg_rnas("GUIDE")

    targets_change_seq_filtered = [target for target in targets_change_seq if target not in targets_guide_seq]
    print("CHANGE-seq - number of sgRNAs:", len(targets_change_seq))
    print("GUIDE-seq - number of sgRNAs:", len(targets_guide_seq))
    print("CHANGE-seq - number of sgRNAs - filtered:", len(targets_change_seq_filtered))

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format("CHANGEseq"), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format("CHANGEseq"), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    positive_df = positive_df[positive_df['target'].isin(targets_change_seq_filtered)]
    negative_df = negative_df[negative_df['target'].isin(targets_change_seq_filtered)]

    save_model_dir_path_prefix = \
        'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    save_model_dir_path_prefix = \
        "train_CHANGE_seq_on_non_overlapping_targets_with_GUIDE_seq/" + save_model_dir_path_prefix
    for model_type in models_options:
        path_prefix = save_model_dir_path_prefix + model_type + "/"
        for include_distance_feature in (True, False):
            train(positive_df, negative_df, targets_change_seq, nucleotides_to_position_mapping, data_type="CHANGEseq",
                  model_type=model_type, k_fold_number=1, include_distance_feature=include_distance_feature,
                  include_sequence_features=True, balanced=False, trans_type=trans_type, path_prefix=path_prefix,
                  encoding="NPM", **kwargs)


def incremental_train_models_folds(models_options=("regression_with_negatives", "classifier"),
                                   exclude_on_targets=False, trans_type="ln_x_plus_one_trans",
                                   transfer_learning_types=(None, "add", "update"), pretrain=True,
                                   seeds=(i for i in range(1, 11)), **kwargs):
    """
    Training the TL models from CHANGE-seq to GUIDE-seq in incremental way. For each model type, we're training n
    (where n is the number sgRNA we dedicated for training the TL models) models. we start with one sgRNA in a train,
    and then increase the number until we train the final model with n sgRNAs.
    :param models_options: models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier").
        Default: ("regression_with_negatives", "classifier")
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param trans_type: str. define which transformation is applied on the read counts.
        Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balaced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param transfer_learning_types: bool. Transfer learning types. support these options: (None, "add", "update").
        When None is included, then models are trained with TL. Default: (None, "add", "update")
    :param pretrain: bool. pretrain the model if True. If False, we assume the pretrained models are exists.
        Default: True
    :param seeds: tuple. tuple of seeds numbers (n numbers). we are training n time the models, each time we shuffle the
        all training set randomly according to the seeds.
    :param kwargs: Additional train settings. See SysEvalOffTarget_src.train_utilities.train for additional arguments.
    :return: None
    """
    # CHANGE-seq
    if pretrain:
        incremental_pretrain_base_models(models_options=models_options,
                                         trans_type=trans_type, exclude_on_targets=exclude_on_targets, **kwargs)

    # GUIDE-seq
    random.seed(general_utilities.SEED)
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    try:
        targets = load_order_sg_rnas("GUIDE")
    except FileNotFoundError:
        targets = order_sg_rnas("GUIDE")

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format("GUIDEseq"), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format("GUIDEseq"), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    test_target_folds_list = np.array_split(targets, 6)
    for fold_i, test_targets_fold in enumerate(test_target_folds_list):
        for seed in seeds:
            train_targets = [target for target in targets if target not in test_targets_fold]
            print("Targets in train:")
            print(train_targets)
            print("Targets in test:")
            print(test_targets_fold)
            print("seed:", seed)
            random.seed(seed)
            random.shuffle(train_targets)
            random.seed(general_utilities.SEED)
            save_model_dir_path_prefix = 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
            save_model_dir_path_prefix = "GUIDEseq/incremental_folds_training/fold_{}/".format(str(fold_i)) + \
                                         save_model_dir_path_prefix
            for transfer_learning_type in transfer_learning_types:
                type_save_model_dir_path_prefix = save_model_dir_path_prefix + \
                                                  "GS_TL_" + transfer_learning_type + '/' if transfer_learning_type is not None \
                    else save_model_dir_path_prefix + "GS/"
                xgb_model_path = general_utilities.FILES_DIR + "models_1fold/" + \
                                 "train_CHANGE_seq_on_non_overlapping_targets_with_GUIDE_seq/"
                xgb_model_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
                for i in range(len(train_targets)):
                    path_prefix = type_save_model_dir_path_prefix + "seed_" + str(seed) + \
                                  "/trained_with_" + str(i + 1) + "_guides_"
                    if "classifier" in models_options:
                        train(positive_df, negative_df, train_targets[0:i + 1], nucleotides_to_position_mapping,
                              data_type='GUIDEseq', model_type="classifier", k_fold_number=1,
                              include_distance_feature=False, include_sequence_features=True, balanced=False,
                              trans_type=trans_type, path_prefix=path_prefix, xgb_model=xgb_model_path +
                                                                                        "classifier"
                                                                                        "/classifier_xgb_model_fold_0_without_Kfold_imbalanced.xgb"
                            if transfer_learning_type is not None else None,
                              transfer_learning_type=transfer_learning_type, encoding="NPM", **kwargs)
                        train(positive_df, negative_df, train_targets[0:i + 1], nucleotides_to_position_mapping,
                              data_type='GUIDEseq', model_type="classifier", k_fold_number=1,
                              include_distance_feature=True, include_sequence_features=True, balanced=False,
                              trans_type=trans_type, path_prefix=path_prefix, xgb_model=xgb_model_path +
                                                                                        "classifier"
                                                                                        "/classifier_xgb_model_fold_0_with_distance_without_Kfold_imbalanced.xgb"
                            if transfer_learning_type is not None else None,
                              transfer_learning_type=transfer_learning_type, encoding="NPM", **kwargs)
                    if "regression_with_negatives" in models_options:
                        train(positive_df, negative_df, train_targets[0:i + 1], nucleotides_to_position_mapping,
                              data_type='GUIDEseq', model_type="regression_with_negatives", k_fold_number=1,
                              include_distance_feature=False, include_sequence_features=True, balanced=False,
                              trans_type=trans_type, path_prefix=path_prefix,
                              xgb_model=xgb_model_path + "regression_with_negatives/"
                                                         "regression_with_negatives_xgb_model_fold_0_without_Kfold_imbalanced.xgb"
                              if transfer_learning_type is not None else None,
                              transfer_learning_type=transfer_learning_type, encoding="NPM", **kwargs)
                        train(positive_df, negative_df, train_targets[0:i + 1], nucleotides_to_position_mapping,
                              data_type='GUIDEseq', model_type="regression_with_negatives", k_fold_number=1,
                              include_distance_feature=True, include_sequence_features=True, balanced=False,
                              trans_type=trans_type, path_prefix=path_prefix,
                              xgb_model=xgb_model_path + "regression_with_negatives/"
                                                         "regression_with_negatives_xgb_model_fold_0_with_distance_without_Kfold_imbalanced.xgb"
                              if transfer_learning_type is not None else None,
                              transfer_learning_type=transfer_learning_type, encoding="NPM", **kwargs)


def incremental_union_train_models_folds(models_options=("regression_with_negatives", "classifier"),
                                         include_distance_feature_options=(True, False),
                                         include_sequence_features_options=(True, False),
                                         exclude_on_targets=False, trans_type="ln_x_plus_one_trans",
                                         seeds=(i for i in range(1, 11)),
                                         n_trees=1000, **kwargs):
    """
    training the CS-GS-Union models in incremental way. For each model type, we're training n
    (where n is the number sgRNA we dedicated for training the TL models) models. we start with one GUIDE-seq sgRNA in a
    train, and then increase the number until we train the final model with n sgRNAs.
    :param models_options: models_options: tuple. A tuple with the model types to train. support these options:
        ("regression_with_negatives", "classifier").
        Default: ("regression_with_negatives", "classifier")
    :param include_distance_feature_options:  tuple. A tuple that determinate whether to add the distance feature.
        The tuple can contain both True and False. Default: (True, False)
    :param include_sequence_features_options:  tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. in can the tuple include False, False can not be included in
        include_distance_feature_options. Default: (True, False)
    :param exclude_on_targets: bool. Exclude on-targets from training in case of True. Default: False
    :param trans_type: str. define which transformation is applied on the read counts.
        Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param seeds: tuple. tuple of seeds numbers (n numbers). we are training n time the models, each time we shuffle the
        GUIDE-seq training set randomly according to the seeds.
    :param n_trees: int. number XGBoost trees estimators. Default: 1000
    :param kwargs: Additional train settings. See SysEvalOffTarget_src.train_utilities.train for additional arguments.
    :return: None
    """
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    # Train CHANGE-seq/GUIDE-seq model
    try:
        targets_change_seq = load_order_sg_rnas("CHANGEseq")
        targets_guide_seq = load_order_sg_rnas("GUIDEseq")
    except FileNotFoundError:
        targets_change_seq = order_sg_rnas("CHANGEseq")
        targets_guide_seq = order_sg_rnas("GUIDEseq")

    test_target_folds_list = np.array_split(targets_guide_seq, 6)
    for fold_i, test_targets_fold in enumerate(test_target_folds_list):
        targets_change_seq_train = [target for target in targets_change_seq if target not in targets_guide_seq]
        for seed in seeds:
            targets_guide_seq_train = [target for target in targets_guide_seq if target not in test_targets_fold]
            print("GUIDE-seq targets in train:")
            print(targets_guide_seq_train)
            print("GUIDE-seq targets in test:")
            print(test_targets_fold)
            print("seed:", seed)
            random.seed(seed)
            random.shuffle(targets_guide_seq_train)
            random.seed(general_utilities.SEED)
            for i in range(len(targets_guide_seq_train)):
                targets_guide_seq_train_i = targets_guide_seq_train[:i + 1]
                targets_train = targets_guide_seq_train_i + targets_change_seq_train
                print("targets in train:", len(targets_train))
                print(targets_train)

                datasets_dir_path = general_utilities.DATASETS_PATH
                datasets_dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'

                positive_change_seq_df = pd.read_csv(
                    datasets_dir_path + '{}_positive.csv'.format("CHANGEseq"), index_col=0)
                negative_change_seq_df = pd.read_csv(
                    datasets_dir_path + '{}_negative.csv'.format("CHANGEseq"), index_col=0)
                negative_change_seq_df = negative_change_seq_df[negative_change_seq_df["offtarget_sequence"].str.find(
                    'N') == -1]  # some off_targets contains N's. we drop them
                positive_change_seq_df = positive_change_seq_df[
                    positive_change_seq_df["target"].isin(targets_change_seq_train)]
                negative_change_seq_df = negative_change_seq_df[
                    negative_change_seq_df["target"].isin(targets_change_seq_train)]

                positive_guide_seq_df = pd.read_csv(
                    datasets_dir_path + '{}_positive.csv'.format("GUIDEseq"), index_col=0)
                negative_guide_seq_df = pd.read_csv(
                    datasets_dir_path + '{}_negative.csv'.format("GUIDEseq"), index_col=0)
                negative_guide_seq_df = negative_guide_seq_df[negative_guide_seq_df["offtarget_sequence"].str.find(
                    'N') == -1]  # some off_targets contains N's. we drop them
                positive_guide_seq_df = positive_guide_seq_df[
                    positive_guide_seq_df["target"].isin(targets_guide_seq_train_i)]
                negative_guide_seq_df = negative_guide_seq_df[
                    negative_guide_seq_df["target"].isin(targets_guide_seq_train_i)]

                positive_guide_seq_df = positive_guide_seq_df[
                    ["chrom", "chromStart", "GUIDEseq_reads", "target", "offtarget_sequence", "distance", "label"]]
                # just for simplicity, assume GUIDEseq_reads are CHANGEseq_reads
                positive_guide_seq_df = positive_guide_seq_df.rename({"GUIDEseq_reads": "CHANGEseq_reads"},
                                                                     axis="columns")
                positive_change_seq_df = positive_change_seq_df[
                    ["chrom", "chromStart", "CHANGEseq_reads", "target", "offtarget_sequence", "distance", "label"]]
                positive_df = pd.concat([positive_change_seq_df, positive_guide_seq_df])

                negative_guide_seq_df = negative_guide_seq_df[
                    ["chrom", "chromStart", "target", "offtarget_sequence", "distance", "label"]]
                negative_change_seq_df = negative_change_seq_df[
                    ["chrom", "chromStart", "target", "offtarget_sequence", "distance", "label"]]
                negative_df = pd.concat([negative_change_seq_df, negative_guide_seq_df])

                save_model_dir_path_prefix = 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
                save_model_dir_path_prefix = \
                    "GUIDE_and_CHANGE_seq_{}_trees/incremental_folds_training/fold_{}/".format(str(n_trees),
                                                                                               str(fold_i)) + save_model_dir_path_prefix
                data_type = 'CHANGEseq'
                for model_type in models_options:
                    path_prefix = save_model_dir_path_prefix + "seed_" + str(seed) + \
                                  "/trained_with_" + str(i + 1) + "_guides_"
                    for include_distance_feature in include_distance_feature_options:
                        for include_sequence_features in include_sequence_features_options:
                            if (not include_distance_feature) and (not include_sequence_features):
                                continue
                            train(positive_df, negative_df, targets_train, nucleotides_to_position_mapping,
                                  data_type=data_type, model_type=model_type, k_fold_number=1,
                                  include_distance_feature=include_distance_feature,
                                  include_sequence_features=include_sequence_features, balanced=False,
                                  trans_type=trans_type, path_prefix=path_prefix, n_trees=n_trees, encoding="NPM",
                                  **kwargs)


def main():
    """
    main function
    """
    # some training examples. You need to run prepre_data.py before trying to train.
    # See regular_train_models, incremental_train_models_folds, and incremental_union_train_models_folds 
    # to train other options.
    # regular_train_models(
    #     models_options=tuple(("regression_without_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Read counts log transformation improves prediction performance
    #
    # # Figure C/D/E/F: CHANGE-seq/Regression-seq-dist/log transformation
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")
    #
    # # Figure C/D/E/F: CHANGE-seq/Regression-seq-dist/without log transformation
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     trans_type="no_trans",
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Including potential OTSs with no reads in regression model training improves prediction performance

    # # Figure A/B/C/D: CHANGE-seq/Regression-seq-dist/with inactive sites
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")
    #
    # # Figure A/B/C/D: CHANGE-seq/Regression-seq-dist/without inactive sites
    # regular_train_models(
    #     models_options=tuple(("regression_without_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # The combination of sequence and distance
    # features achieves the best prediction performance

    # # Figure A/B/C/D: Combination  of sequence and distance features on both classification and regression tasks
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Evaluating regression task performance of models trained with the distance feature only

    # # FIGURE A/B CHANGE-seq/Regression-dist/with distance feature
    # # Figure C Linear function compared with the function learned by the model for regression-dist
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(False,),
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Training of Regression-seq, Regression-seq-dist, and Classification-seq, Classification-seq-dist models
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq",
    #     encoding="NPM")

    # # Training of Regression-seq, Regression-seq-dist, and Classification-seq, Classification-seq-dist models
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq",
    #     encoding="OneHot")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Trying to improve the prediction performance of GUIDEseq regression-seq-dist using hyperparameter tuning
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="GUIDEseq")

    # # Trying to improve the prediction performance of CHANGEseq regression-seq-dist using hyperparameter tuning
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Training the model using a decision tree
    # regular_train_models(
    #     models_options=tuple(("regression_with_negatives", "classifier", "regression_without_negatives")),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Training of Classification-seq-dist and Regression-seq-dist using onehot 5 channel encoding
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq",
    #     encoding="OneHot5Channel")

    # # Training of Classification-seq-dist and Regression-seq-dist using onehot encoding
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq",
    #     encoding="OneHot")

    # # Training of Classification-seq-dist and Regression-seq-dist using kmer encoding
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq",
    #     encoding="kmer")

    # # Training of Classification-seq-dist and Regression-seq-dist using onehot encoding with vertical stacking
    # regular_train_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_type="CHANGEseq",
    #     encoding="OneHotVstack")

    # Training of Classification-seq-dist and Regression-seq-dist using onehot encoding with vertical stacking
    regular_train_models(
        models_options=tuple(("regression_with_negatives",)),
        include_distance_feature_options=(False,),
        include_sequence_features_options=(True,),
        k_fold_number=10, data_type="CHANGEseq",
        encoding="OneHotVstack")
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if __name__ == '__main__':
    main()
