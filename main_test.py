"""
 Contains function for testing the models. Note this code show partial examples of the test options.
 You can see the options in the function's documentation.
"""

import random
import numpy as np
import pandas as pd
from SysEvalOffTarget_src.test_utilities import evaluation, model_folds_predictions

from SysEvalOffTarget_src.utilities import create_nucleotides_to_position_mapping, order_sg_rnas, load_order_sg_rnas
from SysEvalOffTarget_src import general_utilities


random.seed(general_utilities.SEED)


def regular_test_models(
        models_options=("regression_with_negatives", "classifier", "regression_without_negatives"),
        include_distance_feature_options=(True, False), include_sequence_features_options=(True, False),
        trans_type="ln_x_plus_one_trans", trans_all_fold=False,
        trans_only_positive=False, exclude_targets_without_positives=False,
        train_exclude_on_targets=False, test_exclude_on_targets=False, k_fold_number=10,
        task="evaluation", data_types=('CHANGEseq', 'GUIDEseq'), intersection=None,
        encoding="NPM"):
    """
    Function for testing the models. The corresponding function to regular_train_models.
    This function produces results for the regression and classification models trained only on the CHANGE-seq dataset.
    Saves the results in the files directory according to the type if the model.
    The parameters help locate the models, and define on which dataset the test will be applied.
    intersection: can be "CHANGE_GUIDE_intersection_by_both" or "CHANGE_GUIDE_intersection_by_GUIDE"
    :param models_options: models_options: tuple. A tuple with the model types to test. support these options:
        ("regression_with_negatives", "classifier", "regression_without_negatives").
        Default: ("regression_with_negatives", "classifier", "regression_without_negatives")
    :param include_distance_feature_options: tuple. A tuple that determinate whether to add the distance
        feature. The tuple can contain both True and False. In case False is included in the tuple, False cannot be
        included in include_sequence_features_options. Default: (True, False)
    :param include_sequence_features_options: tuple. A tuple that determinate whether to add the sequence
        feature. The tuple can contain both True and False. In case False is included in the tuple, False cannot be
        included in include_distance_feature_options. Default: (True, False)
    :param trans_type: str. define which transformation is applied on the read counts.
        Note the issue in the parametric transformers: we used the inverse transformation which is learned on the test
        set. Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param trans_all_fold: bool. apply the data transformation on each sgRNA dataset if False,
        else apply on the entire test fold. Default: False
    :param trans_only_positive: bool. apply the data transformation on only on active sites if True. Default: False
    :param exclude_targets_without_positives: bool. exclude sgRNAs data without positives if True. Default: False
    :param train_exclude_on_targets: test the models trained without on-targets in case of True. Default: False
    :param test_exclude_on_targets: test on the dataset without the on-targets in case of True. Default: False
    :param k_fold_number: int. number of k-folds of the tested models. Default: 10
    :param task: str. options: "evaluation" or "prediction". If task="evaluation", we evaluate the model performance
        and save the results. If task="prediction", prediction of the models on the dataset samples is made.
        The predictions are saved into a table. Default: "evaluation"
    :param data_types: tuple. The dataset on which we perform the evaluation/predictions.
        Options: ('CHANGEseq', 'GUIDEseq'). Default: ('CHANGEseq', 'GUIDEseq')
    :param intersection: str or None. In case it is not none, ignores data_types, and perform the test over an
        intersection dataset of GUIDE-seq and CHANGE-seq. works only with task="evaluation". Options:
        "CHANGE_GUIDE_intersection_by_both" or "CHANGE_GUIDE_intersection_by_GUIDE".
        See prepare_data file for description. Default: None
    :param encoding: str. encoding type. Options: "NPM" or "OneHot". Default: "NPM"
    :return: None
    """
    if intersection is not None and task == "prediction":
        raise ValueError("prediction task does not support prediction on the intersection")

    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    # test CHANGE-seq model
    try:
        targets_change_seq = load_order_sg_rnas()
    except FileNotFoundError:
        targets_change_seq = order_sg_rnas()

    for data_type in data_types:
        datasets_dir_path = general_utilities.DATASETS_PATH
        datasets_dir_path += 'exclude_on_targets/' if test_exclude_on_targets else 'include_on_targets/'
        dataset_type = data_type if intersection is None else intersection
        positive_df = pd.read_csv(
            datasets_dir_path + '{}_positive.csv'.format(dataset_type), index_col=0)
        negative_df = pd.read_csv(
            datasets_dir_path + '{}_negative.csv'.format(dataset_type), index_col=0)
        negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
            'N') == -1]  # some off_targets contains N's. we drop them

        # the function supports evaluation of the models trained only on the CHANGE-seq dataset
        save_model_dir_path_prefix = 'CHANGEseq/exclude_on_targets/' if \
            train_exclude_on_targets else 'CHANGEseq/include_on_targets/'
        # predictions path
        prediction_results_path = general_utilities.FILES_DIR + "models_{}_fold/".format(k_fold_number) + \
                                  save_model_dir_path_prefix
        prediction_results_path += \
            'predictions_exclude_on_targets/' if test_exclude_on_targets else 'predictions_include_on_targets/'
        prediction_results_path += "{0}_results_all_{1}_folds.csv".format(data_type, k_fold_number)

        for model_type in models_options:
            models_path_prefix = save_model_dir_path_prefix + model_type + "/"
            # evaluation path
            evaluation_results_path_prefix = save_model_dir_path_prefix + model_type + "/"
            evaluation_results_path_prefix += \
                'test_results_exclude_on_targets/' if test_exclude_on_targets else 'test_results_include_on_targets/'
            evaluation_results_path_prefix += intersection + "_" if intersection is not None else ""
            # make the evaluations/predictions
            for include_distance_feature in include_distance_feature_options:
                for include_sequence_features in include_sequence_features_options:
                    if (not include_distance_feature) and (not include_sequence_features):
                        continue
                    call_args = (positive_df, negative_df, targets_change_seq, nucleotides_to_position_mapping)
                    call_kwargs = {"data_type": data_type, "model_type": model_type, "k_fold_number": k_fold_number,
                                   "include_distance_feature": include_distance_feature,
                                   "include_sequence_features": include_sequence_features,
                                   "balanced": False, "trans_type": trans_type,
                                   "trans_all_fold": trans_all_fold,
                                   "trans_only_positive": trans_only_positive,
                                   "exclude_targets_without_positives": exclude_targets_without_positives,
                                   "encoding": encoding}
                    if task == "evaluation":
                        call_kwargs.update({"models_path_prefix": models_path_prefix,
                                            "results_path_prefix": evaluation_results_path_prefix})
                        print("Call args", call_args)
                        print("Call kwargs", call_kwargs)
                        evaluation(*call_args, **call_kwargs)
                    elif task == "prediction":
                        call_kwargs.update({"add_to_results_table": True,
                                            "results_table_path": prediction_results_path,
                                            "save_results": True, "path_prefix": models_path_prefix,
                                            "encoding": encoding})
                        model_folds_predictions(*call_args, **call_kwargs)
                    else:
                        raise ValueError("Invalid task argument value")


def load_incremental_test_folds_data(test_exclude_on_targets):
    """
    utility function of incremental_test_models_folds and incremental_union_test_models_folds
    """
    random.seed(general_utilities.SEED)
    nucleotides_to_position_mapping = create_nucleotides_to_position_mapping()
    try:
        targets = load_order_sg_rnas("GUIDE")
    except FileNotFoundError:
        targets = order_sg_rnas("GUIDE")

    datasets_dir_path = general_utilities.DATASETS_PATH
    datasets_dir_path += 'exclude_on_targets/' if test_exclude_on_targets else 'include_on_targets/'
    positive_df = pd.read_csv(
        datasets_dir_path + '{}_positive.csv'.format("GUIDEseq"), index_col=0)
    negative_df = pd.read_csv(
        datasets_dir_path + '{}_negative.csv'.format("GUIDEseq"), index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find(
        'N') == -1]  # some off_targets contains N's. we drop them

    test_target_folds_list = np.array_split(targets, 6)

    return nucleotides_to_position_mapping, positive_df, negative_df, targets, test_target_folds_list


def incremental_test_evaluate_folds(
        model_type, positive_df, negative_df, test_targets_fold, trans_type, n_targets,
        test_exclude_on_targets, include_distance_feature_options, nucleotides_to_position_mapping,
        tl_or_union_type, evaluation_results_path_prefix, path_prefix, seed, **kwargs):
    """
    utility function of incremental_test_models_folds and incremental_union_test_models_folds
    """
    # evaluation path
    evaluation_results_path_prefix += \
        'test_results_exclude_on_targets/' if test_exclude_on_targets else 'test_results_include_on_targets/'
    evaluation_results_path_prefix += "seed_" + str(seed) + "/" + model_type + "/trained_with_" + str(n_targets + 1) + \
                                      "_guides_"
    for include_distance_feature in include_distance_feature_options:
        call_args = (positive_df, negative_df, test_targets_fold, nucleotides_to_position_mapping)
        call_kwargs = {"data_type": "GUIDEseq", "model_type": model_type, "k_fold_number": 1,
                       "include_distance_feature": include_distance_feature,
                       "include_sequence_features": True,
                       "balanced": False, "trans_type": trans_type
                       }
        call_kwargs.update({"models_path_prefix": path_prefix,
                            "results_path_prefix": evaluation_results_path_prefix})
        try:
            evaluation(*call_args, **call_kwargs, **kwargs)
        except ValueError as error:
            print("got exception:", error)
            print("in seed", seed, "number of sgRNAs:", n_targets, "model_type:", model_type)
            print("include_distance_feature:", include_distance_feature)
            print("TL/union-model type:", tl_or_union_type)


def incremental_test_models_folds(
        models_options=("regression_with_negatives", "classifier"), include_distance_feature_options=(True, False),
        train_exclude_on_targets=False, test_exclude_on_targets=False, trans_type="ln_x_plus_one_trans",
        transfer_learning_types=(None, "add", "update"), seeds=(i for i in range(1, 11)), **kwargs):
    """
    Function for testing the models. The corresponding function to incremental_train_models_folds.
    Save the results in the files directory according to the type if the model.
    :param models_options: tuple. A tuple with the model types to test. support these options:
        ("regression_with_negatives", "classifier").
        Default: ("regression_with_negatives", "classifier")
    :param include_distance_feature_options: tuple. A tuple that determinate whether to add the distance
        feature. The tuple can contain both True and False. In case False is included in the tuple, False cannot be
        included in include_sequence_features_options. Default: (True, False)
    :param train_exclude_on_targets: test the models trained without on-targets in case of True. Default: False
    :param test_exclude_on_targets: test on the dataset without the on-targets in case of True. Default: False
    :param trans_type: str. define which transformation is applied on the read counts.
        Note the issue in the parametric transformers: we used the inverse transformation which is learned on the test
        set. Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param transfer_learning_types: bool. Transfer learning types. support these options: (None, "add", "update").
        When None is included, then models are trained with TL. Default: (None, "add", "update")
    :param seeds: tuple. tuple of seeds numbers (n numbers). we are training n time the models, each time we shuffle the
        GUIDE-seq training set randomly according to the seeds.
    :param kwargs: Additional evaluation settings. See SysEvalOffTarget_src.test_utilities.evaluation for additional
        arguments. Note that in this evaluation, we support only changing the settings of trans_all_fold,
        trans_only_positive and exclude_targets_without_positives.
    :return: None
    """
    nucleotides_to_position_mapping, positive_df, negative_df, targets, test_target_folds_list = \
        load_incremental_test_folds_data(test_exclude_on_targets)
    for fold_i, test_targets_fold in enumerate(test_target_folds_list):
        for seed in seeds:
            train_targets = [target for target in targets if target not in test_targets_fold]
            save_model_dir_path_prefix = 'exclude_on_targets/' if train_exclude_on_targets else 'include_on_targets/'
            save_model_dir_path_prefix = "GUIDEseq/incremental_folds_training/fold_{}/".format(str(fold_i)) + \
                                         save_model_dir_path_prefix
            for transfer_learning_type in transfer_learning_types:
                type_save_model_dir_path_prefix = save_model_dir_path_prefix + \
                                                  "GS_TL_" + transfer_learning_type + '/' if transfer_learning_type is not None else \
                    save_model_dir_path_prefix + "GS/"
                for i in range(len(train_targets)):
                    path_prefix = type_save_model_dir_path_prefix + "seed_" + str(seed) + \
                                  "/trained_with_" + str(i + 1) + "_guides_"
                    for model_type in models_options:
                        incremental_test_evaluate_folds(
                            model_type, positive_df, negative_df, test_targets_fold, trans_type, i,
                            test_exclude_on_targets, include_distance_feature_options, nucleotides_to_position_mapping,
                            transfer_learning_type, type_save_model_dir_path_prefix, path_prefix, seed, **kwargs)


def incremental_union_test_models_folds(
        models_options=("regression_with_negatives", "classifier"), include_distance_feature_options=(True, False),
        train_exclude_on_targets=False, test_exclude_on_targets=False, trans_type="ln_x_plus_one_trans",
        seeds=(i for i in range(1, 11)), n_trees=1000, **kwargs):
    """
    Function for testing the models. The corresponding function to incremental_union_train_models_folds.
    Save the results in the files directory according to the type if the model.
    :param models_options: tuple. A tuple with the model types to test. support these options:
        ("regression_with_negatives", "classifier").
        Default: ("regression_with_negatives", "classifier")
    :param include_distance_feature_options: tuple. A tuple that determinate whether to add the distance
        feature. The tuple can contain both True and False. In case False is included in the tuple, False cannot be
        included in include_sequence_features_options. Default: (True, False)
    :param train_exclude_on_targets: test the models trained without on-targets in case of True. Default: False
    :param test_exclude_on_targets: test on the dataset without the on-targets in case of True. Default: False
    :param trans_type: str. define which transformation is applied on the read counts.
        Note the issue in the parametric transformers: we used the inverse transformation which is learned on the test
        set. Options: "no_trans" - no transformation; "ln_x_plus_one_trans" - log(x+1) where x the read count;
        "ln_x_plus_one_and_max_trans" - log(x+1)/log(MAX_READ_COUNT_IN_TRAIN_SET) where x the read count;
        "standard_trans" - Z-score; "max_trans" - x/MAX_READ_COUNT_IN_TRAIN_SET;
        "box_cox_trans" - Box-Cox; or "yeo_johnson_trans" - Yeo-Johnson. In Box-Cox and Yeo-Johnson, the transformation
        is learned on a balanced set of active and inactive sites (if trans_only_positive=False).
        Default: "ln_x_plus_one_trans"
    :param seeds: tuple. tuple of seeds numbers (n numbers). we are training n time the models, each time we shuffle the
        GUIDE-seq training set randomly according to the seeds.
    :param n_trees: int. number XGBoost trees estimators. Default: 1000
    :param kwargs: Additional evaluation settings. See SysEvalOffTarget_src.test_utilities.evaluation for additional
        arguments. Note that in this evaluation, we support only changing the settings of trans_all_fold,
        trans_only_positive and exclude_targets_without_positives.
    :return: None
    """
    # GUIDE-seq
    nucleotides_to_position_mapping, positive_df, negative_df, targets, test_target_folds_list = \
        load_incremental_test_folds_data(test_exclude_on_targets)

    for fold_i, test_targets_fold in enumerate(test_target_folds_list):
        for seed in seeds:
            train_targets = [target for target in targets if target not in test_targets_fold]
            for i in range(len(train_targets)):
                save_model_dir_path_prefix = 'exclude_on_targets/' if train_exclude_on_targets else \
                    'include_on_targets/'
                save_model_dir_path_prefix = "GUIDE_and_CHANGE_seq_{}_trees/incremental_folds_training/fold_{}/".format(
                    str(n_trees), str(fold_i)) + save_model_dir_path_prefix
                for model_type in models_options:
                    path_prefix = save_model_dir_path_prefix + "seed_" + str(seed) + \
                                  "/trained_with_" + str(i + 1) + "_guides_"
                    incremental_test_evaluate_folds(
                        model_type, positive_df, negative_df, test_targets_fold, trans_type, i,
                        test_exclude_on_targets, include_distance_feature_options, nucleotides_to_position_mapping,
                        n_trees, save_model_dir_path_prefix, path_prefix, seed, **kwargs)


def main():
    """
    main function
    """
    # some testing examples. The models you are testing must be existed before
    # See regular_test_models, incremental_test_models_folds, and incremental_union_test_models_folds
    # to test other options.
    # regular_test_models(
    #     models_options=tuple(("regression_without_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))
    # regular_test_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # to obtain the predictions we change the task into prediction
    # regular_test_models(
    #     models_options=tuple(("regression_without_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="prediction",
    #     data_types=('CHANGEseq', 'GUIDEseq'))
    # regular_test_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="prediction",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Read counts log transformation improves prediction performance

    # # Figure C/D: CHANGE-seq/Classification/Regression-seq-dist/log transformation
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # # Figure C/D: CHANGE-seq/Classification/Regression-seq-dist/without log transformation
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     trans_type="no_trans",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # # Figure E/F: CHANGE-seq/Regression/Regression-seq-dist/log transformation
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # # Figure E/F: CHANGE-seq/Regression/Regression-seq-dist/without log transformation
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     trans_type="no_trans", task="evaluation",
    #     k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Including potential OTSs with no reads in regression model training improves prediction performance

    # # Figure A/B: CHANGE-seq/Classification/Regression-seq-dist/log transformation/ with potential OTSs
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # # Figure A/B: CHANGE-seq/Classification/Regression-seq-dist/log transformation/ without potential OTSs
    # regular_test_models(
    #     models_options=tuple(("regression_without_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # # Figure C/D: CHANGE-seq/Regression/Regression-seq-dist/log transformation/ with potential OTSs
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))
    #
    # # Figure C/D: CHANGE-seq/Regression/Regression-seq-dist/log transformation/ without potential OTSs
    # regular_test_models(
    #     models_options=tuple(("regression_without_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # The combination of sequence and distance
    # features achieves the best prediction performance

    # # Figure A/B/C/D: Combination  of sequence and distance features on both classification and regression tasks
    # regular_test_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, task="evaluation",
    #     data_types=('CHANGEseq', 'GUIDEseq'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Evaluating regression task performance of models trained with the distance feature only

    # # FIGURE A/B CHANGE-seq/Regression-dist/with distance feature
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(False,),
    #     k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'))

    # # Figure C Linear function compared with the function learned by the model for regression-dist
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(False,),
    #     k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'),
    #     task="prediction")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Trying to improve the prediction performance of GUIDEseq regression-seq-dist using hyperparameter tuning
    # regular_test_models(
    #     models_options=tuple(("regression_with_negatives",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_types=('GUIDEseq',))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Testing the Decision Tree model on the CHANGE-seq and GUIDE-seq dataset
    # regular_test_models(
    #     models_options=tuple(("classifier",)),
    #     include_distance_feature_options=(True,),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # # Evaluating the performance of models trained with One Hot encoding
    # regular_test_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'),
    #     encoding="OneHot")

    # # Evaluating the performance of models trained with One Hot 5 channel encoding
    # regular_test_models(
    #     models_options=tuple(("classifier", "regression_with_negatives")),
    #     include_distance_feature_options=(True, False),
    #     include_sequence_features_options=(True,),
    #     k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'),
    #     encoding="OneHot5Channel")

    # Evaluating the performance of models trained with kmer encoding
    regular_test_models(
        models_options=tuple(("classifier", "regression_with_negatives")),
        include_distance_feature_options=(True, False),
        include_sequence_features_options=(True,),
        k_fold_number=10, data_types=('CHANGEseq', 'GUIDEseq'),
        encoding="kmer")


if __name__ == '__main__':
    main()
