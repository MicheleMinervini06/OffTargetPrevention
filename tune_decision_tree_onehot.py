"""
Tune a Decision Tree baseline on OneHot encoding and save only best hyperparameters.

This script intentionally reuses the existing project pipeline:
- data loading from main_train.load_train_datasets
- feature extraction from utilities.build_sequence_features
- regression preprocessing from train_utilities.data_preprocessing
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from main_train import load_train_datasets
from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.train_utilities import data_preprocessing
from SysEvalOffTarget_src.utilities import (
    build_sampleweight,
    build_sequence_features,
    create_nucleotides_to_position_mapping,
)

{'splitter': 'random', 'min_samples_split': np.int64(55), 'min_samples_leaf': np.int64(17), 'max_features': 'log2', 'max_depth': 20, 'criterion': 'absolute_error', 'ccp_alpha': np.float64(0.011)}
def build_dataset(
    model_type: str,
    data_type: str,
    exclude_on_targets: bool,
    include_distance_feature: bool,
    include_sequence_features: bool,
    trans_type: str,
):
    """Build X, y (and optional sample weights) using the existing training pipeline."""
    _, positive_df, negative_df = load_train_datasets(
        union_model=False,
        data_type=data_type,
        exclude_on_targets=exclude_on_targets,
    )

    encoding = "OneHot"
    nt_mapping = create_nucleotides_to_position_mapping(encoding=encoding)

    positive_features = build_sequence_features(
        positive_df,
        nt_mapping,
        include_distance_feature=include_distance_feature,
        include_sequence_features=include_sequence_features,
        encoding=encoding,
    )

    sample_weight = None

    if model_type in ("classifier", "regression_with_negatives"):
        negative_features = build_sequence_features(
            negative_df,
            nt_mapping,
            include_distance_feature=include_distance_feature,
            include_sequence_features=include_sequence_features,
            encoding=encoding,
        )
        x = np.concatenate((negative_features, positive_features))
    else:
        x = positive_features

    negative_class = negative_df["label"].values
    positive_class = positive_df["label"].values

    if model_type == "classifier":
        y = np.concatenate((negative_class, positive_class))
        sample_weight = build_sampleweight(y)
    elif model_type == "regression_with_negatives":
        positive_df, negative_df = data_preprocessing(
            positive_df,
            negative_df,
            trans_type=trans_type,
            data_type=data_type,
            trans_all_fold=False,
            trans_only_positive=False,
        )
        y = np.concatenate(
            (
                negative_df[f"{data_type}_reads"].values,
                positive_df[f"{data_type}_reads"].values,
            )
        )
        class_labels = np.concatenate((negative_class, positive_class))
        sample_weight = build_sampleweight(class_labels)
    elif model_type == "regression_without_negatives":
        positive_df, _ = data_preprocessing(
            positive_df,
            negative_df,
            trans_type=trans_type,
            data_type=data_type,
            trans_all_fold=False,
            trans_only_positive=True,
        )
        y = positive_df[f"{data_type}_reads"].values
    else:
        raise ValueError("model_type must be one of: classifier, regression_with_negatives, regression_without_negatives")

    return x, y, sample_weight


def run_random_search(
    x,
    y,
    sample_weight,
    model_type: str,
    n_iter: int,
    cv: int,
    n_jobs: int,
    seed: int,
):
    """Run randomized search and return best params dict."""
    common_params = {
        "max_depth": [None, 4, 6, 8, 10, 12, 16, 20, 30],
        "min_samples_split": np.arange(2, 61),
        "min_samples_leaf": np.arange(1, 31),
        "max_features": [None, "sqrt", "log2", 0.5, 0.7, 0.9],
        "splitter": ["best", "random"],
        "ccp_alpha": np.linspace(0.0, 0.02, 41),
    }

    if model_type == "classifier":
        estimator = DecisionTreeClassifier(random_state=seed)
        param_dist = {
            **common_params,
            "criterion": ["gini", "entropy", "log_loss"],
            "class_weight": [None, "balanced"],
        }
        scoring = "roc_auc"
    else:
        estimator = DecisionTreeRegressor(random_state=seed)
        param_dist = {
            **common_params,
            "criterion": ["squared_error", "friedman_mse", "absolute_error"],
        }
        scoring = "neg_mean_absolute_error"

    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        verbose=2,
        random_state=seed,
        n_jobs=n_jobs,
    )

    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    search.fit(x, y, **fit_kwargs)
    return search.best_params_


def parse_args():
    parser = argparse.ArgumentParser(
        description="Random search for Decision Tree baseline using OneHot encoding only."
    )
    parser.add_argument(
        "--model-type",
        default="classifier",
        choices=["classifier", "regression_with_negatives", "regression_without_negatives"],
        help="Task/model variant to optimize.",
    )
    parser.add_argument(
        "--data-type",
        default="CHANGEseq",
        choices=["CHANGEseq", "GUIDEseq"],
        help="Dataset used for tuning.",
    )
    parser.add_argument(
        "--exclude-on-targets",
        action="store_true",
        help="Use exclude_on_targets split when loading data.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=60,
        help="Number of random-search iterations.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Cross-validation folds for random search.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--trans-type",
        default="ln_x_plus_one_trans",
        choices=[
            "no_trans",
            "ln_x_plus_one_trans",
            "ln_x_plus_one_and_max_trans",
            "standard_trans",
            "max_trans",
            "box_cox_trans",
            "yeo_johnson_trans",
        ],
        help="Transformation used only for regression modes.",
    )
    parser.add_argument(
        "--include-distance-feature",
        action="store_true",
        default=True,
        help="Include distance feature (default: True).",
    )
    parser.add_argument(
        "--no-distance-feature",
        action="store_false",
        dest="include_distance_feature",
        help="Disable distance feature.",
    )
    parser.add_argument(
        "--output",
        default="best_hyperparameters_decision_tree_OneHot.joblib",
        help="Output file for best hyperparameters.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    seed = general_utilities.SEED

    x, y, sample_weight = build_dataset(
        model_type=args.model_type,
        data_type=args.data_type,
        exclude_on_targets=args.exclude_on_targets,
        include_distance_feature=args.include_distance_feature,
        include_sequence_features=True,
        trans_type=args.trans_type,
    )

    best_params = run_random_search(
        x,
        y,
        sample_weight,
        model_type=args.model_type,
        n_iter=args.n_iter,
        cv=args.cv,
        n_jobs=args.n_jobs,
        seed=seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_params, output_path)

    print("OneHot Decision Tree tuning completed.")
    print(f"Best params saved to: {output_path}")
    print(best_params)


if __name__ == "__main__":
    main()
