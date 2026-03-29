"""
Evaluate XGBoost models across different encodings and produce summary tables.

For each encoding, runs 10-fold cross-validation evaluation on both CHANGEseq and
GUIDEseq datasets for classification and regression tasks (with distance feature only).

Produces 4 summary CSV files:
  - summary_classification_CHANGEseq.csv
  - summary_classification_GUIDEseq.csv
  - summary_regression_CHANGEseq.csv
  - summary_regression_GUIDEseq.csv

Each file has one row per encoding and columns: metric_mean, metric_std.
"""

import random
import pandas as pd
import numpy as np
from pathlib import Path

from SysEvalOffTarget_src import general_utilities
from SysEvalOffTarget_src.utilities import (
    create_nucleotides_to_position_mapping, load_order_sg_rnas, order_sg_rnas,
    extract_model_results_path,
)
from glob import glob

random.seed(general_utilities.SEED)

# ── Configuration ────────────────────────────────────────────────────────────

# Encodings to evaluate; extend this list when models for other encodings exist
ENCODINGS = ["NPM", "OneHot", "OneHot5Channel", "LabelEncodingPairwise", "OneHotVstack", "MM", "kmer", "bulges"]

DATA_TYPES = ["CHANGEseq", "GUIDEseq"]
MODEL_TYPES = ["classifier", "regression_with_negatives"]

# Model backends to evaluate (xgb, catboost, decision_tree)
# Set to None to process all backends, or specify a list like ["xgb", "catboost"]
MODEL_BACKENDS = ["xgb", "catboost", "decision_tree"]  # Add "decision_tree" if you have those results

K_FOLD = 10
INCLUDE_DISTANCE = True       # default: test "with distance" models
INCLUDE_SEQUENCE = True
BALANCED = False
TRANS_TYPE = "ln_x_plus_one_trans"

# Some encodings were trained with a feature-count bug: despite having "_with_distance_"
# in the model filename, the actual model contains 63 features (kmer-only, no distance).
# For those encodings we must match the real training input (no distance feature).
# Keys are encoding names; value is the include_distance_feature to use for that encoding.
ENCODING_DISTANCE_OVERRIDE: dict = {
}

# Metrics to include in the summary tables (map internal column name → display name)
# Classifier results also include pearson/spearman (labels vs predicted proba) and
# reads-to-proba correlations for the positive set, mirroring regular_test_models behaviour.
CLF_METRIC_MAP = {
    "aupr":                                  "aupr",
    "auc":                                   "roc_auc",
    "accuracy":                              "accuracy",
    "precision":                             "precision",
    "recall":                                "recall",
    "f1_score":                              "f1_score",
    # "pearson":                               "pearson",
    # "spearman":                              "spearman",
    # "pearson_reads_to_proba_for_positive_set":  "pearson_reads_to_proba",
    # "spearman_reads_to_proba_for_positive_set": "spearman_reads_to_proba",
}

# Regression results also carry classification-proxy metrics (regression score used
# as a continuous classifier), mirroring regular_test_models behaviour.
REG_METRIC_MAP = {
    #"pearson":              "pearson",
    "pearson_after_inv_trans": "pearson",
    #"spearman":             "spearman",
    "spearman_after_inv_trans": "spearman",
    #"rmse":                 "rmse",
    "rmse_after_inv_trans": "rmse",
    # "reg_to_class_auc":     "reg_to_class_auc",
    # "reg_to_class_aupr":    "reg_to_class_aupr",
    # "reg_to_class_pearson": "reg_to_class_pearson",
    # "reg_to_class_spearman": "reg_to_class_spearman",
}

OUTPUT_DIR = Path(general_utilities.FILES_DIR) / "encoding_comparison"

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_changeseq_targets():
    """Load CHANGE-seq sgRNA ordering (always used for fold assignment)."""
    try:
        return load_order_sg_rnas()  # default = CHANGE-seq
    except FileNotFoundError:
        return order_sg_rnas()


def _load_dataset(data_type: str):
    """Return (positive_df, negative_df) for the given data type."""
    datasets_dir = Path(general_utilities.DATASETS_PATH) / "include_on_targets"
    positive_df = pd.read_csv(datasets_dir / f"{data_type}_positive.csv", index_col=0)
    negative_df = pd.read_csv(datasets_dir / f"{data_type}_negative.csv", index_col=0)
    negative_df = negative_df[negative_df["offtarget_sequence"].str.find("N") == -1]
    return positive_df, negative_df


def _include_distance(encoding: str) -> bool:
    """Return the effective include_distance_feature flag for a given encoding."""
    return ENCODING_DISTANCE_OVERRIDE.get(encoding, INCLUDE_DISTANCE)


def _get_backend_label(filename: str) -> str:
    """Extract backend label from filename (xgb, catboost, or decision_tree)."""
    filename_lower = filename.lower()
    if "catboost" in filename_lower:
        return "catboost"
    elif "xgb" in filename_lower:
        return "xgb"
    elif "decision_tree" in filename_lower or "dt" in filename_lower:
        return "decision_tree"
    else:
        return "unknown"


def _results_csv_path(encoding: str, data_type: str, model_type: str) -> Path:
    """Return the path where evaluation() saves its per-sgRNA results CSV."""
    path_prefix = f"CHANGEseq/include_on_targets/{model_type}/test_results_include_on_targets/"
    csv_path = extract_model_results_path(
        model_type, data_type, K_FOLD,
        _include_distance(encoding), INCLUDE_SEQUENCE,
        BALANCED, TRANS_TYPE,
        trans_all_fold=False, trans_only_positive=False,
        exclude_targets_without_positives=False,
        evaluate_only_distance=None, suffix_add="",
        path_prefix=path_prefix, encoding=encoding,
    )
    return Path(csv_path)


def _results_csv_path_for(encoding: str, data_type: str, model_type: str,
                          include_distance: bool = True, include_sequence: bool = True) -> Path:
    """Return the path for a specific model variant (e.g. '-seq-dist')."""
    path_prefix = f"CHANGEseq/include_on_targets/{model_type}/test_results_include_on_targets/"
    csv_path = extract_model_results_path(
        model_type, data_type, K_FOLD,
        include_distance, include_sequence,
        BALANCED, TRANS_TYPE,
        trans_all_fold=False, trans_only_positive=False,
        exclude_targets_without_positives=False,
        evaluate_only_distance=None, suffix_add="",
        path_prefix=path_prefix, encoding=encoding,
    )
    return Path(csv_path)


def _summarize(csv_path: Path, metric_map: dict) -> dict:
    """
    Load per-sgRNA results CSV and return {metric_mean: value, metric_std: value, ...}.
    Excludes the last aggregate row (consistent with summarize_encoding_results.py)
    so stats are computed per-sgRNA.
    """
    df = pd.read_csv(csv_path, index_col=0)
    # Exclude the final aggregate row (same approach as summarize_encoding_results.py)
    if len(df) > 0:
        df = df.iloc[:-1].copy()
    else:
        df = df.copy()

    row = {}
    for col_name, display_name in metric_map.items():
        if col_name not in df.columns:
            row[f"{display_name}_mean"] = np.nan
            row[f"{display_name}_std"] = np.nan
            continue
        values = pd.to_numeric(df[col_name], errors="coerce").dropna()
        row[f"{display_name}_mean"] = round(float(values.mean()), 4) if len(values) else np.nan
        row[f"{display_name}_std"]  = round(float(values.std()),  4) if len(values) else np.nan
    return row


# ── Main ──────────────────────────────────────────────────────────────────────

def run_all():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Models are trained on CHANGE-seq folds: always use CHANGE-seq sgRNA ordering
    # for fold assignment, exactly as regular_test_models does.
    targets_change_seq = _load_changeseq_targets()
    
    # Determine which backends to process
    backends_to_process = MODEL_BACKENDS if MODEL_BACKENDS is not None else ["xgb", "catboost", "decision_tree"]

    # Initialize collection structure for each backend
    if not hasattr(run_all, "_collected"):
        run_all._collected = {
            backend: {
                "classifier": {dt: [] for dt in DATA_TYPES},
                "regression_with_negatives": {dt: [] for dt in DATA_TYPES}
            }
            for backend in backends_to_process
        }

    for encoding in ENCODINGS:
        print(f"\n{'='*60}")
        print(f"Encoding: {encoding}")
        nucleotides_to_position_mapping = create_nucleotides_to_position_mapping(encoding=encoding)

        for data_type in DATA_TYPES:
            # We read precomputed result CSVs from the results directory for each encoding
            results_dir = Path("files") / "models_10_fold" / "new_models" / "CHANGEseq" / "include_on_targets" / "results" / encoding
            if not results_dir.exists():
                print(f"  WARNING: results dir not found for encoding {encoding}: {results_dir}")
                # append empty rows for all backends
                for backend in backends_to_process:
                    run_all._collected[backend]["classifier"][data_type].append({"encoding": encoding, "backend": backend})
                    run_all._collected[backend]["regression_with_negatives"][data_type].append({"encoding": encoding, "backend": backend})
                continue

            csv_files = list(results_dir.glob(f"{data_type}*.csv"))
            
            # Process each backend separately
            for backend in backends_to_process:
                print(f"  Processing backend: {backend} - {data_type}")
                
                # Select classifier file for this backend
                clf_candidates = [
                    p for p in csv_files 
                    if "classifier" in p.name.lower() 
                    and "with_distance" in p.name.lower()
                    and _get_backend_label(p.name) == backend
                ]
                if not clf_candidates:
                    clf_candidates = [
                        p for p in csv_files 
                        if "classifier" in p.name.lower()
                        and _get_backend_label(p.name) == backend
                    ]
                clf_csv = clf_candidates[0] if clf_candidates else None

                # Select regression file for this backend
                reg_candidates = [
                    p for p in csv_files 
                    if "regression" in p.name.lower() 
                    and "with_distance" in p.name.lower()
                    and _get_backend_label(p.name) == backend
                ]
                if not reg_candidates:
                    reg_candidates = [
                        p for p in csv_files 
                        if "regression" in p.name.lower()
                        and _get_backend_label(p.name) == backend
                    ]
                reg_csv = reg_candidates[0] if reg_candidates else None

                # Process classifier results
                if clf_csv is None:
                    print(f"    WARNING: no classifier CSV found for {encoding} / {data_type} / {backend}")
                    run_all._collected[backend]["classifier"][data_type].append({
                        "encoding": encoding, 
                        "backend": backend
                    })
                else:
                    clf_row = _summarize(clf_csv, CLF_METRIC_MAP)
                    clf_row["encoding"] = encoding
                    clf_row["backend"] = backend
                    run_all._collected[backend]["classifier"][data_type].append(clf_row)
                    print(f"    ✓ Loaded classifier: {clf_csv.name}")

                # Process regression results
                if reg_csv is None:
                    print(f"    WARNING: no regression CSV found for {encoding} / {data_type} / {backend}")
                    run_all._collected[backend]["regression_with_negatives"][data_type].append({
                        "encoding": encoding,
                        "backend": backend
                    })
                else:
                    reg_row = _summarize(reg_csv, REG_METRIC_MAP)
                    reg_row["encoding"] = encoding
                    reg_row["backend"] = backend
                    run_all._collected[backend]["regression_with_negatives"][data_type].append(reg_row)
                    print(f"    ✓ Loaded regression: {reg_csv.name}")

    # Build and save summary CSVs for each backend
    collected = getattr(run_all, "_collected", None)
    if collected is None:
        print("No results collected.")
        return

    for backend in backends_to_process:
        print(f"\n{'='*60}")
        print(f"Generating summaries for backend: {backend.upper()}")
        print(f"{'='*60}")
        
        for model_type in ["classifier", "regression_with_negatives"]:
            task_label = "classification" if model_type == "classifier" else "regression"
            for data_type in DATA_TYPES:
                rows = collected[backend][model_type][data_type]
                summary_df = pd.DataFrame(rows)
                
                # Put 'encoding' and 'backend' as first columns if present
                first_cols = [c for c in ["encoding", "backend"] if c in summary_df.columns]
                other_cols = [c for c in summary_df.columns if c not in first_cols]
                summary_df = summary_df[first_cols + other_cols]

                out_path = OUTPUT_DIR / f"summary_{task_label}_{data_type}_{backend}.csv"
                summary_df.to_csv(out_path, index=False)
                print(f"\nSaved: {out_path}")
                print(summary_df.to_string(index=False))
    
    # Also create combined summary with all backends
    print(f"\n{'='*60}")
    print(f"Generating combined summaries (all backends)")
    print(f"{'='*60}")
    
    for model_type in ["classifier", "regression_with_negatives"]:
        task_label = "classification" if model_type == "classifier" else "regression"
        for data_type in DATA_TYPES:
            all_rows = []
            for backend in backends_to_process:
                all_rows.extend(collected[backend][model_type][data_type])
            
            summary_df = pd.DataFrame(all_rows)
            
            # Put 'encoding' and 'backend' as first columns if present
            first_cols = [c for c in ["encoding", "backend"] if c in summary_df.columns]
            other_cols = [c for c in summary_df.columns if c not in first_cols]
            summary_df = summary_df[first_cols + other_cols]

            out_path = OUTPUT_DIR / f"summary_{task_label}_{data_type}_all_backends.csv"
            summary_df.to_csv(out_path, index=False)
            print(f"\nSaved: {out_path}")
            print(summary_df.to_string(index=False))


if __name__ == "__main__":
    run_all()
