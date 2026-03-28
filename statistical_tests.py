"""
Statistical Testing Framework for Comparing Encodings and Model Backends

This module provides statistical tests to compare:
1. Model backends pairwise for each encoding (XGBoost, CatBoost, Decision Tree)
2. Different encodings within the same backend
3. Generate comprehensive reports with significance levels

Tests used:
- Wilcoxon Signed-Rank Test: for paired comparisons (same sgRNA targets)
- Friedman Test: for comparing >2 encodings simultaneously
- Bonferroni correction: for multiple comparisons
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, shapiro
import itertools
from pathlib import Path
import os
from typing import Dict, List, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


def normalize_backend_name(backend: str) -> str:
    """Normalize backend aliases to the filename convention used in results."""
    backend_norm = backend.lower()
    if backend_norm in ("xgboost", "xgb"):
        return "xgb"
    if backend_norm in ("decisiontree", "decision-tree", "decision_tree"):
        return "decision_tree"
    if backend_norm == "catboost":
        return "catboost"
    return backend_norm


def backend_display_name(backend: str) -> str:
    """Pretty display names for reports."""
    backend_norm = normalize_backend_name(backend)
    if backend_norm == "xgb":
        return "XGBoost"
    if backend_norm == "catboost":
        return "CatBoost"
    if backend_norm == "decision_tree":
        return "DecisionTree"
    return backend


def get_encoding_suffix(encoding: str) -> str:
    """
    Get the encoding suffix based on the encoding type.
    Matches the logic from utilities.prefix_and_suffix_path()
    """
    if encoding == "NPM":
        return ""  # NPM has no encoding suffix
    elif encoding == "OneHot":
        return "_with_OneHotEncoding"
    elif encoding == "OneHot5Channel":
        return "_with_OneHotEncoding5Channel"
    elif encoding == "kmer":
        return "_with_kmerEncoding"
    elif encoding == "OneHotVstack":
        return "_with_OneHotEncodingVstack"
    elif encoding == "LabelEncodingPairwise":
        return "_with_LabelEncodingPairwise"
    elif encoding == "bulges":
        return "_with_bulgesEncoding"
    elif encoding == "MM":
        return "_with_MMEncoding"
    else:
        return f"_with_{encoding}Encoding"


def load_per_target_metrics(
    encoding: str,
    model_backend: str,
    metric: str,
    data_type: str = 'CHANGEseq',
    model_type: str = 'classifier',
    with_distance: bool = True
) -> Optional[np.ndarray]:
    """
    Load per-target metrics from CSV file.
    Excludes the last row (All Targets aggregate).
    
    Parameters:
    -----------
    encoding : str
        Encoding type (OneHot, kmer, bulges, etc.)
    model_backend : str
        Model backend (xgb, catboost, or decision_tree)
    metric : str
        Metric name (aupr, accuracy, pearson, etc.)
    data_type : str
        Dataset type (CHANGEseq or GUIDEseq)
    model_type : str
        Model type (classifier or regression_with_negatives)
    with_distance : bool
        Whether distance feature is included
    
    Returns:
    --------
    np.ndarray or None
        Array of metric values per target, or None if file not found
    """
    model_backend = normalize_backend_name(model_backend)

    # Build filename based on actual file naming convention
    # Format: {data_type}_{model_type}_results_{backend}_model_all_10_folds[_with_distance]_imbalanced{encoding_suffix}.csv
    
    distance_str = "_with_distance" if with_distance else ""
    encoding_suffix = get_encoding_suffix(encoding)
    
    filename = (f"{data_type}_{model_type}_results_{model_backend}_model_"
               f"all_10_folds{distance_str}_imbalanced{encoding_suffix}.csv")
    
    # Use os.path.join for Windows compatibility
    base_path = os.path.join("files", "models_10_fold", "new_models", data_type, 
                              "include_on_targets", "results", encoding)
    file_path = os.path.join(base_path, filename)
    
    if not os.path.exists(file_path):
        print(f"Warning: File not found: {file_path}")
        return None
    
    try:
        df = pd.read_csv(file_path)
        # Remove the last row (All Targets aggregate)
        df = df.iloc[:-1]
        
        if metric not in df.columns:
            print(f"Warning: Metric '{metric}' not found in {file_path}")
            return None
        
        return df[metric].values
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def check_normality(data: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
    """
    Check if data is normally distributed using Shapiro-Wilk test.
    
    Returns:
    --------
    (is_normal, p_value)
    """
    if len(data) < 3:
        return False, 0.0
    
    statistic, p_value = shapiro(data)
    is_normal = p_value > alpha
    return is_normal, p_value


def compare_two_methods(
    method1_data: np.ndarray,
    method2_data: np.ndarray,
    method1_name: str,
    method2_name: str,
    metric_name: str,
    test_type: str = 'wilcoxon'
) -> Dict:
    """
    Compare two methods using paired statistical tests.
    
    Parameters:
    -----------
    method1_data : np.ndarray
        Metric values for method 1
    method2_data : np.ndarray
        Metric values for method 2
    method1_name : str
        Name of method 1
    method2_name : str
        Name of method 2
    metric_name : str
        Name of the metric being compared
    test_type : str
        Type of test ('wilcoxon' or 'paired_ttest')
    
    Returns:
    --------
    dict : Results dictionary with statistics and p-value
    """
    if len(method1_data) != len(method2_data):
        raise ValueError("Data arrays must have the same length")
    
    results = {
        'method1': method1_name,
        'method2': method2_name,
        'metric': metric_name,
        'n_targets': len(method1_data),
        'mean_method1': np.nanmean(method1_data),
        'mean_method2': np.nanmean(method2_data),
        'median_method1': np.nanmedian(method1_data),
        'median_method2': np.nanmedian(method2_data),
        'std_method1': np.nanstd(method1_data),
        'std_method2': np.nanstd(method2_data),
        'mean_diff': np.nanmean(method1_data - method2_data),
        'median_diff': np.nanmedian(method1_data - method2_data),
    }
    
    # Remove pairs where either value is NaN before statistical testing
    valid_mask = ~(np.isnan(method1_data) | np.isnan(method2_data))
    method1_clean = method1_data[valid_mask]
    method2_clean = method2_data[valid_mask]
    
    results['n_valid_pairs'] = len(method1_clean)
    results['n_nan_removed'] = len(method1_data) - len(method1_clean)
    
    # Check normality on clean data
    is_normal1, p_norm1 = check_normality(method1_clean)
    is_normal2, p_norm2 = check_normality(method2_clean)
    results['normality_method1'] = is_normal1
    results['normality_method2'] = is_normal2
    
    if test_type == 'wilcoxon':
        try:
            if len(method1_clean) < 3:
                print(f"Warning: Too few valid pairs ({len(method1_clean)}) for Wilcoxon test")
                results['test'] = 'Insufficient data'
                p_value = np.nan
                results['statistic'] = np.nan
            else:
                statistic, p_value = wilcoxon(
                    method1_clean, method2_clean,
                    alternative='two-sided',
                    zero_method='wilcox'
                )
                results['test'] = 'Wilcoxon'
                results['statistic'] = statistic
        except Exception as e:
            print(f"Wilcoxon test failed: {e}")
            results['test'] = 'Failed'
            p_value = np.nan
            results['statistic'] = np.nan
            
    elif test_type == 'paired_ttest':
        if len(method1_clean) < 3:
            print(f"Warning: Too few valid pairs ({len(method1_clean)}) for t-test")
            results['test'] = 'Insufficient data'
            p_value = np.nan
            results['statistic'] = np.nan
        else:
            statistic, p_value = stats.ttest_rel(method1_clean, method2_clean)
            results['test'] = 'Paired t-test'
            results['statistic'] = statistic
    
    results['p_value'] = p_value
    results['significant_0.05'] = p_value < 0.05
    results['significant_0.01'] = p_value < 0.01
    results['significant_0.001'] = p_value < 0.001
    
    # Effect size (Cohen's d for paired samples)
    diff = method1_data - method2_data
    # Remove NaN values for Cohen's d calculation
    diff_clean = diff[~np.isnan(diff)]
    if len(diff_clean) > 0 and np.nanstd(diff_clean) > 0:
        results['cohens_d'] = np.nanmean(diff_clean) / np.nanstd(diff_clean)
    else:
        results['cohens_d'] = 0
    
    # Winner
    if p_value < 0.05:
        results['winner'] = method1_name if np.nanmean(method1_data) > np.nanmean(method2_data) else method2_name
    else:
        results['winner'] = 'No significant difference'
    
    return results


def compare_multiple_methods(
    methods_dict: Dict[str, np.ndarray],
    method_names: List[str],
    metric_name: str
) -> Dict:
    """
    Compare multiple encodings using Friedman test with post-hoc pairwise comparisons.
    
    Parameters:
    -----------
    methods_dict : dict
        Dictionary mapping method names to their metric arrays
    method_names : list
        List of method names to compare
    metric_name : str
        Name of the metric being compared
    
    Returns:
    --------
    dict : Results with Friedman test and pairwise comparisons
    """
    # Prepare data for Friedman test
    data_arrays = [methods_dict[name] for name in method_names]
    
    # Check that all arrays have the same length
    lengths = [len(arr) for arr in data_arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"All methods must have the same number of targets. Found: {lengths}")
    
    # Remove rows where ANY method has NaN (Friedman requires complete cases)
    # Stack all arrays into a matrix and find rows with no NaN
    data_matrix = np.column_stack(data_arrays)
    valid_rows = ~np.isnan(data_matrix).any(axis=1)
    
    if valid_rows.sum() < 3:
        print(f"Warning: Too few complete cases ({valid_rows.sum()}) for Friedman test")
        return {
            'metric': metric_name,
            'n_methods': len(method_names),
            'n_targets': len(data_arrays[0]),
            'n_valid_cases': int(valid_rows.sum()),
            'friedman_statistic': np.nan,
            'friedman_p_value': np.nan,
            'significant': False,
            'error': 'Insufficient complete cases'
        }
    
    # Filter all arrays to keep only valid rows
    data_arrays_clean = [arr[valid_rows] for arr in data_arrays]
    
    # Friedman test
    try:
        statistic, p_value = friedmanchisquare(*data_arrays_clean)
    except Exception as e:
        print(f"Friedman test failed: {e}")
        return {
            'metric': metric_name,
            'n_methods': len(method_names),
            'n_targets': len(data_arrays[0]),
            'n_valid_cases': int(valid_rows.sum()),
            'friedman_statistic': np.nan,
            'friedman_p_value': np.nan,
            'significant': False,
            'error': str(e)
        }
    
    results = {
        'metric': metric_name,
        'n_methods': len(method_names),
        'n_targets': len(data_arrays[0]),
        'n_valid_cases': int(valid_rows.sum()),
        'n_nan_removed': len(data_arrays[0]) - int(valid_rows.sum()),
        'friedman_statistic': statistic,
        'friedman_p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Mean and median for each method
    for name in method_names:
        results[f'mean_{name}'] = np.nanmean(methods_dict[name])
        results[f'median_{name}'] = np.nanmedian(methods_dict[name])
    
    # If significant, perform post-hoc pairwise comparisons
    if p_value < 0.05:
        pairwise_results = []
        pairs = list(itertools.combinations(method_names, 2))
        n_comparisons = len(pairs)
        bonferroni_alpha = 0.05 / n_comparisons
        
        for m1, m2 in pairs:
            try:
                # Remove NaN pairs
                valid_mask = ~(np.isnan(methods_dict[m1]) | np.isnan(methods_dict[m2]))
                m1_clean = methods_dict[m1][valid_mask]
                m2_clean = methods_dict[m2][valid_mask]
                
                if len(m1_clean) < 3:
                    print(f"  Warning: Skipping {m1} vs {m2} - too few valid pairs ({len(m1_clean)})")
                    continue
                
                stat, p = wilcoxon(m1_clean, m2_clean)
                
                # Determine winner
                mean_diff = np.nanmean(methods_dict[m1]) - np.nanmean(methods_dict[m2])
                winner = m1 if mean_diff > 0 else m2
                
                pairwise_results.append({
                    'method1': m1,
                    'method2': m2,
                    'p_value': p,
                    'p_value_bonferroni': min(p * n_comparisons, 1.0),  # Bonferroni correction
                    'significant_0.05': p < 0.05,
                    'significant_bonferroni': p < bonferroni_alpha,
                    'mean_diff': mean_diff,
                    'median_diff': np.nanmedian(methods_dict[m1] - methods_dict[m2]),
                    'winner': winner if p < 0.05 else 'No diff'
                })
            except Exception as e:
                print(f"Pairwise comparison {m1} vs {m2} failed: {e}")
        
        results['pairwise_comparisons'] = pd.DataFrame(pairwise_results)
    
    return results


def run_backend_comparison(
    encodings: List[str],
    backends: List[str],
    metric: str,
    data_type: str = 'CHANGEseq',
    model_type: str = 'classifier',
    with_distance: bool = True
) -> pd.DataFrame:
    """
    Compare backends pairwise for each encoding.
    
    Returns:
    --------
    pd.DataFrame : Results for all encoding comparisons
    """
    results_list = []
    backends_norm = [normalize_backend_name(backend) for backend in backends]
    backends_norm = list(dict.fromkeys(backends_norm))

    if len(backends_norm) < 2:
        print("Warning: Need at least 2 backends for backend comparison")
        return pd.DataFrame(results_list)
    
    print(f"\n{'='*80}")
    print(f"Backend Comparison (Pairwise) - {data_type} - {model_type}")
    print(f"Metric: {metric} | Distance: {with_distance}")
    print(f"Backends: {', '.join(backend_display_name(b) for b in backends_norm)}")
    print(f"{'='*80}\n")
    
    for encoding in encodings:
        backend_data = {}
        for backend in backends_norm:
            data = load_per_target_metrics(
                encoding, backend, metric, data_type, model_type, with_distance
            )
            if data is not None:
                backend_data[backend] = data

        if len(backend_data) < 2:
            continue

        backend_pairs = list(itertools.combinations(backend_data.keys(), 2))
        n_comparisons = len(backend_pairs)
        bonferroni_alpha = 0.05 / n_comparisons if n_comparisons > 0 else 0.05

        for b1, b2 in backend_pairs:
            result = compare_two_methods(
                backend_data[b1], backend_data[b2],
                backend_display_name(b1),
                backend_display_name(b2),
                metric
            )
            result['encoding'] = encoding
            result['backend1'] = b1
            result['backend2'] = b2
            if pd.isna(result['p_value']):
                result['p_value_bonferroni'] = np.nan
                result['significant_bonferroni'] = False
            else:
                result['p_value_bonferroni'] = min(result['p_value'] * n_comparisons, 1.0)
                result['significant_bonferroni'] = result['p_value'] < bonferroni_alpha
            results_list.append(result)

            p_val = result['p_value']
            if pd.isna(p_val):
                marker = 'na'
                p_print = 'nan'
            else:
                marker = '***' if result['significant_0.001'] else '**' if result['significant_0.01'] else '*' if result['significant_0.05'] else 'ns'
                p_print = f"{p_val:.4f}"
            print(
                f"{encoding:20s} | {result['method1']}: {result['mean_method1']:.4f} "
                f"vs {result['method2']}: {result['mean_method2']:.4f} "
                f"| p={p_print} {marker} | Winner: {result['winner']}"
            )
    
    return pd.DataFrame(results_list)


def run_encoding_comparison(
    encodings: List[str],
    backend: str,
    metric: str,
    data_type: str = 'CHANGEseq',
    model_type: str = 'classifier',
    with_distance: bool = True
) -> Dict:
    """
    Compare multiple encodings for a given backend using Friedman test.
    
    Returns:
    --------
    dict : Friedman test results and pairwise comparisons
    """
    methods_dict = {}
    
    for encoding in encodings:
        data = load_per_target_metrics(
            encoding, backend, metric, data_type, model_type, with_distance
        )
        if data is not None:
            methods_dict[encoding] = data
    
    if len(methods_dict) < 2:
        print(f"Not enough encodings with data for {backend}")
        return {}
    
    print(f"\n{'='*80}")
    print(f"Encoding Comparison - {backend.upper()} - {data_type} - {model_type}")
    print(f"Metric: {metric} | Distance: {with_distance}")
    print(f"{'='*80}\n")
    
    # Print descriptive statistics
    for enc in methods_dict:
        print(f"{enc:20s} | Mean: {np.nanmean(methods_dict[enc]):.4f} ± {np.nanstd(methods_dict[enc]):.4f} "
              f"| Median: {np.nanmedian(methods_dict[enc]):.4f}")
    
    if len(methods_dict) > 2:
        result = compare_multiple_methods(
            methods_dict,
            list(methods_dict.keys()),
            metric
        )
        
        print(f"\nFriedman Test: χ² = {result.get('friedman_statistic', np.nan):.4f}, "
              f"p = {result.get('friedman_p_value', np.nan):.4f}")
        
        if result.get('significant', False) and 'pairwise_comparisons' in result:
            print("\nPairwise Comparisons (Bonferroni corrected):")
            print("-" * 100)
            df = result['pairwise_comparisons']
            for _, row in df.iterrows():
                sig_marker = '***' if row['p_value'] < 0.001 else '**' if row['p_value'] < 0.01 else '*' if row['p_value'] < 0.05 else 'ns'
                print(f"{row['method1']:20s} vs {row['method2']:20s} | "
                      f"p={row['p_value']:.4f} (adj: {row['p_value_bonferroni']:.4f}) {sig_marker:3s} | "
                      f"Winner: {row['winner']}")
        
        return result
    
    elif len(methods_dict) == 2:
        # Just two encodings, use pairwise comparison
        enc1, enc2 = list(methods_dict.keys())
        result = compare_two_methods(
            methods_dict[enc1], methods_dict[enc2],
            enc1, enc2, metric
        )
        print(f"\nWilcoxon Test: p = {result['p_value']:.4f} | Winner: {result['winner']}")
        return {'pairwise_result': result}
    
    return {}


def generate_summary_table(
    encodings: List[str],
    backends: List[str],
    metrics: List[str],
    data_types: List[str] = ['CHANGEseq', 'GUIDEseq'],
    model_type: str = 'classifier',
    with_distance: bool = True,
    output_file: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a comprehensive summary table with all comparisons.
    
    Returns:
    --------
    pd.DataFrame : Summary table
    """
    summary_rows = []
    
    for data_type in data_types:
        for metric in metrics:
            # Backend comparison
            backend_df = run_backend_comparison(
                encodings, backends, metric, data_type, model_type, with_distance
            )
            
            if not backend_df.empty:
                for _, row in backend_df.iterrows():
                    summary_rows.append({
                        'data_type': data_type,
                        'metric': metric,
                        'comparison_type': 'Backend_pairwise',
                        'encoding': row['encoding'],
                        'method1': row['method1'],
                        'method2': row['method2'],
                        'p_value': row['p_value'],
                        'p_value_bonferroni': row.get('p_value_bonferroni', np.nan),
                        'winner': row['winner'],
                        'mean_method1': row['mean_method1'],
                        'mean_method2': row['mean_method2'],
                        'significant': row['significant_0.05'],
                        'significant_bonferroni': row.get('significant_bonferroni', False)
                    })
            
            # Encoding comparison for each backend
            for backend in backends:
                enc_result = run_encoding_comparison(
                    encodings, backend, metric, data_type, model_type, with_distance
                )
                
                if 'pairwise_comparisons' in enc_result:
                    for _, row in enc_result['pairwise_comparisons'].iterrows():
                        summary_rows.append({
                            'data_type': data_type,
                            'metric': metric,
                            'comparison_type': f'Encoding_{backend}',
                            'encoding': 'Multiple',
                            'method1': row['method1'],
                            'method2': row['method2'],
                            'p_value': row['p_value'],
                            'p_value_bonferroni': row.get('p_value_bonferroni', np.nan),
                            'winner': row['winner'],
                            'mean_method1': np.nan,
                            'mean_method2': np.nan,
                            'significant': row['significant_0.05'],
                            'significant_bonferroni': row.get('significant_bonferroni', False)
                        })
                elif 'pairwise_result' in enc_result:
                    row = enc_result['pairwise_result']
                    summary_rows.append({
                        'data_type': data_type,
                        'metric': metric,
                        'comparison_type': f'Encoding_{backend}',
                        'encoding': 'Pairwise',
                        'method1': row['method1'],
                        'method2': row['method2'],
                        'p_value': row['p_value'],
                        'p_value_bonferroni': np.nan,
                        'winner': row['winner'],
                        'mean_method1': row['mean_method1'],
                        'mean_method2': row['mean_method2'],
                        'significant': row['significant_0.05'],
                        'significant_bonferroni': False
                    })
    
    summary_df = pd.DataFrame(summary_rows)
    
    if output_file:
        summary_df.to_csv(output_file, index=False)
        print(f"\nSummary table saved to: {output_file}")
    
    return summary_df


def main():
    """
    Main function to run all statistical tests.
    """
    # Define parameters
    encodings = ['NPM', 'OneHot', 'OneHot5Channel', 'OneHotVstack', 'kmer', 
                 'LabelEncodingPairwise', 'bulges', 'MM']
    backends = ['xgb', 'catboost', 'decision_tree']
    
    # Classification metrics
    classification_metrics = ['aupr', 'auc', 'accuracy', 'pearson', 'spearman', 
                             'precision', 'recall', 'f1_score']
    
    # Regression metrics
    regression_metrics = ['pearson', 'spearman', 'rmse', 
                         'pearson_after_inv_trans', 'spearman_after_inv_trans']
    
    print("\n" + "="*80)
    print("STATISTICAL TESTING FRAMEWORK - OFF-TARGET PREDICTION")
    print("="*80 + "\n")
    
    # Run classification tests
    print("\n" + "#"*80)
    print("# CLASSIFICATION MODELS")
    print("#"*80 + "\n")
    
    summary_class = generate_summary_table(
        encodings=encodings,
        backends=backends,
        metrics=['aupr', 'auc', 'pearson', 'spearman'],  # Key metrics
        data_types=['CHANGEseq', 'GUIDEseq'],
        model_type='classifier',
        with_distance=True,
        output_file='statistical_summary_classification.csv'
    )
    
    # Run regression tests
    print("\n" + "#"*80)
    print("# REGRESSION MODELS")
    print("#"*80 + "\n")
    
    summary_reg = generate_summary_table(
        encodings=encodings,
        backends=backends,
        metrics=['pearson_after_inv_trans', 'spearman_after_inv_trans'],  # Key metrics
        data_types=['CHANGEseq', 'GUIDEseq'],
        model_type='regression_with_negatives',
        with_distance=True,
        output_file='statistical_summary_regression.csv'
    )
    
    print("\n" + "="*80)
    print("STATISTICAL TESTING COMPLETED")
    print("="*80)
    print(f"\nTotal comparisons performed: {len(summary_class) + len(summary_reg)}")
    print(f"Significant results (p < 0.05): {(summary_class['significant'].sum() + summary_reg['significant'].sum())}")
    
    # Generate winners table
    print("\n" + "="*80)
    print("TOP PERFORMING METHODS")
    print("="*80 + "\n")
    
    # Find best encoding per metric (classification)
    for metric in ['aupr', 'pearson']:
        print(f"\n{metric.upper()} - CHANGEseq Classifier:")
        for backend in backends:
            means = {}
            for enc in encodings:
                data = load_per_target_metrics(enc, backend, metric, 'CHANGEseq', 'classifier', True)
                if data is not None:
                    means[enc] = np.nanmean(data)
            
            if len(means) > 0:
                best_enc = max(means, key=means.get)
                print(f"  {backend.upper():10s}: {best_enc:20s} (mean = {means[best_enc]:.4f})")


if __name__ == "__main__":
    main()
