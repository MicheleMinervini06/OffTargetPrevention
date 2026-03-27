# Statistical Testing Framework - Quick Guide

## ⚠️ TROUBLESHOOTING

Se tutti i risultati mostrano `NaN`, verifica:

1. **Working Directory**: Assicurati di eseguire gli script dalla root del progetto
   ```bash
   cd c:\Users\mikim\OneDrive\Desktop\Uni\Tesi\OffTargetPrevention
   python statistical_tests.py
   ```

2. **Test Path Access**: Esegui il test diagnostico
   ```bash
   python check_paths_step_by_step.py
   ```
   Questo mostrerà se i file vengono trovati correttamente.

3. **File Structure**: Verifica che la struttura sia:
   ```
   files/
     models_10_fold/
       new_models/
         CHANGEseq/
           include_on_targets/
             results/
               OneHot/
                 CHANGEseq_classifier_results_catboost_model_all_10_folds_with_distance_imbalanced_with_OneHotEncoding.csv
               NPM/
                 CHANGEseq_classifier_results_catboost_model_all_10_folds_with_distance_imbalanced.csv
               ...
   ```

---

## Quick Start

- **`statistical_tests.py`**: Main module with all statistical testing functions
- **`run_statistical_tests_example.py`**: Example script showing common use cases
- **`test_stat_paths.py`**: Quick test to verify file paths are correct

## Quick Start

### 1. Run Complete Analysis

```bash
python statistical_tests.py
```

This runs all comparisons and generates:
- `statistical_summary_classification.csv` - Summary for classification models
- `statistical_summary_regression.csv` - Summary for regression models

### 2. Run Specific Examples

```bash
python run_statistical_tests_example.py
```

This demonstrates:
- Backend comparison (XGBoost vs CatBoost)
- Encoding comparison (all encodings for each backend)
- Specific pairwise comparisons

### 3. Test File Paths

```bash
python test_stat_paths.py
```

Verifies that all result files can be found.

## Statistical Tests Used

### Wilcoxon Signed-Rank Test
- **Purpose**: Compare two methods on the same targets (paired samples)
- **Advantages**: Non-parametric, robust to outliers, doesn't assume normality
- **When**: Comparing XGBoost vs CatBoost for the same encoding

### Friedman Test
- **Purpose**: Compare >2 methods simultaneously (omnibus test)
- **Advantages**: Non-parametric version of repeated-measures ANOVA
- **When**: Comparing all encodings for a given backend

### Post-hoc Pairwise Comparisons
- **Purpose**: Identify which specific pairs differ after significant Friedman test
- **Correction**: Bonferroni correction for multiple comparisons
- **When**: Friedman test shows significant differences

## Key Metrics

### Classification Models
- **AUPR** (Area Under Precision-Recall curve) - Primary metric
- **AUC** (Area Under ROC curve)
- **Pearson** correlation
- **Spearman** correlation
- Accuracy, Precision, Recall, F1-score

### Regression Models
- **Pearson** (after inverse transformation) - Primary metric
- **Spearman** (after inverse transformation)
- **RMSE** (Root Mean Square Error)
- Metrics computed on positive samples only

## Configuration

### Available Encodings
- NPM (Neural Position Matrix)
- OneHot
- OneHot5Channel
- OneHotVstack
- kmer
- LabelEncodingPairwise
- bulges
- MM (Mismatch Matrix)

### Available Backends
- xgb (XGBoost)
- catboost (CatBoost)

### Datasets
- CHANGEseq
- GUIDEseq

### Model Types
- classifier
- regression_with_negatives

## Custom Usage

```python
from statistical_tests import (
    load_per_target_metrics,
    compare_two_methods,
    run_backend_comparison,
    run_encoding_comparison
)

# Load data for specific configuration
data = load_per_target_metrics(
    encoding='OneHot',
    model_backend='catboost',
    metric='aupr',
    data_type='CHANGEseq',
    model_type='classifier',
    with_distance=True
)

# Compare two specific methods
onehot_data = load_per_target_metrics('OneHot', 'catboost', 'aupr', 'CHANGEseq', 'classifier', True)
kmer_data = load_per_target_metrics('kmer', 'catboost', 'aupr', 'CHANGEseq', 'classifier', True)

result = compare_two_methods(
    onehot_data, kmer_data,
    'OneHot', 'kmer',
    'aupr'
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Winner: {result['winner']}")
```

## Output Files

### Summary CSVs
- **Backend comparison**: One row per encoding showing XGBoost vs CatBoost
- **Encoding comparison**: Pairwise comparisons between all encoding pairs

### Columns in Results
- `method1`, `method2`: Methods being compared
- `p_value`: Statistical significance
- `significant_0.05`, `significant_0.01`, `significant_0.001`: Significance levels
- `mean_method1`, `mean_method2`: Mean metric values
- `median_method1`, `median_method2`: Median metric values
- `winner`: Which method performs better (if significant)
- `cohens_d`: Effect size measure

## Interpretation Guidelines

### P-values
- **p < 0.001**: Very strong evidence (***) 
- **p < 0.01**: Strong evidence (**)
- **p < 0.05**: Moderate evidence (*)
- **p ≥ 0.05**: No significant difference (ns)

### Effect Size (Cohen's d)
- **|d| < 0.2**: Small effect
- **0.2 ≤ |d| < 0.5**: Medium effect
- **|d| ≥ 0.5**: Large effect

### Multiple Comparisons
When comparing many encodings, use the **Bonferroni-corrected p-values** to control false discovery rate.

## Troubleshooting

### Files Not Found
Check that:
1. Result files exist in `files/models_10_fold/new_models/{dataset}/include_on_targets/results/{encoding}/`
2. File naming matches expected pattern
3. Run `test_stat_paths.py` to verify paths

### No Significant Results
This may indicate:
- True lack of difference between methods
- Small sample size (only ~14-18 targets)
- High variance within methods
- Need for larger effect sizes to detect differences

### NaN Values in Results
Occurs when:
- Files are missing for that configuration
- Metrics are not computed for that model type
- Data loading failed (check warnings)

## Citation

If you use this statistical testing framework in your research, please cite the original work.

## Contact

For questions or issues, please contact the project maintainer.
