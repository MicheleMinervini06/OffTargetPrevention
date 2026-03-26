"""
Summarize best AUPR and Pearson results per encoding.

For each encoding folder under:
  files/models_10_fold/new_models/CHANGEseq/include_on_targets/results/

Reads the 8 result CSV files (4 models × 2 datasets) and computes, for each
file, the MEAN across all per-target rows (excluding the last "All Targets"
aggregate row — consistent with how plot_data.bar_plot works).

Metrics extracted:
  - AUPR (classification task): `aupr` for classifier files,
                                 `reg_to_class_aupr` for regression files
  - Pearson (regression task):   `pearson_reads_to_proba_for_positive_set`
                                  for classifier files,
                                 `pearson_only_positives_after_inv_trans`
                                  for regression files

Both metrics are computed for CHANGE-seq and GUIDE-seq independently.
The best mean value across all 4 models is reported, along with which model
achieved it.

Output: files/encoding_comparison/summary_best_results.csv
"""

import os
import glob
import pandas as pd

RESULTS_DIR = os.path.join(
    "files", "models_10_fold", "new_models",
    "CHANGEseq", "include_on_targets", "results"
)

OUTPUT_PATH = os.path.join("files", "encoding_comparison", "summary_best_results.csv")

# ── helpers ──────────────────────────────────────────────────────────────────

def get_model_label(filename: str) -> str:
    """Return a human-readable model label from the CSV filename."""
    name = os.path.basename(filename).lower()
    is_dist = "with_distance" in name
    if "classifier" in name:
        return "Classification-seq-dist" if is_dist else "Classification-seq"
    else:  # regression_with_negatives
        return "Regression-seq-dist" if is_dist else "Regression-seq"


def is_classifier_file(filename: str) -> bool:
    return "classifier" in os.path.basename(filename).lower()


def extract_metrics(csv_path: str):
    """
    Return (aupr_mean, pearson_mean, model_label) computed as the mean over
    all per-target rows (iloc[:-1], excluding the last "All Targets" row),
    consistent with how plot_data.bar_plot represents performance.
    """
    label = get_model_label(csv_path)
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  WARNING: could not read {csv_path}: {e}")
        return None, None, label

    # Drop the last row ("All Targets" aggregate) — same as iloc[:-1] in bar_plot
    per_target = df.iloc[:-1]

    if per_target.empty:
        print(f"  WARNING: no per-target rows in {csv_path}")
        return None, None, label

    if is_classifier_file(csv_path):
        aupr_col    = "aupr"
        pearson_col = "pearson_reads_to_proba_for_positive_set"
    else:
        aupr_col    = "reg_to_class_aupr"
        pearson_col = "pearson_only_positives_after_inv_trans"

    aupr_val    = per_target[aupr_col].mean()    if aupr_col    in per_target.columns else None
    pearson_val = per_target[pearson_col].mean() if pearson_col in per_target.columns else None

    return (
        float(aupr_val)    if aupr_val    is not None and pd.notna(aupr_val)    else None,
        float(pearson_val) if pearson_val is not None and pd.notna(pearson_val) else None,
        label,
    )


def best_across_models(values_labels: list[tuple]) -> tuple:
    """
    Given a list of (value, model_label), return (best_value, best_model_label).
    Ignores None values.
    """
    valid = [(v, lbl) for v, lbl in values_labels if v is not None]
    if not valid:
        return None, None
    return max(valid, key=lambda x: x[0])


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(RESULTS_DIR):
        print(f"ERROR: results directory not found: {RESULTS_DIR}")
        return

    encoding_dirs = sorted(
        d for d in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, d))
    )

    if not encoding_dirs:
        print("No encoding subdirectories found.")
        return

    print(f"Found {len(encoding_dirs)} encoding(s): {', '.join(encoding_dirs)}\n")

    rows = []

    for encoding in encoding_dirs:
        enc_dir = os.path.join(RESULTS_DIR, encoding)
        csv_files = glob.glob(os.path.join(enc_dir, "*.csv"))

        if not csv_files:
            print(f"[{encoding}] No CSV files found, skipping.")
            continue

        print(f"[{encoding}] Processing {len(csv_files)} file(s)...")

        # Separate by dataset
        change_files = [f for f in csv_files if os.path.basename(f).upper().startswith("CHANGESEQ")]
        guide_files  = [f for f in csv_files if os.path.basename(f).upper().startswith("GUIDESEQ")]

        # ── CHANGE-seq ──────────────────────────────────────────────────────
        change_aupr_candidates    = []
        change_pearson_candidates = []
        for fp in change_files:
            aupr, pearson, label = extract_metrics(fp)
            change_aupr_candidates.append((aupr, label))
            change_pearson_candidates.append((pearson, label))

        best_change_aupr,    best_change_aupr_model    = best_across_models(change_aupr_candidates)
        best_change_pearson, best_change_pearson_model = best_across_models(change_pearson_candidates)

        # ── GUIDE-seq ───────────────────────────────────────────────────────
        guide_aupr_candidates    = []
        guide_pearson_candidates = []
        for fp in guide_files:
            aupr, pearson, label = extract_metrics(fp)
            guide_aupr_candidates.append((aupr, label))
            guide_pearson_candidates.append((pearson, label))

        best_guide_aupr,    best_guide_aupr_model    = best_across_models(guide_aupr_candidates)
        best_guide_pearson, best_guide_pearson_model = best_across_models(guide_pearson_candidates)

        rows.append({
            "encoding":                    encoding,
            "CHANGE_best_aupr":            best_change_aupr,
            "CHANGE_best_aupr_model":      best_change_aupr_model,
            "GUIDE_best_aupr":             best_guide_aupr,
            "GUIDE_best_aupr_model":       best_guide_aupr_model,
            "CHANGE_best_pearson":         best_change_pearson,
            "CHANGE_best_pearson_model":   best_change_pearson_model,
            "GUIDE_best_pearson":          best_guide_pearson,
            "GUIDE_best_pearson_model":    best_guide_pearson_model,
        })

        print(
            f"  CHANGE → AUPR: {best_change_aupr:.4f} ({best_change_aupr_model})  |  "
            f"Pearson: {best_change_pearson:.4f} ({best_change_pearson_model})"
            if best_change_aupr is not None else f"  CHANGE → no data"
        )
        print(
            f"  GUIDE  → AUPR: {best_guide_aupr:.4f} ({best_guide_aupr_model})  |  "
            f"Pearson: {best_guide_pearson:.4f} ({best_guide_pearson_model})"
            if best_guide_aupr is not None else f"  GUIDE  → no data"
        )

    if not rows:
        print("No results collected.")
        return

    summary_df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    summary_df.to_csv(OUTPUT_PATH, index=False, float_format="%.6f")
    print(f"\nSummary saved to: {OUTPUT_PATH}")
    print("\n" + summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
