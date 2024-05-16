import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter_plot(csv1, csv2, title, xlabel, ylabel, metric, task):
    # Load the data from the first CSV file
    df1 = pd.read_csv(csv1)

    # Load the data from the second CSV file
    df2 = pd.read_csv(csv2)

    X = df1[metric]
    Y = df2[metric]

    # Calculate mean values
    mean_pearsons_df1 = X.mean()
    mean_pearsons_df2 = Y.mean()

    # Calculate median values
    medidan_pearsons_df1 = X.median()
    medidan_pearsons_df2 = Y.median()

    # Plotting
    plt.figure(figsize=(8, 6))
    sp = plt.scatter(X, Y, c=np.log(df1['positives']), cmap='viridis', alpha=0.5)
    plt.plot([-1, 1], [-1, 1], color='black', linestyle='dotted')  # Dotted line from (-1, -1) to (1, 1)

    # Color bar
    cbar = plt.colorbar(sp)
    cbar.set_label('log(positives)')

    plt.title(title)
    plt.legend(
        ['Pearson\n' f'X - Mean: {mean_pearsons_df1:.3f}, Median: {medidan_pearsons_df1:.3f}\n'
         f'Y - Mean: {mean_pearsons_df2:.3f}, Median:{medidan_pearsons_df2:.3f}'], loc='best')
    plt.xlabel("X - " + xlabel)
    plt.ylabel("Y - " + ylabel)
    plt.grid(False)
    if task == "classification":
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.xlim(-1, 1)  # Set x-axis limits
        plt.ylim(-1, 1)  # Set y-axis limits
    plt.show()


def bar_plot(csv1, csv2, csv3, csv4, labels, task, title):
    # Read data from the csv files
    x1 = pd.read_csv(csv1)
    x2 = pd.read_csv(csv2)
    x3 = pd.read_csv(csv3)
    x4 = pd.read_csv(csv4)

    # Using the correct column names from your dataset
    if task == "classification":
        x1_aupr = x1['aupr'].rename(labels[0])
        x2_aupr = x2['aupr'].rename(labels[1])
        x3_aupr = x3['reg_to_class_aupr'].rename(labels[2])
        x4_aupr = x4['reg_to_class_aupr'].rename(labels[3])
    else:
        x1_aupr = x1['pearson'].rename(labels[0])
        x2_aupr = x2['pearson'].rename(labels[1])
        x3_aupr = x3['pearson'].rename(labels[2])
        x4_aupr = x4['pearson'].rename(labels[3])

    data = pd.concat([x1_aupr, x2_aupr, x3_aupr, x4_aupr], axis=1)

    # Create the boxplot
    sns.boxplot(data=data, gap=0, width=0.5, showmeans=True, meanline=False,
                linecolor='#808080', linewidth=2.5,
                palette="pastel",
                meanprops={"markerfacecolor": "black", "markeredgecolor": "black"},
                flierprops={"markerfacecolor": "black"},
                )

    # Annotate the mean value
    for i, column in enumerate(data.columns):
        mean_val = data[column].mean()
        plt.text(i, mean_val, f'{mean_val:.3f}', color='black', ha='center', va='bottom', weight='bold')

    # Annotate the median value
    for i, column in enumerate(data.columns):
        median_val = data[column].median()
        right_offset = 0.4
        plt.text(i + right_offset, median_val, f'{median_val:.3f}', color='#808080', rotation='vertical', ha='right',
                 va='center', weight='bold')

    # # Calculate means and standard deviations
    means = data.mean()
    stds = data.std()

    n_groups = len(data.columns)

    # Add error bars for 1 standard deviation
    for i in range(n_groups):
        # Determine the x positions for the error bars
        # This position is the center of each boxplot.
        x = i - 0.4

        # Calculate the error bar positions
        mean = means[i]
        error = stds[i]

        # Draw error bars (1 SD)
        plt.errorbar(x, mean, yerr=error, color='black', capsize=4)

    # # Adjust these values based on your dataset
    # max_ylim = max([max(group) for group in data]) + 0.1
    # add_stat_annotation(0, 1, max_ylim, 0.02, '****')
    # add_stat_annotation(2, 3, max_ylim, 0.02, '****')

    if task == "classification":
        plt.ylim(0, 1.2)
        plt.ylabel('AUPR')
    else:
        plt.ylim(-1, 1.5)
        plt.ylabel('Pearson')
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=30)

    plt.show()


def main():
    # Read counts log transformation improves
    # prediction performance

    # # C
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets/CHANGE_no_trans.csv",
    #           title="CHANGE-seq: Classification task",
    #           xlabel="Regression-seq-dist with log transformation",
    #           ylabel="Regression-seq-dist without log transformation",
    #           metric="reg_to_class_aupr",
    #           task="classification")

    # # D
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets/GUIDE_no_trans.csv",
    #           title="GUIDE-seq: Classification task",
    #           xlabel="Regression-seq-dist with log transformation",
    #           ylabel="Regression-seq-dist without log transformation",
    #           metric="reg_to_class_aupr",
    #           task="classification")

    # # E
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets/CHANGE_no_trans.csv",
    #           title="CHANGE-seq: Regression task",
    #           xlabel="Regression-seq-dist with log transformation",
    #           ylabel="Regression-seq-dist without log transformation",
    #           metric="pearson",
    #           task="regression")

    # # F
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets/GUIDE_no_trans.csv",
    #           title="GUIDE-seq: Regression task",
    #           xlabel="Regression-seq-dist with log transformation",
    #           ylabel="Regression-seq-dist without log transformation",
    #           metric="pearson",
    #           task="regression")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Including potential OTSs with no reads in regression model training improves prediction performance

    # # A
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_without_negatives"
    #           "/test_results_include_on_targets"
    #           "/CHANGEseq_regression_without_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           title="CHANGE-seq: Classification task",
    #           xlabel="Regression-seq-dist with inactive off-targets",
    #           ylabel="Regression-seq-dist without inactive off-targets",
    #           metric="reg_to_class_aupr",
    #           task="classification")

    # # B
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_without_negatives"
    #           "/test_results_include_on_targets"
    #           "/GUIDEseq_regression_without_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           title="GUIDE-seq: Classification task",
    #           xlabel="Regression-seq-dist with inactive off-targets",
    #           ylabel="Regression-seq-dist without inactive off-targets",
    #           metric="reg_to_class_aupr",
    #           task="classification")

    # # C
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets/CHANGE_no_trans.csv",
    #           title="CHANGE-seq: Regression task",
    #           xlabel="Regression-seq-dist with inactive off-targets",
    #           ylabel="Regression-seq-dist without inactive off-targets",
    #           metric="pearson",
    #           task="regression")

    # # D
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #           "/test_results_include_on_targets"
    #           "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           "files/models_10fold/CHANGEseq/include_on_targets/regression_without_negatives"
    #           "/test_results_include_on_targets"
    #           "/GUIDEseq_regression_without_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #           title="GUIDE-seq: Regression task",
    #           xlabel="Regression-seq-dist with inactive off-targets",
    #           ylabel="Regression-seq-dist without inactive off-targets",
    #           metric="pearson",
    #           task="regression")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # The combination of sequence and distance features achieves the best prediction performance

    # Labels for the boxplot
    labels = ['Classification-seq', 'Classification-seq-dist', 'Regression-seq', 'Regression-seq-dist']

    # # A
    # bar_plot("files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #          "/CHANGEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #          "/CHANGEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #          "/test_results_include_on_targets"
    #          "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #          "/test_results_include_on_targets"
    #          "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #          labels, task="classification", title="CHANGE-seq: Classification task")

    # # B
    # bar_plot("files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #          "/GUIDEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #          "/GUIDEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #          "/test_results_include_on_targets"
    #          "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #          "/test_results_include_on_targets"
    #          "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #          labels, task="classification", title="GUIDE-seq: Classification task")

    # # C
    # bar_plot("files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #          "/CHANGEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #          "/CHANGEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #          "/test_results_include_on_targets"
    #          "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #          "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #          "/test_results_include_on_targets"
    #          "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #          labels, task="regression", title="CHANGE-seq: Regression task")

    # D
    bar_plot("files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
             "/GUIDEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
             "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
             "/GUIDEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
             "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
             "/test_results_include_on_targets"
             "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
             "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
             "/test_results_include_on_targets"
             "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
             labels, task="regression", title="GUIDE-seq: Regression task")


if __name__ == '__main__':
    main()
