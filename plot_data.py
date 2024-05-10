import pandas as pd
import matplotlib.pyplot as plt


def main_plot(csv1, csv2, title, xlabel, ylabel):
    # Load the data from the first CSV file
    df1 = pd.read_csv(csv1)

    # Load the data from the second CSV file
    df2 = pd.read_csv(csv2)

    # Calculate mean values
    mean_pearsons_df1 = df1['pearson'].mean()
    mean_pearsons_df2 = df2['pearson'].mean()

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(df1['pearson'], df2['pearson'], color='blue', alpha=0.5)
    plt.plot(df1['pearson'], df1['pearson'], color='black', linestyle='dotted')
    plt.title(title)
    plt.legend(
        ['y=x', f'X Mean: {mean_pearsons_df1:.2f}', f'Y Mean: {mean_pearsons_df2:.2f}'])  # Corrected legend labels
    plt.xlabel("X - " + xlabel)
    plt.ylabel("Y - " + ylabel)
    plt.grid(False)
    plt.show()


def main():
    main_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
              "/test_results_include_on_targets"
              "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
              "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
              "/test_results_include_on_targets"
              "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
              title="CHANGE-seq: Regression task",
              xlabel="Regression-seq-dist with log transformation",
              ylabel="Regression-seq-dist without log transformation")


if __name__ == '__main__':
    main()
