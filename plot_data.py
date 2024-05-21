import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def load_data(file_path, metric):
    """Load and validate required data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        if metric not in df.columns:
            raise ValueError(f"Metric {metric} not found in the data.")
        return df
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except pd.errors.EmptyDataError:
        print(f"No data: {file_path}")
        raise


# Function to calculate Pearson correlation for each group
def calculate_pearson(group):
    if len(group) > 1:  # Ensure there are at least two data points to calculate correlation
        if 'CHANGEseq_reads' in group.columns:
            correlation, _ = pearsonr(group['distance'], group['CHANGEseq_reads'])
        else:
            correlation, _ = pearsonr(group['distance'], group['GUIDEseq_reads'])
        return correlation
    else:
        return None  # Not enough data to calculate correlation


def scatter_plot(csv1, csv2, title, xlabel, ylabel, metric, task):
    """
    Generates a scatter plot comparing two metrics from two CSV files.

    Parameters:
    - csv1, csv2: Paths to the CSV files.
    - title: Plot title.
    - xlabel, ylabel: Labels for the x and y axes.
    - metric: Column name for the metric to plot.
    - task: Type of task, affects axis scales ('classification' or 'regression').
    """
    df1 = load_data(csv1, metric)
    df2 = load_data(csv2, metric)

    # Extract the metrics values and exclude the last row (ALL TARGETS)
    X = df1[metric].iloc[:-1]  # All rows except the last one from df1
    Y = df2[metric].iloc[:-1]  # All rows except the last one from df2

    # Calculate statistics
    stats = {
        'mean_1': X.mean(), 'median_1': X.median(),
        'mean_2': Y.mean(), 'median_2': Y.median()
    }

    plt.figure(figsize=(8, 6))
    # Using log-plus-one transformation to the color scale in order to avoid error if zeros are present
    sp = plt.scatter(X, Y, c=np.log(df1['positives'].iloc[:-1] + 1), cmap='viridis', alpha=0.5)
    plt.plot([-1, 1], [-1, 1], 'k--')  # Dotted line from (-1, -1) to (1, 1)

    cbar = plt.colorbar(sp)
    cbar.set_label('log(positives + 1)')

    plt.title(title)
    plt.legend([
        f'Pearson\n'
        f'X - Mean: {stats["mean_1"]:.3f}, Median: {stats["median_1"]:.3f}\n'
        f'Y - Mean: {stats["mean_2"]:.3f}, Median: {stats["median_2"]:.3f}'
    ], loc='best')
    plt.xlabel(f"X - {xlabel}")
    plt.ylabel(f"Y - {ylabel}")

    if task == "classification":
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

    plt.grid(False)
    plt.show()


def bar_plot(csv_info, task, title):
    data_frames = []
    labels = []

    # Read data from each CSV file based on the provided info
    for info in csv_info:
        if info['label'] != 'Reads-distance-corr':
            df = pd.read_csv(info['path'])
            column_name = info['column']
            label = info['label']
            labels.append(label)

            if task == "classification":
                # Extract the specific column for classification-related tasks
                data_series = df[column_name].iloc[:-1]
            else:
                # Extract the specific column for other tasks
                data_series = df[column_name].iloc[:-1]
        else:
            df = pd.read_excel(info['path'])

            # Remove inactive off-targets with reads <= 100
            if 'CHANGEseq_reads' in df.columns:
                df = df[df['CHANGEseq_reads'] > 100]
            else:
                df = df[df['GUIDEseq_reads'] > 100]

            # transformer = transformer_generator(np.where(df['CHANGEseq_reads'] > 100, 1, 0), "ln_x_plus_one_trans")
            # transform(df['distance'], transformer, inverse=True)

            df['distance'] = -df['distance']

            # Replace inf/-inf with 0
            df.replace([np.inf, -np.inf], 0, inplace=True)

            # Group by 'target' and apply the calculation
            data_series = df.iloc[:-1].groupby('target').apply(calculate_pearson, include_groups=False)

            label = info['label']
            labels.append(label)

            print("Type: ", type(data_series))
            print("Data series: ", data_series)

        data_series.rename(label, inplace=True)
        data_frames.append(data_series)

    # Combine all data into a single DataFrame
    data = pd.concat(data_frames, axis=1)

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

    # Error bars
    means = data.mean()
    stds = data.std()

    n_groups = len(data.columns)

    # Add error bars for 1 standard deviation
    for i in range(n_groups):
        # Determine the x positions for the error bars
        # This position is the center of each boxplot.
        x = i - 0.4

        # Calculate the error bar positions
        mean = means.iloc[i]
        error = stds.iloc[i]

        # Draw error bars (1 SD)
        plt.errorbar(x, mean, yerr=error, color='black', capsize=4)

    if task == "classification":
        plt.ylim(0, 1.2)
        plt.ylabel('AUPR')
    else:
        plt.ylim(-1, 1.5)
        plt.ylabel('Pearson')
    plt.title(title)
    plt.xticks(range(len(labels)), labels, rotation=30)

    plt.show()


def linear_regression_plot(csv1):

    df1 = pd.read_csv(csv1)

    # Apply inverse transformation to the 'Regression-dist' column
    df1['Regression-dist'] = np.expm1(df1['Regression-dist'])

    df1_filtered = df1[df1['distance'] > 0]

    # Group by 'distance' and calculate the mean of 'Regression-dist'
    grouped_means = df1_filtered.groupby('distance')['Regression-dist'].mean()

    # Plotting
    plt.figure(figsize=(8, 5))  # Set the figure size
    plt.plot(grouped_means.index, grouped_means.values, 'o--',
             color='g')  # 'o--' denotes dotted line with circle markers
    plt.xlabel('Distance')  # Label for the x-axis
    plt.ylabel('Read counts')  # Label for the y-axis
    plt.grid(True)  # Enable grid for better readability

    # Train a linear regression model
    linear_model = LinearRegression()
    linear_model.fit(df1['distance'].values.reshape(-1, 1), df1['Regression-dist'].values)

    # Predict the values using the linear model
    linear_predictions = linear_model.predict(df1['distance'].values.reshape(-1, 1))

    # Combine the predictions with the distance data
    df1['Linear-regression'] = linear_predictions

    df1_filtered = df1[df1['distance'] > 0]

    # Group by 'distance' and calculate the mean of 'Regression-dist'
    # df1_filtered['Linear-regression'] = np.expm1(df1_filtered['distance'])
    grouped_means = df1_filtered.groupby('distance')['Linear-regression'].mean()

    # Plotting
    plt.plot(grouped_means.index, grouped_means.values, 'o--',
             color='b')  # 'o--' denotes dotted line with circle markers
    plt.show()


def main():
    # Read counts log transformation improves
    # prediction performance

    # # C
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets/CHANGE_no_trans.csv",
    #              title="CHANGE-seq: Classification task",
    #              xlabel="Regression-seq-dist with log transformation",
    #              ylabel="Regression-seq-dist without log transformation",
    #              metric="reg_to_class_aupr",
    #              task="classification")
    #
    # # D
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets/GUIDE_no_trans.csv",
    #              title="GUIDE-seq: Classification task",
    #              xlabel="Regression-seq-dist with log transformation",
    #              ylabel="Regression-seq-dist without log transformation",
    #              metric="reg_to_class_aupr",
    #              task="classification")
    #
    # # E
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets/CHANGE_no_trans.csv",
    #              title="CHANGE-seq: Regression task",
    #              xlabel="Regression-seq-dist with log transformation",
    #              ylabel="Regression-seq-dist without log transformation",
    #              metric="pearson_only_positives_after_inv_trans",
    #              task="regression")
    #
    # # F
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets/GUIDE_no_trans.csv",
    #              title="GUIDE-seq: Regression task",
    #              xlabel="Regression-seq-dist with log transformation",
    #              ylabel="Regression-seq-dist without log transformation",
    #              metric="pearson_only_positives_after_inv_trans",
    #              task="regression")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Including potential OTSs with no reads in regression model training improves prediction performance

    # # A
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_without_negatives"
    #              "/test_results_include_on_targets"
    #              "/CHANGEseq_regression_without_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              title="CHANGE-seq: Classification task",
    #              xlabel="Regression-seq-dist with inactive off-targets",
    #              ylabel="Regression-seq-dist without inactive off-targets",
    #              metric="reg_to_class_aupr",
    #              task="classification")
    #
    # # B
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_without_negatives"
    #              "/test_results_include_on_targets"
    #              "/GUIDEseq_regression_without_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              title="GUIDE-seq: Classification task",
    #              xlabel="Regression-seq-dist with inactive off-targets",
    #              ylabel="Regression-seq-dist without inactive off-targets",
    #              metric="reg_to_class_aupr",
    #              task="classification")
    #
    # # C
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets/CHANGE_no_trans.csv",
    #              title="CHANGE-seq: Regression task",
    #              xlabel="Regression-seq-dist with inactive off-targets",
    #              ylabel="Regression-seq-dist without inactive off-targets",
    #              metric="pearson_only_positives_after_inv_trans",
    #              task="regression")
    #
    # # D
    # scatter_plot("files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #              "/test_results_include_on_targets"
    #              "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              "files/models_10fold/CHANGEseq/include_on_targets/regression_without_negatives"
    #              "/test_results_include_on_targets"
    #              "/GUIDEseq_regression_without_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #              title="GUIDE-seq: Regression task",
    #              xlabel="Regression-seq-dist with inactive off-targets",
    #              ylabel="Regression-seq-dist without inactive off-targets",
    #              metric="pearson_only_positives_after_inv_trans",
    #              task="regression")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # The combination of sequence and distance features achieves the best prediction performance

    # Labels for the boxplot
    labels = ['Classification-seq', 'Classification-seq-dist', 'Regression-seq', 'Regression-seq-dist']

    # Labels for the boxplot
    labels = ['Classification-seq', 'Classification-seq-dist', 'Regression-seq', 'Regression-seq-dist']

    # # A - CHANGE-seq: Classification task
    # csv_info_a = [
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/CHANGEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'aupr', 'label': labels[0]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/CHANGEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'aupr', 'label': labels[1]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'reg_to_class_aupr', 'label': labels[2]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'reg_to_class_aupr', 'label': labels[3]}
    # ]
    # bar_plot(csv_info_a, task="classification", title="CHANGE-seq: Classification task")
    #
    # # B - GUIDE-seq: Classification task
    # csv_info_b = [
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/GUIDEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'aupr', 'label': labels[0]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/GUIDEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'aupr', 'label': labels[1]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'reg_to_class_aupr', 'label': labels[2]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'reg_to_class_aupr', 'label': labels[3]}
    # ]
    # bar_plot(csv_info_b, task="classification", title="GUIDE-seq: Classification task")
    #
    # # C - CHANGE-seq: Regression task
    # csv_info_c = [
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/CHANGEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'pearson_reads_to_proba_for_positive_set', 'label': labels[0]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/CHANGEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'pearson_reads_to_proba_for_positive_set', 'label': labels[1]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': labels[2]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': labels[3]}
    # ]
    # bar_plot(csv_info_c, task="regression", title="CHANGE-seq: Regression task")
    #
    # # D - GUIDE-seq: Regression task
    # csv_info_d = [
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/GUIDEseq_classifier_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'pearson_reads_to_proba_for_positive_set', 'label': labels[0]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/classifier/test_results_include_on_targets"
    #                 "/GUIDEseq_classifier_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'pearson_reads_to_proba_for_positive_set', 'label': labels[1]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_imbalanced.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': labels[2]},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': labels[3]}
    # ]
    # bar_plot(csv_info_d, task="regression", title="GUIDE-seq: Regression task")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Evaluating regression task performance of models trained with the distance feature only

    # # A - CHANGE-seq: Regression task
    # csv_info_a = [
    #     {
    #         'path': "files/datasets/CHANGE-seq.xlsx",
    #         'label': 'Reads-distance-corr'},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/CHANGEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': 'Regression-seq-dist'},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets/CHANGEseq_only_distance.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': 'Regression-dist'}
    # ]
    #
    # bar_plot(csv_info_a, task="regression", title="CHANGE-seq: Regression task")
    #
    # # B - GUIDE-seq: Regression task
    # csv_info_b = [
    #     {
    #         'path': "files/datasets/GUIDE-seq.xlsx",
    #         'label': 'Reads-distance-corr'},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets"
    #                 "/GUIDEseq_regression_with_negatives_results_xgb_model_all_10_folds_with_distance_imbalanced.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': 'Regression-seq-dist'},
    #     {
    #         'path': "files/models_10fold/CHANGEseq/include_on_targets/regression_with_negatives"
    #                 "/test_results_include_on_targets/GUIDEseq_only_distance.csv",
    #         'column': 'pearson_only_positives_after_inv_trans', 'label': 'Regression-dist'}
    # ]
    #
    # bar_plot(csv_info_b, task="regression", title="GUIDE-seq: Regression task")

    # C - Linear regression plot to compare a linear model vs the regression-dist
    linear_regression_plot("files/models_10_fold/CHANGEseq/include_on_targets/predictions_include_on_targets"
                           "/CHANGEseq_results_all_10_folds.csv")


if __name__ == '__main__':
    main()
