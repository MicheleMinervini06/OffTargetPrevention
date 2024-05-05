"""
Generates the datasets for training and testing the models
"""

from pathlib import Path
import pandas as pd
from Bio.Seq import Seq
from SysEvalOffTarget_src import general_utilities


def create_positives(dataset_excel_path=general_utilities.CHANGE_SEQ_PATH, data_type="CHANGEseq",
                     read_threshold=None, exclude_on_targets=False, save_sets=False):
    """
    Generates the positive set of the CHANGE-seq or GUIDE-seq according the requested parameters.
    :param dataset_excel_path: the Path the samples before any filtering.
    :param data_type: str. The type of the dataset that the positive set is crated for.
    :param read_threshold: int or None. The read count threshold for filtering. samples with lower will be filtered.
        if None, no filtering is applied. Usually read_threshold=100. Default: None
    :param exclude_on_targets: bool. exclude the on-targets from the positive set in case of True. Default: False
    :param save_sets: bool. Save the positive set generated in case of True. Otherwise, return the table as DataFrame.
        Default: True
    :return: DataFrame. The positive set that was generated.
    """
    dataset_df = pd.read_excel(dataset_excel_path)
    # exclude bulges
    # drop off targets with len not equal to 23 and with '-'
    dataset_df = dataset_df[dataset_df["offtarget_sequence"].str.len() == 23]
    dataset_df = dataset_df[dataset_df["offtarget_sequence"].str.find('-') == -1]
    # set the condition for splitting to positive and undefined sets
    if read_threshold is not None:
        read_threshold_conds = dataset_df["{}_reads".format(data_type)] <= read_threshold
        undefined_mask = (read_threshold_conds | (dataset_df['distance'] == 0)) if exclude_on_targets \
            else read_threshold_conds
    elif exclude_on_targets:
        undefined_mask = dataset_df['distance'] == 0
    else:
        undefined_mask = None
    # splitting to positive and undefined sets
    if undefined_mask is not None:
        # take the undefined set
        # .copy() and .loc added to avoid SettingWithCopyWarning
        # Pandas is not sure whether dataset_undefined_df is a view or a copy of the original DataFrame
        dataset_undefined_df = dataset_df[undefined_mask].copy()
        dataset_undefined_df.loc[:, 'label'] = -1
        # take the positive set
        dataset_positive_df = dataset_df[~ undefined_mask].copy()
        dataset_positive_df.loc[:, 'label'] = 1
    else:
        dataset_undefined_df = None
        dataset_positive_df = dataset_df
        dataset_positive_df['label'] = 1
    # save the sets
    if save_sets:
        dir_path = general_utilities.DATASETS_PATH
        dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        dataset_positive_df.to_csv(dir_path + '{}_positive.csv'.format(data_type))
        if dataset_undefined_df is not None:
            dataset_undefined_df.to_csv(dir_path + '{}_undefined.csv'.format(data_type))

    return dataset_positive_df, dataset_undefined_df


def create_negatives(experiment_df, cas_offinder_optional_offtargets_path=general_utilities.DATASETS_PATH +
                     "output_file_pam_change.txt", data_type="CHANGEseq", save_sets=False,
                     exclude_on_targets=False):
    """
    Generates the negative set of the CHANGE-seq or GUIDE-seq according the requested parameters.
    :param experiment_df: DataFrame. the Dataframe that contains the samples as we obtained from the experiment (e.g.,
        without any filtering).
    :param cas_offinder_optional_offtargets_path: str. The path to table that contains all the options off-targets
        obtained by CAS-OFFinder.
    :param data_type: str. The type of the dataset that the negative set is crated for.
        Options: "CHANGEseq" or "GUIDEseq". Default: "CHANGEseq"
    :param save_sets: bool. Save the negative set generated in case of True. Otherwise, return the table as DataFrame.
        Default: True
    :param exclude_on_targets: This argument just define where to save the dataset. The on-targets are always filtered
        from the negative set. Default: False
    :return: DataFrame. The negative set that was generated.
    """
    negative_df = pd.read_table(cas_offinder_optional_offtargets_path)
    negative_df['label'] = 0
    negative_df = negative_df[['chrom', 'chromStart', 'strand',
                               'offtarget_sequence', 'distance', 'target', 'label']]
    negative_df['offtarget_sequence'] = negative_df['offtarget_sequence'].str.upper()

    print("number of optional off targets before filtering: ", len(negative_df))

    print("Dropping chroms which do not appear in the experiment")
    chroms = experiment_df["chrom"].unique()
    negative_df = negative_df.drop(
        negative_df[~negative_df['chrom'].isin(chroms)].index)
    print("number of optional off targets after this stage: ", len(negative_df))

    print("Dropping off-targets which their target doesn't appear in the experiment_df")
    targets = experiment_df["target"].unique()
    negative_df = negative_df[negative_df["target"].isin(targets)]
    print("number of optional off targets after this stage: ", len(negative_df))

    print(
        "Dropping for each Target the optional off-targets which their sequences\
         (or their tt complement) appear in the experiment (without connection to chromStart)")
    # The filter on the reverse complement is for very rare
    # situations (probably do not exist due to the number of mismatch allowed)
    for target in targets:
        df_target = experiment_df[experiment_df['target'] == target]
        target_change_seq_off_targets = df_target["offtarget_sequence"]
        target_change_seq_reverse_off_targets = df_target["offtarget_sequence"].apply(
            lambda seq: str(Seq(seq).reverse_complement()))
        negative_df = negative_df.drop(
            negative_df[(negative_df['target'] == target) & ((negative_df['offtarget_sequence'].isin(
                target_change_seq_off_targets)) | (negative_df['offtarget_sequence'].isin(
                    target_change_seq_reverse_off_targets)))].index)
    print("number of optional off targets after this stage: ", len(negative_df))

    # This is done since some CHANGE-seq (or other experiment) off-target
    # share same chromStart with optional off-target from cas-offinder but their sequences does not agree.
    print("Dropping for each chrom the optional off-targets which their chromStart appear in the experiment")
    chroms = experiment_df["chrom"].unique()
    for chrom in chroms:
        negative_df = negative_df.drop(negative_df[(negative_df['chrom'] == chrom) & (
            negative_df['chromStart'].isin(experiment_df["chromStart"]))].index)
    print("number of optional off targets after this stage: ", len(negative_df))

    # The machine learning can not different between same
    # optional off-target with different position for the same target
    print("Dropping for each Target duplicates of optional off-targets")
    targets = experiment_df["target"].unique()
    for target in targets:
        negative_df = negative_df[~(
                (experiment_df['target'] == target) & negative_df['offtarget_sequence'].duplicated())]
    print("number of optional off targets after this stage: ", len(negative_df))

    print("dropping on-targets if exists (at all and after all the previous stages(")
    negative_df = negative_df[negative_df["distance"] != 0]
    print("number of optional off targets after this stage: ", len(negative_df))

    if save_sets:
        dir_path = general_utilities.DATASETS_PATH
        dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        negative_df.to_csv(dir_path + '{}_negative.csv'.format(data_type))

    return negative_df


def intersection_creation(exclude_on_targets=False):
    """
    crate the intersection datasets of the CHANGE-seq and GUIDE-seq. we define the positive sample as the positives
    in both datasets or in the GUIDE-seq (the code generate the two options).
    :param exclude_on_targets: exclude the on-target from the intersection dataset in case of True. Default: False
    :return: None
    """
    dir_path = general_utilities.DATASETS_PATH
    dir_path += 'exclude_on_targets/' if exclude_on_targets else 'include_on_targets/'

    guide_seq_positives_df = pd.read_csv(
        dir_path + 'GUIDEseq_positive.csv', index_col=0)
    guide_seq_positives_df = guide_seq_positives_df.drop(
        ["chromEnd", "name", "genomic_coordinate", "run"], axis=1)
    guide_seq_negatives_df = pd.read_csv(
        dir_path + 'GUIDEseq_negative.csv', index_col=0)
    guide_seq_negatives_df["GUIDEseq_reads"] = 0

    change_seq_positives_df = pd.read_csv(
        dir_path + 'CHANGEseq_positive.csv', index_col=0)
    change_seq_positives_df = change_seq_positives_df.drop(
        ["chromEnd", "name", "chromStart:chromEnd"], axis=1)
    change_seq_negatives_df = pd.read_csv(
        dir_path + 'CHANGEseq_negative.csv', index_col=0)
    change_seq_negatives_df["CHANGEseq_reads"] = 0

    guide_seq_df = pd.concat(
        [guide_seq_positives_df, guide_seq_negatives_df], ignore_index=True, sort=False)
    change_seq_df = pd.concat(
        [change_seq_positives_df, change_seq_negatives_df], ignore_index=True, sort=False)
    # inner for intersection
    change_guide_intersection = pd.merge(
        change_seq_df, guide_seq_df, how='inner',
        on=['chrom', 'chromStart', 'strand', 'offtarget_sequence', 'distance', 'target'],
        suffixes=("_CHANGE", "_GUIDE"))
    change_guide_intersection.to_csv(dir_path + 'CHANGE_GUIDE_intersection.csv')

    # .copy() and .loc added to avoid SettingWithCopyWarning
    # Pandas is not sure whether dataset_undefined_df is a view or a copy of the original DataFrame
    change_guide_intersection_positives_by_guide = \
        change_guide_intersection[change_guide_intersection['GUIDEseq_reads'] > 0].copy()
    change_guide_intersection_negatives_by_guide = \
        change_guide_intersection[change_guide_intersection['GUIDEseq_reads'] == 0].copy()
    change_guide_intersection_positives_by_guide.loc[:, 'label'] = 1
    change_guide_intersection_negatives_by_guide.loc[:, 'label'] = 0
    change_guide_intersection_positives_by_guide.to_csv(
        dir_path + 'CHANGE_GUIDE_intersection_by_GUIDE_positive.csv')
    change_guide_intersection_negatives_by_guide.to_csv(
        dir_path + 'CHANGE_GUIDE_intersection_by_GUIDE_negative.csv')

    change_guide_intersection_positives_by_both = \
        change_guide_intersection[
            (change_guide_intersection['GUIDEseq_reads'] > 0) & (change_guide_intersection['CHANGEseq_reads'] > 0)].copy()
    change_guide_intersection_negatives_by_both = \
        change_guide_intersection[
            (change_guide_intersection['GUIDEseq_reads'] == 0) & (change_guide_intersection['CHANGEseq_reads'] == 0)].copy()
    change_guide_intersection_positives_by_both.loc[:, 'label'] = 1
    change_guide_intersection_negatives_by_both.loc[:, 'label'] = 0
    change_guide_intersection_positives_by_both.to_csv(
        dir_path + 'CHANGE_GUIDE_intersection_by_both_positive.csv')
    change_guide_intersection_negatives_by_both.to_csv(
        dir_path + 'CHANGE_GUIDE_intersection_by_both_negative.csv')


def main():
    """
    main function
    """
    tasks = ["CHANGE", "intersection", "GUIDE"]
    if "CHANGE" in tasks:
        print("create CHANGE-seq dataset")
        create_positives(dataset_excel_path=general_utilities.CHANGE_SEQ_PATH, data_type="CHANGEseq",
                         read_threshold=100, exclude_on_targets=False, save_sets=True)
        change_seq_df = pd.read_excel(general_utilities.CHANGE_SEQ_PATH)
        # drop off targets that contains '-'
        change_seq_df = change_seq_df[change_seq_df["offtarget_sequence"].str.len() == 23]
        change_seq_df = change_seq_df[change_seq_df["offtarget_sequence"].str.find('-') == -1]
        create_negatives(change_seq_df, cas_offinder_optional_offtargets_path=general_utilities.DATASETS_PATH +
                         "output_file_pam_change.txt", data_type="CHANGEseq", save_sets=True, exclude_on_targets=False)

    if "GUIDE" in tasks:
        print("create GUIDE-seq dataset")
        create_positives(dataset_excel_path=general_utilities.GUIDE_SEQ_PATH, data_type="GUIDEseq",
                         read_threshold=None, exclude_on_targets=False, save_sets=True)
        guide_seq_df = pd.read_excel(general_utilities.GUIDE_SEQ_PATH)
        # drop off targets that contains '-' and ith len not equal to 23
        guide_seq_df = guide_seq_df[guide_seq_df["offtarget_sequence"].str.len() == 23]
        guide_seq_df = guide_seq_df[guide_seq_df["offtarget_sequence"].str.find('-') == -1]
        create_negatives(guide_seq_df, cas_offinder_optional_offtargets_path=general_utilities.DATASETS_PATH +
                         "output_file_pam_change.txt",  data_type="GUIDEseq", save_sets=True, exclude_on_targets=False)

    if "intersection" in tasks:
        print("create CHANGE-GUIDE intersection datasets")
        intersection_creation(exclude_on_targets=False)


if __name__ == '__main__':
    main()
