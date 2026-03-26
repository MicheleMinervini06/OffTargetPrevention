"""
general utilizes
"""

import pathlib

# ##############paths######################
HOME_DIR = str(pathlib.Path(__file__).parent.parent.absolute()) + \
    "/"  # project path
FILES_DIR = HOME_DIR+"files/"


DATASETS_PATH = FILES_DIR+"datasets/"
CHANGE_SEQ_PATH = DATASETS_PATH+"CHANGE-seq.xlsx"
GUIDE_SEQ_PATH = DATASETS_PATH+"GUIDE-seq.xlsx"

FIGURES_PATH = FILES_DIR+"figures/"


# #################constants##################
SEED = 10
