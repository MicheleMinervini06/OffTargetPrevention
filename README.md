# SysEvalOffTarget

The code developed for the manuscript:
[A systematic evaluation of data processing and problem formulation of CRISPR off-target site prediction](https://academic.oup.com/bib/article-abstract/23/5/bbac157/6589868)

The code contains multiple options for training and testing models, including a transfer learning model for CHANGE-seq to GUIDE-seq. 

# Usage
Once you have cloned this repository and installed all the requirements, unzip the potential off-target sites dataset located in:
```
files/datasets/output_file_pam_change.zip
```
Then, you should execute
```
python prepare_data.py
```
to generate the active and inactive datasets that will be used in the training and testing stages.

To train the models, you should execute
```
python main_train.py
```
This will train some examples models. You can modify the `main_train.py` to run all the possible options we have formed.

Once you trained your models. you can evaluate the prefoamce of the models by executing
```
python main_test.py
```
In the current form, `main_test.py` will evaluate the example models and save the prediction performance tables located in files. To evaluate other models variants, please modidy `main_test.py` accordingly.

# Requirements:
The code was tested with:\
Python interpreter == 3.6.6\
Python packages required (other versions may work as well):\
   numpy == 1.18.5\
   pandas == 1.1.2\
   biopython==1.78\
   scikit-learn == 0.23.0\
   xgboost==1.3.3

Note: There might be other packages needed. Please contact us in case of any problem.
