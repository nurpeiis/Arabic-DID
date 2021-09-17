# Arabic-DID

The main goal of this repository is to create

## Setting Up the Environment

First you will have to download the conda environment from the file

```
conda env create -f environment.yml
```

Secondly you will have to activate the environment 

```
conda activate  arabic-did
```

## Directory description

- `labels` - this is the folder that contains labels space for the dialects at each level
  - `label_space.tsv` - table that contains dependency between city, province, country and region 
- `data_process` - directory that contains data  processing scripts, where each file is a script for the dataset
- `bottom_up` - if the label is at city level it will go up to country and region level using `label_space`
- `compute_distances` - given geographical locations it will return distance between those locations
- `data_utils` - utility functions for data it will encode those data and labels
- `data_aggregation` - functions to collect data based on the training or validation and datasets
- `did_dataset` - helper `Dataset` object
- `finetuning_utils` - finetuning utility functions
- `experiment` - functions to run training and evaluating scripts
- `run_experiment` - functions to run various experiments
- `google_sheet_utils` - utility functions to record experimental results  into google  sheet through the google's API
- `filter_logit` - functions to filter logit
- `notebooks` - jupyter notebooks to experiment
  
## Data Processing
## Future Works

- Learn how to assign weights based on the distance 
- jerusalem ps gives some location in london
- al_suwayda is not syrian location, it is as suwayda
- what should we do with msa distance? equidistant with everyone
  - msa will never give or get anything. It will only give to msa
  

- Distance penalty: alpha = distance/max distance
  - soft margin svm
  - the further it goes away the more penalty, distance exponentially


## Main Questions

1. Unbalanced data
2. Not all cities available

## Data Explanation
1. Dev has no zagazig and giza because `ldc_callhome_arabic_trans_1997_t19`  was splitted originally

## KenLM Instructions

Install KenLM binary on C++  using the following commands:

```
cd
wget -O - https://kheafield.com/code/kenlm.tar.gz |tar xz
mkdir kenlm/build
cd kenlm/build
cmake ..
make -j2
```

To install python package run following command:

`pip install https://github.com/kpu/kenlm/archive/master.zip`
## Running on Greene HPC

1. Make sure to activate conda env
   ```
   source $SCRATCH/miniconda3/bin/activate
   conda activate name_env (mllu right now)
   ```
2. Then install all package dependencies
## Results

1. Salameh
```
{'accuracy': 0.6765384615384615, 'f1_micro': 0.6765384615384615, 'f1_macro': 0.6779509849008464, 'recall_micro': 0.6765384615384615, 'recall_macro': 0.6765384615384615, 'precision_micro': 0.6765384615384615, 'precision_macro': 0.6836726852465969}
```
2. Salameh + aggregated_city on aggregated_city:
```
{'accuracy': 0.6759615384615385, 'f1_micro': 0.6759615384615385, 'f1_macro': 0.6771667896947418, 'recall_micro': 0.6759615384615385, 'recall_macro': 0.6759615384615384, 'precision_micro': 0.6759615384615385, 'precision_macro': 0.6826720348902401}
```
3. Salameh + aggregated_city on madar 26:
```
{'accuracy': 0.68, 'f1_micro': 0.68, 'f1_macro': 0.6814004149650975, 'recall_micro': 0.68, 'recall_macro': 0.6800000000000002, 'precision_micro': 0.68, 'precision_macro': 0.687314065312601}
``` 

4. Salameh but use proper char division as lamda function:
```
{'accuracy': 0.6765384615384615, 'f1_micro': 0.6765384615384615, 'f1_macro': 0.6779509849008464, 'recall_micro': 0.6765384615384615, 'recall_macro': 0.6765384615384615, 'precision_micro': 0.6765384615384615, 'precision_macro': 0.6836726852465969}
```
                                       
## Training idea:
1. Use particular separate levels to have ripple carrying effect
2. Salameh + then gradually add data
3. For Salameh use your hierarchical labels that you created to evaluate in different levels
4. for every input repeat the sequence twice or n number of times to see how it affects the signal !!!

## Evaluation:
- Evaluate at country & region level, i.e. even if city level increases by 0.3%, there might be larger increase at higher levels 
- Have 113 evaluation too, potentially significantly lower, but  higher level label is better
- Statistical significance TODO: MUST READ ABOUT HOW THIS IS DONE
  - Look into T-test in particular
  - Find p value
  - How two sets are different, predictions of two different models
  - https://statisticsbyjim.com/hypothesis-testing/comparing-hypothesis-tests-data-types/
- Divide datasets by annotation type.
  - Madar extra vs All other city level data

-  Generate table data with all variables