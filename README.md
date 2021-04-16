# Arabic-DID


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

- `bottom_up` - if the label is at city level it will go up to country and region level using `label_space`
- `compute_distances` - given geographical locations it will return distance between those locations
- `data_utils` - utility functions for data it will encode those data and labels
- `collect_data` - functions to collect data based on the training or validation and datasets
- `did_dataset` - helper `Dataset` object
- `finetuning_utils` - finetuning utility functions
  
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


## Main Questions:

1. Unbalanced data
2. Not all cities available