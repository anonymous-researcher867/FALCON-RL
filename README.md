# FALCON-RL

Code for the paper "FALCON-RL: Federated Aligned Language-guided CONcept-oriented Rule Learning"

FALCON-RL is a language model-guided federated CBMs method designed to extract interpretable textual concept-based rules for image classification. 

## Environments

We run our experiments using Python 3.11. You can install the required packages using:
``` bash
conda create --name falconrl python=3.11
conda activate falconrl
pip install -r requirements.txt
```

## Datasets

The file structure of each dataset folder is:
``` 
dataset/
|-- images/
|-- concepts/
```
**Note**:
All dataset images should be placed in the `images/` directory. The `concepts/` directory contains a file named `concepts.csv`, which specifies the candidate concepts associated with the dataset.
You need to download the dataset files and store the images in the corresponding directory: `datasets/{dataset name}/images/`.
Alternatively, the dataset and concept file paths can be customized by modifying the configuration settings in `config.py`.

You can download the datasets from the following links:
| **Dataset**       | **Description**         | **Download Link**                                                                 |
|-------------------|-------------------------|-----------------------------------------------------------------------------------|
| Aircraft          | FGVC-Aircraft dataset        | [click to download](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz) |
| Flower            | Flower-102 dataset          | [click to download](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz) |
| HAM10000          | HAM10000 dataset        | [click to download](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T#) |
| RESISC45          | RESISC45 dataset        | [click to download](https://1drv.ms/u/s!AmgKYzARBl5ca3HNaHIlzp_IXjs)              |
| CUB               | CUB dataset             | [click to download](https://data.caltech.edu/records/20098)                       |


## Directories
    .
    ├── config.py              # Global configuration and hyperparameters
    ├── dataset.py             # Dataset loading, preprocessing, and FL sharding
    ├── extract_features.py    # Step 1: CLIP feature extraction
    ├── fl_core.py             # Federated learning algorithms and model definitions
    ├── rules.py               # Rule extraction and evaluation utilities
    ├── train.py               # Step 2: Federated training and rule generation
    ├── utils.py               # Helper functions and utilities
    ├── requirements.txt       # Python dependencies
    └── results/               # Output directory for features, models, and rules

## FALCON-RL Training
To train FALCON-RL, you first need to extract the image and text features using CLIP. This step reduces computational cost and provides the feature representations required for federated training.

Execute the following command to extract the features:
``` python
python extract_features.py
```
The extracted features will be stored in `./results/clip_features/`

Next, initiate the federated training process by running:
``` python
python train.py
```
Upon completion of the maximum number of communication rounds, the model checkpoint with the highest validation accuracy will be saved to `./results/model_weights/`

## Acknowledgments
This repository is based on the following repositories:
[CLIP](https://github.com/openai/CLIP).
[Labo](https://github.com/YueYANG1996/LaBo/tree/main).

