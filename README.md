# FALCON-RL: Federated Aligned Language-guided CONcept-oriented Rule Learning

This repository provides the implementation of **Falcon-RL**, a
federated learning framework for extracting **interpretable
concept-based rules** from distributed image datasets. The method
leverages **CLIP** to compute concept embeddings and trains a
**federated concept bottleneck model** to derive human-readable
classification rules.

The framework enables interpretable learning in decentralized
environments while preserving data locality across clients.

------------------------------------------------------------------------

# Repository Structure

The codebase is organized into modular components to facilitate
reproducibility, debugging, and extensibility.

    .
    ├── config.py              # Global configuration and hyperparameters
    ├── dataset.py             # Dataset loading, preprocessing, and FL sharding
    ├── extract_features.py    # Step 1: EVA-CLIP feature extraction
    ├── fl_core.py             # Federated learning algorithms and model definitions
    ├── rules.py               # Rule extraction and evaluation utilities
    ├── train.py               # Step 2: Federated training and rule generation
    ├── utils.py               # Helper functions and utilities
    ├── requirements.txt       # Python dependencies
    └── results/               # Output directory for features, models, and rules

------------------------------------------------------------------------

# Prerequisites

The implementation requires:

-   **Python 3.8 or later**
-   PyTorch
-   EVA-CLIP

Install all required dependencies using:

``` bash
pip install -r requirements.txt
```

Ensure that the EVA-CLIP library is installed and accessible in your
Python environment.

------------------------------------------------------------------------

# Data Preparation

Before running the pipeline, the dataset and concept files must be
prepared.

## 1. Image Dataset

Download the **NWPU-RESISC45** dataset and place it in the repository
root directory (or update the dataset path in `config.py`).

Expected directory structure:

    DATA_ROOT/
     └── NWPU-RESISC45/
          ├── airplane/
          ├── airport/
          ├── beach/
          └── ...

## 2. Concept File

Provide the concept descriptions in JSON format.

Example:

    RESISC45.json

This file contains the candidate concepts associated with each dataset
class.

The dataset location and concept file path can be modified in
`config.py`.

------------------------------------------------------------------------

# Running the Pipeline

The workflow consists of **two stages**.

Separating feature extraction from training reduces computational cost
and allows faster experimentation.

------------------------------------------------------------------------

# Step 1 --- Feature Extraction

EVA-CLIP is used to extract visual and textual embeddings from the
dataset.

Run:

``` bash
python extract_features.py
```

This step:

-   partitions the dataset into federated clients
-   computes image and text embeddings using EVA-CLIP
-   verifies the absence of data leakage across clients
-   saves extracted embeddings

Output directory:

    ./results/clip_features/

Feature extraction is performed **once** and reused in subsequent
experiments.

------------------------------------------------------------------------

# Step 2 --- Federated Training and Rule Extraction

Once features are extracted, the federated concept model can be trained.

Run:

``` bash
python train.py
```

This step:

-   loads precomputed EVA-CLIP embeddings
-   performs federated averaging (FedAvg) across clients
-   trains the concept bottleneck classifier
-   applies a gravity-based sparsity regularization
-   extracts human-readable decision rules
-   evaluates both neural and rule-based performance

Output directory:

    ./results/model_weights/

Generated outputs include:

-   trained model weights
-   extracted rule sets
-   evaluation metrics

------------------------------------------------------------------------

# Configuration

All hyperparameters are centralized in:

    config.py

Key parameters include:

  -----------------------------------------------------------------------
  Parameter               Description                 Default
  ----------------------- --------------------------- -------------------
  NUM_CLIENTS             Number of federated clients 10

  ROUNDS                  Number of communication     1000
                          rounds                      

  LOCAL_EPOCHS            Local training epochs per   90
                          client                      

  TARGET_LAMBDA           Sparsity regularization     1e-4
                          coefficient                 

  TARGET_CLASSES          Subset of classes used in   \["airplane",
                          training                    "airport",
                                                      "beach"\]
  -----------------------------------------------------------------------

Adjust these parameters depending on the dataset size and experimental
setup.

------------------------------------------------------------------------

# Results

The framework produces:

-   interpretable **concept bottleneck classifiers**
-   **human-readable rules** derived from learned concept activations
-   performance metrics for both neural and symbolic models

These outputs enable the analysis of **concept-level reasoning in
federated learning systems**.

------------------------------------------------------------------------

# License

This project is released under the **MIT License**. See the `LICENSE`
file for additional details.

------------------------------------------------------------------------

# Citation

If you use this repository in your research, please cite the associated
paper (to be released).
