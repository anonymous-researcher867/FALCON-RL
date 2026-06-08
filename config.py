import torch

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Federated Learning Hyperparameters
NUM_CLIENTS = 10
ROUNDS = 500
LOCAL_EPOCHS = 90
BATCH_SIZE = 128
TARGET_LAMBDA = 1e-4 

# Data & Classes
TARGET_CLASSES = sorted(["airplane", "airport", "beach"]) #Due to implementation constraints associated with AutoCoRe-FL, we selected three classes from the RESISC45 dataset
# TARGET_CLASSES = sorted(["monkshood", "snapdragon", "petunia"]) #Classes of the Flower dataset
# TARGET_CLASSES = sorted(["tornado", "spitfire", "metroliner"]) #Classes of the Aircraft dataset
# TARGET_CLASSES = sorted(["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]) #Classes of the HAM10000 dataset

NUM_CLASSES = len(TARGET_CLASSES)

# Paths (Update these to your local/server setup)
DATA_ROOT = "datasets/RESISC45"  
CONCEPT_FILE = "concepts.json" 
FEATURE_PATH = "results/RESISC45/features.pt"
MODEL_SAVE_PATH = "results/RESISC45/weights.pth"
