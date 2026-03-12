import torch

# Hardware
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Federated Learning Hyperparameters
NUM_CLIENTS = 10
ROUNDS = 100
LOCAL_EPOCHS = 90
BATCH_SIZE = 128
TARGET_LAMBDA = 1e-4 

# Data & Classes
TARGET_CLASSES = sorted(["airplane", "airport", "beach"])
NUM_CLASSES = len(TARGET_CLASSES)

# Paths (Update these to your local/server setup)
DATA_ROOT = "./NWPU-RESISC45"  
CONCEPT_FILE = "./RESISC45.json" 
FEATURE_PATH = "./results/clip_features/resisc_EVA02-CLIP-bigE-14-plus_fl_extracted_features.pt"
MODEL_SAVE_PATH = "./results/model_weights/final_resisc_concept_model.pth"
