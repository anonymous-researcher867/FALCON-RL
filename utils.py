import random
import torch
import numpy as np
import json
import re

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_absolute_indices(dataset):
    if hasattr(dataset, 'indices') and hasattr(dataset, 'dataset'):
        parent_indices = get_absolute_indices(dataset.dataset)
        return [parent_indices[i] for i in dataset.indices]
    elif hasattr(dataset, 'subset'):
        return get_absolute_indices(dataset.subset)
    return list(range(len(dataset)))

def pil_collate_fn(batch):
    images = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return images, torch.tensor(labels)

def canonicalize(name):
    if "." in name: name = name.split(".", 1)[1]
    return name.replace("_", " ").lower().strip()

def normalize_concept_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_valid_concept(text):
    if len(text) < 3: return False
    artifacts = ["add concepts from sentence", "extract concepts", "divide concepts"]
    for artifact in artifacts:
        if artifact in text: return False
    return True
    
def load_and_normalize_concepts(concept_file_path, target_classes):
    with open(concept_file_path) as f: raw_concepts = json.load(f)
    canonical_concepts = {}
    for key, concepts in raw_concepts.items():
        canon_key = canonicalize(key)
        if canon_key in canonical_concepts: canonical_concepts[canon_key].extend(concepts)
        else: canonical_concepts[canon_key] = concepts.copy()

    selected_concepts = set()
    for cls, concepts in canonical_concepts.items():
        if cls.lower() not in target_classes: continue
        cls_unique_concepts = []
        for c in concepts:
            norm_c = normalize_concept_text(c)
            if norm_c != "" and is_valid_concept(norm_c) and norm_c not in cls_unique_concepts:
                cls_unique_concepts.append(norm_c)
        selected_concepts.update(cls_unique_concepts)
        
    return sorted(list(selected_concepts))
