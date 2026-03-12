import os
import torch
import config
from utils import set_all_seeds, load_and_normalize_concepts
from dataset import get_federated_data, verify_no_data_leakage
from fl_core import client_local_concept_extraction_eva18b
from eva_clip import create_model_and_transforms, get_tokenizer

def main():
    print("Initializing Federated Concept Extraction Framework...")
    set_all_seeds(42) 
    
    global_concepts = load_and_normalize_concepts(config.CONCEPT_FILE, config.TARGET_CLASSES)
    input_dim = len(global_concepts)
    print(f"Loaded {input_dim} unique concepts.")

    data_struct = get_federated_data(
        root=config.DATA_ROOT, 
        num_clients=config.NUM_CLIENTS, 
        noise_client_percentage=0, 
        noise_label_shuffle_degree=0, 
        num_classes=config.NUM_CLASSES,
        target_classes=config.TARGET_CLASSES
    )
    verify_no_data_leakage(data_struct)

    model_name, pretrained = "EVA02-CLIP-bigE-14-plus", "eva_clip"
    clip_model, _, preprocess = create_model_and_transforms(
        model_name, pretrained=pretrained, force_custom_clip=True, precision="fp16", device=config.DEVICE
    )
    clip_model.eval()
    tokenizer = get_tokenizer(model_name)

    extracted_features = {
        'server': {},
        'clients': [],
        'global_concepts': global_concepts,
        'target_classes': config.TARGET_CLASSES
    }

    print("\n=== EXTRACTING SERVER DATA ===")
    s_val_x, s_val_y = client_local_concept_extraction_eva18b(data_struct['server']['val'], global_concepts, clip_model, preprocess, tokenizer, config.DEVICE)
    extracted_features['server']['val'] = (s_val_x, s_val_y)
    
    s_test_x, s_test_y = client_local_concept_extraction_eva18b(data_struct['server']['test'], global_concepts, clip_model, preprocess, tokenizer, config.DEVICE)
    extracted_features['server']['test'] = (s_test_x, s_test_y)

    print("\n=== EXTRACTING CLIENT DATA ===")
    for c in data_struct['clients']:
        print(f" -> Extracting Client {c['id']:02d}")
        c_tr_x, c_tr_y = client_local_concept_extraction_eva18b(c['train'], global_concepts, clip_model, preprocess, tokenizer, config.DEVICE)
        c_v_x, c_v_y = client_local_concept_extraction_eva18b(c['val'], global_concepts, clip_model, preprocess, tokenizer, config.DEVICE)
        c_te_x, c_te_y = client_local_concept_extraction_eva18b(c['test'], global_concepts, clip_model, preprocess, tokenizer, config.DEVICE)
        
        extracted_features['clients'].append({
            'id': c['id'],
            'is_noisy': c['is_noisy'],
            'train': (c_tr_x, c_tr_y),
            'val': (c_v_x, c_v_y),
            'test': (c_te_x, c_te_y)
        })

    os.makedirs(os.path.dirname(config.FEATURE_PATH), exist_ok=True)
    torch.save(extracted_features, config.FEATURE_PATH)
    print("Done! Features saved.")

if __name__ == "__main__":
    main()
