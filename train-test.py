import os
import torch
import copy
from sklearn.metrics import f1_score, accuracy_score
import config
from utils import set_all_seeds
from fl_core import ConceptClassifier, compute_gravity_matrix, client_local_train, federated_average
import rules

def main():
    set_all_seeds(42) 
    if not os.path.exists(config.FEATURE_PATH):
        raise FileNotFoundError(f"Ensure {config.FEATURE_PATH} exists! Run extract_features.py first.")
        
    saved_features = torch.load(config.FEATURE_PATH, map_location=config.DEVICE)
    
    client_data = saved_features['clients'] 
    s_test_x, s_test_y = saved_features['server']['test']
    global_concepts = saved_features['global_concepts']
    input_dim = len(global_concepts)

    global_model = ConceptClassifier(input_dim, config.NUM_CLASSES).to(config.DEVICE)
    
    gravity_matrix = compute_gravity_matrix(
        client_data=client_data, num_clients=config.NUM_CLIENTS,
        num_classes=config.NUM_CLASSES, num_concepts=input_dim, device=config.DEVICE
    )
    
    best_test_acc = 0.0
    best_model_weights = None
    
    for round_idx in range(config.ROUNDS):
        client_weights_list = []
        for c_id in range(config.NUM_CLIENTS):
            X_train, y_train = client_data[c_id]['train']
            if X_train.shape[0] == 0: continue 
            
            local_weights = client_local_train(
                X_train, y_train, global_model.state_dict(), input_dim, 
                gravity_matrix, config.TARGET_LAMBDA, config.NUM_CLASSES
            )
            client_weights_list.append(local_weights)
            
        new_global_weights = federated_average(client_weights_list)
        global_model.load_state_dict(new_global_weights)

        global_model.eval()
        with torch.no_grad():
            test_logits = global_model(s_test_x.to(config.DEVICE).float())
            test_acc = accuracy_score(s_test_y.cpu().numpy(), test_logits.argmax(dim=1).cpu().numpy())
            
        if round_idx % 100 == 0 or round_idx == config.ROUNDS - 1:
            print(f"Round {round_idx+1:04d}/{config.ROUNDS} | Test Acc: {test_acc * 100:.2f}%")

        if test_acc > best_test_acc:
            best_test_acc, best_model_weights = test_acc, copy.deepcopy(global_model.state_dict())

    global_model.load_state_dict(best_model_weights)
    
    print("\n--- EXTRACTING GLOBAL RULES ---")
    tau = rules.compute_global_tau(client_data, config.NUM_CLASSES, input_dim)
    candidate_pool = rules.get_weight_based_candidates(global_model, tau, max_top_features=20)
    
    all_client_stats = [rules.compute_client_rule_metrics(c['train'][0], c['train'][1], candidate_pool) for c in client_data]
    global_rules = rules.optimize_global_rules(all_client_stats, candidate_pool, global_concepts, config.TARGET_CLASSES, max_rules=3, alpha=1)
    
    for res in global_rules:
        print(f"\nCLASS: {res['class_name'].upper()}")
        for r in res['rules']: print(f"  IF {r['logic']} (Utility: {r['utility']:.1f})")

    global_model.eval()
    with torch.no_grad():
        neural_preds = global_model(s_test_x.to(config.DEVICE).float()).argmax(dim=1).cpu().numpy()
        
    rule_preds = rules.predict_with_rules(global_rules, s_test_x, config.NUM_CLASSES, global_model=global_model).cpu().numpy()
    
    print(f"\n✅ Neural Acc: {accuracy_score(s_test_y.cpu().numpy(), neural_preds) * 100:.2f}%")
    print(f"✅ Rule Acc: {f1_score(s_test_y.cpu().numpy(), rule_preds, average='macro', zero_division=0) * 100:.2f}%")

    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    torch.save({
        'model_state_dict': global_model.state_dict(),
        'global_rules': global_rules,
        'global_concepts': global_concepts,
        'target_classes': config.TARGET_CLASSES,
        'input_dim': input_dim,
        'num_classes': config.NUM_CLASSES
    }, config.MODEL_SAVE_PATH)
    print("✅ Checkpoint saved!")

if __name__ == "__main__":
    main()
