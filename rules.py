import torch
import random
import config

def compute_global_tau(client_data, num_classes, num_concepts):
    sum_z = torch.zeros(num_classes, num_concepts).to(config.DEVICE)
    counts = torch.zeros(num_classes).to(config.DEVICE)

    for c_id in range(config.NUM_CLIENTS):
        X, y = client_data[c_id]['train']
        X, y = X.to(config.DEVICE), y.to(config.DEVICE)
        for cls in range(num_classes):
            mask = (y == cls)
            if mask.any():
                sum_z[cls] += X[mask].sum(dim=0)
                counts[cls] += mask.sum()

    sim = sum_z / (counts.unsqueeze(1) + 1e-8)
    global_mean = sim.mean(dim=0)
    max_class_mean = sim.max(dim=0).values
    return global_mean + (max_class_mean - global_mean) * 0.25 

def get_weight_based_candidates(model, tau, max_top_features=12):
    W = model.linear.weight.data.cpu()
    actual_num_classes = W.shape[0] 
    
    candidates = {}
    for y in range(actual_num_classes):
        abs_weights = torch.abs(W[y])
        sorted_indices = torch.argsort(abs_weights, descending=True)[:max_top_features]
        
        candidates[y] = []
        for idx in sorted_indices:
            sign = 1 if W[y, idx] > 0 else -1
            candidates[y].append([{
                'concept_idx': idx.item(),
                'threshold': tau[idx].item(),
                'sign': sign
            }])
    return candidates

def compute_client_rule_metrics(X, y, candidate_pool):
    num_classes = len(candidate_pool)
    results = {y_idx: [] for y_idx in range(num_classes)}
    
    for y_idx, candidates in candidate_pool.items():
        is_pos = (y == y_idx)
        for cand_list in candidates:
            triggered = torch.ones(X.shape[0], dtype=torch.bool, device=X.device)
            for cond in cand_list:
                c_score = X[:, cond['concept_idx']]
                cond_triggered = (c_score > cond['threshold']) if cond['sign'] == 1 else (c_score <= cond['threshold'])
                triggered &= cond_triggered
            
            results[y_idx].append({
                'cov': torch.sum(triggered & is_pos).item(),
                'spill': torch.sum(triggered & ~is_pos).item()
            })
    return results

def optimize_global_rules(all_client_stats, candidate_pool, global_concepts, target_classes, max_rules=3, alpha=1):
    final_rulebook = []
    
    for class_idx in range(len(target_classes)):
        class_name = target_classes[class_idx]
        class_candidates = candidate_pool.get(class_idx, [])
        
        if not class_candidates:
            final_rulebook.append({'class_idx': class_idx, 'class_name': class_name, 'rules': []})
            continue
            
        scored_rules = []
        for rule_idx, cand_list in enumerate(class_candidates):
            total_cov = 0
            total_spill = 0
            
            for c_stats in all_client_stats:
                if class_idx in c_stats and rule_idx < len(c_stats[class_idx]):
                    stats = c_stats[class_idx][rule_idx]
                    total_cov += stats.get('cov', 0)
                    total_spill += stats.get('spill', 0)
            
            utility = total_cov - (alpha * total_spill)
            cond = cand_list[0]
            cid = cond['concept_idx']
            
            operator_str = "IS PRESENT (>)" if cond['sign'] > 0 else "IS NOT PRESENT (<=)"
            logic_str = f"'{global_concepts[cid]}' {operator_str} {cond['threshold']:.2f}"
            precision = total_cov / (total_cov + total_spill + 1e-6)
            
            scored_rules.append({
                'rule_id': f"c{class_idx}_r{rule_idx}",
                'logic': logic_str,
                'utility': utility,
                'precision': precision,
                'cov': total_cov,
                'spill': total_spill,
                'concept_idx': cid,
                'cond_list': cand_list
            })
            
        scored_rules = sorted(scored_rules, key=lambda x: x['utility'], reverse=True)[:max_rules]
        final_rulebook.append({'class_idx': class_idx, 'class_name': class_name, 'rules': scored_rules})
        
    return final_rulebook

def predict_with_rules(rulebook, X, num_classes, global_model=None):
    predictions = torch.zeros(X.shape[0], dtype=torch.long)
    X_gpu = X.to(X.device).float() 
    
    if global_model is not None:
        global_model.eval()
        with torch.no_grad():
            neural_logits = global_model(X_gpu)
            neural_fallback_preds = neural_logits.argmax(dim=1).cpu()

    for i in range(X.shape[0]):
        sample = X[i]
        best_class = -1
        max_utility_score = -999.0
        max_avg_precision = -1.0 
        
        for rule_class in rulebook:
            c_idx = rule_class['class_idx']
            total_utility = 0
            triggered_precisions = []
            
            for r in rule_class['rules']:
                cond_list = r.get('cond_list')
                if not cond_list: continue
                
                rule_triggered = True
                for cond in cond_list:
                    val = sample[cond['concept_idx']].item()
                    cond_triggered = (val > cond['threshold']) if cond['sign'] == 1 else (val <= cond['threshold'])
                    if not cond_triggered:
                        rule_triggered = False
                        break
                
                if rule_triggered:
                    total_utility += r['utility']
                    triggered_precisions.append(r['precision'])
            
            avg_precision = sum(triggered_precisions) / len(triggered_precisions) if triggered_precisions else 0
            
            if total_utility > max_utility_score and total_utility > 0:
                max_utility_score = total_utility
                max_avg_precision = avg_precision
                best_class = c_idx
            elif total_utility == max_utility_score and total_utility > 0:
                if avg_precision > max_avg_precision:
                    max_avg_precision = avg_precision
                    best_class = c_idx

        if best_class == -1 or max_utility_score <= 0:
            predictions[i] = neural_fallback_preds[i] if global_model is not None else random.randint(0, num_classes - 1)
        else:
            predictions[i] = best_class
            
    return predictions
