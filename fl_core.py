import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import pil_collate_fn
import config

def client_local_concept_extraction_eva18b(client_dataset, received_concepts, model, preprocess, tokenizer, device):
    loader = DataLoader(client_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=pil_collate_fn)
    concept_prompts = [f"A photo of a {c}" for c in received_concepts]
    text_embeddings_list = []

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        for i in range(0, len(concept_prompts), 256):
            batch_prompts = concept_prompts[i : i+256]
            text_inputs = tokenizer(batch_prompts).to(device)
            batch_embs = model.encode_text(text_inputs)
            batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
            text_embeddings_list.append(batch_embs)

    text_embeddings = torch.cat(text_embeddings_list, dim=0)
    all_scores, all_labels = [], []
    
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float32):
        for raw_pil_batch, batch_labels in tqdm(loader, desc="   Extracting", leave=False):
            image_tensors = torch.stack([preprocess(img) for img in raw_pil_batch]).to(device)
            image_embeddings = model.encode_image(image_tensors)
            image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True)

            concept_scores = torch.matmul(image_embeddings, text_embeddings.T)
            all_scores.append(concept_scores.cpu())
            all_labels.append(batch_labels)

    return torch.cat(all_scores), torch.cat(all_labels)

class ConceptClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x): 
        return self.linear(x)

def compute_gravity_matrix(client_data, num_clients, num_classes, num_concepts, device):
    client_class_avgs = torch.zeros((num_clients, num_classes, num_concepts), device=device)
    client_has_class = torch.zeros((num_clients, num_classes), device=device)
    
    for c_id in range(num_clients):
        cx, cy = client_data[c_id]['train']
        cx, cy = cx.to(device), cy.to(device)
        for c in range(num_classes):
            mask = (cy == c)
            if mask.sum() > 0:
                client_class_avgs[c_id, c] = cx[mask].mean(dim=0)
                client_has_class[c_id, c] = 1.0 

    S = torch.zeros((num_classes, num_concepts), device=device)
    for c in range(num_classes):
        num_participating = client_has_class[:, c].sum()
        if num_participating > 0:
            S[c] = client_class_avgs[:, c].sum(dim=0) / num_participating

    s_max, _ = S.max(dim=1, keepdim=True)
    gravity_matrix = s_max - S
    return torch.clamp(gravity_matrix, min=0.0)
    
def client_local_train(train_x, train_y, global_weights, input_dim, gravity_matrix, current_lambda, num_classes):
    model = ConceptClassifier(input_dim, num_classes).to(config.DEVICE)
    model.load_state_dict(global_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_x, train_y = train_x.to(config.DEVICE).to(torch.float32), train_y.to(config.DEVICE)
    gravity_matrix = gravity_matrix.to(config.DEVICE)
    model.train()
    
    for _ in range(config.LOCAL_EPOCHS):
        optimizer.zero_grad()
        logits = model(train_x)
        ce_loss = criterion(logits, train_y)
        W = model.linear.weight  
        gravity_loss = current_lambda * torch.sum(gravity_matrix * torch.abs(W))
        loss = ce_loss + gravity_loss 
        loss.backward()
        optimizer.step()

    return model.state_dict()

def federated_average(client_weights_list):
    avg_weights = {}
    for k in client_weights_list[0].keys():
        avg_weights[k] = torch.stack([w[k] for w in client_weights_list]).mean(dim=0)
    return avg_weights
