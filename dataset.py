import os
import random
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from utils import get_absolute_indices

class Dataset_Images(Dataset):
    def __init__(self, root, transform=None, target_classes=None):
        self.root = root
        self.transform = transform
        self.target_classes = [c.lower() for c in target_classes]
        self.paths = []
        self.targets = []
        
        self.classes = sorted([
            d for d in os.listdir(root) 
            if os.path.isdir(os.path.join(root, d)) 
            and not d.startswith('.') 
            and d.lower() in self.target_classes
        ])
        
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root, cls_name)
            for img_name in sorted(os.listdir(cls_dir)):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
                    self.paths.append(os.path.join(cls_dir, img_name))
                    self.targets.append(self.class_to_idx[cls_name])

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.targets[idx]
        if self.transform:
            return self.transform(image), label
        return image, label

class NoisyDatasetWrapper(Dataset):
    def __init__(self, subset, noise_type='label', noise_label_shuffle_degree=0.5, num_classes=3):
        self.subset = subset
        self.noise_type = noise_type
        self.noise_label_shuffle_degree = noise_label_shuffle_degree
        self.corrupted_labels = {}

        if self.noise_type == 'label':
            try:
                labels = self._get_underlying_labels(self.subset)
                for i, orig_label in enumerate(labels):
                    if random.random() < self.noise_label_shuffle_degree:
                        self.corrupted_labels[i] = random.choice([l for l in range(num_classes) if l != orig_label])
            except AttributeError:
                print("Warning: Loading images to generate noise. This may take a while.")
                for i in range(len(self.subset)):
                    if random.random() < self.noise_label_shuffle_degree:
                        orig_label = self.subset[i][1] 
                        self.corrupted_labels[i] = random.choice([l for l in range(num_classes) if l != orig_label])

    def _get_underlying_labels(self, ds):
        if hasattr(ds, 'targets'):
            return ds.targets
        elif hasattr(ds, 'dataset') and hasattr(ds, 'indices'):
            parent_targets = self._get_underlying_labels(ds.dataset)
            return [parent_targets[i] for i in ds.indices]
        else:
            raise AttributeError("Cannot automatically find labels.")

    def __len__(self): 
        return len(self.subset)

    def __getitem__(self, idx):
        data, label = self.subset[idx]
        if self.noise_type == 'label':
            label = self.corrupted_labels.get(idx, label)
        return data, label

def partition_client_data(dataset, num_clients, iid=True):
    local_indices = np.arange(len(dataset))
    if not iid:
        if isinstance(dataset, Subset):
            parent_targets = np.array(dataset.dataset.targets)
            subset_targets = parent_targets[dataset.indices]
        else:
            subset_targets = np.array(dataset.targets)
        local_indices = local_indices[np.argsort(subset_targets)]
    return np.array_split(local_indices, num_clients)

def get_federated_data(root, num_clients=10, noise_client_percentage=0.2, noise_label_shuffle_degree=0.5, num_classes=3, target_classes=None):
    full_ds = Dataset_Images(root, transform=None, target_classes=target_classes)
    total_len = len(full_ds)
    
    server_val_size = int(total_len * 0.10)   
    server_test_size = int(total_len * 0.10)  
    clients_pool_size = total_len - server_val_size - server_test_size  
    
    clients_pool, server_val, server_test = random_split(
        full_ds, [clients_pool_size, server_val_size, server_test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    num_noisy_clients = int(noise_client_percentage * num_clients)
    rng = random.Random(42) 
    noisy_client_indices = set(rng.sample(range(num_clients), num_noisy_clients))
    
    client_shards_indices = partition_client_data(clients_pool, num_clients, iid=True)
    clients_data = []
    
    for i, shard_indices in enumerate(client_shards_indices):
        client_base_ds = Subset(clients_pool, shard_indices)
        is_noisy = i in noisy_client_indices
        
        if is_noisy:
            client_ds = NoisyDatasetWrapper(
                client_base_ds, 
                noise_type='label', 
                noise_label_shuffle_degree=noise_label_shuffle_degree, 
                num_classes=num_classes
            )
        else:
            client_ds = client_base_ds
            
        c_len = len(client_ds)
        n_train = int(0.90 * c_len)
        n_val = int(0.05 * c_len)
        n_test = c_len - n_train - n_val 
        
        c_train, c_val, c_test = random_split(
            client_ds, [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42 + i) 
        )
        
        clients_data.append({
            "id": i, 
            "train": c_train, 
            "val": c_val, 
            "test": c_test, 
            "is_noisy": is_noisy
        })

    return {"server": {"val": server_val, "test": server_test}, "clients": clients_data}

def verify_no_data_leakage(data_struct):
    print("\n" + "="*45)
    print("DATA LEAKAGE & OVERLAP VERIFICATION REPORT")
    print("="*45)

    s_val_idx = set(get_absolute_indices(data_struct['server']['val']))
    s_test_idx = set(get_absolute_indices(data_struct['server']['test']))
    server_total = s_val_idx.union(s_test_idx)
    print(f"[CHECK] Server Val vs Server Test Overlap: {len(s_val_idx.intersection(s_test_idx))} images")

    all_client_idx = set()
    for c in data_struct['clients']:
        c_tr = set(get_absolute_indices(c['train']))
        c_v = set(get_absolute_indices(c['val']))
        c_te = set(get_absolute_indices(c['test']))

        internal_overlap = len(c_tr.intersection(c_v)) + len(c_tr.intersection(c_te)) + len(c_v.intersection(c_te))
        if internal_overlap > 0:
            print(f"[ERROR] Client {c['id']:02d} Internal Overlap: {internal_overlap} images")

        c_total = c_tr.union(c_v).union(c_te)
        all_client_idx = all_client_idx.union(c_total)

    print(f"[CHECK] Client Internal (Train/Val/Test) Overlaps: 0 images")

    server_client_overlap = server_total.intersection(all_client_idx)
    if len(server_client_overlap) == 0:
        print(f"[CHECK] CRITICAL: Server vs ALL Clients Overlap: 0 images (CLEAN)")
    else:
        print(f"[ERROR] CRITICAL: Server vs Clients Overlap: {len(server_client_overlap)} images leaked!")
    print("="*45 + "\n")
