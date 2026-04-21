import gc
import torch
import torch.nn as nn
import open_clip
import sys
import os
import glob
import copy
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn import svm 
from sklearn import metrics as sk_metrics
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV 
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
import openpyxl
#================================
#       HARD CODED VALUES 
#================================
LEVELS_V1 = [1,3,5,7,9,11,13,15,17,19,21,23]
LEVELS_V2 = [3,7,11,15,19,23]

real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = "/oblivion/Datasets/FFHQ/generated/sdv1_4/images1024x1024"
ACC_THRESHOLD = 0.5

struct_sets_versions = {
    "v1": { 
        "fake_stylegan1": fake_data_StyleGAN1_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    },
    "v2": {
        "fake_stylegan1": fake_data_StyleGAN1_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    },
    "v3": { # Aggiungi i percorsi modificati per V3 se necessario
        "fake_stylegan1": fake_data_StyleGAN1_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    }
}

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
repo_path = os.path.join(parent_dir, 'ClipBased-SyntheticImageDetection')
if repo_path not in sys.path:
    sys.path.append(repo_path)
from networks import openclipnet
from compute_metrics import compute_metrics, dict_metrics

# ==========================================
#               DATALOADING
# ==========================================



#Function to create dataset embeddings by processing images through the OpenCLIP model and extracting features from specified layers.
def create_dataset_embeddings(img_dir, model, label, device='cpu'):
    tensors = []
    model.to(device)
    model.eval()
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='commonpool_xl_s13b_b90k')
    sorted_layer_keys = [f'block_{i}' for i in sorted(model.layers_to_extract)]
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    with torch.no_grad():
        for fname in tqdm(files, desc=f"Processing {os.path.basename(img_dir)}"):
            img_path = os.path.join(img_dir, fname)
            try:
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).unsqueeze(0).to(device)
                features_dict = model.forward_features(img)
                layers_list = [features_dict[key].squeeze(0).cpu() for key in sorted_layer_keys if key in features_dict]
                stacked_embeddings = torch.stack(layers_list, dim=0)
                tensors.append({"image": fname, "label": int(label), "embeddings": stacked_embeddings})
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue    
            model.intermediate_features = {}
    return tensors


# For each dataset (real, fake_stylegan1, fake_stablediffusion) and for each split (train_set, val_set, test_set), this function creates embeddings using the OpenCLIP model and saves them as .pt files in a structured directory format.
def create_embeddings(version="v1"):
    emb_folder = f"dataset_embeddings_{version}"
    if not os.path.exists(emb_folder):
        levels_to_extract = LEVELS_V1 if version == "v1" else LEVELS_V2
        token_mode = 'default' if version == "v1" else 'corners_centers'
        
        model = openclipnet.OpenClipLinear(layer_to_extract=levels_to_extract, token_mode=token_mode)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for embedding creation (Version: {version})")

        classes = { 
            "real": (real_data_FFHQ_path, 0),
            "fake_stylegan1": (struct_sets_versions[version]["fake_stylegan1"], 1),
            "fake_stablediffusion": (struct_sets_versions[version]["fake_stablediffusion"], 1)
        }
        
        splits = ['train_set', 'val_set', 'test_set']
        for cls, (base_path, label) in tqdm(classes.items()):
            for split in splits:
                img_dir = os.path.join(base_path, split)
                out_dir = os.path.join(emb_folder, cls, split)
                os.makedirs(out_dir, exist_ok=True)
                data = create_dataset_embeddings(img_dir, model, label, device=device)
                torch.save(data, os.path.join(out_dir, "embeddings.pt"))
                print(f"Saved embeddings for class '{cls}' split '{split}' to '{out_dir}/embeddings.pt'")



# Class custom to load embeddings from .pt files, normalize them, and provide labels and image names for training/testing.
class DataLoaderEmbeddings(torch.utils.data.Dataset):
    def __init__(self, embeddings_file_path):
        print("Loading embeddings from:", embeddings_file_path)
        raw_data = torch.load(embeddings_file_path)
        self.embeddings = torch.stack([item['embeddings'].detach().cpu() for item in raw_data])
        self.labels = torch.tensor([item['label'] for item in raw_data], dtype=torch.long)
        self.image_names = [item['image'] for item in raw_data]

        print(f"Loaded {len(self.embeddings)} embeddings. Shape: {self.embeddings[0].shape}")
        del raw_data
        gc.collect()

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        return emb, self.labels[idx], self.image_names[idx]



# Function to load dataloaders for each dataset (real, fake_stylegan1, fake_stablediffusion) based on the specified split (train_set, val_set, test_set).
def get_separated_dataloaders(embeddings_base_path, batch_size=32, split='train_set', target_datasets=None):    
    loader = {}
    if not os.path.exists(embeddings_base_path):
        raise FileNotFoundError(f"Embeddings path '{embeddings_base_path}' does not exist.")
    
    datasets_names = [d for d in os.listdir(embeddings_base_path) if os.path.isdir(os.path.join(embeddings_base_path, d))]
    if target_datasets is not None:
        datasets_names = [d for d in datasets_names if d == "real" or d == target_datasets]

    for name in datasets_names:
        pt_path = os.path.join(embeddings_base_path, name, split, "embeddings.pt")
        if os.path.exists(pt_path):
           ds = DataLoaderEmbeddings(pt_path)
           is_train = (split == 'train_set')
           dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=2, pin_memory=True)
           loader[name] = dl
    return loader    



# ==========================================
#    EXPERIMENT V1 (Global Token / CLS)
# ==========================================



def validation_all_patches(models, data_val, level_idx, num_levels, num_patches, input_dim, patiences, device):
    for m in models: m.eval()
    criterion = nn.CrossEntropyLoss()
    val_losses = [0.0] * num_patches
    total = 0
    
    with torch.no_grad():
        for embeddings, labels, _ in data_val:
            labels = labels.to(device)
            embeddings = embeddings.view(embeddings.size(0), num_levels, num_patches, input_dim)
            
            for p_idx in range(num_patches):
                if patiences[p_idx] >= 3: 
                    continue
                
                emb_patch = embeddings[:, level_idx, p_idx, :].to(device)
                val_losses[p_idx] += criterion(models[p_idx](emb_patch), labels).item() * embeddings.size(0)
            
            total += embeddings.size(0)           
    for p_idx in range(num_patches):
        if patiences[p_idx] < 3:
            val_losses[p_idx] /= total
        else:
            val_losses[p_idx] = float('inf') 
    return val_losses 


# This function trains a classificator for each specified level in LEVELS_V1
def train(model_string='mlp', device=None, num_epochs=10, batch_size=32, train_dataset="stylegan1", version="v1", levels=LEVELS_V1):
    version_patch ="CLS" if version == "v1" else "eight_patches"
    save_dir = f"classificators/{model_string}/{train_dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_separated_dataloaders(f"dataset_embeddings_{version}", batch_size=batch_size, split='train_set')
    val_loader = get_separated_dataloaders(f"dataset_embeddings_{version}", batch_size=batch_size, split='val_set')
    ds_train = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[f'fake_{train_dataset}'].dataset])
    ds_val = torch.utils.data.ConcatDataset([val_loader['real'].dataset, val_loader[f'fake_{train_dataset}'].dataset])
    
    data_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    data_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
    input_dim = ds_train[0][0].shape[-1]
    sample = ds_train[0][0]
    num_patches = sample.numel() // (len(levels) * input_dim)
    patch_names = ["CLS"] if num_patches == 1 else ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    print (f"Input dimension: {input_dim}, Number of patches: {num_patches} (Sample shape: {sample.shape})")
    
    for level_idx, level in enumerate(levels):
        print(f"Training {version} classificator for level {level}\n")

        if model_string not in ["mlp", "linear"]:
            models = []
            for _ in range(num_patches):
                if model_string == "mlp":
                    models.append(nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2)).to(device))
                else:
                    models.append(nn.Linear(input_dim, 2).to(device))

            optimizers = [torch.optim.AdamW(m.parameters(), lr=0.001) for m in models]
            criterion = nn.CrossEntropyLoss()    

            best_val_losses = [float('inf')] * num_patches
            patiences = [0] * num_patches
            best_states = [None] * num_patches

            for epoch in range(num_epochs):
                if all(p >= 3 for p in patiences): break

                for m in models: m.train()
                
                for embeddings , labels in tqdm(data_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                        labels = labels.to(device)
                        embeddings = embeddings.view(embeddings.size(0), len(levels), num_patches, input_dim)

                        for p_idx in range(num_patches):
                            if patiences[p_idx] >= 3: continue

                            emb_patch = embeddings[:, level_idx, p_idx, :]
                            optimizers[p_idx].zero_grad()
                            loss = criterion(models[p_idx](emb_patch),labels)
                            loss.backward()
                            optimizers[p_idx].step()
                
                val_losses = validation_all_patches(models=models, data_val=data_val, level_idx=level_idx, 
                    num_levels=len(levels), num_patches=num_patches, input_dim=input_dim, patiences=patiences, device=device )
                
                for p_idx in range(num_patches):
                    if patiences[p_idx] >= 3: continue
    
                    if val_losses[p_idx] < best_val_losses[p_idx]:
                        best_val_losses[p_idx] = val_losses[p_idx]
                        patiences[p_idx] = 0
                        best_states[p_idx] = copy.deepcopy(models[p_idx].state_dict())
                    else:
                        patiences[p_idx] += 1

            for p_idx in range(num_patches):
                if best_states[p_idx]: 
                    models[p_idx].load_state_dict(best_states[p_idx])
                torch.save(models[p_idx].state_dict(), f'{save_dir}/classificator_level_{level}_{patch_names[p_idx]}.pt')
             
        elif model_string == "svm":
            # Per l'SVM salviamo i dati in liste separate per ogni patch
            all_embeddings = [[] for _ in range(num_patches)]
            all_labels = []
            
            for embeddings, labels, _ in tqdm(data_train):
                embeddings = embeddings.view(embeddings.size(0), len(levels), num_patches, input_dim)
                for p_idx in range(num_patches):
                    all_embeddings[p_idx].append(embeddings[:, level_idx, p_idx, :].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            y = np.concatenate(all_labels, axis=0)
            
            for p_idx in range(num_patches):
                X = np.concatenate(all_embeddings[p_idx], axis=0)
                classificator = CalibratedClassifierCV(SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3), cv=3)
                classificator.fit(X, y)
                joblib.dump(classificator, f'{save_dir}/classificator_level_{level}_{patch_names[p_idx]}.pkl')

                     

def test(cross_validate=False, device=None, model_string="mlp", batch_size=64, test_dataset="stylegan1", version="unified", levels=LEVELS_V1):

    target_fake = "fake_stablediffusion" if (test_dataset == "stylegan1" and cross_validate) or (test_dataset == "stablediffusion" and not cross_validate) else "fake_stylegan1"
    string_cross_val = "_vs_Stable_Diffusion_" if cross_validate and test_dataset=="stylegan1" else ("_vs_SG_" if cross_validate else "_")
    
    emb_dir = f"dataset_embeddings_{version}"
    
    test_loader = get_separated_dataloaders(emb_dir, batch_size=batch_size, split='test_set')
    ds_test = torch.utils.data.ConcatDataset([test_loader['real'].dataset, test_loader[target_fake].dataset])
    data_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    input_dim = test_loader['real'].dataset[0][0].shape[-1]
    sample_embedding = test_loader['real'].dataset[0][0]
    num_patches = sample_embedding.numel() // (len(levels) * input_dim)
    patch_names = ["CLS"] if num_patches == 1 else ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    load_dir = f"classificators_{version}/{model_string}/{test_dataset}"

    print(f"Loading {len(levels) * num_patches} models...")
    loaded_models = [[None for _ in range(num_patches)] for _ in range(len(levels))]
    
    for l_idx, level in enumerate(levels):
        for p_idx, patch_name in enumerate(patch_names):
            model_path = f'{load_dir}/classificator_level_{level}_{patch_name}'
            
            if model_string in ["mlp", "linear"]:
                model = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2)).to(device) if model_string == "mlp" else nn.Linear(input_dim, 2).to(device)
                model.load_state_dict(torch.load(f"{model_path}.pt", map_location=device, weights_only=True))
                model.eval()
                loaded_models[l_idx][p_idx] = model
            elif model_string == "svm":
                loaded_models[l_idx][p_idx] = joblib.load(f"{model_path}.pkl")

    all_labels = []
    all_outputs = [[[] for _ in range(num_patches)] for _ in range(len(levels))]
    
    with torch.no_grad():
        for embeddings, labels, _ in tqdm(data_test, desc="Evaluating"):
            all_labels.append(labels.cpu())
            
            embeddings = embeddings.view(embeddings.size(0), len(levels), num_patches, input_dim)
            
            for l_idx in range(len(levels)):
                for p_idx in range(num_patches):
                    emb_patch = embeddings[:, l_idx, p_idx, :]
                    
                    if model_string in ["mlp", "linear"]:
                        probs = torch.softmax(loaded_models[l_idx][p_idx](emb_patch.to(device)), dim=1)[:, 1].cpu()
                    else:
                        probs = torch.tensor(loaded_models[l_idx][p_idx].predict_proba(emb_patch.numpy())[:, 1])
                    
                    all_outputs[l_idx][p_idx].append(probs)

    # 5. Calcolo delle metriche e salvataggio
    all_labels = torch.cat(all_labels).numpy()
    results = []
    
    for l_idx, level in enumerate(levels):
        for p_idx, patch_name in enumerate(patch_names):
            # Uniamo tutte le probabilità dei batch per questa specifica patch
            preds = torch.cat(all_outputs[l_idx][p_idx]).numpy() >= 0.5 
            acc = sk_metrics.accuracy_score(all_labels, preds)
            results.append({'Level': level, 'Patch': patch_name, 'Accuracy': acc})
    
    os.makedirs("csv_results", exist_ok=True)
    csv_name = f"csv_results/accuracy{string_cross_val}{version}_{test_dataset}_{model_string}.csv"
    pd.DataFrame(results).to_csv(csv_name, index=False)
    print(f"\nResults saved successfully to {csv_name}!")


# ==========================================
# 4. MAIN & ARGPARSE UNIFICATO
# ==========================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Script for Deepfake Detection (V1 & V2)")
    parser.add_argument("--experiment_version", type=str, choices=["v1", "v2", "v3"], default="v2", help="Choose the logic to execute")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--create_embeddings", action='store_true')
    parser.add_argument("--mode", type=str, choices=["train", "test"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dataset", type=str, choices=["stylegan1", "stablediffusion"], default="stylegan1")
    parser.add_argument("--cross_validate", action='store_true', help="Enable cross validation on testing")
    parser.add_argument("--number of levels", type=int, default=12, help="Number of levels to extract (max 12 for V1)")
    
    # Argomenti esclusivi V1
    parser.add_argument("--classificator_model", type=str, choices=["mlp","svm","linear"], default="linear", help="(V1 Only) Model type")
    parser.add_argument("--cross_validate", action='store_true', help="(V1 Only) Enable cross validation on testing")

    args = vars(parser.parse_args())
    device = torch.device(args['device'])
    version = args['experiment_version']

    print(f"--- RUNNING EXPERIMENT {version.upper()} ---")

    if args['create_embeddings']:
        create_embeddings(version=version)
        sys.exit(0)

    if args['mode'] == "train":
        train(model_string=args['classificator_model'], device=device, num_epochs=args['num_epochs'], batch_size=args['batch_size'], train_dataset=args['dataset'], version=version, levels=LEVELS_V1 if version == "v1" else LEVELS_V2)
    elif args['mode'] == "test":
        test(cross_validate=args['cross_validate'], device=device, model_string=args['classificator_model'], batch_size=args['batch_size'], test_dataset=args['dataset'], version=version, levels=LEVELS_V1 if version == "v1" else LEVELS_V2)