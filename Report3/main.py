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

# --- COSTANTI E PERCORSI ---
LEVELS_V1 = [1,3,5,7,9,11,13,15,17,19,21,23]
LEVELS_V2 = [3,7,11,15,19,23]

real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StyleGAN3_path = "/oblivion/Datasets/FFHQ/generated/stylegan3-psi-0.5/images1024x1024"
fake_data_StyleGANXL_path = "/oblivion/Datasets/FFHQ/generated/styleganxl-psi-0.5/images1024x1024"

fake_data_StableDiffusion_path = "/oblivion/Datasets/FFHQ/generated/sdv1_4/images1024x1024"
ACC_THRESHOLD = 0.5

struct_sets_versions = {
    "v1": { 
        "fake_stylegan": fake_data_StyleGAN1_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    },
    "v2": {
        "fake_stylegan": fake_data_StyleGAN1_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    },
    "v3": { 
        "fake_stylegan": fake_data_StyleGAN3_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    },
    "v4": {
        "fake_stylegan": fake_data_StyleGANXL_path,
        "fake_stablediffusion": fake_data_StableDiffusion_path
    }
}

# Fix per importare dal repo locale (mantenuto da entrambi)
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
repo_path = os.path.join(parent_dir, 'ClipBased-SyntheticImageDetection')
if repo_path not in sys.path:
    sys.path.append(repo_path)
from networks import openclipnet
from compute_metrics import compute_metrics, dict_metrics

# ==========================================
# 1. CLASSI E FUNZIONI CONDIVISE (DATALOADING)
# ==========================================

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
           dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=0, pin_memory=True)
           loader[name] = dl
    return loader    

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
            "fake_stylegan": (struct_sets_versions[version]["fake_stylegan"], 1),
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


# ==========================================
# 2. LOGICA EXPERIMENT V1 (Global Token / CLS)
# ==========================================

def train_v1(model_string='mlp', device=None, num_epochs=10, batch_size=32, train_dataset="stylegan"):
    save_dir = f"classificators/{model_string}/{train_dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='train_set')
    val_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='val_set')
    ds = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[f'fake_{train_dataset}'].dataset])
    ds_val = torch.utils.data.ConcatDataset([val_loader['real'].dataset, val_loader[f'fake_{train_dataset}'].dataset])
    
    data_train = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
    data_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
    input_dim = ds[0][0].shape[-1]
    
    for level_idx, level in enumerate(LEVELS_V1):
        print(f"Training V1 classificator for level {level}\n")
        if model_string in ["mlp", "linear"]:
            classificator = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2)).to(device) if model_string == "mlp" else nn.Linear(input_dim, 2).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(classificator.parameters(), lr=0.001)

            best_val_loss = float('inf')
            patience_counter = 0
            best_model_state = None

            for epoch in range(num_epochs):
                classificator.train()
                running_loss = 0.0
                for embeddings, labels, _ in tqdm(data_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                    embeddings_level, labels = embeddings[:, level_idx, :].to(device), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(classificator(embeddings_level), labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() * embeddings.size(0)

                val_loss = validation_classificator(classificator, data_val, level_idx, device)
                if val_loss < best_val_loss:
                    best_val_loss, patience_counter, best_model_state = val_loss, 0, copy.deepcopy(classificator.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= 3: break
            
            if best_model_state: classificator.load_state_dict(best_model_state)            
            torch.save(classificator.state_dict(), f'{save_dir}/classificator_level_{level}.pt')

        elif model_string == "svm":
            all_embeddings, all_labels = [], []
            for embeddings, labels, _ in tqdm(data_train):
                all_embeddings.append(embeddings[:, level_idx, :].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            X, y = np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)
            classificator = CalibratedClassifierCV(SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3), cv=3)
            classificator.fit(X, y)
            joblib.dump(classificator, f'{save_dir}/classificator_level_{level}.pkl')

def test_v1(cross_validate, device=None, model_string="mlp", batch_size=64, test_dataset="stylegan1"):    
    test_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='test_set')
    input_dim = test_loader['real'].dataset[0][0].shape[-1]
    arrays_classificators = []
    
    for level in LEVELS_V1:
        if model_string == "mlp":
            model = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2)).to(device)
            model.load_state_dict(torch.load(f'classificators/{model_string}/{test_dataset}/classificator_level_{level}.pt'))
        elif model_string == "linear":
            model = nn.Linear(input_dim, 2).to(device)
            model.load_state_dict(torch.load(f'classificators/{model_string}/{test_dataset}/classificator_level_{level}.pt'))
        elif model_string == "svm":
            model = joblib.load(f'classificators/{model_string}/{test_dataset}/classificator_level_{level}.pkl')
        
        if hasattr(model, 'eval'): model.eval()
        arrays_classificators.append(model)

    target_fake = "fake_stablediffusion" if (test_dataset == "stylegan1" and cross_validate) or (test_dataset == "stablediffusion" and not cross_validate) else "fake_stylegan1"
    string_cross_val = "_vs_Stable_Diffusion_" if cross_validate and test_dataset=="stylegan1" else ("_vs_SG_" if cross_validate else "_")
    
    ds_test = torch.utils.data.ConcatDataset([test_loader['real'].dataset, test_loader[target_fake].dataset])
    data_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)

    all_labels, all_outputs = [], [[] for _ in LEVELS_V1]
    with torch.no_grad():
        for embeddings, labels, _ in tqdm(data_test):
            all_labels.append(labels.cpu())
            for level_idx, classificator in enumerate(arrays_classificators):
                emb_level = embeddings[:, level_idx, :]
                if model_string in ["mlp", "linear"]:
                    probs = torch.softmax(classificator(emb_level.to(device)), dim=1)[:, 1]
                else:
                    probs = classificator.predict_proba(emb_level.numpy())[:, 1]
                all_outputs[level_idx].append(probs.cpu() if isinstance(probs, torch.Tensor) else torch.tensor(probs))

    all_labels = torch.cat(all_labels).numpy()
    results = []
    for level_idx, level in enumerate(LEVELS_V1):
        preds = torch.cat(all_outputs[level_idx]).numpy() > 0.5
        results.append({'level': level, 'accuracy': sk_metrics.accuracy_score(all_labels, preds)})
    
    os.makedirs("csv_results", exist_ok=True)
    pd.DataFrame(results).to_csv(f"csv_results/accuracy{string_cross_val}data_{model_string}.csv", index=False)


# ==========================================
# 3. LOGICA EXPERIMENT V2 (Patch Specifiche)
# ==========================================

def validation_classificator(model, data_val, level_idx, device, patch=None):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss=  0.0 
    total = 0
    with torch.no_grad():
        for embeddings, labels, _ in data_val:
            if len(embeddings.shape) == 3 and embeddings.shape[-1] == 8192:
                embeddings = embeddings.view(embeddings.size(0), -1, 8, 1024)
            
            if patch is not None:
                embeddings_level, labels = embeddings[:, level_idx, patch, :].to(device)
            else:
                embeddings_level, labels = embeddings[:, level_idx, :, :].to(device)

            labels.to(device)    
            val_loss += criterion(model(embeddings_level), labels).item() * labels.size(0)
            total += labels.size(0)
    return val_loss / total

def train_v2(device=None, num_epochs=10, batch_size=32, train_dataset="stylegan1", version="v2"):
    save_dir = f"classificators_{version}/{train_dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    emb_dir = f"dataset_embeddings_{version}"
    train_loader = get_separated_dataloaders(emb_dir, batch_size=batch_size, split='train_set')
    val_loader = get_separated_dataloaders(emb_dir, batch_size=batch_size, split='val_set')
    
    train_ds = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[f'fake_{train_dataset}'].dataset])
    val_ds = torch.utils.data.ConcatDataset([val_loader['real'].dataset, val_loader[f'fake_{train_dataset}'].dataset])
    
    dl_train = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
    dl_val = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, num_workers=0, shuffle=True)

    models = [[nn.Linear(1024, 2).to(device) for _ in range(8)] for _ in range(len(LEVELS_V2))]
    optimizers = [[torch.optim.Adam(models[i][j].parameters(), lr=0.001, weight_decay=1e-4) for j in range(8)] for i in range(len(LEVELS_V2))]
    criterion = nn.CrossEntropyLoss()

    patch_names = ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]

    for epoch in range(num_epochs):
        for embs, labels, _ in tqdm(dl_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embs, labels = embs.to(device), labels.to(device)
            if len(embs.shape) == 3 and embs.shape[-1] == 8192:
                embs = embs.view(embs.size(0), len(LEVELS_V2), 8, 1024)
            
            for i, _ in enumerate(LEVELS_V2):
                for j in range(8):
                    opt, model = optimizers[i][j], models[i][j]
                    opt.zero_grad()
                    loss = criterion(model(embs[:, i, j, :]), labels)
                    loss.backward()
                    opt.step()
                
                    

    print("Saving V2 models...")
    for i, level_val in enumerate(LEVELS_V2):
        for j in range(8):
            torch.save(models[i][j].state_dict(), os.path.join(save_dir, f"lvl_{level_val}_{patch_names[j]}.pt"))

def test_v2(device=None, batch_size=64, test_dataset="stylegan1", version="v2"):    
    os.makedirs(f"results_csv/{test_dataset}", exist_ok=True)
    target_name = "fake_stablediffusion" if test_dataset == "stylegan1" else "fake_stylegan1"
    
    test_loader = get_separated_dataloaders(f"dataset_embeddings_{version}", batch_size=batch_size, split='test_set', target_datasets=target_name)
    ds_test = torch.utils.data.ConcatDataset([test_loader['real'].dataset, test_loader[target_name].dataset])
    dl_full = torch.utils.data.DataLoader(ds_test, batch_size=batch_size)
    
    all_embeddings, all_labels = [], []
    for embs, lbls, _ in tqdm(dl_full, desc="Loading Test Data"):
        all_embeddings.append(embs)
        all_labels.append(lbls)

    full_X = torch.cat(all_embeddings, dim=0) 
    full_y = torch.cat(all_labels, dim=0).numpy()
    if len(full_X.shape) == 3 and full_X.shape[-1] == 8192:
         full_X = full_X.view(full_X.size(0), len(LEVELS_V2), 8, 1024)
    
    patch_names = ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    all_results = []
    
    for i, level_val in enumerate(tqdm(LEVELS_V2, desc="Evaluating Models")):
        for patch_idx, patch_name in enumerate(patch_names):
            model_path = f"classificators_{version}/{test_dataset}/lvl_{level_val}_{patch_name}.pt"
            if not os.path.exists(model_path): continue
            
            model = nn.Linear(1024, 2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            with torch.no_grad():
                probs = torch.softmax(model(full_X[:, i, patch_idx, :].to(device)), dim=1)[:, 1].cpu().numpy()
            
            acc = sk_metrics.accuracy_score(full_y, (probs >= ACC_THRESHOLD).astype(int))
            all_results.append({"Level": level_val, "Patch": patch_name, "ACC": acc})
            
    pd.DataFrame(all_results).to_csv(f"results_csv/{test_dataset}/test_results_{test_dataset}.csv", index=False)


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
        if version == "v1":
            train_v1(model_string=args['classificator_model'], device=device, num_epochs=args['num_epochs'], batch_size=args['batch_size'], train_dataset=args['dataset'])
        else:
            train_v2(device=device, num_epochs=args['num_epochs'], batch_size=args['batch_size'], train_dataset=args['dataset'], version=version)
            
    elif args['mode'] == "test":
        if version == "v1":
            test_v1(cross_validate=args['cross_validate'], device=device, model_string=args['classificator_model'], batch_size=args['batch_size'], test_dataset=args['dataset'])
        else:
            test_v2(device=device, batch_size=args['batch_size'], test_dataset=args['dataset'], version=version)