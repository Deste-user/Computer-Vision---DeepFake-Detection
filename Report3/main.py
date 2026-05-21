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
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.patheffects as path_effects
import openpyxl
import wandb as wb

#================================
#       HARD CODED VALUES 
#================================
LEVELS_V1 = [1,3,5,7,9,11,13,15,17,19,21,23]
LEVELS_V2 = [3,7,11,15,19,23]
PHYSICAL_PATCH_ORDER = [
    "Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", 
    "Center_TL", "Center_TR", "Center_BL", "Center_BR"
]

real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = "/oblivion/Datasets/FFHQ/generated/sdv1_4/images1024x1024"
fake_data_SG3_path = "/oblivion/Datasets/FFHQ/generated/stylegan3-psi-0.5/images1024x1024"
fake_data_SGXL_path = "/oblivion/Datasets/FFHQ/generated/styleganxl-psi-0.5/images1024x1024"
fake_data_SG2_path = "/oblivion/Datasets/FFHQ/generated/stylegan2-psi-0.5/images1024x1024"
fake_data_StableDiffusion2_path= "/oblivion/Datasets/FFHQ/generated/sdv2_1/images1024x1024"

ACC_THRESHOLD = 0.5

emb_path_array = ["dataset_embeddings_v1\fake1\test_set","dataset_embeddings_v1\fake2\test_set", "dataset_embeddings_v2\fake1\test_set","dataset_embeddings_v2\fake2\test_set",
                    "dataset_embeddings_v3\fake1\test_set","dataset_embeddings_v3\fake2\test_set"]

struct_sets_versions = {
    "v1": { 
        "fake_1": fake_data_StyleGAN1_path,
        "fake_2": fake_data_StableDiffusion_path,
        "names": {"fake_1": "StyleGAN 1", "fake_2": "Stable Diffusion v1.4"},
        "patch_attention": ["CLS"]
    },
    "v2": {
        "fake_1": fake_data_StyleGAN1_path,
        "fake_2": fake_data_StableDiffusion_path,
        "names": {"fake_1": "StyleGAN 1", "fake_2": "Stable Diffusion v1.4"},
        "patch_attention": ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    },
    "v3": { 
        "fake_1": fake_data_SG3_path,
        "fake_2": fake_data_SGXL_path,
        "names": {"fake_1": "StyleGAN 3", "fake_2": "StyleGAN XL"},
        "test_embedding_path": {"fake_1": "dataset_embeddings_v3/test_set/fake_1/embeddings.pt", "fake_2": "dataset_embeddings_v3/test_set/fake_2/embeddings.pt"},
        "patch_attention": ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    },

    "v4": {
        "fake_1": fake_data_SG2_path,
        "fake_2": fake_data_StableDiffusion2_path,
        "names": {"fake_1": "StyleGAN 2.1", "fake_2": {"StableDiffusion2.1"} },
        "test_embedding_path": {"fake_1": "dataset_embeddings_v4/test_set/fake_1/embeddings.pt", "fake_2": "dataset_embeddings_v4/test_set/fake_2/embeddings.pt"},
        "patch_attention": ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
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


# For each dataset and for each split (train_set, val_set, test_set), this function creates embeddings using the OpenCLIP model and saves them as .pt files in a structured directory format.
def create_embeddings(version="v1"):
    emb_folder = f"dataset_embeddings_{version}"
    if not os.path.exists(emb_folder):
        levels_to_extract = LEVELS_V1 if version == "v1" else LEVELS_V2
        token_mode = 'cls' if version == "v1" else 'corners_centers'
        
        model = openclipnet.OpenClipLinear(layer_to_extract=levels_to_extract, token_mode=token_mode)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device} for embedding creation (Version: {version})")

        classes = { 
            "real": (real_data_FFHQ_path, 0),
            "fake_1": (struct_sets_versions[version]["fake_1"], 1),
            "fake_2": (struct_sets_versions[version]["fake_2"], 1)
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



# =============================================
# EXPERIMENT TRAIN/EVAL/TEST (Global Token/CLS)
# =============================================



def validation_all_patches(models, data_val, level_idx, num_levels, num_patches, input_dim, patiences, device,patch_names):
    for m in models: m.eval()
    criterion = nn.CrossEntropyLoss()
    val_losses = [0.0] * num_patches
    val_corrects = [0.0] * num_patches
    total = 0
    
    with torch.no_grad():
        for embeddings, labels, _ in data_val:
            labels = labels.to(device)
            data_num_patches = embeddings.shape[-1] // input_dim
            embeddings = embeddings.view(embeddings.size(0), num_levels, data_num_patches, input_dim)
            
            for p_idx in range(num_patches):
                if patiences[p_idx] >= 3: 
                    continue
                physical_idx = 0 if patch_names[p_idx] == "CLS" else PHYSICAL_PATCH_ORDER.index(patch_names[p_idx])
                emb_patch = embeddings[:, level_idx, physical_idx, :].to(device)

                outputs = models[p_idx](emb_patch)
                loss = criterion(outputs, labels)
                
                val_losses[p_idx] += loss.item() * embeddings.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_corrects[p_idx] += (preds == labels).sum().item()
            
            total += embeddings.size(0)           
    
    val_accuracies = [0.0] * num_patches
    for p_idx in range(num_patches):
        if patiences[p_idx] < 3:
            val_losses[p_idx] /= total
            val_accuracies[p_idx] = val_corrects[p_idx] / total
        else:
            val_losses[p_idx] = float('inf')
            val_accuracies[p_idx] = 0.0
    return val_losses, val_accuracies


# This function trains a classificator for each specified level in LEVELS_V1
def train(model_string='mlp', device=None, num_epochs=10, batch_size=32, train_dataset="fake_1", version="v1", levels=LEVELS_V1):
    
    save_dir = f"classificators_{version}/{model_string}/{train_dataset}"
    os.makedirs(save_dir, exist_ok=True)
    
    train_loader = get_separated_dataloaders(f"dataset_embeddings_{version}", batch_size=batch_size, split='train_set')
    val_loader = get_separated_dataloaders(f"dataset_embeddings_{version}", batch_size=batch_size, split='val_set')
    ds_train = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[train_dataset].dataset])
    ds_val = torch.utils.data.ConcatDataset([val_loader['real'].dataset, val_loader[train_dataset].dataset])
    
    data_train = torch.utils.data.DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=2)
    data_val = torch.utils.data.DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=2)
    input_dim = 1024
    sample = ds_train[0][0]
    data_num_patches = sample.shape[-1] // input_dim
    patch_names = struct_sets_versions[version]["patch_attention"]
    num_patches = len(patch_names)
    
    print (f"Input dimension: {input_dim}, Patches in data: {data_num_patches}, Training {num_patches} patches.")
    

    


    for level_idx, level in enumerate(levels):
        print(f"Training {version} classificator with {model_string} for level {level}\n")
        should_log = level in [3, 7, 11]
        
        if should_log:
            wb.init(
                project="Deepfake-Patch-Detection",
                name=f"{version}_Level_{level}_{train_dataset}_{model_string}",
                group=f"Experiment_{train_dataset}",
                reinit=True
            )

        if model_string in ["mlp", "linear"]:
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

                epoch_train_losses = [0.0] * num_patches 
                epoch_train_corrects = [0.0] * num_patches 
                train_total = 0
                
                for embeddings , labels, _ in tqdm(data_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                        labels = labels.to(device)
                        embeddings = embeddings.view(embeddings.size(0), len(levels), data_num_patches, input_dim)
                        train_total += embeddings.size(0)

                        for p_idx in range(num_patches):
                            if patiences[p_idx] >= 3: continue
                            physical_idx = 0 if patch_names[p_idx] == "CLS" else PHYSICAL_PATCH_ORDER.index(patch_names[p_idx])
                            emb_patch = embeddings[:, level_idx, physical_idx, :].to(device)
                            optimizers[p_idx].zero_grad()
                            outputs = models[p_idx](emb_patch)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizers[p_idx].step()

                            epoch_train_losses[p_idx] += loss.item() * embeddings.size(0)
                            preds = torch.argmax(outputs, dim=1)
                            epoch_train_corrects[p_idx] += (preds == labels).sum().item()
                
                val_losses, val_accuracies = validation_all_patches(models=models, data_val=data_val, level_idx=level_idx, 
                    num_levels=len(levels), num_patches=num_patches, input_dim=input_dim, patiences=patiences, device=device, patch_names=patch_names)

                if should_log: 
                    log_metrics = {"epoch": epoch}
                    for p_idx in range(num_patches):
                        if patiences[p_idx] < 3: 
                            avg_train_loss = epoch_train_losses[p_idx] / train_total
                            train_acc = epoch_train_corrects[p_idx] / train_total
                            
                            p_name = patch_names[p_idx]
                            log_metrics[f"Loss/Training.{p_name}"] = avg_train_loss
                            log_metrics[f"Loss/Validation.{p_name}"] = val_losses[p_idx]
                            log_metrics[f"Accuracy/Training.{p_name}"] = train_acc
                            log_metrics[f"Accuracy/Validation.{p_name}"] = val_accuracies[p_idx]
                            
                    wb.log(log_metrics)


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
                embeddings = embeddings.view(embeddings.size(0), len(levels), data_num_patches, input_dim)
                for p_idx in range(num_patches):
                    physical_idx = 0 if patch_names[p_idx] == "CLS" else PHYSICAL_PATCH_ORDER.index(patch_names[p_idx])
                    all_embeddings[p_idx].append(embeddings[:, level_idx, physical_idx, :].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            y = np.concatenate(all_labels, axis=0)
            
            for p_idx in range(num_patches):
                X = np.concatenate(all_embeddings[p_idx], axis=0)
                classificator = CalibratedClassifierCV(SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3), cv=3)
                classificator.fit(X, y)
                joblib.dump(classificator, f'{save_dir}/classificator_level_{level}_{patch_names[p_idx]}.pkl')
        if should_log:
            wb.finish()              
        
                     

def test(cross_validate=False, device=None, model_string="mlp", batch_size=32, path_train=None, path_test=None):

    train_parts = path_train.replace("\\", "/").split('/')
    train_vers = train_parts[0].replace("dataset_embeddings_", "")
    train_target = train_parts[1]
    train_name = struct_sets_versions[train_vers]["names"][train_target].replace(" ", "")
    
    test_parts = path_test.replace("\\", "/").split('/')
    test_vers = test_parts[0].replace("dataset_embeddings_", "")
    test_target = test_parts[1]
    test_name = struct_sets_versions[test_vers]["names"][test_target].replace(" ", "")

    levels_dict = {
        "v1": LEVELS_V1, 
        "v2": LEVELS_V2, 
        "v3": LEVELS_V2,
    }
    
    train_levels = levels_dict[train_vers]
    test_levels = levels_dict[test_vers]
    train_patches = struct_sets_versions[train_vers]["patch_attention"]
    test_patches = struct_sets_versions[test_vers]["patch_attention"]

    common_levels = [l for l in train_levels if l in test_levels]
    common_patches = [p for p in train_patches if p in test_patches]

    token_type = "_CLS" if common_patches[0] == "CLS" else ""

    if not common_levels or not common_patches:
        raise ValueError(f"CRITICAL ERROR: Nessuna intersezione. Livelli comuni: {len(common_levels)}, Patch comuni: {len(common_patches)}.")

    string_cross_val = f"Train-{train_name}_Test-{test_name}_" if cross_validate else f"{test_name}_"
    emb_dir = f"dataset_embeddings_{test_vers}"
    
    test_loader = get_separated_dataloaders(emb_dir, batch_size=batch_size, split='test_set')
    ds_test = torch.utils.data.ConcatDataset([test_loader['real'].dataset, test_loader[test_target].dataset])
    data_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False)
    
    input_dim = 1024  
    sample = test_loader['real'].dataset[0][0]
    data_num_patches = sample.shape[-1] // input_dim
    load_dir = f"classificators_{train_vers}/{model_string}/{train_target}"

    print(f"Loading {len(common_levels) * len(common_patches)} intersection models...")
    
    loaded_models = {}
    for level in common_levels:
        for patch_name in common_patches:
            model_path = f'{load_dir}/classificator_level_{level}_{patch_name}'
            
            if model_string in ["mlp", "linear"]:
                model = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Linear(256, 2)).to(device) if model_string == "mlp" else nn.Linear(input_dim, 2).to(device)
                model.load_state_dict(torch.load(f"{model_path}.pt", map_location=device, weights_only=True))
                model.eval()
                loaded_models[(level, patch_name)] = model
            elif model_string == "svm":
                loaded_models[(level, patch_name)] = joblib.load(f"{model_path}.pkl")

    all_labels = []
    all_outputs = { (l, p): [] for l in common_levels for p in common_patches }
    
    with torch.no_grad():
        for embeddings, labels, _ in tqdm(data_test, desc="Evaluating"):
            all_labels.append(labels.cpu())
            
            embeddings = embeddings.view(embeddings.size(0), len(test_levels), data_num_patches, input_dim)
            
            for level in common_levels:
                l_idx_test = test_levels.index(level)
                
                for patch_name in common_patches:
                    p_idx_test = 0 if patch_name == "CLS" else PHYSICAL_PATCH_ORDER.index(patch_name)
                    
                    emb_patch = embeddings[:, l_idx_test, p_idx_test, :]
                    model = loaded_models[(level, patch_name)]
                    
                    if model_string in ["mlp", "linear"]:
                        probs = torch.softmax(model(emb_patch.to(device)), dim=1)[:, 1].cpu()
                    else:
                        probs = torch.tensor(model.predict_proba(emb_patch.numpy())[:, 1])
                    
                    all_outputs[(level, patch_name)].append(probs)

    all_labels = torch.cat(all_labels).numpy()
    results = []
    
    for level in common_levels:
        for patch_name in common_patches:
            probs_concat = torch.cat(all_outputs[(level, patch_name)]).numpy() 
            preds = probs_concat >= 0.5 
            acc = sk_metrics.accuracy_score(all_labels, preds)
            auc = sk_metrics.roc_auc_score(all_labels, probs_concat)
            results.append({'Level': level, 'Patch': patch_name, 'Accuracy': acc, 'AUC': auc})
    
    os.makedirs("csv_results", exist_ok=True)
    csv_name = f"csv_results/{string_cross_val}{token_type}{model_string}.csv"
    pd.DataFrame(results).to_csv(csv_name, index=False)
    print(f"\nResults saved successfully to {csv_name}!")

#===========================================
#       Auxiliary functions 
#===========================================

def plot_graph(dir, title, metric):
    dir_imgs_results = f"plots_results/{metric}"

    if not os.path.exists(dir_imgs_results):
        os.makedirs(dir_imgs_results)

    df = pd.read_csv(dir)
    df = df.sort_values(by=[ 'Patch', 'Level'])
    plt.figure(figsize=(10, 6))
    for patch in df['Patch'].unique():
        patch_data = df[df['Patch'] == patch]
        plt.plot(patch_data['Level'], patch_data[metric], marker='o', label=patch)

    plt.title(title.replace("_", " "), fontsize=14)
    plt.xlabel('Level') 
    plt.ylabel(metric)
    plt.xticks(df['Level'].unique())
    plt.grid()
    plt.legend()
    save_path = os.path.join(dir_imgs_results, f"{title.replace(' ', '_')}.png")
    plt.savefig(save_path)
    plt.close()

def plot_all_results(CLS= False):
    csv_dir = "csv_results"
    for file in os.listdir(csv_dir):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, file))
            title_base = file.replace(".csv", "")
            plot_graph(os.path.join(csv_dir, file), f"{title_base}-ACC", "Accuracy")
            plot_graph(os.path.join(csv_dir, file), f"{title_base}-AUC", "AUC")
    return                 


def choosen_acc_to_plot(array_dirs, metric="Accuracy"):
    common_levels = None
    dataframes = {}

    for dir_path in array_dirs:
        df = pd.read_csv(dir_path)
        dataframes[dir_path] = df
        
        df_levels = df['Level'].unique()
        if common_levels is None:
            common_levels = set(df_levels)
        else:    
            common_levels = common_levels.intersection(set(df_levels))
            
    common_levels = sorted(list(common_levels))
    
    if not common_levels:
        print("ATTENZIONE: Nessun livello in comune tra i file forniti!")
        return

    plt.figure(figsize=(10, 6))

    for dir_path, df in dataframes.items():
        df_common = df[df['Level'].isin(common_levels)].sort_values(by=['Patch', 'Level'])

        file_name = os.path.basename(dir_path).replace('.csv', '')

        model_type = file_name.split("_")[-1]

        for patch in df_common['Patch'].unique():
            patch_data = df_common[df_common['Patch'] == patch]

            label_name = f"{patch} ({model_type})"
            
            plt.plot(patch_data['Level'], patch_data[metric], marker='o', label=label_name)
    plt.xticks(common_levels)
    plt.xlabel('Level', fontsize=12)
    plt.ylabel(metric, fontsize=12)
    title =  input("\nName for the plot (without extension): ")
    name = title.replace(" ", "_")
    plt.title(f'{title}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Sposto la legenda fuori dal grafico se ci sono tante linee
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    #Insert name of the plot based on the files compared

    
    plt.savefig(f"plots_results/{name}.png", dpi=300)




def evidence_patch(img_path, folder_name,idx ,patch_dim=14, resize_to=224):
    img = Image.open(img_path).convert('RGB').resize((resize_to, resize_to), Image.BICUBIC)
    
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = img.width, img.height
    mid_x, mid_y = w // 2, h // 2
    
    # Dizionario con sintassi corretta: "Nome": ((coord_x, coord_y), "colore")
    patches_data = {
        # ANGOLI (Corners)
        "Corner_TL": ((0, 0), "green"),                                     # Top-Left
        "Corner_TR": ((w - patch_dim, 0), "red"),                           # Top-Right
        "Corner_BL": ((0, h - patch_dim), "blue"),                          # Bottom-Left
        "Corner_BR": ((w - patch_dim, h - patch_dim), "orange"),            # Bottom-Right
        
        # CENTRI (Centers)
        "Center_TL": ((mid_x - patch_dim, mid_y - patch_dim), "pink"),      # Centro Top-Left
        "Center_TR": ((mid_x, mid_y - patch_dim), "grey"),                  # Centro Top-Right
        "Center_BL": ((mid_x - patch_dim, mid_y), "purple"),                # Centro Bottom-Left
        "Center_BR": ((mid_x, mid_y), "brown")                              # Centro Bottom-Right
    }
    
    alpha = 100
    for _ , ((x, y), color) in patches_data.items():
        r, g, b = ImageColor.getrgb(color)
        
        fill_color = (r, g, b, alpha)
        draw.rectangle([x, y, x + patch_dim, y + patch_dim],fill=fill_color ,outline=color,width=1)

    img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')

    img.save(f"./visualization_dataset/{folder_name}/patches_visualization_{idx}.jpeg", quality=95)
    return img


# ==========================================
#               MAIN & ARGPARSE 
# ==========================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unified Script for Deepfake Detection (V1 & V2)")
    parser.add_argument("--experiment_version", type=str, choices=["v1", "v2", "v3"], default="v2", help="Choose the logic to execute")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--create_embeddings", action='store_true')
    parser.add_argument("--mode", type=str, choices=["train", "test","draw"])
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--dataset", type=str, choices=["fake_1", "fake_2"], default="fake_1", help="Select which fake dataset to use as target (based on version dict)")
    parser.add_argument("--cross_validate", action='store_true', help="Enable cross validation on testing")
    parser.add_argument("--number_of_levels", type=int, default=12, help="Number of levels to extract")
    parser.add_argument("--classificator_model", type=str, choices=["mlp","svm","linear"], default="linear", help="(V1 Only) Model type")
    parser.add_argument("--plot_results", action='store_true', help="Plot all results from CSV files in the csv_results directory")
    parser.add_argument("--evidence_patch", action='store_true', help="Path to an image to visualize the patch locations on it")
    parser.add_argument("--plt_acc_spec", nargs='+', help="Provide paths to specific CSV files to plot accuracy comparison (only if --plot_results is set)")
    args = vars(parser.parse_args())

    if(args['evidence_patch']):
        if not os.path.exists("./visualization_dataset"):
            print("No directory named 'visualization_dataset' found. Please create it and add an image to visualize.")
        for fold in os.listdir("./visualization_dataset"):
            c=0
            for file in os.listdir(os.path.join("./visualization_dataset", fold)):
                if file.lower().endswith((".png", ".jpg")):
                    img_path = os.path.join("./visualization_dataset", fold, file)
                    print(f"Visualizing patches on image: {img_path}")
                    c+=1
                    evidence_patch(img_path, folder_name=fold, idx=c)
        sys.exit(0)


    if (args['plt_acc_spec']):
        choosen_acc_to_plot(args['plt_acc_spec'], metric="Accuracy")
        sys.exit(0)

    device = torch.device(args['device'])
    version = args['experiment_version']


    if args['create_embeddings']:
        create_embeddings(version=version)
        sys.exit(0)

    if args['mode'] == "train":
        train(model_string=args['classificator_model'], device=device, num_epochs=args['num_epochs'], batch_size=args['batch_size'], train_dataset=args['dataset'], version=version, levels=LEVELS_V1 if version == "v1" else LEVELS_V2)
    elif args['mode'] == "test":

        emb_path_options = []
        menu_descriptions = []

        for v_key, config in struct_sets_versions.items():
            for f_key in ["fake_1", "fake_2"]:
                path = f"dataset_embeddings_{v_key}/{f_key}/test_set"
                emb_path_options.append(path)
                
                name = config["names"][f_key]
                patches = ", ".join(config["patch_attention"])
                menu_descriptions.append(f"[{v_key.upper()}] {name} (Tokens: {patches})")

        dataset_chosen = {"training":None , "testing":None}
        
        if args['cross_validate']:
            print("\n" + "="*60)
            print("CROSS-DATASET MODE: Select two different datasets")
            print("="*60)
            for i in range(2):
                role = "TRAIN (Source Weights)" if i == 0 else "TEST (Target Evaluation)"
                for idx, desc in enumerate(menu_descriptions):
                    print(f"{idx + 1}. {desc}")
                
                choice = int(input(f"\nSelect dataset for {role}: ")) - 1
                key = "training" if i == 0 else "testing"
                dataset_chosen[key] = emb_path_options[choice]
                print(f"-> Selected: {dataset_chosen[key]}\n")
        else:
            print("\n" + "="*60)
            print("STANDARD TEST MODE (NO CROSS-DATASET): Select the dataset")
            print("="*60)
            for idx, desc in enumerate(menu_descriptions):
                print(f"{idx + 1}. {desc}")
            
            choice = int(input("\nEnter the dataset number: ")) - 1
            dataset_chosen["training"] = emb_path_options[choice]
            dataset_chosen["testing"] = emb_path_options[choice]

        test(cross_validate=args['cross_validate'], device=device, model_string=args['classificator_model'], batch_size=args['batch_size'], path_train=dataset_chosen["training"], path_test=dataset_chosen["testing"])
    elif args['mode'] == "draw":
        plot_all_results()