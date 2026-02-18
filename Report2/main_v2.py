import gc
import torch
import torch.nn as nn
import open_clip
import sys
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn import svm 
from sklearn import metrics as sk_metrics
import joblib
from PIL import Image
import glob
import openpyxl
import copy
import matplotlib.pyplot as plt
# Half elements of the previous level array.
levels = [3,7,11,15,19,23]
real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = "/oblivion/Datasets/FFHQ/generated/sdv1_4/images1024x1024"
ACC_THRESHOLD = 0.5

script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
repo_path = os.path.join(script_dir, 'ClipBased-SyntheticImageDetection')
sys.path.append(repo_path)
from networks import openclipnet
from compute_metrics import compute_metrics, dict_metrics

class DataLoaderEmbeddings(torch.utils.data.Dataset):
    def __init__(self, embeddings_file_path):
        print("Loading embeddings from:", embeddings_file_path)
        
        raw_data = torch.load(embeddings_file_path)
        self.embeddings = torch.stack([item['embeddings'].detach().cpu() for item in raw_data])
        self.labels = torch.tensor([item['label'] for item in raw_data], dtype=torch.long)
        self.image_names = [item['image'] for item in raw_data]

        print(f"Loaded {len(self.embeddings)} embeddings.")
        print(f"Embedding shape: {self.embeddings[0].shape}")
        print(f"Loaded labels: {len(self.labels)}")
        del raw_data
        gc.collect()
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        emb = self.embeddings[idx]

        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        return emb, self.labels[idx], self.image_names[idx]


def get_separated_dataloaders(embeddings_base_path, batch_size=32,split='train_set', target_datasets=None):    
    loader = {}

    if not os.path.exists(embeddings_base_path):
        raise FileNotFoundError(f"Embeddings path '{embeddings_base_path}' does not exist.")
    
    datasets_names=[d for d in os.listdir(embeddings_base_path) if os.path.isdir(os.path.join(embeddings_base_path,d))]
    print (datasets_names)
    if target_datasets is not None:
        target_fake = f"fake_{target_datasets}"
        datasets_names = [d for d in datasets_names if d == "real" or d == target_fake]

    for name  in datasets_names:
        pt_path=os.path.join(embeddings_base_path,name,split,"embeddings.pt")
        if os.path.exists(pt_path):
           ds = DataLoaderEmbeddings(pt_path)
           is_train = (split=='train_set')

           dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=0,pin_memory=True)
           loader[name] = dl
    return loader    


def create_dataset_embeddings(img_dir, model, label, device='cpu'):
    tensors = []
    model.to(device)
    model.eval()

    # Preprocessing per CLIP
    _, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='commonpool_xl_s13b_b90k')

    sorted_layer_keys = [f'block_{i}' for i in sorted(model.layers_to_extract)]
    
    files = [f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    with torch.no_grad():
        for fname in tqdm(files, desc=f"Processing {os.path.basename(img_dir)}"):
            img_path = os.path.join(img_dir, fname)
        
            try:
                # Carica e preprocessa l'immagine
                img = Image.open(img_path).convert('RGB')
                img = preprocess(img).unsqueeze(0).to(device)

                features_dict = model.forward_features(img)

                layers_list = [features_dict[key].squeeze(0).cpu() for key in sorted_layer_keys if key in features_dict]
                stacked_embeddings = torch.stack(layers_list, dim=0)
                
                tensors.append({
                    "image": fname,
                    "label": int(label),
                    "embeddings": stacked_embeddings
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue    

            model.intermediate_features = {}
    return tensors


def create_embeddings():
    if not os.path.exists("dataset_embeddings_v2"):
        #Install default the clip version 14 ViT-g-14
        model = openclipnet.OpenClipLinear(layer_to_extract=levels,token_mode='corners_centers')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        real_imgs_db=os.listdir(real_data_FFHQ_path)    
        fake_imgs_db_stylegan1=os.listdir(fake_data_StyleGAN1_path)
        fake_imgs_db_stablediffusion=os.listdir(fake_data_StableDiffusion_path)

        classes = { "real": (real_data_FFHQ_path, 0),
            "fake_stylegan1": (fake_data_StyleGAN1_path, 1),
            "fake_stablediffusion": (fake_data_StableDiffusion_path, 1)}
        
        splits = ['train_set', 'val_set', 'test_set']

        for cls, (base_path, label) in tqdm(classes.items()):
            for split in splits:
                img_dir = os.path.join(base_path, split)
                out_dir = os.path.join("dataset_embeddings_v2", cls, split)
                os.makedirs(out_dir, exist_ok=True)

                data = create_dataset_embeddings(img_dir, model, label, device=device)
                torch.save(data, os.path.join(out_dir, "embeddings.pt"))
                print(f"Saved embeddings for class '{cls}' split '{split}' to '{out_dir}/embeddings.pt'")


def train_classificators(model_string='mlp', device=None, num_epochs=10,batch_size=32, train_dataset="stylegan1"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training.\n", flush=True)
    else:
        print("Using CPU for training.\n", flush=True)
        device = torch.device("cpu")

    save_dir = f"classificators_v2/{train_dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loader = get_separated_dataloaders("dataset_embeddings_v2", batch_size=batch_size, split='train_set',target_datasets=train_dataset)
    val_loader = get_separated_dataloaders("dataset_embeddings_v2", batch_size=batch_size, split='val_set', target_datasets=train_dataset)
    full_dataset = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[f'fake_{train_dataset}'].dataset, val_loader['real'].dataset, val_loader[f'fake_{train_dataset}'].dataset])
    BATCH_SIZE = batch_size
    
    dl_full = torch.utils.data.DataLoader(full_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=True)
    print (f"Training classificators on dataset '{train_dataset}'")
    print ("Training classificators on dataset with", len(full_dataset), "samples.")

    models = [[nn.Linear(1024, 2).to(device) for _ in range(8)] for _ in range(len(levels))]
    optimizers = [[torch.optim.Adam(models[i][j].parameters(), lr=0.001, weight_decay=1e-4) for j in range(8)] for i in range(len(levels))]
    criterion = nn.CrossEntropyLoss()

    patch_names = [
        "Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", 
        "Center_TL", "Center_TR", "Center_BL", "Center_BR"
    ]

    for epoch in range(num_epochs):
        for embs, labels, _ in tqdm(dl_full, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embs = embs.to(device)    # Shape: [Batch, 6, 8, 1024]
            labels = labels.to(device) # Shape: [Batch]
            if len(embs.shape) == 3 and embs.shape[-1] == 8192:
                embs = embs.view(embs.size(0), len(levels), 8, 1024)
            
            # Per ogni batch, aggiorniamo tutti e 48 i modelli
            for i, level_val in enumerate(levels):
                for j in range(8):
                    model = models[i][j]
                    opt = optimizers[i][j]
                    
                    # Estraiamo la fetta di tensore corretta
                    inputs = embs[:, i, j, :]
                    
                    # Forward + Backward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    

    print("Saving models...")
    for i, level_val in enumerate(levels):
        for j in range(8):
            filename = f"lvl_{level_val}_{patch_names[j]}.pt"
            torch.save(models[i][j].state_dict(), os.path.join(save_dir, filename))
        


def test_classificators_in_dataset(cross_validate, device=None, model_string="mlp",batch_size=64, test_dataset="stylegan1"):    
    print(f"--- AVVIO TEST: Patch-wise con nn.Linear ---")
    
    test_loader = get_separated_dataloaders("dataset_embeddings_v2", batch_size=batch_size, split='test_set')
    ds_test_real = test_loader['real']
    
    if test_dataset == "stylegan1":
        target_name = "fake_stablediffusion" 
    elif test_dataset == "stablediffusion":
        target_name = "fake_stylegan1" 
        
    ds_test_fake = test_loader[target_name]
    ds_test = torch.utils.data.ConcatDataset([ds_test_real.dataset, ds_test_fake.dataset])
    
    print("Loading Test Set...")
    dl_full = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=4)
    
    all_embeddings = []
    all_labels = []
    for embs, lbls, _ in tqdm(dl_full, desc="Loading Test Data"):
        all_embeddings.append(embs)
        all_labels.append(lbls)
        
    full_X = torch.cat(all_embeddings, dim=0) # [N, Levels, Patches, 1024]
    full_y = torch.cat(all_labels, dim=0)     # [N]
    
    # 3. Setup Matrice Risultati
    patch_names = ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", 
                   "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    
    
    train_source = test_dataset
    
    model_dir = f"classificators/{test_dataset}"
    print(f"Loading models from: {model_dir}")

    for i, level_val in enumerate(tqdm(levels, desc="Evaluating")):
        for patch_idx in range(8):
            patch_name = patch_names[patch_idx]
            
            model_path = os.path.join(model_dir, f"lvl_{level_val}_{patch_name}.pt")
            if not os.path.exists(model_path):
                continue
            
            model = nn.Linear(1024, 2).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            inputs = full_X[:, i, patch_idx, :].to(device)
            targets = full_y.numpy() # Sklearn vuole numpy per le metriche
            
            with torch.no_grad():
                outputs = model(inputs)
                # Softmax per avere probabilità tra 0 e 1
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            # Salva risultati in CSV
            results_df = pd.DataFrame({                
                "Level": [level_val] * len(probs),
                "Patch": [patch_name] * len(probs),
                "Probs": probs,
                "Labels": targets
            })
            output_file = os.path.join(model_dir, f"results_{level_val}_{patch_name}.csv")
            results_df.to_csv(output_file, index=False)
            


def _read_metrics_file(filepath):
    df = pd.read_csv(filepath)
    if not {"Levels", "AUC", "ACC"}.issubset(df.columns):
        raise ValueError(f"File {filepath} must contain columns: Levels, AUC, ACC")
    df = df.copy()
    df["LevelNum"] = df["Levels"].str.replace("level_", "", regex=False).astype(int)
    df.sort_values("LevelNum", inplace=True)
    return df


def build_metrics_summary_table(metrics_dir="metrics_results", output_xlsx="metrics_summary.xlsx"):
    files = sorted(glob.glob(os.path.join(metrics_dir, "metrics_*.csv")))
    if not files:
        print("Nessun file metrics_*.csv trovato.")
        return None

    #os.makedirs(os.path.dirname(output_xlsx), exist_ok=True)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        row_pointer = 0
        for f in files:
            df = _read_metrics_file(f)
            level_labels = df["LevelNum"].apply(lambda n: f"level_{n}").tolist()
            table = pd.DataFrame([df["AUC"].values, df["ACC"].values],
                                 index=["AUC", "ACC"],
                                 columns=level_labels)
            table.index.name = "Metrics"

            title = os.path.basename(f)
            pd.DataFrame([title]).to_excel(writer, sheet_name="Results",
                                           startrow=row_pointer, header=False, index=False)
            row_pointer += 1
            table.to_excel(writer, sheet_name="Results", startrow=row_pointer)
            row_pointer += len(table.index) + 3

    print(f"Report Excel salvato in: {output_xlsx}")
    return output_xlsx


def plot_metrics_curves(metrics_dir="metrics_results", output_dir=None):
    files = sorted(glob.glob(os.path.join(metrics_dir, "metrics_*.csv")))
    if not files:
        print("Nessun file metrics_*.csv trovato.")
        return

    if output_dir is None:
        output_dir = os.path.join(metrics_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)

    for f in files:
        df = _read_metrics_file(f)
        title = os.path.splitext(os.path.basename(f))[0]

        plt.figure(figsize=(8, 4.5))
        plt.plot(df["LevelNum"], df["AUC"], marker="o", label="AUC")
        plt.plot(df["LevelNum"], df["ACC"], marker="o", label="ACC")
        plt.xlabel("Level")
        plt.ylabel("Score")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(df["LevelNum"], [f"level_{n}" for n in df["LevelNum"]], rotation=45)
        out_path = os.path.join(output_dir, f"{title}.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Plot salvato in: {out_path}")





if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    parser.add_argument("--classificator_model",type=str, choices=["mlp","svm"], default="mlp", help="Type of classificator model to use")
    parser.add_argument("--create_embeddings", action='store_true', help="Flag to create embeddings")
    parser.add_argument("--mode", type=str, choices=["train", "test"], help="Mode: train or test the classificator")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--metrics", action='store_true', help="Flag to compute metrics after testing and save results")
    parser.add_argument("--cross_validate", action='store_true', help="Flag to cross validate on stable diffusion data during testing")
    parser.add_argument("--dataset", type=str, choices=["stylegan1", "stablediffusion"], default="stylegan1", help="Dataset to use for training/testing")
    parser.add_argument("--report", action='store_true', help="Create a complete report of all experiments")
    parser.add_argument("--graphs", action='store_true', help="Create all graphs")


    args= vars(parser.parse_args())
    dataset_name = args['dataset']

    device = torch.device(args['device'])

    if args['create_embeddings']:
        create_embeddings()
        print("Embeddings created. Exiting.")
        sys.exit(0)

    if args['cross_validate']:
        cross_val = True
        print("Cross validation enabled.")
    else:
        print("Cross validation disabled.")
        cross_val = False    

    model_string = args['classificator_model']

    if args['mode'] == "train":
        train_classificators(model_string, device, num_epochs=args['num_epochs'], batch_size=args['batch_size'],train_dataset=args['dataset'])
    elif args['mode'] == "test":
        test_classificators_in_dataset(cross_val, device, model_string, batch_size=args['batch_size'], test_dataset=args['dataset'])

    if args['metrics']:
        string_cross_val = ""
        if dataset_name == "stylegan1":
            if not cross_val:
                string_cross_val = "_StyleGAN1_data_"
            else:
                string_cross_val = "_SG_vs_Stable_Diffusion_data_"
        elif dataset_name == "stablediffusion":
            if not cross_val:
                string_cross_val = "_Stable_Diffusion_data_"
            else:
                string_cross_val = "_Stable_Diffusion_vs_SG_data_"
        
        os.makedirs("metrics_results",exist_ok=True)
        csv_filename = f"csv_results/test_results{string_cross_val}{model_string}.csv" 

        if os.path.exists(csv_filename):
            tab_ACC = compute_metrics(csv_filename, csv_filename, dict_metrics['acc']).drop(columns=['AVG'])
            tab_ACC.columns = ["ACC"]
            result = pd.concat([tab_ACC], axis=1)
            result.index.name = "Levels"
            result.to_csv(f"metrics_results/metrics_{string_cross_val}{model_string}.csv")
            #create_report()
            print("Metrics report created.")
        else:
            print(f"File {csv_filename} not found. Cannot compute metrics.")

    if args['report']:
        build_metrics_summary_table()
    if args['graphs']:
        plot_metrics_curves(output_dir="result_images")