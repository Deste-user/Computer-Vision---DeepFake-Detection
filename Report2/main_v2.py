import gc
from matplotlib import patches
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
    if target_datasets is not None:
        target_fake = target_datasets
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


def validation_classificator(model, data_val, level_idx, patch, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    total = 0

    with torch.no_grad():
        for embeddings, labels, _ in data_val:
            embeddings = embeddings.to(device)
            labels = labels.to(device)

            if len(embeddings.shape) == 3 and embeddings.shape[-1] == 8192:
                embeddings = embeddings.view(embeddings.size(0), len(levels), 8, 1024)

            embeddings_level = embeddings[:, level_idx, patch, :]
            outputs = model(embeddings_level)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            val_loss += loss.item() * batch_size
            total += batch_size

    return val_loss / total

# Plot training and validation loss curves during epochs for central level.
def plot_loss_curves(array_train_loss, array_eval_loss, num_epochs, train_dataset, patch_names):
    df_train_loss = pd.DataFrame(array_train_loss)
    df_eval_loss = pd.DataFrame(array_eval_loss)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    epochs = range(1, num_epochs + 1)
    for patch in patch_names:
        patch_train_loss = df_train_loss[(df_train_loss["Patch"] == patch) & (df_train_loss["Level"] == levels[2])]["Loss"].values
        plt.plot(epochs, patch_train_loss, marker="o", label=f"{patch} Train")
    
    plt.title("Training Loss Curves (Central Level)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    for patch in patch_names:
        patch_eval_loss = df_eval_loss[(df_eval_loss["Patch"] == patch) & (df_eval_loss["Level"] == levels[2])]["Loss"].values
        plt.plot(epochs, patch_eval_loss, marker="o", label=f"{patch} Val")
        
    plt.title("Validation Loss Curves (Central Level)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()

    os.makedirs("result_images", exist_ok=True)
    plt.savefig(f"result_images/loss_curves_{train_dataset}.png", dpi=300, bbox_inches='tight')          


def train_classificators(device=None, num_epochs=10,batch_size=32, train_dataset="stylegan1"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training.\n", flush=True)
    else:
        print("Using CPU for training.\n", flush=True)
        device = torch.device("cpu")

    save_dir = f"classificators_v2/{train_dataset}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    train_loader = get_separated_dataloaders("dataset_embeddings_v2", batch_size=batch_size, split='train_set')
    val_loader = get_separated_dataloaders("dataset_embeddings_v2", batch_size=batch_size, split='val_set')
    print(train_loader.keys())
    train_ds = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[f'fake_{train_dataset}'].dataset])
    val_ds = torch.utils.data.ConcatDataset([ val_loader['real'].dataset, val_loader[f'fake_{train_dataset}'].dataset])
    BATCH_SIZE = batch_size
    
    dl_train = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=True)
    print (f"Training classificators on dataset '{train_dataset}'")
    print ("Training classificators on dataset with", len(train_ds), "samples.")
    dl_val = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, pin_memory=True)

    models = [[nn.Linear(1024, 2).to(device) for _ in range(8)] for _ in range(len(levels))]
    optimizers = [[torch.optim.Adam(models[i][j].parameters(), lr=0.001, weight_decay=1e-4) for j in range(8)] for i in range(len(levels))]
    criterion = nn.CrossEntropyLoss()

    patch_names = [
        "Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", 
        "Center_TL", "Center_TR", "Center_BL", "Center_BR"
    ]

    array_train_loss = []
    array_eval_loss = []


    for epoch in range(num_epochs):
        epoch_train_losses = [[0.0 for _ in range(8)] for _ in range(len(levels))]

        for embs, labels, _ in tqdm(dl_train, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embs = embs.to(device)    # Shape: [Batch, 6, 8, 1024] => [Batch, Levels, Patches * 1024]
            labels = labels.to(device) # Shape: [Batch]
            if len(embs.shape) == 3 and embs.shape[-1] == 8192:
                #Reform the dimention of embeddings.
                #The size of the batch - number of levels - number of patches and dimention of embeddings
                embs = embs.view(embs.size(0), len(levels), 8, 1024)
            
            # For all the level we take all the embeddings of i level and j patch.
            for i, level_val in enumerate(levels):
                for j in range(8):
                    model = models[i][j]
                    opt = optimizers[i][j]
                    
                    inputs = embs[:, i, j, :]
                    
                    # Forward + Backward
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    epoch_train_losses[i][j] += loss.item()

        num_batches = len(dl_train)
        for i, level_val in enumerate(levels):
            for j in range(8):
                avg_train_loss = epoch_train_losses[i][j] / num_batches
                array_train_loss.append({"Epoch": epoch, "Level": level_val, "Patch": patch_names[j], "Loss": avg_train_loss})


        for i, level_val in enumerate(levels):
            for j in range(8):
                val_loss = validation_classificator(models[i][j], dl_val, i, j, device=device)
                array_eval_loss.append({"Epoch": epoch, "Level": level_val, "Patch": patch_names[j], "Loss": val_loss})

    print("\nCreation of graphs...")
    plot_loss_curves(array_train_loss, array_eval_loss, num_epochs, train_dataset, patch_names)

    print("Saving models...")
    for i, level_val in enumerate(levels):
        for j in range(8):
            filename = f"lvl_{level_val}_{patch_names[j]}.pt"
            torch.save(models[i][j].state_dict(), os.path.join(save_dir, filename))
        

def test_classificators_in_dataset(device=None, batch_size=64, test_dataset="stylegan1"):    
    os.makedirs(f"results_csv/{test_dataset}", exist_ok=True)
    # Sistemato il path per non ripetere il nome del dataset due volte
    results_csv_path = os.path.join("results_csv", test_dataset, f"test_results_{test_dataset}.csv")

    if test_dataset == "stylegan1":
        target_name = "fake_stablediffusion" 
    elif test_dataset == "stablediffusion":
        target_name = "fake_stylegan1"
    else:
        target_name = f"fake_{test_dataset}" 
    
    test_loader = get_separated_dataloaders("dataset_embeddings_v2", batch_size=batch_size, split='test_set', target_datasets=target_name)
    ds_test_real = test_loader['real']        
    ds_test_fake = test_loader[target_name]

    ds_test = torch.utils.data.ConcatDataset([ds_test_real.dataset, ds_test_fake.dataset])
    
    print(f"Loading Test Set (Target: {target_name})...")
    dl_full = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, num_workers=4)
    
    all_embeddings = []
    all_labels = []
    
    for embs, lbls, _ in tqdm(dl_full, desc="Loading Test Data"):
        all_embeddings.append(embs)
        all_labels.append(lbls)

    # Concatenate all embeddings and labels into single tensors    
    full_X = torch.cat(all_embeddings, dim=0) # [N, Levels, Patches, 1024] (o forma piatta)
    full_y = torch.cat(all_labels, dim=0)     # [N]
    

    if len(full_X.shape) == 3 and full_X.shape[-1] == 8192:
         full_X = full_X.view(full_X.size(0), len(levels), 8, 1024)
    
    patch_names = ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", 
                   "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    
    model_dir = f"classificators_v2/{test_dataset}" 
    print(f"Loading models from: {model_dir}")

    all_results_df = []
    targets = full_y.numpy() 

    for i, level_val in enumerate(tqdm(levels, desc="Evaluating Models")):
        for patch_idx in range(8):
            patch_name = patch_names[patch_idx]
            
            model_path = os.path.join(model_dir, f"lvl_{level_val}_{patch_name}.pt")
            if not os.path.exists(model_path):
                print(f"Model not found: {model_path}")
                continue
            
            model = nn.Linear(1024, 2).to(device)
            # weights_only=True è una buona pratica di sicurezza per PyTorch
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
            model.eval()
            
            inputs = full_X[:, i, patch_idx, :].to(device)
            
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()

            acc = sk_metrics.accuracy_score(targets, (probs >= ACC_THRESHOLD).astype(int))
            
            patch_df = pd.DataFrame({
                "Level": [level_val],
                "Patch": [patch_name],
                "ACC": [acc],
            })
            all_results_df.append(patch_df)
            
    final_df = pd.concat(all_results_df, ignore_index=True)
    
    # --- FIX 4: Stampa corretta delle quantità ---
    print(f"\nNumber of test samples evaluated: {len(targets)}") 
    print(f"Number of models tested (rows in CSV): {len(final_df)}") 
    
    final_df.to_csv(results_csv_path, index=False, header=True)
    print(f"Results successfully saved to: {results_csv_path}\n")





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


def plot_metrics_curves(metrics_dir="../Report1/metrics_results",output_dir=None,dataset=None):
    if dataset_name == 'stylegan1':
        files = os.path.join(metrics_dir, "metrics_.csv")
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



def plot_acc_results(output_dir='result_images', dataset_name="stylegan1"):
    file = os.path.join('results_csv',dataset_name ,f"test_results_{dataset_name}.csv")
    if dataset_name == 'stylegan1':
        file_CLS = os.path.join('../Report1/csv_results', "accuracy__SG_vs_Stable_Diffusion_data_linear.csv")
    else:
        file_CLS = os.path.join('../Report1/csv_results', "accuracy__Stable_Diffusion_vs_SG_data_linear.csv")

    df = pd.read_csv(file)
    df_CSL = pd.read_csv(file_CLS)

    patch_names = ["Corner_TL", "Corner_TR", "Corner_BL", "Corner_BR", 
                   "Center_TL", "Center_TR", "Center_BL", "Center_BR"]
    acc_CSL = [df_CSL[df_CSL["level"] == l]["accuracy"].iloc[0] for l in levels]

    
    figure = plt.figure(figsize=(12, 8))
    for p in patch_names:
        df_patch = df[df["Patch"] == p]
        plt.plot(levels,df_patch['ACC'], marker="o", label=p)
    plt.plot(levels, acc_CSL, marker="s", label="CLS", linestyle="--", color="black")
    plt.xlabel("Level")
    plt.ylabel("ACC")
    plt.title(f"Score ACC with models trained on {dataset_name}")
    plt.grid(True, alpha=0.3)
    plt.xticks(levels, [f"level {n}" for n in df_patch["Level"]], rotation=45)
    plt.legend()
    output_file =os.path.join(output_dir,f"ACC_result_{dataset_name}.png")    
    plt.savefig(output_file,dpi=150)    

               


        




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    parser.add_argument("--create_embeddings", action='store_true', help="Flag to create embeddings")
    parser.add_argument("--mode", type=str, choices=["train", "test"], help="Mode: train or test the classificator")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--dataset", type=str, choices=["stylegan1", "stablediffusion"], default="stylegan1", help="Dataset to use for training/testing")
    parser.add_argument("--graphs", action='store_true', help="Create all graphs")


    args= vars(parser.parse_args())
    dataset_name = args['dataset']

    device = torch.device(args['device'])

    if args['create_embeddings']:
        create_embeddings()
        print("Embeddings created. Exiting.")
        sys.exit(0)


    if args['mode'] == "train":
        train_classificators( device, num_epochs=args['num_epochs'], batch_size=args['batch_size'],train_dataset=args['dataset'])
    elif args['mode'] == "test":
        test_classificators_in_dataset( device, batch_size=args['batch_size'], test_dataset=args['dataset'])
        
    if args['graphs']:
        plot_acc_results(dataset_name=dataset_name)