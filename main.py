from sklearn.linear_model import SGDClassifier
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

levels = [1,3,5,7,9,11,13,15,17,19,21,23]
real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = "/oblivion/Datasets/FFHQ/generated/sdv1_4/images1024x1024"
OUTPUT_FILE = "report_result.xlsx"
ACC_THRESHOLD = 0.5

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_path = os.path.join(script_dir, 'ClipBased-SyntheticImageDetection')
sys.path.append(repo_path)
from networks import openclipnet

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
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx], self.image_names[idx]


def get_separated_dataloaders(embeddings_base_path, batch_size=32,split='train_set'):    
    loader = {}

    if not os.path.exists(embeddings_base_path):
        raise FileNotFoundError(f"Embeddings path '{embeddings_base_path}' does not exist.")
    datasets_names=[d for d in os.listdir(embeddings_base_path) if os.path.isdir(os.path.join(embeddings_base_path,d))]
    print (datasets_names)

    for name  in datasets_names:
        pt_path=os.path.join(embeddings_base_path,name,split,"embeddings.pt")
        if os.path.exists(pt_path):
           ds = DataLoaderEmbeddings(pt_path)
           is_train = (split=='train_set')

           dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=4,pin_memory=True)
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
    if not os.path.exists("dataset_embeddings"):
        #Install default the clip version 14 ViT-g-14
        model = openclipnet.OpenClipLinear(layer_to_extract=levels)
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
                out_dir = os.path.join("dataset_embeddings", cls, split)
                os.makedirs(out_dir, exist_ok=True)

                data = create_dataset_embeddings(img_dir, model, label,device=device)
                torch.save(data, os.path.join(out_dir, "embeddings.pt"))
                print(f"Saved embeddings for class '{cls}' split '{split}' to '{out_dir}/embeddings.pt'")

def train_classificators(model_string='mlp', device=None, num_epochs=10,batch_size=32, train_dataset="stylegan1"):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for training.\n", flush=True)
    else:
        print("Using CPU for training.\n", flush=True)
        device = torch.device("cpu")

    if not os.path.exists(f"classificators/{model_string}/{train_dataset}"):
        os.makedirs(f"classificators/{model_string}/{train_dataset}")

    train_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='train_set')
    val_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='val_set')
    ds = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader[f'fake_{train_dataset}'].dataset])
    ds_val = torch.utils.data.ConcatDataset([ds, val_loader['real'].dataset, val_loader[f'fake_{train_dataset}'].dataset])
    BATCH_SIZE = batch_size
    
    data_train = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    data_val = torch.utils.data.DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print (f"Training classificators on dataset '{train_dataset}'")
    print ("Training classificators on dataset with", len(data_train.dataset), "samples.")

    arrays_classificators = []
    input_dim = ds[0][0].shape[-1]
    N_LEVELS = len(levels)
    for level_idx in range(N_LEVELS):
        print(f"Training classificator for level {levels[level_idx]}\n")
        if model_string == "mlp":
            classificator = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            ).to(device)
            # In this Loss is already included Softmax.
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(classificator.parameters(), lr=0.001)

            # Early stopping parameters
            best_val_loss = float('inf')
            patience = 3  # Numero di epoch senza miglioramento prima di fermarsi
            patience_counter = 0
            best_model_state = None

            for epoch in range(num_epochs):
                classificator.train()
                running_loss = 0.0
                for embeddings, labels, _ in tqdm(data_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                    embeddings_level = embeddings[:, level_idx, :].to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    #Forward pass
                    outputs = classificator(embeddings_level)

                    #Criterio di loss
                    loss = criterion(outputs, labels)
                    
                    #avg gradients
                    loss.backward()

                    # Update weights
                    optimizer.step()

                    running_loss += loss.item() * embeddings.size(0)

                epoch_loss = running_loss / len(data_train.dataset)

                # Validation
                classificator.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for embeddings, labels, _ in data_val:
                        embeddings_level = embeddings[:, level_idx, :].to(device)
                        labels = labels.to(device)
                        outputs = classificator(embeddings_level)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item() * embeddings.size(0)
                val_loss /= len(data_val.dataset)
                print(f"\nEpoch {epoch+1}/{num_epochs}, Loss : {epoch_loss:.4f}, Val Loss : {val_loss:.4f}\n")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = classificator.state_dict().copy()
                    print("  Saving best model...\n", flush=True)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping triggered.\n", flush=True)
                        break

            torch.save(classificator.state_dict(), f'classificators/{model_string}/{train_dataset}/classificator_level_{levels[level_idx]}.pt')
        elif model_string == "svm":
            from sklearn.svm import LinearSVC
            from sklearn.calibration import CalibratedClassifierCV    

            all_embeddings = []
            all_labels = []
            for embeddings, labels, _ in tqdm(data_train):
                all_embeddings.append(embeddings[:, level_idx, :].cpu().numpy())
                all_labels.append(labels.cpu().numpy())
            
            X = np.concatenate(all_embeddings, axis=0)
            y = np.concatenate(all_labels, axis=0)

            print(f"Training LinearSVC on {X.shape[0]} samples...", flush=True)
            
            classificator = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3)
            classificator = CalibratedClassifierCV(classificator, cv=3)
            classificator.fit(X, y)
            joblib.dump(classificator, f'classificators/{model_string}/{train_dataset}/classificator_level_{levels[level_idx]}.pkl')
            print(f"Saved!", flush=True)


#TODO: Implement the train with stable diffusion data as well
def test_classificators_in_dataset(cross_validate, device=None, model_string="mlp",batch_size=64, test_dataset="stylegan1"):    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU for testing.")
    else:
        device = torch.device("cpu")    
    
    #Load classificators
    arrays_classificators = []
    for level in levels:
        if model_string == "mlp":
            classificator = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            ).to(device)
            classificator.load_state_dict(torch.load(f'classificators/{model_string}/{test_dataset}/classificator_level_{level}.pt'))
            classificator.eval()
        elif model_string == "svm":
            classificator = joblib.load(f'classificators/{model_string}/{test_dataset}/classificator_level_{level}.pkl')
        

        arrays_classificators.append(classificator)



   ## Testing the classificator on the test set.
    test_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='test_set')
    ds_test_real = test_loader['real']
    string_cross_val = None
    label = None

    if test_dataset == "stylegan1":
        if cross_validate is not True:
            ds_test_fake = test_loader[f'fake_{test_dataset}']
            string_cross_val="_StyleGAN1_data_"
            label = "stylegan1"    
        else:
            ds_test_fake = test_loader['fake_stablediffusion']
            string_cross_val="SG_vs_Stable_Diffusion_data_"
            label = "stablediffusion"    
    elif test_dataset == "stablediffusion":
        if cross_validate is not True:
            ds_test_fake = test_loader[f'fake_{test_dataset}']
            string_cross_val="_Stable_Diffusion_data_"
            label = "stablediffusion"
        else:
            ds_test_fake = test_loader['fake_stylegan1']
            string_cross_val="Stable_Diffusion_vs_SG_data_"
            label = "stylegan1"
    else:
        print("Test dataset not recognized.")
        return        

    print (f"Testing classificators on dataset '{test_dataset}' with cross validation = {cross_validate}")
   

    ds_test = torch.utils.data.ConcatDataset([ds_test_real.dataset, ds_test_fake.dataset])
    data_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=0)

    all_labels = []
    all_filenames = []
    all_types = []
    all_outputs = [[] for _ in levels]

    with torch.no_grad():
        for embeddings, labels, filename in tqdm(data_test):
            all_labels.append(labels.cpu())
            all_filenames.extend(filename)

            types = ['real' if l == 0 else label for l in labels.cpu().numpy()]    
            all_types.extend(types)

            for level_idx, classificator in enumerate(arrays_classificators):
                embeddings_level = embeddings[:, level_idx, :]

                if model_string == "mlp":
                    embeddings_level = embeddings_level.to(device)
                    outputs = classificator(embeddings_level)
                    probs = torch.softmax(outputs, dim=1)[:, 1]  
                    all_outputs[level_idx].append(probs.cpu())
                elif model_string == "svm":
                    probs = classificator.predict_proba(embeddings_level.cpu().numpy())[:, 1] 
                    all_outputs[level_idx].append(torch.tensor(probs))



   # TODO: Save results to a CSV file
   # Create a file .csv with the result of test
    all_labels = torch.cat(all_labels).numpy()
    for level_idx in range(len(levels)):
        all_outputs[level_idx] = torch.cat(all_outputs[level_idx]).numpy()

    
    results = {'filename': all_filenames, 'typ': all_types}
    for level_idx, level in enumerate(levels):
        results[f'level_{level}'] = all_outputs[level_idx]

    df = pd.DataFrame(results)
    df.to_csv("test_results"+string_cross_val+model_string+".csv", index=False)
    print("Test results saved to test_results"+string_cross_val+model_string+".csv")

def compute_metrics_for_file(filepath):
    """Legge un CSV e restituisce un DataFrame 2 righe x N livelli (AUC e ACC)."""
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Errore lettura {filepath}: {e}")
        return None

    if 'typ' not in df.columns:
        return None

    # Ground Truth
    y_true = (df['typ'] != 'real').astype(int)
    
    # Trova e ordina le colonne dei livelli (level_1, level_3...)
    level_cols = [c for c in df.columns if c.startswith('level_')]
    # Ordinamento numerico (per evitare che level_11 venga prima di level_3)
    level_cols.sort(key=lambda x: int(x.split('_')[1]))

    # Dizionario per i risultati
    results = {'Metric': ['AUC', 'ACC']}
    
    # Liste per calcolare la media (AVG) alla fine
    auc_avgs = []
    acc_avgs = []

    for lvl in level_cols:
        y_score = df[lvl].values
        col_vals = []

        # 1. Calcolo AUC
        try:
            if np.all(np.isfinite(y_score)):
                val_auc = metrics.roc_auc_score(y_true, y_score)
            else:
                val_auc = np.nan
        except: val_auc = np.nan
        
        # 2. Calcolo ACC (con soglia 0.5)
        try:
            val_acc = metrics.balanced_accuracy_score(y_true, y_score > ACC_THRESHOLD)
        except: val_acc = np.nan

        # Aggiungi alla colonna corrente
        results[lvl] = [val_auc, val_acc]
        
        if not np.isnan(val_auc): auc_avgs.append(val_auc)
        if not np.isnan(val_acc): acc_avgs.append(val_acc)

    # Crea DataFrame
    res_df = pd.DataFrame(results)
    
    # Imposta la metrica come indice (per pulizia visiva)
    res_df.set_index('Metric', inplace=True)
    
    return res_df

def create_report():
    files = glob.glob("test_results_*.csv")
    files.sort() # Ordine alfabetico dei file
    
    if not files:
        print("Nessun file CSV trovato.")
        return

    print(f"Trovati {len(files)} file. Scrittura su Excel...")

    try:
        with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
            row_pointer = 0
            
            for f in files:
                print(f"Processing: {f}")
                
                # Calcola la tabellina per questo file
                df_res = compute_metrics_for_file(f)
                
                if df_res is not None:
                    # 1. Scrivi il nome del file (come titolo)
                    # Creiamo un piccolo dataframe solo per scrivere il titolo nella cella
                    pd.DataFrame([f]).to_excel(writer, sheet_name='Results', 
                                               startrow=row_pointer, header=False, index=False)
                    row_pointer += 1
                    
                    # 2. Scrivi la tabella dei risultati sotto il titolo
                    df_res.to_excel(writer, sheet_name='Results', 
                                    startrow=row_pointer, float_format="%.4f")
                    
                    # 3. Aggiorna il puntatore per lasciare spazio (Tabella + Titolo + Spazio vuoto)
                    # 2 righe di dati + 1 di header + 2 di spazio = 5 righe di offset
                    row_pointer += 5 
                    
        print(f"\nFatto! File salvato come: {OUTPUT_FILE}")
        
    except Exception as e:
        print(f"Errore salvataggio Excel: {e}")

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
        from compute_metrics import compute_metrics, dict_metrics
        
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
        
        csv_filename = f"test_results{string_cross_val}{model_string}.csv"

        if os.path.exists(csv_filename):
            #print(f"Computing metrics {args['metrics']} on: {csv_filename}")
            #tab_AUC = compute_metrics(csv_filename, csv_filename, dict_metrics['auc']).drop(columns=['AVG'])
            #tab_ACC = compute_metrics(csv_filename, csv_filename, dict_metrics['acc']).drop(columns=['AVG'])
            #tab_AUC.rename(columns={dataset_name: 'AUC'+string_cross_val}, inplace=True)
            #tab_ACC.rename(columns={dataset_name: 'ACC'+string_cross_val}, inplace=True)
            #result = pd.concat([tab_AUC, tab_ACC], axis=1)
            #result.index.name = "Levels"
            #print(result)
            #result.to_excel(f"metrics_results.xlsx") 
            #result.to_csv(f"metrics_results.csv")
            #TODO: REVIEW THIS FUNCTION.
            create_report()
            print("Metrics report created.")
        else:
            print(f"File {csv_filename} not found. Cannot compute metrics.")



