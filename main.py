import torch
import torch.nn as nn
#import open_clip
import sys
from tqdm import tqdm
import os
import numpy as np
import pandas as pd
from sklearn import svm 
from sklearn import metrics as sk_metrics
import joblib
from PIL import Image

levels = [1,3,5,7,9,11,13,15,17,19,21,23]
real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = "/oblivion/Datasets/FFHQ/generated/sdv1_4/images1024x1024"
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


def get_separated_dataloaders(embeddings_base_path, batch_size=32,split='train'):    
    loader = {}

    if not os.path.exists(embeddings_base_path):
        raise FileNotFoundError(f"Embeddings path '{embeddings_base_path}' does not exist.")
    datasets_names=[d for d in os.listdir(embeddings_base_path) if os.path.isdir(os.path.join(embeddings_base_path,d))]

    for name  in datasets_names:
        pt_path=os.path.join(embeddings_base_path,name,split,"embeddings.pt")
        if os.path.exists(pt_path):
           ds = DataLoaderEmbeddings(pt_path)
           is_train = (split=='train')

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

                data = create_dataset_embeddings(img_dir, model, label)
                torch.save(data, os.path.join(out_dir, "embeddings.pt"))
                print(f"Saved embeddings for class '{cls}' split '{split}' to '{out_dir}/embeddings.pt'")

def train_classificators(model_string, device, num_epochs=10,batch_size=64):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    if not os.path.exists(f"classificators/{model_string}"):
        os.makedirs(f"classificators/{model_string}")

        train_loader = get_separated_dataloaders("dataset_embeddings", batch_size=64, split='train')
        ds = torch.utils.data.ConcatDataset([train_loader['real'].dataset, train_loader['fake_stylegan1'].dataset])
        BATCH_SIZE = 128
        data_train = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

        arrays_classificators = []
        N_LEVELS = len(levels)
        for level_idx in range(N_LEVELS):
            print(f"Training classificator for level {levels[level_idx]}")
            if model_string == "mlp":
                classificator = nn.Sequential(
                    nn.Linear(768, 256),
                    nn.ReLU(),
                    nn.Linear(256, 2)
                ).to(device)

                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(classificator.parameters(), lr=0.001)

                for epoch in tqdm(range(num_epochs)):
                    classificator.train()
                    running_loss = 0.0
                    for embeddings, labels, _ in data_train:
                        embeddings_level = embeddings[:, level_idx, :].to(device)
                        labels = labels.to(device)

                        optimizer.zero_grad()

                        #Forward pass
                        outputs = classificator(embeddings_level)

                        #Criiterio di loss
                        loss = criterion(outputs, labels)
                        
                        #avg gradients
                        loss.backward()

                        # Update weights
                        optimizer.step()

                        running_loss += loss.item() * embeddings.size(0)

                    epoch_loss = running_loss / len(data_train.dataset)
                    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

                torch.save(classificator.state_dict(), f'classificators/{model_string}/classificator_level_{levels[level_idx]}.pt')
            elif model_string == "svm":
                all_embeddings = []
                all_labels = []
                for embeddings, labels, _ in data_train:
                    all_embeddings.append(embeddings[:, level_idx, :].cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                
                X = np.concatenate(all_embeddings, axis=0)
                y = np.concatenate(all_labels, axis=0)
                
                classificator = svm.SVC(probability=True)
                classificator.fit(X, y)
                joblib.dump(classificator, f'classificators/{model_string}/classificator_level_{levels[level_idx]}.pkl')

                    


def test_classificators_in_dataset(device, model_string="svm",batch_size=64):    
    #Load classificators
    arrays_classificators = []
    for level in levels:
        if model_string == "mlp":
            classificator = nn.Sequential(
                nn.Linear(768, 256),
                nn.ReLU(),
                nn.Linear(256, 2)
            ).to(device)
            classificator.load_state_dict(torch.load(f'classificators/{model_string}/classificator_level_{level}.pt'))
            classificator.eval()
        elif model_string == "svm":
            classificator = joblib.load(f'classificators/{model_string}/classificator_level_{level}.pkl')
        

        arrays_classificators.append(classificator)



   ## Testing the classificator on the test set.
    test_loader = get_separated_dataloaders("dataset_embeddings", batch_size=batch_size, split='test')
    ds_test_real = test_loader['real']
    ds_test_fake_stylegan1 = test_loader['fake_stylegan1']

    ds_test = torch.utils.data.ConcatDataset([ds_test_real.dataset, ds_test_fake_stylegan1.dataset])
    data_test = torch.utils.data.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

    all_labels = []
    all_filenames = []
    all_types = []
    all_outputs = [[] for _ in levels]

    with torch.no_grad():
        for embeddings, labels,filename in data_test:
            all_labels.append(labels.cpu())
            all_filenames.extend(filename)

            types = ['real' if l == 0 else 'fake_stylegan1' for l in labels.cpu().numpy()]
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
    df.to_csv("test_results.csv", index=False)
    print("Test results saved to test_results.csv")



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation")
    parser.add_argument("--classificator_model",type=str, choices=["mlp","svm"], default="mlp", help="Type of classificator model to use")
    parser.add_argument("--create_embeddings", action='store_true', help="Flag to create embeddings")
    parser.add_argument("--mode", type=str, choices=["train", "test"], default="test", help="Mode: train or test the classificator")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--metrics", action='store_true', help="Flag to compute metrics after testing")

    args= vars(parser.parse_args())

    device = torch.device(args['device'])

    if args['create_embeddings']:
        create_embeddings()
        print("Embeddings created. Exiting.")
        sys.exit(0)

    model_string = args['classificator_model']

    if args['mode'] == "train":
        train_classificators(model_string, device, num_epochs=args['num_epochs'], batch_size=args['batch_size'])
    elif args['mode'] == "test":
        test_classificators_in_dataset(device, model_string, batch_size=args['batch_size'])

    if args['metrics']:
        from compute_metrics import compute_metrics, dict_metrics
        tab_metrics = compute_metrics("test_results.csv", "test_results.csv", dict_metrics['auc'])
        print(tab_metrics.to_string(float_format=lambda x: '%5.3f'%x))

        acc_05 = lambda label, score: sk_metrics.balanced_accuracy_score(label, score > 0.5)
        tab_acc = compute_metrics("test_results.csv", "test_results.csv", acc_05)
        print("\n=== Balanced Accuracy (threshold=0.5) ===")
        print(tab_acc.to_string(float_format=lambda x: '%5.3f'%x))



