import torch
import torch.nn as nn
#import open_clip
import sys
import os

real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = " /oblivion/Datasets/FFHQ/generate/sdv1_4/images1024x1024"
repo_path = os.path.join(os.getcwd(), 'ClipBased-SyntheticImageDetection')
sys.path.append(repo_path)

from networks import openclipnet

class DataLoaderEmbeddings(torch.utils.data.Dataset):
    def __init__(self, embeddings_file_path):
        print("Loading embeddings from:", embeddings_file_path)
        
        raw_data = torch.load(embeddings_file_path)
        self.embeddings = torch.stack([item['embeddings'].detach().cpu() for item in raw_data])
        self.labels = torch.tensor([item['label'] for item in raw_data], dtype=torch.long)

        print(f"Loaded {len(self.embeddings)} embeddings.")
        print(f"Embedding shape: {self.embeddings[0].shape}")
        print(f"Loaded labels: {len(self.labels)}")
    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]



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




def create_dataset_embeddings(img_dir, model, real):
    tensors = []
    model.to(device)
    model.eval()

    sorted_layer_keys = [f'block_{i}' for i in sorted(model.layers_to_extract)]
    with torch.no_grad():
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")): continue    
            img_path = os.path.join(img_dir, fname)
        
            try:
                img = openclipnet.load_image(img_path)
                img = img.unsqueeze(0).to(device)

                print(f"Processing image: {img_path}")
                features_dict = model.features(img)

                # Free the VRAM used by the image    
                layers_list = [features_dict[key].squeeze(0).cpu() for key in sorted_layer_keys if key in features_dict]
                stacked_embeddings = torch.stack(layers_list,dim=0)
                
                tensors.append({
                    "image": fname,
                    "label": 0 if real else 1,
                    "embeddings": stacked_embeddings
                })
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue    

            model.intermediate_features = {}
            # Save or process the embeddings as needed
    return tensors


def create_embeddings():
    levels = [1,3,5,7,9,11,13,15,17,19,21,23]
    #Install default the clip version 14 ViT-g-14
    model = openclipnet.OpenClipLinear(layer_to_extract=levels)

    real_imgs_db=os.listdir(real_data_FFHQ_path)    
    fake_imgs_db_stylegan1=os.listdir(fake_data_StyleGAN1_path)
    fake_imgs_db_stablediffusion=os.listdir(fake_data_StableDiffusion_path)

    classes = { "real": (real_data_FFHQ_path, 0),
        "fake_stylegan1": (fake_data_StyleGAN1_path, 1),
        "fake_stablediffusion": (fake_data_StableDiffusion_path, 1)}
    
    splits = ['train', 'val', 'test']

    for cls, (base_path, label) in classes.items():
        for split in splits:
            img_dir = os.path.join(base_path, split)
            out_dir = os.path.join("dataset_embeddings", cls, split)
            os.makedirs(out_dir, exist_ok=True)

            data = create_dataset_embeddings(img_dir, model, label)
            torch.save(data, os.path.join(out_dir, "embeddings.pt"))


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    

    create_embeddings()

    ## NOW we have to create a dataLoader to load these embeddings and train a classifier on top of them.
    train_loader = get_separated_dataloaders("dataset_embeddings", batch_size=64, split='train')
    val_loader = get_separated_dataloaders("dataset_embeddings", batch_size=64, split='val')
    test_loader = get_separated_dataloaders("dataset_embeddings", batch_size=64, split='test')
    
    
    ## Then we have to create a training script to train a classifier on top of these embeddings.
   
    ## We have to train a classificator for each level of embeddings extracted.


   ## Testing the classificator on the test set.



