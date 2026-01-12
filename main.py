import torch
import torch.nn as nn
#import open_clip
import sys
import os

real_data_FFHQ_path = "/oblivion/Datasets/FFHQ/images1024x1024"
fake_data_StyleGAN1_path = "/oblivion/Datasets/FFHQ/generated/ stylegan1-psi-0.5/images1024x1024"
fake_data_StableDiffusion_path = " /oblivion/Datasets/FFHQ/generate/sdv1_4/images1024x1024"
repo_path = os.path.join(os.getcwd(), 'ClipBased-SyntheticImageDetection')
sys.path.append(repo_path)

from networks import openclipnet

model = openclipnet.OpenClipLinear(layer_to_extract=23)





