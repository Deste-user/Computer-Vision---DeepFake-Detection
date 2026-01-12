import ClipBased-SyntheticImageDetection/networks/openclipnet.py as openclipnet
import torch
import torch.nn as nn
import open_clip
import sys
import os

repo_path = os.path.join(os.getcwd(), 'ClipBased-SyntheticImageDetection')
sys.path.append(repo_path)

from networks import openclipnet

levels = {0,5,10,15,20,23}
model = openclipnet.OpenClipLinear()


