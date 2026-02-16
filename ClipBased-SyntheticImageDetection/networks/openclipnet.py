'''                                        
Copyright 2024 Image Processing Research Group of University Federico
II of Naples ('GRIP-UNINA'). All rights reserved.
                        
Licensed under the Apache License, Version 2.0 (the "License");       
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at                    
                                           
    http://www.apache.org/licenses/LICENSE-2.0
                                                      
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,    
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.                         
See the License for the specific language governing permissions and
limitations under the License.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from .resnet_mod import ChannelLinear

dict_pretrain = {
    'clipL14openai'     : ('ViT-L-14', 'openai'),
    'clipL14laion400m'  : ('ViT-L-14', 'laion400m_e32'),
    'clipL14laion2B'    : ('ViT-L-14', 'laion2b_s32b_b82k'),
    'clipL14datacomp'   : ('ViT-L-14', 'laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K', 'open_clip_pytorch_model.bin'),
    'clipL14commonpool' : ('ViT-L-14', "laion/CLIP-ViT-L-14-CommonPool.XL-s13B-b90K", 'open_clip_pytorch_model.bin'),
    'clipaL14datacomp'  : ('ViT-L-14-CLIPA', 'datacomp1b'),
    'cocaL14laion2B'    : ('coca_ViT-L-14', 'laion2b_s13b_b90k'),
    'clipg14laion2B'    : ('ViT-g-14', 'laion2b_s34b_b88k'),
    'eva2L14merged2b'   : ('EVA02-L-14', 'merged2b_s4b_b131k'),
    'clipB16laion2B'    : ('ViT-B-16', 'laion2b_s34b_b88k'),
}

# It takes the corners and four patch central
def extract_corner_center_tokens(output,grid_size = 14):
    #(batch_size, seq_len, dim)
    batch_size = output.shape[0]
    seq_len = output.shape[1]
    dim = output.shape[2]
    
    # Validate the sequence length
    expected_seq_len = 1 + grid_size * grid_size
    if seq_len != expected_seq_len:
        #print(f"Warning: expected seq_len={expected_seq_len}, got {seq_len}. Attempting to adjust grid_size.")
        # Auto-detect grid_size from seq_len
        grid_size = int((seq_len - 1) ** 0.5)
    
    # Skip the CLS token (index 0), patches start from index 1
    patches = output[:, 1:, :].reshape(batch_size, grid_size, grid_size, dim)
    
    # Extract 4 corners
    top_left = patches[:, 0, 0, :]
    top_right = patches[:, 0, -1, :]
    bottom_left = patches[:, -1, 0, :]
    bottom_right = patches[:, -1, -1, :]
    
    # Extract 4 center patches
    center_start = grid_size // 2 - 1
    center_end = grid_size // 2 + 1
    
    center_top_left = patches[:, center_start, center_start, :]
    center_top_right = patches[:, center_start, center_end - 1, :]
    center_bottom_left = patches[:, center_end - 1, center_start, :]
    center_bottom_right = patches[:, center_end - 1, center_end - 1, :]
    
    selected_tokens = torch.stack([top_left, top_right, bottom_left, bottom_right,
                                   center_top_left, center_top_right, center_bottom_left, center_bottom_right],
                                  dim=1)
    return selected_tokens.reshape(batch_size, -1)
    

    
    


class OpenClipLinear(nn.Module):
    def __init__(self, num_classes=1, pretrain='clipL14commonpool', normalize=True, next_to_last=False,
                  layer_to_extract=None, token_mode='cls'):
        
        super(OpenClipLinear, self).__init__()

        # Load backbone or download all pretrained weights
        if len(dict_pretrain[pretrain])==2:
            backbone = open_clip.create_model(dict_pretrain[pretrain][0], pretrained=dict_pretrain[pretrain][1])
        else:
            from huggingface_hub import hf_hub_download
            backbone = open_clip.create_model(dict_pretrain[pretrain][0], pretrained=hf_hub_download(*dict_pretrain[pretrain][1:]))
        
        # If next_to_last is True, remove the final projection layer
        if next_to_last:
            self.num_features = backbone.visual.proj.shape[0]
            backbone.visual.proj = None
        else:
            self.num_features = backbone.visual.output_dim
        
        # Put the backbone in a list  to not optimize its parameters
        self.bb = [backbone, ]
        self.normalize = normalize
        self.token_mode = token_mode

        self.layers_to_extract = layer_to_extract
        self.intermediate_features = {}

        def get_activation(name):
            def hook(model,input,output):
                if self.token_mode == 'cls':
                    #The output is [seq_len, batch, dim]
                    self.intermediate_features[name] = output[:,0,:].detach().cpu()
                elif self.token_mode == 'corners_centers':
                    self.intermediate_features[name] = extract_corner_center_tokens(output, grid_size=14).detach().cpu()
                else:
                    raise ValueError(f"Unknown token_mode: {self.token_mode}")
            return hook
        
        if self.layers_to_extract is not None:
            for i in self.layers_to_extract:
                if i < len(self.bb[0].visual.transformer.resblocks):
                    # The register forward hook registers a "hook" to extract the output of the layer
                    self.bb[0].visual.transformer.resblocks[i].register_forward_hook(get_activation(f'block_{i}'))
        

        if self.token_mode == 'cls':
            input_dim = self.num_features
        elif self.token_mode == 'corners_centers':
            input_dim = 8 * self.num_features

        # Define the classification head, this is the piece to be trained
        self.fc = ChannelLinear(input_dim, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)

    #To ensure that the backbone and all are also moved to the right device
    def to(self, *args, **kwargs):
        self.bb[0].to(*args, **kwargs)
        super(OpenClipLinear, self).to(*args, **kwargs)
        return self

    
    def forward_features(self, x):
        # No gradient for the backbone
        with torch.no_grad():
            #Go in the eval mode
            self.bb[0].eval()
            # Pass through all the backbone
            self.bb[0].encode_image(x, normalize=self.normalize)
            return self.intermediate_features

    #This try to guess
    def forward_head(self, x):
        return self.fc(x)

    # Take the input x and pass it through the whole network, then return the output - True/False
    def forward(self, x):
        features = self.forward_features(x)
        if isinstance(features, dict):
            return features['final']
        return features 
