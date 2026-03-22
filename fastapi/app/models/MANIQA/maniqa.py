import math
import os
import torch
import torch.nn as nn
import numpy as np
import random
import cv2
from torchvision import transforms
from .models.maniqa import MANIQA

class ImageData(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(ImageData, self).__init__()
        self.img_name = os.path.basename(image_path)
        with open(image_path, 'rb') as f:
            img_bytes = f.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        self.img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.transform = transform

        c, h, w = self.img.shape[2], self.img.shape[0], self.img.shape[1]
        new_h = 224
        new_w = 224

        scale = max(new_h / h, new_w / w)
        if scale > 1:
            new_size = (int(math.ceil(w * scale))), int(math.ceil((h * scale)))
            self.img = cv2.resize(self.img, new_size, interpolation=cv2.INTER_LINEAR)
            h, w = self.img.shape[:2]

        self.img = np.transpose(self.img, (2, 0, 1))
        c, h, w = self.img.shape

        self.img_patches = []
        for _ in range(num_crops):
            top = np.random.randint(0, h - new_h + 1) if h > new_h else 0
            left = np.random.randint(0, w - new_w + 1) if w > new_w else 0
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
class IQAModule(nn.Module):
    def __init__(self, config):
        super(IQAModule, self).__init__()
        self.net = MANIQA(
            embed_dim=config.embed_dim,
            num_outputs=config.num_outputs,
            dim_mlp=config.dim_mlp,
            patch_size=config.patch_size,
            img_size=config.img_size,
            window_size=config.window_size,
            depths=config.depths,
            num_heads=config.num_heads,
            num_tab=config.num_tab,
            scale=config.scale
        )

    def forward(self, x):
        return self.net(x)