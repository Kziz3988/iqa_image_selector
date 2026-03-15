import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from app.utils.paths import get_weight_path

class ResNetFeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        resnet = models.resnet50(weights=None)
        state_dict = torch.load(get_weight_path('resnet50.pth'), map_location=self.device)
        resnet.load_state_dict(state_dict)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove full connection layers
        self.model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])

    def extract(self, img_path):
        img = Image.open(img_path).convert('RGB')
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(x).squeeze().to(self.device).numpy()
        return features / np.linalg.norm(features)