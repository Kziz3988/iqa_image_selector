import torch
from PIL import Image
from torchvision import transforms
from app.utils.paths import get_weight_path
from app.models.VCRNet.IQASolver import demoIQA as VCRNetIQA

class BaseModel:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
    def predict(self, img_path):
        raise NotImplementedError

class VCRNet(BaseModel):
    def __init__(self):
        super().__init__()
        
        self.model = VCRNetIQA().to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def predict(self, img_path):
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(img_tensor)
        return float(score.item())

class IQAFactory:
    @staticmethod
    def get_model(name='VCRNet'):
        if name == 'VCRNet':
            return VCRNet()
        else:
            raise ValueError(f"Unknown IQA model: {name}")